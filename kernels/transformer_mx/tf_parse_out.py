#!/usr/bin/env python3
"""tf_parse_out.py <uartlog> [--hidden .. --expansion .. --heads .. --seq .. --tile_n .. --occupancy ..]

Parse the TFDUMP hex lines host.cpp writes after a transformer_mx run and compare every phase
buffer (q, k, v, ctx, res, ff, out) against golden_tf.py's golden, cell by cell. Reports:
  * per phase: exact / lost (device 0, golden != 0) / non-finite / max abs error over finite
    cells vs golden_tf._phase_tol -- and the FIRST phase out of tolerance, which is where the bug
    enters (every later phase inherits it)
  * for that first bad phase: a per-row table with the owning SIMT thread and warp (every epilogue
    in transformer_mx_impl.hpp gives thread t rows {t, t+thr, ...}; warp = thread // lanes), and
    which columns are affected -- e.g. whole output tiles (TN columns) point at one GEMM tile
Legacy "TFOUT <row> ..." lines (output only) are read as phase "out".
Dims default to the tiny config the checked-in tf_data.h was generated with.
"""
import argparse
import pathlib
import re
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import golden_tf as g                      # noqa: E402
from gen_data import build_operands        # noqa: E402

LANES = 8                                   # MU_NUM_THREADS in lib/include/mu_intrinsics.h
PHASES = ["q", "k", "v", "ctx", "res", "ff", "out"]


def parse_dumps(text):
    rows = {}
    for m in re.finditer(r"^TF(?:DUMP (\w+)|OUT) (\d+)((?: [0-9a-f]{4})+)\s*$", text, re.M):
        name = m.group(1) or "out"
        rows.setdefault(name, {})[int(m.group(2))] = [int(v, 16) for v in m.group(3).split()]
    return rows


def compare(dev_rows, gold):
    R, C = gold.shape
    dev = np.full((R, C), -1, dtype=np.int64)
    for r, vals in dev_rows.items():
        if r < R and len(vals) == C:
            dev[r] = vals
    have = dev >= 0
    d = np.where(have, dev, 0).astype(np.uint16)
    gold = gold.astype(np.int64)
    nonfinite = have & (((d >> 7) & 0xFF) == 0xFF)
    lost = have & ((d & 0x7FFF) == 0) & ((gold & 0x7FFF) != 0)
    exact = have & (dev == gold)
    fin = have & ~nonfinite
    err = np.zeros((R, C))
    err[fin] = np.abs(g.from_bf16(d[fin]).astype(np.float64)
                      - g.from_bf16(gold[fin].astype(np.uint16)))
    return dict(have=have, nonfinite=nonfinite, lost=lost, exact=exact, err=err)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("uartlog")
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--expansion", type=int, default=256)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--seq", type=int, default=32)
    ap.add_argument("--tile_n", type=int, default=64)
    ap.add_argument("--occupancy", type=int, default=2, help="mu_schedule occupancy of the run")
    a = ap.parse_args()

    text = pathlib.Path(a.uartlog).read_text(errors="replace")
    for line in text.splitlines():
        if line.startswith("[host] done="):
            print(line)
    dumps = parse_dumps(text)
    if not dumps:
        sys.exit("no TFDUMP/TFOUT lines found in " + a.uartlog)

    op = build_operands(a.hidden, a.expansion, a.heads, a.seq, a.tile_n)
    golden = g.encoder_layer(op)

    print(f"\n{'phase':5} {'rows':>5} {'exact':>6} {'lost':>5} {'nonfin':>6} {'maxerr':>9} {'tol':>8}  verdict")
    first_bad, results = None, {}
    for ph in PHASES:
        if ph not in dumps:
            print(f"{ph:5} {'-':>5}  not dumped")
            continue
        gold = golden[ph]
        res = compare(dumps[ph], gold)
        results[ph] = res
        tol = g._phase_tol(ph, g.from_bf16(gold))
        nrows = int(res["have"].any(axis=1).sum())
        ok = (nrows == gold.shape[0] and res["nonfinite"].sum() == 0 and res["lost"].sum() == 0
              and res["err"].max() <= tol)
        if not ok and first_bad is None:
            first_bad = ph
        print(f"{ph:5} {nrows:5d} {int(res['exact'].sum()):6d} {int(res['lost'].sum()):5d} "
              f"{int(res['nonfinite'].sum()):6d} {res['err'].max():9.4f} {tol:8.4f}  "
              f"{'OK' if ok else 'BAD'}")

    if first_bad is None:
        print("\nALL DUMPED PHASES WITHIN TOLERANCE")
        return
    res = results[first_bad]
    thr = a.occupancy * LANES
    print(f"\nFIRST BAD PHASE: {first_bad}  (threads = {thr}; row r -> thread r % {thr} -> warp // {LANES})")
    print(f"{'row':>3} {'thr':>3} {'warp':>4} {'exact':>5} {'lost':>4} {'nonfin':>6} {'maxerr':>9}")
    for r in range(res["have"].shape[0]):
        if not res["have"][r].any():
            print(f"{r:3d}  (row missing from dump)")
            continue
        t = r % thr
        print(f"{r:3d} {t:3d} {t // LANES:4d} {int(res['exact'][r].sum()):5d} "
              f"{int(res['lost'][r].sum()):4d} {int(res['nonfinite'][r].sum()):6d} {res['err'][r].max():9.4f}")
    bad = res["lost"] | res["nonfinite"] | (res["err"] > g._phase_tol(first_bad, g.from_bf16(golden[first_bad])))
    cols = np.nonzero(bad.any(axis=0))[0]
    print(f"\ncolumns with bad cells: {len(cols)} of {bad.shape[1]} -> {cols[:32].tolist()}"
          f"{' ...' if len(cols) > 32 else ''}")
    tn = a.tile_n
    print(f"bad cells per {tn}-column output tile: "
          f"{[int(bad[:, j:j + tn].sum()) for j in range(0, bad.shape[1], tn)]}")


if __name__ == "__main__":
    main()
