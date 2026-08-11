/* minimal crt0: set stack and jump to main */
    .section .text.start, "ax"
    .globl _start
_start:
    la   sp, __stack_top
    call main
1:  j    1b

/* weak: if a kernel pulls in -lgloss_htif (e.g. via printf), its strong
 * tohost/fromhost definitions win instead of conflicting with these -- the
 * linker script maps both conventions' sections to the same place either way. */
.section ".tohost","aw",@progbits
.align 6
.weak tohost
tohost: .dword 0
.align 6
.weak fromhost
fromhost: .dword 0
