// Building this on Windows on ARM (aarch64-pc-windows-msvc) needs clang on PATH, and
// MSVC BuildTools alone is not enough. ring and aws-lc-sys assemble their aarch64
// sources through cc-rs, which reaches for clang rather than cl.exe on that target, so
// without it the build dies inside ring with
//
//     ring@0.17.14: failed to find tool "clang": program not found
//
// which names the crate and not the missing toolchain. Install LLVM and put its bin
// directory on PATH. The x64 Windows, macOS and Linux legs do not need it, which is why
// no workflow installs it: none of them build for aarch64 Windows today.
fn main() {
    tauri_build::build()
}
