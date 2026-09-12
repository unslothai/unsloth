// Building for aarch64-pc-windows-msvc needs clang on PATH: ring and aws-lc-sys assemble their
// aarch64 sources through cc-rs, which reaches for clang rather than cl.exe, and the failure
// names ring rather than the missing toolchain. No other leg builds for aarch64 Windows.
fn main() {
    tauri_build::build()
}
