// Building for aarch64-pc-windows-msvc needs clang on PATH: ring and aws-lc-sys assemble their
// aarch64 sources through cc-rs, which reaches for clang rather than cl.exe, and the failure
// names ring rather than the missing toolchain. No other leg builds for aarch64 Windows.
fn main() {
    println!("cargo:rerun-if-changed=windows/app-manifest.xml");
    let windows = tauri_build::WindowsAttributes::new()
        .app_manifest(include_str!("windows/app-manifest.xml"));
    tauri_build::try_build(tauri_build::Attributes::new().windows_attributes(windows))
        .expect("failed to build Tauri application resources");
}
