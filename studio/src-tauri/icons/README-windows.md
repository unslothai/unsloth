# Windows desktop icon

`icon.png` (1024 px) is the original, unchanged brand artwork. `windows-icon.png`
is its Windows-only high-resolution variant with a modestly rounder tile; the
mascot pixels remain unchanged. The `windows-{16,24,32,48,64,256}.png` assets
are independently rasterized native-size frames embedded byte-for-byte as PNG
entries in `icon.ico`. Both the Tauri app and NSIS installer reference that ICO.
The macOS/Linux icons and tray icons are untouched.

Regenerate using Python and Pillow 12.3.0:

```sh
uv run --with pillow==12.3.0 python studio/src-tauri/icons/generate_windows_icon.py
uv run --with pillow==12.3.0 --with pytest python -m pytest studio/src-tauri/tests/test_windows_icon.py
```

The test decodes actual ICO entries, checks transparent corners and recognizable
mascot contrast at 16/24/32 px, compares their embedded PNG bytes to the
per-size assets, and regenerates into a temporary directory to detect drift.
`windows-icon-clarity.yml` separately builds a real Windows executable/NSIS
installer and uses Win32 icon resource extraction on a GitHub-hosted runner.
Its PNG artifact is extracted Windows-resource evidence, not a taskbar screenshot.
