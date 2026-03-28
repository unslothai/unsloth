# Tauri App — Temporary workarounds

## --api-only flag (process.rs)

**Status:** Gated behind `cfg!(debug_assertions)` — only works in dev builds.

**Why:** The `--api-only` CLI flag was added on the `tauri-app` branch but hasn't been published to PyPI yet. Production builds install from PyPI, so the binary doesn't recognize the flag and exits with an error.

**Impact without it:**
- Port discovery falls back to scanning 8888-8908 (~2s slower instead of instant `TAURI_PORT=` output)
- Backend serves its own web frontend alongside Tauri's bundled one (harmless, just wasteful)
- CORS is not restricted to Tauri origins (less strict but functional)

**To fix:** Once a new `unsloth` version with `--api-only` is on PyPI:
1. In `studio/src-tauri/src/process.rs`, change `let use_api_only = cfg!(debug_assertions)` to `let use_api_only = true`
2. Remove this note
