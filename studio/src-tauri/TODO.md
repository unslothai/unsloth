# Tauri App — Notes

## --api-only flag (process.rs)

**Status:** Auto-detected at runtime via `supports_api_only()`.

Dev builds always use `--api-only`. Production builds probe the installed binary's
`--help` output and enable it if supported. No manual change needed when PyPI ships
the flag.
