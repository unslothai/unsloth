from __future__ import annotations

import argparse
import importlib
import json
import sys
import traceback


def _load_callable(path: str):
    if ":" not in path:
        raise ValueError("Expected --func 'package.module:callable_name'")
    mod_name, attr = path.split(":", 1)
    mod = importlib.import_module(mod_name)
    try:
        return getattr(mod, attr)
    except AttributeError as e:
        raise AttributeError(f"{path!r} not found") from e


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a callable in a clean subprocess")
    parser.add_argument("--func", required=True, help="Import path: 'pkg.mod:callable'")
    args = parser.parse_args()

    # Payload format: {"args": [...], "kwargs": {...}}
    try:
        payload = json.load(sys.stdin) if not sys.stdin.isatty() else {}
    except json.JSONDecodeError:
        payload = {}

    call_args = payload.get("args", []) or []
    call_kwargs = payload.get("kwargs", {}) or {}

    func = _load_callable(args.func)

    try:
        # Call user code; allow it to print to stdout/stderr freely.
        func(*call_args, **call_kwargs)
        # Success -> exit 0 (no extra prints so your function's stdout stays clean).
        sys.exit(0)
    except SystemExit:
        # Re-raise so explicit sys.exit propagates
        raise
    except Exception:
        # On failure, send a traceback to stderr for the parent to capture.
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
