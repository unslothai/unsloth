# Fast iteration (Studio terminal output / setup)

You do **not** need to re-run full `unsloth studio setup` every time you change messages or colors.

## `unsloth studio setup` uses the **installed** `studio/setup.sh`

If you edit `studio/setup.sh` in a git clone but run `unsloth studio setup` from a **PyPI** install, you still get the old script until you reinstall from source, e.g.:

```bash
cd /path/to/unsloth   # package root containing pyproject.toml
uv pip install -e .    # or: pip install -e .
unsloth studio setup
```

Verbose setup (full pip output + logs when optional steps like llama.cpp fail):

```bash
unsloth studio setup --verbose
# or: UNSLOTH_VERBOSE=1 ./studio/setup.sh
```

## Easiest tests (copy-paste)

| What you’re tweaking | Command (from repo root: `.../unsloth/`) |
|----------------------|------------------------------------------|
| **Launch / URLs** (after server starts) | `python studio/backend/print_terminal_harness.py` |
| **Install UI** (progress bar, titles, colors in `install_python_stack.py`) | `python studio/backend/preview_install_output.py` |
| **Colors in a pipe / `less`** | `python studio/backend/preview_install_output.py --color on \| less -R` |
| **Plain text (no ANSI)** | `python studio/backend/preview_install_output.py --color off` |

`preview_install_output.py` imports the real `_progress` / `_title` / `_green` from `install_python_stack.py`, so it matches production. It does **not** run pip.

- **`--python-only`** — only the pip-style progress + success line.
- **`--setup-only`** — only the fake `setup.sh`-style box / sections (strings live in the preview script; tweak `setup.sh` itself for exact match).

## Harness (fastest): preview terminal output with no server

**stdlib only** — edits to `studio/backend/startup_banner.py` show up immediately.

From the `unsloth` repo root:

```bash
python studio/backend/print_terminal_harness.py
python studio/backend/print_terminal_harness.py --scenario loopback
python studio/backend/print_terminal_harness.py --port-was-busy --port 8001 --original-port 8000
python studio/backend/print_terminal_harness.py --with-reexec --with-frontend
```

- **`--color on`** sets `FORCE_COLOR=1` (use with `less -R` or CI logs if needed).
- **`--color off`** sets `NO_COLOR=1`.
- **`--scenario`**: `default` | `loopback` | `lan-share` | `custom-bind` | `minimal` (banner only).

The real server uses the same strings via `startup_banner.py` (imported from `run.py`).

## Python (full stack check)

1. Use an **editable** install from your repo once:
   `uv pip install -e .` (or `pip install -e .`) from the `unsloth` package root.
2. Edit `startup_banner.py`, `run.py`, or `unsloth_cli/commands/studio.py`, then **restart only** the server:
   `unsloth studio -p 8000`  
   No reinstall step for text/color tweaks.

## `setup.sh` / `install_python_stack.py`

- **Shell syntax:** `bash -n studio/setup.sh`
- **Installer copy/colors:** run only the Python installer (after your venv exists and matches what setup would use):

  ```bash
  # example: same venv setup.sh uses
  ./studio/.venv/bin/python ./studio/install_python_stack.py
  ```

  Adjust the path to the venv you are testing.

- **Full setup:** only when you change steps that fetch/build dependencies (Node, llama.cpp, etc.), not for echo/color tweaks.

## CI / logs

Set `NO_COLOR=1` in environments where escape codes are undesirable.
