"""Colab cell-magic compatibility for the Unsloth Docker notebooks.

Colab cells often look like:

    #@title Colab Extra Install { display-mode: "form" }
    %%capture
    !pip install ...

In IPython a cell magic (`%%capture`, `%%bash`, ...) is only recognised when it
is the VERY FIRST line of the cell. A leading Colab `#@title`/`#@param` form (or
any comment/blank line) pushes the `%%magic` to line 2, so IPython treats it as a
line magic and raises `UsageError: Line magic function `%%capture` not found.`
and the cell fails.

Fix: register an `input_transformers_cleanup` (runs before magic detection) that
hoists a `%%` cell magic above any leading blank/comment (`#...`, incl. `#@...`)
lines, so the magic lands on line 0 and fires normally. The skipped comment lines
stay in the cell (still inert), just below the magic -- so `%%capture` now also
captures them. Idempotent and fully guarded: any problem returns the input
unchanged, so a cell never breaks because of this helper.

This mirrors unsloth_nb_compat.register_ipython(): it is wired from the baked
IPython startup file (docker/unsloth_ipython_startup.py).
"""

from __future__ import annotations
import sys


def colab_cell_magic_fix(lines):
    """Hoist a `%%` cell magic above leading blank/comment lines.

    `lines` is the IPython cell as a list of strings (each ending in '\\n').
    Returns a (possibly reordered) list of the same lines.
    """
    try:
        skipped = []
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped == "" or stripped.startswith("#"):
                skipped.append(line)  # blank or comment (incl. #@title)
                continue
            # First real line. Only act if it is a cell magic that is not yet on
            # top (i.e. something was skipped before it).
            if stripped.startswith("%%") and i > 0:
                return [line] + skipped + lines[i + 1 :]
            return lines  # already on top, or not a magic
        return lines  # all blank/comment -> nothing to do
    except Exception:
        return lines


def register_ipython():
    """Append the transformer to the running IPython (called from startup)."""
    try:
        ip = get_ipython()  # noqa: F821 (provided by IPython)
    except NameError:
        return
    if ip is None or getattr(ip, "_unsloth_colab_fix", False):
        return
    try:
        ip.input_transformers_cleanup.append(colab_cell_magic_fix)
        ip._unsloth_colab_fix = True
    except Exception as e:  # never break a kernel because of the helper
        print(f"[unsloth-nb] colab-compat hook skipped: {e!r}", file = sys.stderr)
