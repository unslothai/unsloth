# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-Present the Unsloth team. See /studio/LICENSE.AGPL-3.0

"""Colab cell-magic compatibility for the Unsloth Docker notebooks.

IPython recognises a cell magic only on the VERY FIRST line, so a leading Colab
`#@title` form pushes `%%capture` to line 2 and the cell dies. An
`input_transformers_cleanup`, which runs before magic detection, hoists it back.

Restricted to magics whose body runs as code, where the moved-down comment stays
inert; a content magic would write or render it.
"""

from __future__ import annotations
import sys


_SAFE_CELL_MAGICS = frozenset(
    {
        "capture",
        "time",
        "timeit",
        "prun",
        "debug",
        "bash",
        "sh",
        "shell",
        "python",
        "python2",
        "python3",
        "pypy",
    }
)


def colab_cell_magic_fix(lines):
    """Hoist a safe `%%` cell magic above leading blank/comment lines."""
    try:
        skipped = []
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped == "" or stripped.startswith("#"):
                skipped.append(line)
                continue
            if stripped.startswith("%%") and i > 0:
                name = stripped[2:].split(maxsplit = 1)
                name = name[0] if name else ""
                if name in _SAFE_CELL_MAGICS:
                    return [line] + skipped + lines[i + 1 :]
                # content/data magic: moving the comment into its body would render it
                return lines
            return lines
        return lines
    except Exception:
        return lines


def register_ipython():
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
