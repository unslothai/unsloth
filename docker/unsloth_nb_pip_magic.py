# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-Present the Unsloth team. See /studio/LICENSE.AGPL-3.0

"""Route notebook `%pip` / `%uv` / `python -m pip` installs through the shim.

The PATH shim only intercepts `!pip` shell cells, but `%pip` runs pip in-process and
`python -m pip` runs it as a module. `%pip` / `%pip3` / `%uv` are re-registered as
line magics that delegate to the shell -- overriding the MAGIC, not rewriting cell
text, so a `%pip` inside a string is untouched -- and an input transformer rewrites
an explicit `!python -m pip` line to `!pip`.
"""

import re

# Transformers see the RAW cell text (IPython expands `{sys.executable}` later), so
# the braced form and quoted/bare interpreter paths must be matched here too.
_PY_M_PIP = re.compile(
    r"""^(\s*)!\s*
    (?:
        (?:python[0-9.]*|py)                        # literal python / py
      | ["']?\{\s*sys\.executable\s*\}["']?         # {sys.executable}, opt. quoted
      | "(?:[^"]*[/\\])python[0-9.]*(?:\.exe)?"     # quoted interpreter path
      | '(?:[^']*[/\\])python[0-9.]*(?:\.exe)?'
      | \S*[/\\]python[0-9.]*(?:\.exe)?             # bare interpreter path
    )
    \s+-m\s+(pip|uv)\b(.*)$""",
    re.VERBOSE,
)


def _rewrite_python_dash_m(lines):
    """`!python -m pip install X` -> `!pip install X`, which hits the PATH shim."""
    try:
        out = []
        for line in lines:
            body = line.rstrip("\n")
            tail = line[len(body) :]  # keep the trailing newline(s)
            m = _PY_M_PIP.match(body)
            if m:
                out.append(m.group(1) + "!" + m.group(2) + m.group(3) + tail)
            else:
                out.append(line)
        return out
    except Exception:
        return lines


def register_ipython():
    try:
        ip = get_ipython()  # noqa: F821 (provided by IPython)
    except Exception:
        ip = None
    if ip is None or getattr(ip, "_unsloth_pip_magic", False):
        return

    def _make(tool):
        def _magic(line):
            return ip.system(tool + " " + line)

        return _magic

    ip.register_magic_function(_make("pip"), "line", "pip")
    ip.register_magic_function(_make("pip"), "line", "pip3")
    ip.register_magic_function(_make("uv"), "line", "uv")

    if _rewrite_python_dash_m not in ip.input_transformers_cleanup:
        ip.input_transformers_cleanup.append(_rewrite_python_dash_m)

    ip._unsloth_pip_magic = True
