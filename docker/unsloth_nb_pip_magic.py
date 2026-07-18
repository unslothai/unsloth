# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-Present the Unsloth team. See /studio/LICENSE.AGPL-3.0

"""Route notebook `%pip` / `%uv` / `python -m pip` installs through the shim.

The PATH shim (/opt/unsloth-nb/bin/{pip,pip3,uv} -> unsloth_pip_shim.py) only
intercepts `!pip` / `!uv` shell cells. IPython's `%pip` / `%uv` LINE MAGICS run
pip in-process, and `python -m pip` runs pip as a module -- both bypass PATH, so
a notebook could still reinstall torch / transformers / vLLM and clobber the
baked cu128 stack the shim is meant to protect.

This closes that gap two ways, with no clobbering of the shell-escape path:
  * `%pip` / `%pip3` / `%uv` are re-registered as line magics that delegate to
    the shell (`get_ipython().system("pip ...")`); since /opt/unsloth-nb/bin is
    first on PATH, that resolves to the shim. Overriding the real magic (rather
    than rewriting cell text) means we only act when IPython actually dispatches
    the magic -- a `%pip` inside a string is left untouched.
  * a narrow input transformer rewrites an explicit `!python -m pip` /
    `!python -m uv` shell line to `!pip` / `!uv`, so that form hits the shim too.

UNSLOTH_NB_SHIM=1 is already exported by the startup hook and inherited by the
subprocess, so the shim applies. Safe no-op outside IPython.
"""

import re

# Only the explicit `!<python> -m pip|uv ...` shell form. Input transformers see
# the RAW cell text (IPython expands `{sys.executable}` later), so the braced form
# (`!{sys.executable} -m pip install ...`) and absolute interpreter paths, quoted
# or bare, must be matched here too or module-pip bypasses the PATH shim.
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
    """`!python -m pip install X` -> `!pip install X` (so it hits the PATH shim)."""
    try:
        out = []
        for line in lines:
            body = line.rstrip("\n")
            tail = line[len(body) :]  # preserve the trailing newline(s), if any
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
            # /opt/unsloth-nb/bin is first on PATH, so `pip`/`uv` here is the shim.
            return ip.system(tool + " " + line)

        return _magic

    # Override the built-in %pip / %uv so they route through the shim too.
    ip.register_magic_function(_make("pip"), "line", "pip")
    ip.register_magic_function(_make("pip"), "line", "pip3")
    ip.register_magic_function(_make("uv"), "line", "uv")

    if _rewrite_python_dash_m not in ip.input_transformers_cleanup:
        ip.input_transformers_cleanup.append(_rewrite_python_dash_m)

    ip._unsloth_pip_magic = True
