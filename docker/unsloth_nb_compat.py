# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-Present the Unsloth team. See /studio/LICENSE.AGPL-3.0

"""Per-notebook transformers version activation for the Unsloth Docker image.

The image bakes coherent transformers "sidecars" and prepends one to sys.path before
transformers is imported. Mirrors studio/backend/utils/transformers_version.py.

Selection has a FLOOR as well as a ceiling (see sidecar_for): vLLM is version-locked
to transformers, so a sidecar older than the baked vLLM accepts is not an older
transformers, it is an ImportError at `import unsloth`.
"""

from __future__ import annotations
import os, sys, glob, json, re

SIDECAR_ROOT = os.environ.get("UNSLOTH_TF_SIDECAR_ROOT", "/opt/unsloth-venv/tf-sidecars")
MARKER = os.environ.get("UNSLOTH_NB_TF_MARKER", "/tmp/unsloth_nb/requested_transformers")

# Lowest transformers the baked vLLM can import. Written by the Dockerfile's sidecar
# verification step, so it tracks the vLLM the image bakes rather than a literal.
FLOOR_FILE = os.path.join(SIDECAR_ROOT, ".vllm_min_transformers")


def _logging_enabled() -> bool:
    """Sidecar activation is silent by default; UNSLOTH_ENABLE_LOGGING=1 surfaces it."""
    return os.environ.get("UNSLOTH_ENABLE_LOGGING", "").strip().lower() not in (
        "",
        "0",
        "false",
        "no",
        "off",
    )


# fallback when a notebook names a new model but pins nothing; ported from Studio
_TIER_SUBSTRINGS = {
    "5.10.2": ("gemma-4-12b", "gemma4-12b"),
    "5.5.0": ("gemma-4", "gemma4", "qwen3.6"),
    "5.3.0": (
        "ministral-3",
        "glm-4.7-flash",
        "qwen3-30b-a3b",
        "qwen3.5",
        "qwen3-next",
        "qwen3_5",
        "lfm2.5-vl",
    ),
}


def _baked():
    out = {}
    for d in sorted(glob.glob(os.path.join(SIDECAR_ROOT, "t_*"))):
        out[os.path.basename(d)[2:].replace("_", ".")] = d
    return out


def min_version():
    """Lowest transformers this image's vLLM can import; None keeps the pre-floor
    behaviour. UNSLOTH_TF_SIDECAR_MIN overrides."""
    v = os.environ.get("UNSLOTH_TF_SIDECAR_MIN", "").strip()
    if v:
        return v
    try:
        with open(FLOOR_FILE) as f:
            return f.read().strip() or None
    except OSError:
        return None


def _eligible():
    baked = _baked()
    if not baked:
        return []
    try:
        from packaging.version import Version
    except Exception:
        return None
    floor = min_version()
    try:
        low = Version(floor) if floor else None
    except Exception:
        low = None
    rows = []
    for v, d in baked.items():
        try:
            ver = Version(v)
        except Exception:
            continue
        if low is not None and ver < low:
            continue  # vLLM cannot import it
        rows.append((ver, v, d))
    rows.sort()
    return rows


def tier_for_model(model_name: str):
    if not model_name:
        return None
    low = model_name.lower()
    # newest tiers first, so gemma-4-12b wins over gemma-4
    for ver in ("5.10.2", "5.5.0", "5.3.0"):
        if any(s in low for s in _TIER_SUBSTRINGS[ver]):
            return ver
    return None


def sidecar_for(version: str):
    """FLOOR then CEILING: a sidecar the baked vLLM cannot import is never eligible
    whatever the notebook pinned, so a request below the floor clamps UP; among the
    eligible ones take the smallest >= the request, since a model added in X needs at
    least X. Above every eligible sidecar returns None, i.e. the base venv."""
    if not version:
        return None
    rows = _eligible()
    if rows is None:  # no packaging: only an exact, still-eligible match is safe
        baked = _baked()
        d = baked.get(version)
        floor = min_version()
        return d if (d and (not floor or version == floor)) else None
    if not rows:
        return None
    for _ver, v, d in rows:
        if v == version:
            return d
    try:
        from packaging.version import Version
        want = Version(version)
    except Exception:
        return None
    for ver, _v, d in rows:
        if ver >= want:
            return d
    return None


def requested_version():
    try:
        with open(MARKER) as f:
            v = f.read().strip()
        return v or None
    except OSError:
        return None


# --- install-cell pin scanning -------------------------------------------------
# Lives here rather than in unsloth_run because this module is the one copied into
# site-packages, so it is the only one a kernel can import; unsloth_run reads these
# back out. One implementation, so the headless path and the IPython hook cannot
# disagree about what an install line is.

# The requirement NAME is matched loosely and normalised afterwards, because pip
# accepts every PEP 503 spelling of it and unsloth_pip_shim canonicalises before it
# decides what to drop. `Transformers==5.5.0` and `transformers[torch]==5.5.0` install
# exactly what the canonical spelling installs and the shim drops all three, so a
# scanner that only knew the canonical form left the pin unseen while the install was
# still suppressed. The cell then imported the base version, and that import freezes
# the sidecar choice for the life of the kernel with nothing left to correct it.
_PIN_RE = re.compile(
    r"(?<![\w.\-])([A-Za-z0-9][A-Za-z0-9._\-]*)[ \t]*(?:\[[^\]]*\])?[ \t]*=="
    r"[ \t]*([0-9][0-9A-Za-z.\-]*)"
)


def _norm_req(name):
    """PEP 503 normalised distribution name, the rule unsloth_pip_shim._norm_name uses."""
    return re.sub(r"[-_.]+", "-", name.strip()).lower()


def pin_from(text):
    """transformers version pinned by an install command in `text`, else None."""
    for m in _PIN_RE.finditer(_install_lines(text)):
        if _norm_req(m.group(1)) == "transformers":
            return m.group(2)
    return None


# Only an actual install invocation may supply the pin: the pin outranks the model
# tier, so a commented-out install line would pick the wrong sidecar with nothing
# running afterwards to correct it.
_INSTALL_RE = re.compile(
    r"""^[ \t]*(?![ \t]*\#)[!%]?[ \t]*
        (?: uv (?:[ \t]+-{1,2}\S+)* [ \t]+ )?
        (?: \S+ [ \t]+ -m [ \t]+ )?
        pip[0-9.]* [ \t]+
        (?: -{1,2}\S+ [ \t]+ )*
        install (?: [ \t] | $ )""",
    re.VERBOSE,
)


def _strip_comment(line):
    """Drop a trailing `# ...` from a shell/magic line. Only a `#` starting a token
    counts, so a `git+https://...#egg=` fragment survives, as does a quoted #."""
    quote = None
    for i, ch in enumerate(line):
        if quote:
            if ch == quote:
                quote = None
        elif ch in "'\"":
            quote = ch
        elif ch == "#" and (i == 0 or line[i - 1].isspace()):
            return line[:i]
    return line


# A triple-quoted body is data, not code. Blanked keeping newlines, so the
# continuation logic below is unaffected.
_TRIPLE_RE = re.compile(r'("""|\'\'\')(?:.|\n)*?\1')


def _live_source(src):
    """Triple-quoted bodies and comments blanked out. Single-quoted strings stay
    intact: the model name unsloth_run scans for lives inside one."""
    blanked = _TRIPLE_RE.sub(lambda m: re.sub(r"[^\n]", " ", m.group(0)), src)
    return "\n".join(_strip_comment(line) for line in blanked.splitlines())


def _install_lines(src):
    kept, cont = [], False
    for line in src.splitlines():
        if cont or _INSTALL_RE.match(line):
            code = _strip_comment(line)
            kept.append(code)
            cont = code.rstrip().endswith("\\")
        else:
            cont = False
    return "\n".join(kept)


def pin_in_cell(source):
    """transformers version pinned by an install command IN this cell, else None.

    Same scan unsloth_run runs over a whole notebook, applied to one cell. Anything
    that installs nothing has to lose: once transformers is imported the sidecar
    choice is frozen for the life of the kernel, so a wrong guess is unrecoverable
    while no guess merely leaves the base venv."""
    if not source:
        return None
    return pin_from(_live_source(source))


def activate(version: str | None, *, quiet: bool = False):
    """Prepend the matching sidecar to sys.path, if transformers is not imported yet."""
    if not version:
        return None
    d = sidecar_for(version)
    if not d:
        return None
    if "transformers" in sys.modules:
        if not quiet:
            print(
                f"[unsloth-nb] transformers already imported; cannot switch to "
                f"{version} in-process (restart the kernel, or use `unsloth-run`).",
                file = sys.stderr,
            )
        return None
    if d not in sys.path:
        sys.path.insert(0, d)
    # guarded like the sys.path insert above: the hook runs on EVERY cell until
    # transformers is imported, so an unconditional prepend grows PYTHONPATH by one
    # copy per cell and every child process inherits the pile
    _pp = os.environ.get("PYTHONPATH", "")
    if d not in _pp.split(os.pathsep):
        os.environ["PYTHONPATH"] = d + os.pathsep + _pp
    if not quiet and _logging_enabled():
        print(f"[unsloth-nb] activated transformers sidecar for {version}: {d}")
    return d


def resolve(model_name: str | None = None):
    return requested_version() or tier_for_model(model_name or "")


def _pre_run_cell(info = None):
    """Activate before the cell's first statement runs.

    The marker is the shim's record of a PREVIOUS cell's install, so it cannot help a
    cell that installs and then imports in one go: the shim writes it from a child
    process partway through that same cell, long after this hook has returned. Reading
    the pin out of the cell we are about to run covers that shape, which is how the
    notebooks pin a new model. Without it the cell silently ran the base transformers
    and every later cell got "already imported; cannot switch"."""
    if "transformers" in sys.modules:
        return
    # The cell's own pin outranks the marker, which is a record of an install that has
    # ALREADY run. Within one notebook a later cell can pin a different version, and
    # the marker still holds the earlier one; the marker path also falls back to
    # pid-<pid> when the ipykernel connection file cannot be read, and /tmp is never
    # swept, so a recycled pid inherits someone else's pin.
    v = pin_in_cell(getattr(info, "raw_cell", None)) or requested_version()
    if v:
        activate(v)


def register_ipython():
    try:
        ip = get_ipython()  # noqa: F821 (provided by IPython)
    except NameError:
        return
    if ip is not None and not getattr(ip, "_unsloth_tf_hook", False):
        ip.events.register("pre_run_cell", _pre_run_cell)
        ip._unsloth_tf_hook = True
