# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-Present the Unsloth team. See /studio/LICENSE.AGPL-3.0

"""Baked IPython startup hook (copied to the profile's startup/ dir).

Runs once per kernel. Registers a pre_run_cell event that activates the right
transformers sidecar before the first model cell, using the version the
notebook's own install cell asked for (recorded by the pip/uv shim). Safe no-op
outside IPython, when no version was requested, or once transformers is imported.
"""

try:
    import os

    # Tell the pip/uv shim it's running inside a notebook kernel, so a cell's
    # `!pip install ...` / `!uv pip install ...` (which inherits this env) gets
    # the safe-install behaviour. Unset everywhere else => shim is a passthrough.
    os.environ["UNSLOTH_NB_SHIM"] = "1"

    # Scope the transformers-request marker to THIS kernel so concurrent notebooks
    # don't read each other's pin. The pip/uv shim (a child of this kernel)
    # inherits UNSLOTH_NB_TF_MARKER, so writer and reader agree on the path. Falls
    # back to the shared default when unset (e.g. `unsloth-run`, one notebook/process).
    if not os.environ.get("UNSLOTH_NB_TF_MARKER"):
        # A kernel id that is stable for the kernel's lifetime and unique per
        # kernel: the ipykernel connection file name, else the kernel PID.
        _kid = ""
        try:
            from ipykernel import get_connection_file  # type: ignore
            _kid = os.path.splitext(os.path.basename(get_connection_file()))[0]
        except Exception:
            _kid = ""
        _kid = _kid or ("pid-%d" % os.getpid())
        os.environ["UNSLOTH_NB_TF_MARKER"] = "/tmp/unsloth_nb/requested_transformers." + _kid

    import unsloth_nb_compat

    unsloth_nb_compat.register_ipython()

    # Re-point the %pip / %uv line magics and `!python -m pip` at the same shim,
    # so the in-process / module install paths cannot bypass the PATH shim and
    # overwrite the baked torch/vLLM stack. Independent of the sidecar hook.
    import unsloth_nb_pip_magic

    unsloth_nb_pip_magic.register_ipython()
except Exception as _e:  # never break a kernel because of the helper
    import sys
    print(f"[unsloth-nb] startup hook skipped: {_e!r}", file = sys.stderr)

# Colab cell-magic compatibility (hoist `%%capture` above a leading `#@title`
# form so it fires instead of raising UsageError). Independent try/except so a
# failure here never disables the transformers-sidecar hook above and vice versa.
try:
    import unsloth_colab_compat
    unsloth_colab_compat.register_ipython()
except Exception as _e:  # never break a kernel because of the helper
    import sys
    print(f"[unsloth-nb] colab-compat hook skipped: {_e!r}", file = sys.stderr)
