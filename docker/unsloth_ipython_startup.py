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
    import unsloth_nb_compat

    unsloth_nb_compat.register_ipython()
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
