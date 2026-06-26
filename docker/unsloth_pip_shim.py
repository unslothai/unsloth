#!/opt/unsloth-venv/bin/python
"""pip / uv shim for the Unsloth Docker notebook environment.

Installed earlier on PATH than the real tools so a notebook's `!pip install ...`
or `!uv pip install ...` cell becomes SAFE + idempotent instead of clobbering the
carefully-resolved cu128 torch/vLLM/transformers stack:

  * `transformers==X`  -> NOT installed into the base venv. The version X is
    recorded so the sidecar mechanism (unsloth_nb_compat) activates it for the
    model cells. The base stack stays intact.
  * torch / torchvision / torchaudio / triton / xformers / vllm / bitsandbytes /
    flashinfer / nvidia-* -> SKIPPED (the baked, ABI-matched versions are kept;
    a notebook reinstall here only ever breaks the GPU stack).
  * everything else (omegaconf, snac, causal-conv1d, ...) -> passed through to the
    real tool unchanged, so notebooks that genuinely need extra packages still
    get them.

Real tools are at /opt/unsloth-venv/bin/{pip,uv}; this shim invokes them by
absolute path so there is no recursion. `python -m pip` / `%pip` bypass PATH and
are not intercepted -- the driven `unsloth-run` handles those by parsing the
notebook directly.
"""

import os, re, sys, subprocess

REAL = {"pip": "/opt/unsloth-venv/bin/pip", "uv": "/opt/unsloth-venv/bin/uv"}
MARKER = os.environ.get("UNSLOTH_NB_TF_MARKER", "/tmp/unsloth_nb/requested_transformers")

# Packages whose baked version must never be changed by a notebook install cell.
_KEEP = {
    "torch",
    "torchvision",
    "torchaudio",
    "triton",
    "triton-rocm",
    "pytorch-triton",
    "xformers",
    "vllm",
    "bitsandbytes",
    "flashinfer",
    "flashinfer-python",
    "unsloth",
    "unsloth-zoo",
    "unsloth_zoo",
}
_KEEP_PREFIX = ("nvidia-", "nvidia_")
# pip/uv flags that consume the following token as a value (so we don't mistake
# that value for a requirement).
_VALUE_FLAGS = {
    "-r",
    "--requirement",
    "-c",
    "--constraint",
    "-i",
    "--index-url",
    "--extra-index-url",
    "-f",
    "--find-links",
    "--target",
    "-t",
    "--python",
    "-p",
    "--prefix",
    "--index-strategy",
    "--upgrade-package",
    "-P",
    "--no-binary",
    "--only-binary",
    "--platform",
    "--python-version",
    "--abi",
    "--implementation",
}
# Of those value-flags, the ones whose VALUE is itself an install target: a
# requirements file pulls real requirements. An index-url / find-links /
# constraint / target value is an option, not something to install.
_REQ_FILE_FLAGS = {"-r", "--requirement"}


def _canon(token):
    """Extract the lowercased distribution name from a requirement token, or None
    if the token is not a plain pkg spec (url / path / vcs / option)."""
    if token.startswith("-"):
        return None
    if re.match(r"^[a-z]+\+", token) or "://" in token or token.startswith((".", "/")):
        return None  # vcs / url / local path -> let it pass through
    # strip extras and any version/marker tail
    name = re.split(r"[<>=!~\[\s;@]", token, 1)[0].strip()
    return name.lower().replace("_", "-") or None


def _version_pin(token):
    """Return the pinned version for a `pkg==X` token, else None."""
    m = re.search(r"==\s*([0-9][0-9A-Za-z.\-]*)", token)
    return m.group(1) if m else None


def main():
    tool = "uv" if os.path.basename(sys.argv[0]).startswith("uv") else "pip"
    argv = sys.argv[1:]

    # Only intercept inside a notebook kernel (UNSLOTH_NB_SHIM is set by the baked
    # IPython startup and by `unsloth-run`). EVERYWHERE else -- install.sh during
    # the image build, internal tooling, an interactive shell -- behave exactly
    # like the real tool, so we never disturb the build or system package mgmt.
    if os.environ.get("UNSLOTH_NB_SHIM") != "1":
        os.execv(REAL[tool], [REAL[tool]] + argv)
        return

    # Locate the `install` verb (uv: `uv pip install ...`; pip: `pip install ...`).
    try:
        if tool == "uv":
            # skip a leading `pip` subcommand
            i = argv.index("install")
        else:
            i = argv.index("install")
    except ValueError:
        os.execv(REAL[tool], [REAL[tool]] + argv)  # not an install -> passthrough
        return

    head, tail = argv[: i + 1], argv[i + 1 :]
    keep_args, dropped, recorded = [], [], None
    has_target = False
    skip_next = False
    prev_flag = None
    for tok in tail:
        if skip_next:
            keep_args.append(tok)
            # The value of -r/--requirement pulls real requirements (a target);
            # the value of an index-url / find-links / constraint / etc. flag is
            # an option, not something to install.
            if prev_flag in _REQ_FILE_FLAGS:
                has_target = True
            skip_next = False
            prev_flag = None
            continue
        if tok in _VALUE_FLAGS:
            keep_args.append(tok)
            skip_next = True
            prev_flag = tok
            continue
        name = _canon(tok)
        if name is None:
            keep_args.append(tok)  # bare flag, or a positional url / path / vcs
            if not tok.startswith("-"):
                has_target = True  # standalone . / ./pkg / git+... / *.whl
            continue
        if name == "transformers":
            v = _version_pin(tok)
            if v:
                recorded = v
            dropped.append(tok)
            continue
        if name in _KEEP or name.startswith(_KEEP_PREFIX):
            dropped.append(tok)
            continue
        keep_args.append(tok)
        has_target = True  # a kept package spec

    if recorded:
        try:
            os.makedirs(os.path.dirname(MARKER), exist_ok = True)
            with open(MARKER, "w") as f:
                f.write(recorded)
            print(
                f"[unsloth-nb] notebook requested transformers=={recorded}; will "
                f"activate its sidecar for the model cells (base stack kept)."
            )
        except OSError:
            pass
    if dropped:
        print("[unsloth-nb] kept baked versions, skipped: " + " ".join(dropped))

    # Anything left to actually install? `has_target` was set during the scan for
    # a kept package spec, a positional url / path / vcs / editable target, or a
    # -r/--requirement file. A line carrying only baked packages plus option flags
    # (e.g. `--extra-index-url <url> torch`) leaves no target, so no-op instead of
    # exec'ing a bare `pip install --extra-index-url <url>` that would fail.
    if not has_target:
        print("[unsloth-nb] nothing to install after keeping the baked stack; ok.")
        return
    cmd = [REAL[tool]] + head + keep_args
    sys.stdout.flush()
    os.execv(REAL[tool], cmd)


if __name__ == "__main__":
    main()
