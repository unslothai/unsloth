# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""`from_pretrained(..., gguf_file = ...)` checks shared by the GPU, CPU and MLX entry points.

No torch / transformers import at module level: the MLX branch of `unsloth/__init__.py` uses this too.
"""

import importlib.util
import os

__all__ = [
    "is_gguf_path",
    "require_gguf_package",
    "refuse_gguf_on_mlx",
    "has_tokenizer_files",
    "load_gguf_tokenizer_if_missing",
]

# transformers' own floor for reading GGUF files.
GGUF_MIN_VERSION = "0.10.0"

# Files a tokenizer builds its vocabulary from. tokenizer_config.json alone is not one: transformers 5 then
# builds an empty tokenizer (vocab 1).
_TOKENIZER_VOCAB_FILES = (
    "tokenizer.json",
    "tokenizer.model",
    "vocab.json",
    "vocab.txt",
    "spiece.model",
    "sentencepiece.bpe.model",
    "tekken.json",
)


def is_gguf_path(model_name):
    return isinstance(model_name, (str, os.PathLike)) and os.fspath(model_name).lower().endswith(
        ".gguf"
    )


def _gguf_version():
    try:
        from importlib.metadata import version
        return version("gguf")
    except Exception:
        return None


def require_gguf_package(gguf_file):
    """transformers needs the `gguf` package to read a GGUF file, and Unsloth does not install it by default.
    Raise one clear error before anything downloads instead of transformers' ImportError after the weights."""
    if not gguf_file:
        return
    installed = _gguf_version() if importlib.util.find_spec("gguf") is not None else None
    if installed is not None:
        try:
            from packaging.version import Version
            if Version(installed) >= Version(GGUF_MIN_VERSION):
                return
        except Exception:
            return
    found = "is not installed" if installed is None else f"is {installed}, which is too old"
    raise ImportError(
        f"Unsloth: loading `gguf_file = {gguf_file!r}` needs the `gguf` package (>= {GGUF_MIN_VERSION}), "
        f'which {found}. Install it with `pip install "gguf>={GGUF_MIN_VERSION}"` '
        f'(or `pip install "unsloth[gguf]"`), then load again.'
    )


def refuse_gguf_on_mlx(model_name, gguf_file):
    """Apple Silicon's MLX loader (mlx-lm) reads safetensors only. Without this it drops `gguf_file`: a GGUF-only
    repo fails with "No safetensors found", a .gguf path with NotADirectoryError, and a repo holding both silently
    loads and 4-bit quantizes the safetensors instead of the GGUF that was asked for."""
    if not gguf_file and not is_gguf_path(model_name):
        return
    asked = f"gguf_file = {gguf_file!r}" if gguf_file else f"{os.fspath(model_name)!r}"
    raise NotImplementedError(
        f"Unsloth: the MLX backend on Apple Silicon cannot load GGUF files ({asked}): mlx-lm reads "
        "safetensors checkpoints only. Options:\n"
        "  1. Load the original safetensors model without `gguf_file` (for example `unsloth/Qwen3-0.6B` instead of "
        "`unsloth/Qwen3-0.6B-GGUF`). MLX quantizes it to 4-bit at load with `load_in_4bit = True`.\n"
        "  2. To fine-tune the GGUF's own weights, use Unsloth's PyTorch CPU backend: set "
        "`UNSLOTH_FORCE_GPU_PATH=1` and `UNSLOTH_FORCE_CPU=1` before `import unsloth`, `pip install gguf`, and "
        'pass the repo or folder as `model_name` with `gguf_file = "<file>.gguf"`. The GGUF is dequantized to '
        "16-bit at load.\n"
        "  3. To only run the GGUF, use llama.cpp or Unsloth Studio."
    )


def has_tokenizer_files(
    model_name,
    revision = None,
    token = None,
    cache_dir = None,
    subfolder = None,
    local_files_only = False,
):
    """Whether a local folder or Hub repo ships a tokenizer vocabulary. GGUF repos often ship only config.json
    next to the .gguf files, whose metadata holds the tokenizer."""
    names = [os.path.join(subfolder, f) if subfolder else f for f in _TOKENIZER_VOCAB_FILES]
    if os.path.isdir(model_name):
        return any(os.path.isfile(os.path.join(model_name, n)) for n in names)
    offline = local_files_only or os.environ.get("HF_HUB_OFFLINE", "0") not in (
        "0",
        "",
        "false",
        "False",
    )
    if not offline:
        try:
            from huggingface_hub import HfApi
            files = set(HfApi().list_repo_files(model_name, revision = revision, token = token))
            return any(n.replace(os.sep, "/") in files for n in names)
        except Exception:
            pass
    # Offline or the Hub is unreachable: what the cache holds.
    try:
        from huggingface_hub import try_to_load_from_cache
    except Exception:
        return True
    for name in names:
        try:
            found = try_to_load_from_cache(model_name, name, cache_dir = cache_dir, revision = revision)
        except Exception:
            continue
        if isinstance(found, str):
            return True
    return False


def load_gguf_tokenizer_if_missing(
    model_name,
    gguf_file,
    revision = None,
    token = None,
    cache_dir = None,
    subfolder = None,
    local_files_only = False,
    **tokenizer_kwargs,
):
    """The tokenizer read from the GGUF's metadata when the repo has no tokenizer files of its own, else None
    (load the repo's tokenizer as usual). Without `gguf_file` the tokenizer load ignores the GGUF: transformers 5
    returns an empty tokenizer (vocab 1, so every forward fails on 0 tokens) and 4.57 raises."""
    if not gguf_file or not isinstance(model_name, str):
        return None
    hub_kwargs = dict(revision = revision, token = token, cache_dir = cache_dir, subfolder = subfolder)
    if has_tokenizer_files(model_name, local_files_only = local_files_only, **hub_kwargs):
        return None
    from transformers import AutoTokenizer

    print(
        f"Unsloth: {model_name} has no tokenizer files, so the tokenizer is read from {gguf_file}."
    )
    hub_kwargs = {k: v for k, v in hub_kwargs.items() if v is not None}
    if local_files_only:
        hub_kwargs["local_files_only"] = True
    return AutoTokenizer.from_pretrained(
        model_name, gguf_file = gguf_file, **hub_kwargs, **tokenizer_kwargs
    )
