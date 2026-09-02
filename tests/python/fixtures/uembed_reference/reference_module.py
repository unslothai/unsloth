"""Test-only adapter for the pinned upstream UEmbed reference.

The upstream Qwen35Embedder loads sparse sidecars with os.path.join, so a Hub ID does not
load them. This adapter resolves the immutable Hub snapshot to a directory before calling
the otherwise unchanged upstream class.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

from huggingface_hub import snapshot_download


_REVISIONS = {
    "Alibaba-NLP/UEmbed-2B": "e7501a4d1be34ac4c7f8d1565cbeaa5b3f5b41b3",
    "Alibaba-NLP/UEmbed-4B": "2fab6202a2fb43481772eeb7a95f4e3d12a8ff3d",
}
_SOURCE = Path(__file__).with_name("qwen35_embedding.py")
_SPEC = importlib.util.spec_from_file_location("uembed_pinned_upstream", _SOURCE)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"Cannot import pinned UEmbed reference source at {_SOURCE}")
_UPSTREAM = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _UPSTREAM
try:
    _SPEC.loader.exec_module(_UPSTREAM)
except BaseException:
    if sys.modules.get(_SPEC.name) is _UPSTREAM:
        del sys.modules[_SPEC.name]
    raise


class Qwen35Embedder(_UPSTREAM.Qwen35Embedder):
    """Resolve a known UEmbed Hub ID to its pinned local snapshot before loading."""

    def __init__(self, model_name_or_path: str, *args, **kwargs):
        revision = os.environ.get("UNSLOTH_UEMBED_REFERENCE_REVISION")
        if revision is None:
            revision = _REVISIONS.get(model_name_or_path)
        if revision is not None and not os.path.isdir(model_name_or_path):
            model_name_or_path = snapshot_download(
                model_name_or_path,
                revision = revision,
                local_files_only = os.environ.get("HF_HUB_OFFLINE") == "1",
            )
        super().__init__(model_name_or_path, *args, **kwargs)
        if self.sparse_lm_heads is not None:
            self.model.to(dtype = self.sparse_lm_heads[0].dtype)

    def encode(self, inputs):
        """Adapt the parity harness's encode surface to upstream ``process`` unchanged."""
        if isinstance(inputs, (str, dict)):
            inputs = [inputs]
        model_inputs = [{"text": item} if isinstance(item, str) else item for item in inputs]
        return self.process(model_inputs)
