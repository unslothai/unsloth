# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""`import unsloth` must not change models it never loaded.

vllm/platforms/cuda.py turns cuDNN SDPA off when imported, and Unsloth's import imports vLLM to probe it,
so the switch reached every model in the process. The import now puts the caller's value back and an
Unsloth load re-applies vLLM's value, so Unsloth models keep the backends they always ran with.

The zoo fused lm_head + cross entropy forward and the causal mask patch replace transformers functions for
the whole process; at import Unsloth asks zoo to keep them for the models it loads or trains (model_marker).
"""

import ast
import os
import types

import pytest
import torch

from unsloth import import_fixes

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture
def cudnn_state(monkeypatch):
    before = torch.backends.cuda.cudnn_sdp_enabled()
    monkeypatch.setattr(import_fixes, "_CUDNN_SDPA_BEFORE_IMPORT", [None])
    monkeypatch.setattr(import_fixes, "_CUDNN_SDPA_OFF_FOR_LOADS", [False])
    yield
    torch.backends.cuda.enable_cudnn_sdp(before)


def test_vllm_switch_off_is_undone_and_reapplied_for_loads(cudnn_state):
    torch.backends.cuda.enable_cudnn_sdp(True)
    import_fixes.snapshot_cudnn_sdpa_before_import()
    torch.backends.cuda.enable_cudnn_sdp(False)  # what importing vllm.platforms.cuda does
    import_fixes.restore_cudnn_sdpa_after_import()
    assert torch.backends.cuda.cudnn_sdp_enabled() is True
    import_fixes.apply_cudnn_sdpa_for_unsloth_load()
    assert torch.backends.cuda.cudnn_sdp_enabled() is False


def test_caller_switch_off_is_kept(cudnn_state):
    torch.backends.cuda.enable_cudnn_sdp(False)  # the caller (or vLLM imported first) already chose
    import_fixes.snapshot_cudnn_sdpa_before_import()
    import_fixes.restore_cudnn_sdpa_after_import()
    assert torch.backends.cuda.cudnn_sdp_enabled() is False
    torch.backends.cuda.enable_cudnn_sdp(True)  # the caller turns it back on later
    import_fixes.apply_cudnn_sdpa_for_unsloth_load()
    assert torch.backends.cuda.cudnn_sdp_enabled() is True


def test_no_switch_off_during_import_leaves_loads_alone(cudnn_state):
    torch.backends.cuda.enable_cudnn_sdp(True)
    import_fixes.snapshot_cudnn_sdpa_before_import()
    import_fixes.restore_cudnn_sdpa_after_import()
    import_fixes.apply_cudnn_sdpa_for_unsloth_load()
    assert torch.backends.cuda.cudnn_sdp_enabled() is True


def _top_level_calls(tree):
    """Line of each import_fixes helper imported at module level (any nesting of if / try)."""
    lines = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "import_fixes":
            for alias in node.names:
                lines.setdefault(alias.name, node.lineno)
        if isinstance(node, ast.ImportFrom) and node.module == "_gpu_init":
            lines.setdefault("_gpu_init", node.lineno)
    return lines


def test_init_snapshots_before_gpu_init_and_restores_last():
    with open(os.path.join(ROOT, "unsloth", "__init__.py"), encoding = "utf-8") as f:
        tree = ast.parse(f.read())
    lines = _top_level_calls(tree)
    assert lines["snapshot_cudnn_sdpa_before_import"] < lines["_gpu_init"]
    restore = lines["restore_cudnn_sdpa_after_import"]
    last_import = max(
        node.lineno
        for node in tree.body
        if isinstance(node, (ast.Import, ast.ImportFrom)) and node.lineno < restore
    )
    assert restore > last_import


def test_every_from_pretrained_starts_with_the_load_hook():
    for rel, owner in (
        ("unsloth/models/loader.py", ("FastLanguageModel", "FastModel")),
        ("unsloth/models/llama.py", ("FastLlamaModel",)),
        ("unsloth/models/vision.py", ("FastBaseModel",)),
    ):
        with open(os.path.join(ROOT, rel), encoding = "utf-8") as f:
            tree = ast.parse(f.read())
        for cls in tree.body:
            if not (isinstance(cls, ast.ClassDef) and cls.name in owner):
                continue
            fn = next(
                n
                for n in cls.body
                if isinstance(n, ast.FunctionDef) and n.name == "from_pretrained"
            )
            first_calls = [
                n.value.func.id
                for n in fn.body[:3]
                if isinstance(n, ast.Expr)
                and isinstance(n.value, ast.Call)
                and isinstance(n.value.func, ast.Name)
            ]
            assert "_unsloth_load_begins" in first_calls, (rel, cls.name)


def test_unsloth_load_marks_every_transformers_model():
    model_marker = pytest.importorskip("unsloth_zoo.model_marker")
    from transformers import LlamaConfig, LlamaForCausalLM
    from unsloth.models._utils import _mark_requested_float32

    config = LlamaConfig(
        vocab_size = 64,
        hidden_size = 16,
        intermediate_size = 32,
        num_hidden_layers = 1,
        num_attention_heads = 2,
        num_key_value_heads = 1,
    )
    marked, plain = LlamaForCausalLM(config), LlamaForCausalLM(config)
    _mark_requested_float32(marked, False)  # every loader.py from_pretrained return passes here
    assert getattr(marked, model_marker.MODEL_MARKER, False) is True
    assert getattr(marked.model, model_marker.MODEL_MARKER, False) is True
    assert not hasattr(plain, model_marker.MODEL_MARKER)


def test_import_requires_the_marker():
    model_marker = pytest.importorskip("unsloth_zoo.model_marker")
    import unsloth  # noqa: F401

    if getattr(unsloth, "_IS_CPU_BACKEND", True):
        pytest.skip("the GPU backend asks for the marker; this run selected the CPU backend")
    assert model_marker.model_marker_required()


def test_trainer_marks_every_model_it_holds():
    model_marker = pytest.importorskip("unsloth_zoo.model_marker")
    import transformers
    from transformers import LlamaConfig, LlamaForCausalLM
    from unsloth.models._utils import _mark_trainer_models

    config = LlamaConfig(
        vocab_size = 64,
        hidden_size = 16,
        intermediate_size = 32,
        num_hidden_layers = 1,
        num_attention_heads = 2,
        num_key_value_heads = 1,
    )
    policy, ref, reward, other = (LlamaForCausalLM(config) for _ in range(4))
    trainer = types.SimpleNamespace(model = policy, ref_model = ref, reward_funcs = [reward, len])
    _mark_trainer_models(trainer)
    for model in (policy, ref, reward):
        assert getattr(model, model_marker.MODEL_MARKER, False) is True
    assert not hasattr(other, model_marker.MODEL_MARKER)
    # Installed on the transformers Trainer at import, before any Unsloth load. Later Unsloth patches
    # (patch_gradient_accumulation_fix wraps __init__) sit on top and call through via __wrapped__.
    for name in ("__init__", "call_model_init", "train"):
        assert _wrapper_chain_marks(getattr(transformers.Trainer, name)), name


def _wrapper_chain_marks(fn):
    seen = set()
    while fn is not None and id(fn) not in seen:
        if getattr(fn, "_unsloth_marks_models", False):
            return True
        seen.add(id(fn))
        fn = getattr(fn, "__wrapped__", None)
    return False
