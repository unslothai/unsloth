# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""GRPO PrefixGrouper engages only when the forward that runs applies prefix_seg_info.

A transformers / generic / compiled forward takes prefix_seg_info through **kwargs and drops it, so
completion 2 of a group attends to completion 1. On a tiny Qwen3 that is a 0.036-nat old / ref
log-prob error, inside the 0.7-nat self-verify, giving step-1 KL 5e-5 instead of 1e-11.
"""

import ast
import functools
import importlib.util
import inspect
import re
import sys
import types
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
UTILS = ROOT / "unsloth" / "utils"
MODELS = ROOT / "unsloth" / "models"
RL_REPLACEMENTS = MODELS / "rl_replacements.py"
MARK = "_unsloth_applies_prefix_seg_info"
FAST_ATTENTION = {
    "llama": "LlamaAttention_fast_forward",
    "qwen3": "Qwen3Attention_fast_forward",
    "mistral": "MistralAttention_fast_forward",
    "gemma2": "Gemma2Attention_fast_forward",
    "cohere": "CohereAttention_fast_forward",
    "granite": "GraniteAttention_fast_forward",
}


def _load_prefix_grouper():
    pkg_name = "_pg_capability_test_pkg"
    if pkg_name not in sys.modules:
        pkg = types.ModuleType(pkg_name)
        pkg.__path__ = [str(UTILS)]
        sys.modules[pkg_name] = pkg
    for name in ("prefix_grouper_kernel", "prefix_grouper"):
        full = f"{pkg_name}.{name}"
        if full not in sys.modules:
            spec = importlib.util.spec_from_file_location(full, UTILS / f"{name}.py")
            module = importlib.util.module_from_spec(spec)
            sys.modules[full] = module
            spec.loader.exec_module(module)
    return sys.modules[f"{pkg_name}.prefix_grouper"]


pg = _load_prefix_grouper()


def _top_level(path, names):
    tree = ast.parse(path.read_text(encoding = "utf-8"), filename = str(path))
    body = [
        node
        for node in tree.body
        if (isinstance(node, ast.FunctionDef) and node.name in names)
        or (
            isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id in names for t in node.targets)
        )
    ]
    namespace = {"re": re, "inspect": inspect, "functools": functools}
    exec(compile(ast.Module(body = body, type_ignores = []), str(path), "exec"), namespace)
    return namespace


def _model(
    attention_forwards,
    layers = None,
    config_layers = "auto",
):
    """A decoder whose layer i runs attention_forwards[i] as its attention class forward."""

    class Config:
        pass

    config = Config()
    if config_layers == "auto":
        config.num_hidden_layers = len(attention_forwards) if layers is None else layers
    elif config_layers is not None:
        config.num_hidden_layers = config_layers

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = config
            self.layers = torch.nn.ModuleList()
            for forward in attention_forwards:
                attn_cls = type("Attention", (torch.nn.Module,), {"forward": forward})
                layer = torch.nn.Module()
                layer.self_attn = attn_cls()
                self.layers.append(layer)

    return Model()


def _plain_forward(self, hidden_states, **kwargs):
    return hidden_states


def _marked_forward(self, hidden_states, **kwargs):
    return hidden_states


setattr(_marked_forward, MARK, True)


def test_unmarked_forwards_do_not_apply_prefix_seg_info():
    assert pg.prefix_seg_info_applied(_model([_plain_forward, _plain_forward])) is False


def test_every_layer_marked_applies_prefix_seg_info():
    assert pg.prefix_seg_info_applied(_model([_marked_forward, _marked_forward])) is True


def test_one_unmarked_layer_is_enough_to_refuse():
    assert pg.prefix_seg_info_applied(_model([_marked_forward, _plain_forward])) is False


def test_unknown_layer_count_refuses():
    assert pg.prefix_seg_info_applied(_model([_marked_forward], config_layers = None)) is False


def test_model_declaration_overrides_detection():
    leak = _model([_plain_forward])
    leak._unsloth_applies_prefix_seg_info = True
    assert pg.prefix_seg_info_applied(leak) is True
    capable = _model([_marked_forward])
    capable._unsloth_applies_prefix_seg_info = False
    assert pg.prefix_seg_info_applied(capable) is False


def test_pre_patch_gate_applies_only_on_marked_modules():
    """pre_patch's gate copies the fast forward's attributes (functools.wraps), so the class method
    carries the mark either way; only modules marked by mark_unsloth_fast_path run it."""
    llama = _top_level(MODELS / "llama.py", {"_gated_method", "running_forward", "_MISSING"})
    gated = llama["_gated_method"](torch.nn.Module, "forward", _marked_forward, _plain_forward)
    assert getattr(gated, MARK, False) is True
    unmarked = _model([gated, gated])
    assert pg.prefix_seg_info_applied(unmarked) is False
    marked = _model([gated, gated])
    for module in marked.modules():
        module._unsloth_fast_path = True
    assert pg.prefix_seg_info_applied(marked) is True
    for module in marked.modules():
        attn_forward = pg._running_forward(module)
        assert attn_forward is llama["running_forward"](module)


def test_prefix_grouper_enabled_checks_the_model(monkeypatch):
    monkeypatch.setattr(pg, "_ENABLED", True)
    monkeypatch.setattr(pg, "compile_check_failed", lambda: None)
    assert pg.prefix_grouper_enabled() is True  # no model: the previous behaviour
    assert pg.prefix_grouper_enabled(_model([_marked_forward, _marked_forward])) is True
    assert pg.prefix_grouper_enabled(_model([_plain_forward, _plain_forward])) is False
    monkeypatch.setattr(pg, "_ENABLED", False)
    assert pg.prefix_grouper_enabled(_model([_marked_forward])) is False


def test_self_verify_tolerances_unchanged():
    source = (UTILS / "prefix_grouper.py").read_text(encoding = "utf-8")
    assert 'os.environ.get("UNSLOTH_GRPO_PREFIX_GROUPER_TOL", "0.7")' in source
    assert pg.TOL_KILL == 1.5


@pytest.mark.parametrize("module, function", sorted(FAST_ATTENTION.items()))
def test_fast_attention_forwards_are_marked(module, function):
    path = MODELS / f"{module}.py"
    tree = ast.parse(path.read_text(encoding = "utf-8"), filename = str(path))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == function)
    decorators = [d.id for d in node.decorator_list if isinstance(d, ast.Name)]
    assert "applies_prefix_seg_info" in decorators
    calls = {
        n.func.id
        for n in ast.walk(node)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
    }
    assert "resolve_prefix_seg_info" in calls


def test_marker_names_agree():
    dispatch = _top_level(
        UTILS / "attention_dispatch.py", {"APPLIES_PREFIX_SEG_INFO", "applies_prefix_seg_info"}
    )
    assert dispatch["APPLIES_PREFIX_SEG_INFO"] == pg.APPLIES_PREFIX_SEG_INFO == MARK

    def forward(self):
        return None

    assert dispatch["applies_prefix_seg_info"](forward) is forward
    assert getattr(forward, MARK) is True


def test_nograd_gate_passes_the_model():
    text = RL_REPLACEMENTS.read_text(encoding = "utf-8")
    node = next(
        n
        for n in ast.parse(text).body
        if isinstance(n, ast.FunctionDef)
        and n.name == "grpo_trainer__get_per_token_logps_and_entropies"
    )
    source = ast.get_source_segment(text, node)
    assert "_pg_enabled_fn(unwrapped_model)" in source
    assert re.search(r"\b_pg_enabled_fn\(\s*\)", source) is None


_PATCH = _top_level(
    RL_REPLACEMENTS,
    {"_GRPO_PREFIX_GROUPER_GATE", "_patch_grpo_accumulated_loss_prefix_grouper_gate"},
)
patch_gate = _PATCH["_patch_grpo_accumulated_loss_prefix_grouper_gate"]

_ZOO_LIKE = """
def grpo_accumulated_loss(trainer, unwrapped_model, _pg_funcs):
    _pg_build_layout, _pg_enabled_fn = _pg_funcs
    _pg_engage = (
        _pg_enabled_fn()
        and True
    )
    if _pg_engage:
        return unwrapped_model(prefix_seg_info = _pg_build_layout())
    return None
"""


def test_zoo_gate_patch_passes_the_model():
    patched = patch_gate(_ZOO_LIKE)
    assert patched.count("_pg_enabled_fn(unwrapped_model)") == 1
    assert "_pg_enabled_fn()" not in patched
    assert patch_gate(patched) == patched  # already patched: unchanged
    namespace = {}
    exec(patched, namespace)
    seen = []

    def enabled(model = None):
        seen.append(model)
        return False

    model = object()
    assert namespace["grpo_accumulated_loss"](None, model, (lambda: None, enabled)) is None
    assert seen == [model]


def test_zoo_gate_patch_leaves_zoo_without_prefix_grouper():
    source = "def grpo_accumulated_loss(unwrapped_model):\n    return unwrapped_model()\n"
    assert patch_gate(source) == source


def test_zoo_gate_patch_refuses_an_unrecognised_gate():
    source = _ZOO_LIKE.replace("_pg_enabled_fn()", "_pg_on")
    with pytest.raises(RuntimeError, match = "PrefixGrouper gate"):
        patch_gate(source)


def test_installed_zoo_gate_is_patched():
    zoo = pytest.importorskip("unsloth_zoo.rl_replacements")
    function = getattr(zoo, "grpo_accumulated_loss", None)
    if function is None:
        function = zoo.RL_REPLACEMENTS.get("grpo_accumulated_loss")
    source = inspect.getsource(function)
    if "prefix_seg_info" not in source:
        pytest.skip("this unsloth_zoo has no gradient PrefixGrouper")
    patched = patch_gate(source)
    assert "_pg_enabled_fn(unwrapped_model)" in patched
    assert re.search(r"\b_pg_enabled_fn\(\s*\)", patched) is None
    compile(patched, "<patched grpo_accumulated_loss>", "exec")
