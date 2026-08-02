# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""`revision` must reach the config, weight and tokenizer loads (issue #3544).

FastLlamaModel.from_pretrained took a `revision` argument and never read it, so the
config, weights and tokenizer silently came from the repo's default branch. These are
AST-structural so they need no GPU, no network and no gated checkpoint; importing
unsloth on a CPU runner is what tests/conftest.py exists to work around.
"""

import ast
import types
from pathlib import Path

import pytest


REPO = Path(__file__).parents[2]
LLAMA = REPO / "unsloth" / "models" / "llama.py"
LOADER = REPO / "unsloth" / "models" / "loader.py"
VISION = REPO / "unsloth" / "models" / "vision.py"
TOKENIZER_UTILS = REPO / "unsloth" / "tokenizer_utils.py"
LOADER_UTILS = REPO / "unsloth" / "models" / "loader_utils.py"


def _tree(path):
    return ast.parse(path.read_text(encoding = "utf-8"))


def _function(
    tree,
    name,
    class_name = None,
):
    body = tree.body
    if class_name is not None:
        classes = [n for n in body if isinstance(n, ast.ClassDef) and n.name == class_name]
        assert classes, f"{class_name} not found"
        body = classes[0].body
    for node in body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{class_name or ''}.{name} not found")


def _params(function):
    return [a.arg for a in function.args.args + function.args.kwonlyargs]


def _calls(function, callee):
    """Every Call whose dotted name ends with `callee`."""
    return [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Call) and ast.unparse(node.func).split(".")[-1] == callee
    ]


def _revision_kwarg(call):
    for keyword in call.keywords:
        if keyword.arg == "revision":
            return keyword
    return None


def test_fast_llama_model_reads_its_revision_argument():
    """The whole of #3544: the parameter existed but had zero reads."""
    function = _function(_tree(LLAMA), "from_pretrained", "FastLlamaModel")
    assert "revision" in _params(function)
    loads = [
        n
        for n in ast.walk(function)
        if isinstance(n, ast.Name) and n.id == "revision" and isinstance(n.ctx, ast.Load)
    ]
    assert loads, "revision is accepted but never read"


@pytest.mark.parametrize(
    "callee, minimum",
    [
        ("AutoConfig", 2),  # checkpoint probe + main config
        ("AutoModelForCausalLM", 2),  # user-config and plain branches
        ("AutoModelForSequenceClassification", 1),
        ("load_correct_tokenizer", 1),
    ],
)
def test_llama_loads_forward_revision(callee, minimum):
    function = _function(_tree(LLAMA), "from_pretrained", "FastLlamaModel")
    calls = (
        _calls(function, "from_pretrained")
        if callee != "load_correct_tokenizer"
        else _calls(function, "load_correct_tokenizer")
    )
    if callee != "load_correct_tokenizer":
        calls = [c for c in calls if ast.unparse(c.func).startswith(callee)]
    assert len(calls) >= minimum, f"expected >= {minimum} {callee} loads, found {len(calls)}"
    for call in calls:
        assert _revision_kwarg(call) is not None, f"{callee} at line {call.lineno} drops revision"


def test_llama_does_not_pass_revision_to_load_vllm():
    """load_vllm has no `revision` parameter and load_vllm_kwargs is not filtered,
    so putting one in that dict is an unconditional TypeError on the vLLM path."""
    function = _function(_tree(LLAMA), "from_pretrained", "FastLlamaModel")
    dicts = [
        node.value
        for node in ast.walk(function)
        if isinstance(node, ast.Assign)
        and any(getattr(t, "id", None) == "load_vllm_kwargs" for t in node.targets)
        and isinstance(node.value, ast.Call)
    ]
    assert dicts, "load_vllm_kwargs assignment not found"
    for call in dicts:
        assert _revision_kwarg(call) is None, "revision is not a load_vllm argument"


def test_fast_base_model_does_not_bind_revision():
    """vision.py's weight load forwards **kwargs, so binding `revision` as a named
    parameter would silently drop it from there and from kwargs.get('revision')."""
    function = _function(_tree(VISION), "from_pretrained", "FastBaseModel")
    assert "revision" not in _params(function)
    assert function.args.kwarg is not None, "**kwargs is what carries revision here"


@pytest.mark.parametrize(
    "callee, minimum",
    [("AutoConfig", 4), ("auto_processor", 2), ("_AutoTokenizer", 2)],
)
def test_vision_loads_forward_revision(callee, minimum):
    function = _function(_tree(VISION), "from_pretrained", "FastBaseModel")
    calls = [
        c for c in _calls(function, "from_pretrained") if ast.unparse(c.func).startswith(callee)
    ]
    assert len(calls) >= minimum, f"expected >= {minimum} {callee} loads, found {len(calls)}"
    for call in calls:
        assert _revision_kwarg(call) is not None, f"{callee} at line {call.lineno} drops revision"


@pytest.mark.parametrize("name", ["load_correct_tokenizer", "_load_correct_tokenizer"])
def test_tokenizer_helpers_accept_revision(name):
    assert "revision" in _params(_function(_tree(TOKENIZER_UTILS), name))


def test_tokenizer_helpers_forward_revision():
    tree = _tree(TOKENIZER_UTILS)
    public = _function(tree, "load_correct_tokenizer")
    inner = _calls(public, "_load_correct_tokenizer")
    assert len(inner) == 1 and _revision_kwarg(inner[0]) is not None

    private = _function(tree, "_load_correct_tokenizer")
    loads = _calls(private, "from_pretrained")
    assert len(loads) >= 2, "expected the slow and fast tokenizer loads"
    for call in loads:
        assert (
            _revision_kwarg(call) is not None
        ), f"tokenizer load at line {call.lineno} drops revision"


def _load_gate():
    """Exec just _revision_for_resolved_repo, so no GPU-bound import is needed."""
    source = LOADER.read_text(encoding = "utf-8")
    function = _function(ast.parse(source), "_revision_for_resolved_repo")
    namespace = {"logger": types.SimpleNamespace(warning_once = lambda *a, **k: None)}
    module = ast.Module(body = [function], type_ignores = [])
    ast.fix_missing_locations(module)
    exec(compile(module, str(LOADER), "exec"), namespace)
    return namespace["_revision_for_resolved_repo"]


def test_revision_survives_when_the_repo_is_unchanged():
    # The reported case: a user's own repo is never in the mapper tables.
    gate = _load_gate()
    assert gate("my-branch", "myorg/my-ft", "myorg/my-ft") == "my-branch"


@pytest.mark.parametrize(
    "model_name, old_model_name",
    [
        ("unsloth/llama-3-8b-bnb-4bit", "meta-llama/Meta-Llama-3-8B"),  # prequant mirror
        ("unsloth/Qwen3-30B-A3B", "unsloth/Qwen3-30B-A3B-bnb-4bit"),  # suffix strip
        ("/tmp/unsloth-fp8-cache/model", "meta-llama/Meta-Llama-3-8B"),  # fp8 temp dir
    ],
)
def test_revision_is_dropped_once_the_repo_is_remapped(model_name, old_model_name):
    # The ref only exists on the repo the caller named, so pinning it elsewhere
    # would 404 or, worse, resolve a same-named branch on a different repo.
    assert _load_gate()("abc123", model_name, old_model_name) is None


def test_no_revision_stays_none_even_when_remapped():
    gate = _load_gate()
    assert gate(None, "unsloth/llama-3-8b-bnb-4bit", "meta-llama/Meta-Llama-3-8B") is None


def test_the_gate_warns_exactly_once_when_it_drops_a_revision():
    source = LOADER.read_text(encoding = "utf-8")
    function = _function(ast.parse(source), "_revision_for_resolved_repo")
    warnings = []
    namespace = {"logger": types.SimpleNamespace(warning_once = lambda m: warnings.append(m))}
    module = ast.Module(body = [function], type_ignores = [])
    ast.fix_missing_locations(module)
    exec(compile(module, str(LOADER), "exec"), namespace)

    namespace["_revision_for_resolved_repo"]("abc123", "unsloth/x-bnb-4bit", "org/x")
    assert len(warnings) == 1
    message = warnings[0]
    # Both repos have to be named or the user cannot tell which load was silently redirected.
    assert "abc123" in message and "org/x" in message and "unsloth/x-bnb-4bit" in message
    assert "use_exact_model_name" in message


def test_both_loader_paths_gate_before_and_after_resolution():
    """The gate has to run before the AutoConfig / PeftConfig probes, or a pinned 4bit
    load fails against the mirror instead of warning, and again after the last remap."""
    tree = _tree(LOADER)
    for class_name in ("FastLanguageModel", "FastModel"):
        function = _function(tree, "from_pretrained", class_name)
        gates = _calls(function, "_revision_for_resolved_repo")
        assert len(gates) == 2, f"{class_name} needs an early and a late gate, found {len(gates)}"
        early, late = sorted(gates, key = lambda c: c.lineno)

        probes = [
            c for c in _calls(function, "from_pretrained")
            if ast.unparse(c.func).split(".")[0] in ("AutoConfig", "PeftConfig")
        ]
        assert probes, f"{class_name} has no config probe"
        gated = 0
        for probe in probes:
            assert probe.lineno > early.lineno, "the gate must precede the config probes"
            keyword = _revision_kwarg(probe)
            if keyword is None:
                continue  # the PEFT base-model probe deliberately pins nothing
            assert getattr(keyword.value, "id", None) == "base_revision", (
                f"probe at line {probe.lineno} uses the ungated revision"
            )
            gated += 1
        assert gated >= 2, f"{class_name} must gate its AutoConfig and PeftConfig probes"

        # The late gate feeds on base_revision so an already-dropped one warns only once.
        assert getattr(late.args[0], "id", None) == "base_revision"


def test_the_late_gate_is_skipped_for_peft():
    """On a PEFT load model_name is necessarily the base model, so the remap warning
    would fire for every versioned adapter while PeftModel loads the ref correctly."""
    tree = _tree(LOADER)
    for class_name in ("FastLanguageModel", "FastModel"):
        function = _function(tree, "from_pretrained", class_name)
        late = sorted(_calls(function, "_revision_for_resolved_repo"), key = lambda c: c.lineno)[-1]
        guards = [
            n for n in ast.walk(function)
            if isinstance(n, ast.If)
            and ast.unparse(n.test).replace(" ", "") == "notis_peft"
            and n.lineno <= late.lineno <= n.end_lineno
        ]
        assert guards, f"{class_name}'s late gate must sit under `if not is_peft`"


def test_the_adapter_load_keeps_the_callers_revision():
    """`revision` names the adapter repo, so PeftModel must get the ungated value."""
    tree = _tree(LOADER)
    for class_name in ("FastLanguageModel", "FastModel"):
        function = _function(tree, "from_pretrained", class_name)
        peft_loads = [
            c for c in _calls(function, "from_pretrained")
            if ast.unparse(c.func).startswith("PeftModel")
        ]
        assert peft_loads, f"{class_name} has no PeftModel load"
        for call in peft_loads:
            keyword = _revision_kwarg(call)
            assert keyword is not None and getattr(keyword.value, "id", None) == "revision"


@pytest.mark.parametrize("path, flag", [(LLAMA, "revision"), (VISION, "_revision")])
def test_a_pinned_load_does_not_mix_refs_with_vllm(path, flag):
    """load_vllm takes no revision, so vLLM fetches the default branch. Pinning only the
    config and tokenizer would put two refs in one model, so the pin is dropped instead."""
    source = path.read_text(encoding = "utf-8")
    tree = ast.parse(source)
    name = "FastLlamaModel" if path is LLAMA else "FastBaseModel"
    function = _function(tree, "from_pretrained", name)
    clears = [
        n for n in ast.walk(function)
        if isinstance(n, ast.Assign)
        and any(getattr(t, "id", None) == flag for t in n.targets)
        and isinstance(n.value, ast.Constant) and n.value.value is None
    ]
    assert clears, f"{path.name} never drops the revision on the vLLM path"
    # It must happen before the config load, or the config is pinned and the weights are not.
    configs = [
        c for c in _calls(function, "from_pretrained")
        if ast.unparse(c.func).startswith("AutoConfig")
    ]
    assert configs
    assert min(c.lineno for c in clears) < min(c.lineno for c in configs)


def test_local_snapshot_resolution_takes_the_revision():
    """A local snapshot dir cannot be re-pointed by a revision handed to from_pretrained,
    so the cache resolution itself has to select the requested ref."""
    tree = _tree(LOADER_UTILS)
    for name in ("_resolve_hub_repo_local_dir", "_hub_repo_or_local_path"):
        function = _function(tree, name)
        assert "revision" in _params(function), f"{name} must accept revision"
    resolver = _function(tree, "_resolve_hub_repo_local_dir")
    downloads = _calls(resolver, "hf_hub_download")
    assert downloads, "expected the cache probe download"
    for call in downloads:
        assert _revision_kwarg(call) is not None
    wrapper = _function(tree, "_hub_repo_or_local_path")
    inner = _calls(wrapper, "_resolve_hub_repo_local_dir")
    assert inner and all(_revision_kwarg(c) is not None for c in inner)
