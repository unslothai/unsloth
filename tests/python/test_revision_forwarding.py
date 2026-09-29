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
SAVE = REPO / "unsloth" / "save.py"


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
    # The ref only exists on the repo the caller named: elsewhere it 404s or, worse, resolves a same-named branch on a
    # different repo.
    assert _load_gate()("abc123", model_name, old_model_name) is None


def test_no_revision_stays_none_even_when_remapped():
    gate = _load_gate()
    assert gate(None, "unsloth/llama-3-8b-bnb-4bit", "meta-llama/Meta-Llama-3-8B") is None


def _gate_with_warnings():
    source = LOADER.read_text(encoding = "utf-8")
    function = _function(ast.parse(source), "_revision_for_resolved_repo")
    warnings = []
    namespace = {"logger": types.SimpleNamespace(warning_once = lambda m: warnings.append(m))}
    module = ast.Module(body = [function], type_ignores = [])
    ast.fix_missing_locations(module)
    exec(compile(module, str(LOADER), "exec"), namespace)
    return namespace["_revision_for_resolved_repo"], warnings


def test_the_gate_warns_exactly_once_when_it_drops_a_revision():
    gate, warnings = _gate_with_warnings()
    gate("abc123", "unsloth/x-bnb-4bit", "org/x", True)
    assert len(warnings) == 1
    message = warnings[0]
    # Both repos have to be named or the user cannot tell which load was silently redirected.
    assert "abc123" in message and "org/x" in message and "unsloth/x-bnb-4bit" in message


def test_exact_name_mode_is_only_offered_when_it_would_help():
    """It gates the mapper substitution alone. The ModelScope download, the
    ALLOW_PREQUANTIZED_MODELS strip and fast_inference_setup all ignore it, so
    recommending it there sends the caller round the same loop."""
    gate, warnings = _gate_with_warnings()
    gate("abc123", "unsloth/x-bnb-4bit", "org/x", True)
    assert "use_exact_model_name" in warnings[0]

    gate, warnings = _gate_with_warnings()
    gate("abc123", "/tmp/modelscope/x", "org/x", False)
    assert "use_exact_model_name" not in warnings[0]


def test_both_loader_paths_pass_the_mapper_flag():
    tree = _tree(LOADER)
    for class_name in ("FastLanguageModel", "FastModel"):
        function = _function(tree, "from_pretrained", class_name)
        assert [
            n
            for n in ast.walk(function)
            if isinstance(n, ast.Assign)
            and any(getattr(t, "id", None) == "mapper_moved_name" for t in n.targets)
        ], f"{class_name} must record whether the mapper moved the name"
        for call in _calls(function, "_revision_for_resolved_repo"):
            names = [getattr(a, "id", None) for a in call.args]
            assert "mapper_moved_name" in names, "the gate needs the flag to tailor its remedy"


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
            c
            for c in _calls(function, "from_pretrained")
            if ast.unparse(c.func).split(".")[0] in ("AutoConfig", "PeftConfig")
        ]
        assert probes, f"{class_name} has no config probe"
        gated = 0
        for probe in probes:
            assert probe.lineno > early.lineno, "the gate must precede the config probes"
            keyword = _revision_kwarg(probe)
            if keyword is None:
                continue  # the PEFT base-model probe deliberately pins nothing
            assert getattr(keyword.value, "id", None) in (
                "base_revision",
                "adapter_revision",
            ), f"probe at line {probe.lineno} uses the ungated revision"
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
            n
            for n in ast.walk(function)
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
            c
            for c in _calls(function, "from_pretrained")
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
        n
        for n in ast.walk(function)
        if isinstance(n, ast.Assign)
        and any(getattr(t, "id", None) == flag for t in n.targets)
        and isinstance(n.value, ast.Constant)
        and n.value.value is None
    ]
    assert clears, f"{path.name} never drops the revision on the vLLM path"
    # It must happen before the config load, or the config is pinned and the weights are not.
    configs = [
        c
        for c in _calls(function, "from_pretrained")
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


def test_the_vllm_drop_only_fires_when_vllm_owns_the_weights():
    """fast_inference is turned off in that same block when vLLM is missing or the GPU
    is too old, and a num_labels load goes through transformers regardless. Both of
    those can honour the pin, so the drop must not be unconditional."""
    function = _function(_tree(LLAMA), "from_pretrained", "FastLlamaModel")
    clears = [
        n
        for n in ast.walk(function)
        if isinstance(n, ast.Assign)
        and any(getattr(t, "id", None) == "revision" for t in n.targets)
        and isinstance(n.value, ast.Constant)
        and n.value.value is None
    ]
    assert clears, "the vLLM revision drop is gone"
    for clear in clears:
        guards = [
            n
            for n in ast.walk(function)
            if isinstance(n, ast.If)
            and n.lineno <= clear.lineno <= n.end_lineno
            and "revision" in ast.unparse(n.test)
        ]
        assert guards, "the drop needs its own condition"
        test = ast.unparse(guards[0].test)
        assert "fast_inference" in test, "must re-check fast_inference"
        assert "num_labels" in test, "a num_labels load runs in-process and can be pinned"


def test_the_tokenizer_revision_is_resolved_by_the_loader():
    """The tokenizer repo is not always the base model's: a PEFT load whose
    tokenizer_name is the adapter keeps the caller's ref, which the base model cannot."""
    tree = _tree(LOADER)
    helper = _function(tree, "_revision_for_tokenizer_repo")
    assert helper, "the loader must resolve the tokenizer repo's revision"
    for class_name in ("FastLanguageModel", "FastModel"):
        function = _function(tree, "from_pretrained", class_name)
        dispatches = [
            c
            for c in _calls(function, "from_pretrained")
            if any(k.arg == "tokenizer_revision" for k in c.keywords)
        ]
        assert dispatches, f"{class_name} must dispatch a tokenizer_revision"
        for call in dispatches:
            keyword = next(k for k in call.keywords if k.arg == "tokenizer_revision")
            assert isinstance(keyword.value, ast.Call), "it has to be the resolved value"


def test_llama_uses_the_dispatched_tokenizer_revision():
    function = _function(_tree(LLAMA), "from_pretrained", "FastLlamaModel")
    assert "tokenizer_revision" in _params(function)
    loads = _calls(function, "load_correct_tokenizer")
    assert loads
    for call in loads:
        keyword = _revision_kwarg(call)
        assert keyword is not None
        assert getattr(keyword.value, "id", None) == "tokenizer_revision"


def test_vision_pops_the_tokenizer_revision_before_the_weight_load():
    """FastBaseModel forwards **kwargs to the weight load, and transformers has no
    tokenizer_revision argument, so it must be popped rather than read."""
    function = _function(_tree(VISION), "from_pretrained", "FastBaseModel")
    pops = [
        c
        for c in ast.walk(function)
        if isinstance(c, ast.Call)
        and ast.unparse(c.func).endswith("kwargs.pop")
        and c.args
        and getattr(c.args[0], "value", None) == "tokenizer_revision"
    ]
    assert pops, "tokenizer_revision must be popped from kwargs"
    weight_loads = [
        c
        for c in _calls(function, "from_pretrained")
        if ast.unparse(c.func).startswith("auto_model")
    ]
    assert weight_loads
    assert pops[0].lineno < min(c.lineno for c in weight_loads)


def _load_tokenizer_gate():
    source = LOADER.read_text(encoding = "utf-8")
    function = _function(ast.parse(source), "_revision_for_tokenizer_repo")
    namespace = {}
    module = ast.Module(body = [function], type_ignores = [])
    ast.fix_missing_locations(module)
    exec(compile(module, str(LOADER), "exec"), namespace)
    return namespace["_revision_for_tokenizer_repo"]


def test_an_adapter_ref_never_reaches_the_base_tokenizer():
    """On a PEFT load the late gate is skipped, so the gated value still names the
    adapter. The base repo's tokenizer must take the model load's ref, which is None."""
    gate = _load_tokenizer_gate()
    # Remote adapter, no explicit tokenizer_name: the tokenizer follows the base model.
    assert gate(None, "org/base", "org/adapter", "v2", None) is None


def test_an_adapter_hosted_tokenizer_keeps_the_callers_ref():
    """An adapter is a separate repo with its own history, so the caller's ref still
    names it even though the base model it sits on cannot answer to it."""
    gate = _load_tokenizer_gate()
    assert gate("org/adapter", "org/base", "org/adapter", "v2", None, True) == "v2"


def test_a_remapped_plain_load_drops_the_tokenizer_pin_too():
    """Naming the requested repo as tokenizer_name must not smuggle the ref back in: the
    weights now come off a mirror's default branch, and a pinned tokenizer beside them is
    the ref mismatch the gate exists to prevent. Only a PEFT adapter is a separate repo."""
    gate = _load_tokenizer_gate()
    assert gate("org/model", "unsloth/model-bnb-4bit", "org/model", "v2", None) is None


def test_a_plain_load_gives_the_tokenizer_the_model_ref():
    gate = _load_tokenizer_gate()
    assert gate(None, "org/model", "org/model", "v2", "v2") == "v2"
    # A third-party tokenizer repo is pinned by neither.
    assert gate("other/tok", "org/model", "org/model", "v2", "v2") is None


def test_both_dispatches_share_one_model_revision():
    """The value handed to the base load and the one the tokenizer resolution sees have
    to be the same, or the PEFT case leaks the adapter ref into the base repo."""
    tree = _tree(LOADER)
    for class_name in ("FastLanguageModel", "FastModel"):
        function = _function(tree, "from_pretrained", class_name)
        assert [
            n
            for n in ast.walk(function)
            if isinstance(n, ast.Assign)
            and any(getattr(t, "id", None) == "model_revision" for t in n.targets)
        ], f"{class_name} must derive one model_revision"
        dispatch = next(
            c
            for c in _calls(function, "from_pretrained")
            if any(k.arg == "tokenizer_revision" for k in c.keywords)
        )
        model_kw = _revision_kwarg(dispatch)
        assert getattr(model_kw.value, "id", None) == "model_revision"
        tok_kw = next(k for k in dispatch.keywords if k.arg == "tokenizer_revision")
        assert "model_revision" in ast.unparse(tok_kw.value)


def test_a_direct_llama_call_still_pins_its_tokenizer():
    """FastLlamaModel is exported and the architecture wrappers forward only `revision`,
    so tokenizer_revision has to fall back to it when the repos are the same."""
    function = _function(_tree(LLAMA), "from_pretrained", "FastLlamaModel")
    fallbacks = [
        n
        for n in ast.walk(function)
        if isinstance(n, ast.Assign)
        and any(getattr(t, "id", None) == "tokenizer_revision" for t in n.targets)
        and getattr(n.value, "id", None) == "revision"
    ]
    assert fallbacks, "no fallback from revision to tokenizer_revision"
    warms = _calls(function, "maybe_prefetch_hf_snapshot")
    tokenizer_warms = [
        c
        for c in warms
        if any(
            k.arg == "revision" and getattr(k.value, "id", None) == "tokenizer_revision"
            for k in c.keywords
        )
    ]
    assert tokenizer_warms, "the tokenizer warm should use the same pin"
    # The fallback must precede the warm, or the warm fetches the wrong ref.
    assert fallbacks[0].lineno < min(c.lineno for c in tokenizer_warms)


@pytest.mark.parametrize(
    "path, cls, name",
    [
        (LLAMA, "FastLlamaModel", "tokenizer_revision"),
        (VISION, "FastBaseModel", "_tokenizer_revision_arg"),
    ],
)
def test_the_vllm_drop_clears_the_tokenizer_pin_too(path, cls, name):
    """Clearing only the model pin left vLLM on the default branch while the tokenizer
    stayed on the requested ref."""
    function = _function(ast.parse(path.read_text(encoding = "utf-8")), "from_pretrained", cls)
    clears = [
        n
        for n in ast.walk(function)
        if isinstance(n, ast.Assign)
        and any(getattr(t, "id", None) == name for t in n.targets)
        and isinstance(n.value, ast.Constant)
        and n.value.value is None
    ]
    assert clears, f"{path.name} never clears {name} on the vLLM path"


def _simulate_loader():
    """Run the loader's two revision decisions the way from_pretrained sequences them."""
    tree = ast.parse(LOADER.read_text(encoding = "utf-8"))
    namespace = {"logger": types.SimpleNamespace(warning_once = lambda *a, **k: None)}
    functions = [
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef)
        and n.name in ("_revision_for_resolved_repo", "_revision_for_tokenizer_repo")
    ]
    module = ast.Module(body = functions, type_ignores = [])
    ast.fix_missing_locations(module)
    exec(compile(module, str(LOADER), "exec"), namespace)
    gate = namespace["_revision_for_resolved_repo"]
    tokenizer_gate = namespace["_revision_for_tokenizer_repo"]

    def run(old_model_name, model_name, is_peft, tokenizer_name, revision, mapper_moved_name):
        base_revision = gate(revision, model_name, old_model_name, mapper_moved_name)
        if not is_peft:
            base_revision = gate(base_revision, model_name, old_model_name, mapper_moved_name)
        model_revision = base_revision if not is_peft else None
        return model_revision, tokenizer_gate(
            tokenizer_name, model_name, old_model_name, revision, model_revision, is_peft
        )

    return run


@pytest.mark.parametrize(
    "label, old_model_name, model_name, is_peft, tokenizer_name, revision, mapper_moved_name,"
    " expected_model, expected_tokenizer",
    [
        ("plain pinned load", "org/m", "org/m", False, None, "v2", False, "v2", "v2"),
        (
            "remapped to a prequant mirror",
            "org/m",
            "unsloth/m-bnb-4bit",
            False,
            None,
            "v2",
            True,
            None,
            None,
        ),
        # The adapter's ref is not the base repo's, and the tokenizer follows the base.
        ("PEFT, remote adapter", "org/ad", "org/base", True, None, "v2", False, None, None),
        # ... unless the tokenizer is the adapter itself, which the caller did pin.
        (
            "PEFT, adapter-hosted tokenizer",
            "org/ad",
            "org/base",
            True,
            "org/ad",
            "v2",
            False,
            None,
            "v2",
        ),
        (
            "plain load, third-party tokenizer",
            "org/m",
            "org/m",
            False,
            "other/tok",
            "v2",
            False,
            "v2",
            None,
        ),
        # Naming the requested repo back does not survive the remap: the weights moved.
        (
            "remapped, tokenizer named as the requested repo",
            "org/m",
            "unsloth/m-bnb-4bit",
            False,
            "org/m",
            "v2",
            True,
            None,
            None,
        ),
        ("no revision at all", "org/m", "unsloth/m-bnb-4bit", False, None, None, True, None, None),
    ],
    ids = lambda v: v if isinstance(v, str) and " " in v else None,
)
def test_the_revision_decision_matrix(
    label,
    old_model_name,
    model_name,
    is_peft,
    tokenizer_name,
    revision,
    mapper_moved_name,
    expected_model,
    expected_tokenizer,
):
    """One table for the whole contract: which repo each pin is allowed to reach."""
    run = _simulate_loader()
    model_revision, tokenizer_revision = run(
        old_model_name, model_name, is_peft, tokenizer_name, revision, mapper_moved_name
    )
    assert model_revision == expected_model, label
    assert tokenizer_revision == expected_tokenizer, label


def test_the_processor_fallback_carries_the_tokenizer_revision():
    """get_auto_processor runs when AutoProcessor raises, so it is a real load path: an
    unpinned one there hands back a default-branch processor beside pinned weights."""
    function = _function(_tree(VISION), "from_pretrained", "FastBaseModel")
    fallbacks = _calls(function, "get_auto_processor")
    assert fallbacks, "the processor fallback must still exist"
    for call in fallbacks:
        keyword = _revision_kwarg(call)
        assert keyword is not None, "the fallback needs the revision too"
        assert getattr(keyword.value, "id", None) == "_tokenizer_revision"


def test_the_fp8_quantizer_takes_the_requested_revision():
    """Its output path replaces model_name, so the gate downstream drops the pin. If it
    did not quantize the pinned ref itself, that ref never reaches the weights at all."""
    function = _function(_tree(LOADER_UTILS), "_offline_quantize_to_fp8")
    assert "revision" in _params(function)
    for callee in ("from_pretrained",):
        loads = _calls(function, callee)
        assert loads
        for call in loads:
            assert _revision_kwarg(call) is not None, ast.unparse(call.func)


def test_the_fp8_cache_name_is_revision_specific():
    """A shared temp dir keyed only on the repo name would serve one ref's artifact to
    another, and the artifact outlives the process that built it."""
    function = _function(_tree(LOADER_UTILS), "_offline_quantize_to_fp8")
    writes = [
        n
        for n in ast.walk(function)
        if isinstance(n, ast.AugAssign) and getattr(n.target, "id", None) == "cache_name"
    ]
    assert writes
    guarded = [
        n
        for n in ast.walk(function)
        if isinstance(n, ast.If)
        and "revision" in ast.unparse(n.test)
        and any(n.lineno <= w.lineno <= n.end_lineno for w in writes)
    ]
    assert guarded, "two revisions of one repo would share a cache entry"


def test_both_loaders_hand_the_fp8_quantizer_the_revision():
    tree = _tree(LOADER)
    for class_name in ("FastLanguageModel", "FastModel"):
        function = _function(tree, "from_pretrained", class_name)
        calls = _calls(function, "_offline_quantize_to_fp8")
        assert calls, f"{class_name} must still quantize on the fly"
        for call in calls:
            keyword = _revision_kwarg(call)
            assert keyword is not None
            assert (
                getattr(keyword.value, "id", None) == "revision"
            ), "the fp8 source is still the caller's own repo here"


def test_the_vllm_drop_happens_before_the_config_probe():
    """model_types, auto_model and the text-only decision all come off the probed config.
    Reading it at a ref vLLM will not fetch picks the dispatch for a different model, so
    the pin has to be gone before the probe, not just before the dispatch."""
    function = _function(_tree(LOADER), "from_pretrained", "FastModel")
    drops = [
        n
        for n in ast.walk(function)
        if isinstance(n, ast.If)
        and "is_vLLM_available" in ast.unparse(n.test)
        and any(
            isinstance(b, ast.Assign)
            and any(getattr(t, "id", None) == "base_revision" for t in b.targets)
            and isinstance(b.value, ast.Constant)
            and b.value.value is None
            for b in n.body
        )
    ]
    assert drops, "FastModel never drops base_revision for the vLLM path"
    probes = [
        c
        for c in _calls(function, "from_pretrained")
        if ast.unparse(c.func).split(".")[0] in ("AutoConfig", "PeftConfig")
    ]
    assert probes
    assert drops[0].end_lineno < min(
        c.lineno for c in probes
    ), "the probe would read a ref the weights will not be at"
    # The probed config goes down untouched again, so nothing may re-gate it at dispatch.
    dispatches = [
        c
        for c in _calls(function, "from_pretrained")
        if ast.unparse(c.func).startswith("FastBaseModel")
    ]
    assert dispatches
    for call in dispatches:
        keyword = next((k for k in call.keywords if k.arg == "auto_config"), None)
        assert keyword is not None
        assert getattr(keyword.value, "id", None) == "model_config"


def test_the_fp8_cache_key_survives_a_lossy_sanitization():
    """The readable half replaces every unsafe character with the same one, so `a/b` and
    `a.b` collapse together. Only a digest of the raw ref keeps them apart."""
    function = _function(_tree(LOADER_UTILS), "_offline_quantize_to_fp8")
    source = ast.unparse(function)
    assert "sha256" in source or "blake2" in source, "the sanitized name alone collides"
    digests = [
        n for n in ast.walk(function) if isinstance(n, ast.Call) and "sha256" in ast.unparse(n.func)
    ]
    assert digests
    assert any("revision" in ast.unparse(n) for n in digests), "hash the ref, not the repo"


def test_the_peft_probe_keeps_the_adapter_ref_under_vllm():
    """The vLLM drop runs before is_peft is known. An adapter is loaded in-process by peft,
    so zeroing its probe would read the default branch and either miss PEFT entirely or
    resolve a different base model before attaching the pinned adapter."""
    tree = _tree(LOADER)
    for class_name in ("FastLanguageModel", "FastModel"):
        function = _function(tree, "from_pretrained", class_name)
        probes = [
            c
            for c in _calls(function, "from_pretrained")
            if ast.unparse(c.func).startswith("PeftConfig")
        ]
        assert probes, f"{class_name} must still probe for an adapter"
        for call in probes:
            keyword = _revision_kwarg(call)
            assert keyword is not None
            assert (
                getattr(keyword.value, "id", None) == "adapter_revision"
            ), "the adapter probe must not take the base model's gated ref"


def test_both_loaders_drop_the_vllm_pin_before_the_probe():
    """model_types picks the architecture class off the probed config, so reading it at a
    ref vLLM will not fetch dispatches the wrong one."""
    tree = _tree(LOADER)
    for class_name in ("FastLanguageModel", "FastModel"):
        function = _function(tree, "from_pretrained", class_name)
        drops = [
            n
            for n in ast.walk(function)
            if isinstance(n, ast.If)
            and any(
                isinstance(b, ast.Assign)
                and any(getattr(t, "id", None) == "base_revision" for t in b.targets)
                and isinstance(b.value, ast.Constant)
                and b.value.value is None
                for b in n.body
            )
        ]
        assert drops, f"{class_name} never drops base_revision for the vLLM path"
        probes = [
            c
            for c in _calls(function, "from_pretrained")
            if ast.unparse(c.func).split(".")[0] in ("AutoConfig", "PeftConfig")
        ]
        assert probes
        assert drops[0].end_lineno < min(
            c.lineno for c in probes
        ), f"{class_name} probes at a ref the weights will not be at"


def test_llama_owns_the_vllm_predicate_the_loader_gates_on():
    """FastLanguageModel also falls back in-process on pre-Volta GPUs and for a num_labels
    load, so the loader cannot gate on `fast_inference and is_vLLM_available()` the way the
    FastModel path does. One helper, used by both, or the two drift apart."""
    tree = _tree(LLAMA)
    helper = _function(tree, "_vllm_will_load_weights")
    source = ast.unparse(helper)
    for token in ("is_vLLM_available", "get_device_capability", "hip", "num_labels"):
        assert token in source, token

    guard = _function(tree, "from_pretrained", "FastLlamaModel")
    assert _calls(guard, "_vllm_will_load_weights"), "llama.py must use its own helper"
    loader = _function(_tree(LOADER), "from_pretrained", "FastLanguageModel")
    assert _calls(loader, "_vllm_will_load_weights"), "the loader must gate on the same one"


def test_a_pinned_tokenizer_is_stamped_for_the_save_path():
    """save.py restores tokenizer.model from tokenizer.name_or_path, which names the repo
    but not the branch, so a merged export would copy the default branch's asset."""
    stamps = _calls(
        _function(_tree(TOKENIZER_UTILS), "load_correct_tokenizer"), "_mark_loaded_revision"
    )
    assert stamps, "the loaded ref has to travel with the tokenizer"
    assert any(
        any(getattr(a, "id", None) == "revision" for a in c.args) for c in stamps
    ), "stamp the ref that was actually loaded"

    tree = _tree(LOADER_UTILS)
    assert _function(tree, "_mark_loaded_revision")
    assert _function(tree, "_tokenizer_revision")
    assert "revision" in _params(_function(tree, "_resolve_hub_repo_cached_file"))


@pytest.mark.parametrize(
    "callee", ["_resolve_hub_repo_cached_file", "hf_hub_download", "model_info"]
)
def test_the_sentencepiece_restore_reads_the_stamped_ref(callee):
    tree = _tree(SAVE)
    functions = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef)
        and n.name in ("_has_tokenizer_model", "_preserve_sentencepiece_tokenizer_assets")
    ]
    assert functions
    calls = [c for f in functions for c in _calls(f, callee)]
    assert calls, f"{callee} not found on the save path"
    for call in calls:
        assert _revision_kwarg(call) is not None, f"{callee} at line {call.lineno} drops the ref"


def test_the_vision_path_stamps_its_pinned_tokenizer_too():
    """FastBaseModel builds its processor without load_correct_tokenizer, so the stamp the
    save path reads has to be applied here as well or a pinned VLM load saves the default
    branch's tokenizer.model. At the return, so a patch fallback cannot lose it."""
    function = _function(_tree(VISION), "from_pretrained", "FastBaseModel")
    stamps = _calls(function, "_mark_loaded_revision")
    assert stamps, "the vision path never stamps its loaded ref"
    for call in stamps:
        assert any(
            getattr(a, "id", None) == "_tokenizer_revision" for a in call.args
        ), "stamp the ref the tokenizer was actually read at"
    returns = [n for n in ast.walk(function) if isinstance(n, ast.Return)]
    assert returns
    assert max(c.lineno for c in stamps) < max(r.lineno for r in returns)
