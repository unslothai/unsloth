# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""`save_method = "lora"` saves the adapter, and `safe_serialization = None` is safetensors.

Two defects, one code path.

`patch_saving_functions` binds `unsloth_generic_save_pretrained_merged` and
`unsloth_generic_push_to_hub_merged` on every model, and the PEFT branch of
`unsloth_generic_save` handed every `save_method` to
`unsloth_zoo.saving_utils.merge_and_overwrite_lora`, which has no `"lora"` branch. The
value matched nothing and fell through to a plain 16bit merge, so a caller asking for an
adapter got a full-size merged checkpoint with no `adapter_config.json` (measured at
2.47 GB for a 1B base). `unsloth_save_model` still had the adapter branch, but nothing
reached it.

`safe_serialization = None` is what Unsloth's own warning and the troubleshooting docs
tell a caller to pass to FORCE safetensors, and `None` is falsy to peft and to
transformers, so it wrote `adapter_model.bin` instead: the advice produced the file it
exists to avoid (unslothai/unsloth#1792).

`unsloth.save` cannot be imported on a GPU-less host, so the functions under test are
extracted with `ast` and exec'd against fakes, like the other tests in this directory.
This file therefore runs on Linux, macOS and Windows with no accelerator and no network.
"""

from __future__ import annotations

import ast
import sys
import types
from pathlib import Path

import pytest


_SAVE_PY = Path(__file__).resolve().parent.parent.parent / "unsloth" / "save.py"
_SOURCE = _SAVE_PY.read_text(encoding = "utf-8")
_TREE = ast.parse(_SOURCE)


def _function_source(name: str) -> str:
    for node in ast.walk(_TREE):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            segment = ast.get_source_segment(_SOURCE, node)
            # Nested definitions carry their enclosing indentation.
            indent = len(segment) - len(segment.lstrip())
            if indent:
                segment = "\n".join(line[indent:] for line in segment.split("\n"))
            # The decorators are outside the segment already; nothing else to strip.
            return segment
    raise AssertionError(f"{name} not found in unsloth/save.py")


def _load(*names, **env):
    """Exec the named top-level functions against `env` and return the namespace."""
    namespace = dict(env)
    for name in names:
        exec(compile(_function_source(name), str(_SAVE_PY), "exec"), namespace)
    return namespace


# --------------------------------------------------------------------------- helpers


@pytest.mark.parametrize(
    "value, expected",
    [(None, True), (True, True), (False, False)],
)
def test_none_means_the_safetensors_default(value, expected):
    """`None` is the documented override, so it must not fall through as falsy."""
    namespace = _load("_normalize_safe_serialization")
    assert namespace["_normalize_safe_serialization"](value) is expected


@pytest.mark.parametrize(
    "value, expected",
    [
        ("lora", True),
        ("LoRA", True),
        (" lora ", True),
        ("Lora", True),
        ("merged_16bit", False),
        ("merged_4bit", False),
        ("merged 16bit", False),
        ("mxfp4", False),
        ("lora_16bit", False),
        ("", False),
        (None, False),
        (17, False),
    ],
)
def test_the_adapter_save_method_is_spelled_the_same_way_everywhere(value, expected):
    """Same normalisation the other `save_method` readers use, and never a crash on None.

    Studio passes `save_method = None` for whisper, so a non-string must answer False
    rather than raise.
    """
    namespace = _load("_is_adapter_save_method")
    assert namespace["_is_adapter_save_method"](value) is expected


def test_push_keywords_this_transformers_cannot_take_are_dropped():
    """transformers 5 removed `use_temp_dir` and `safe_serialization` from push_to_hub."""
    warnings_seen = []
    namespace = _load(
        "_filter_push_to_hub_kwargs",
        logger = types.SimpleNamespace(warning_once = lambda message: warnings_seen.append(message)),
    )

    def transformers_5_push(
        repo_id,
        *,
        commit_message = None,
        commit_description = None,
        private = None,
        token = None,
        revision = None,
        create_pr = False,
        max_shard_size = "50GB",
        tags = None,
    ):
        raise AssertionError("not called")

    kept = namespace["_filter_push_to_hub_kwargs"](
        transformers_5_push,
        dict(
            repo_id = "owner/model",
            use_temp_dir = None,
            safe_serialization = True,
            max_shard_size = "5GB",
            tags = ["unsloth"],
        ),
    )
    assert kept == dict(repo_id = "owner/model", max_shard_size = "5GB", tags = ["unsloth"])
    # Neither loss changes the upload on this transformers, so neither is reported.
    assert warnings_seen == []


def test_a_dropped_pickle_request_is_reported():
    """`safe_serialization = False` asked for a pickle and will not get one: say so."""
    warnings_seen = []
    namespace = _load(
        "_filter_push_to_hub_kwargs",
        logger = types.SimpleNamespace(warning_once = lambda message: warnings_seen.append(message)),
    )

    def transformers_5_push(repo_id, *, token = None):
        raise AssertionError("not called")

    kept = namespace["_filter_push_to_hub_kwargs"](
        transformers_5_push,
        dict(repo_id = "owner/model", safe_serialization = False),
    )
    assert kept == dict(repo_id = "owner/model")
    assert len(warnings_seen) == 1 and "safe_serialization" in warnings_seen[0]


def test_a_var_keyword_signature_keeps_everything():
    """peft's own wrappers take **kwargs, so nothing may be filtered out of them."""
    namespace = _load(
        "_filter_push_to_hub_kwargs", logger = types.SimpleNamespace(warning_once = lambda m: None)
    )

    def anything(repo_id, **kwargs):
        raise AssertionError("not called")

    arguments = dict(repo_id = "owner/model", use_temp_dir = True, safe_serialization = False)
    assert namespace["_filter_push_to_hub_kwargs"](anything, arguments) == arguments


def test_an_unreadable_callable_is_left_alone():
    """An object with no readable signature forwards unchanged, which is what main did."""
    namespace = _load(
        "_filter_push_to_hub_kwargs", logger = types.SimpleNamespace(warning_once = lambda m: None)
    )
    arguments = dict(repo_id = "owner/model", use_temp_dir = True)
    assert namespace["_filter_push_to_hub_kwargs"](object(), arguments) == arguments


# --------------------------------------------------------------- the routing decision


class _PeftModel:
    """Stands in for `peft.PeftModel`; the branch under test is an isinstance check."""

    def __init__(self):
        self.config = types.SimpleNamespace(_name_or_path = "base/model", model_type = "llama")
        self.saved = []

    def state_dict(self):
        return {}

    def save_pretrained(self, directory, **kwargs):
        self.saved.append((directory, kwargs))


class _FullModel:
    """A model with no adapter. Deliberately NOT a subclass of the PeftModel stand-in, so
    the isinstance check the branch turns on answers False here."""

    def __init__(self):
        self.config = types.SimpleNamespace(_name_or_path = "base/model", model_type = "llama")
        self.saved = []

    def state_dict(self):
        return {}

    def save_pretrained(self, directory, **kwargs):
        self.saved.append((directory, kwargs))


def _routing_environment(monkeypatch, model):
    calls = {"merge": [], "adapter": [], "prewarm": []}

    zoo = types.ModuleType("unsloth_zoo.saving_utils")
    zoo.merge_and_overwrite_lora = lambda *args, **kwargs: calls["merge"].append(kwargs)
    monkeypatch.setitem(sys.modules, "unsloth_zoo.saving_utils", zoo)

    namespace = _load(
        "_normalize_safe_serialization",
        "_is_adapter_save_method",
        "unsloth_generic_save",
        PeftModel = _PeftModel,
        PreTrainedTokenizerBase = type("Tokenizer", (), {}),
        ProcessorMixin = type("Processor", (), {}),
        patch_saving_functions = lambda tokenizer: tokenizer,
        get_token = lambda: "fixture-token",
        get_model_name = lambda name: name,
        _push_merged_to_hub_revision = lambda kwargs: calls.setdefault("revision", []).append(kwargs),
        _prewarm_base_model_hub_cache = lambda *args, **kwargs: calls["prewarm"].append(kwargs),
        _is_qwen3_5_vlm = lambda model: False,
        _determine_username = lambda repo, old, token: (repo, "owner"),
        unsloth_save_model = lambda *args, **kwargs: calls["adapter"].append(kwargs),
        logger = types.SimpleNamespace(warning_once = lambda *a, **k: None),
        gc = types.SimpleNamespace(collect = lambda: None),
        torch = types.SimpleNamespace(
            bfloat16 = "bfloat16",
            float16 = "float16",
            save = lambda *args, **kwargs: None,
            cuda = types.SimpleNamespace(is_bf16_supported = lambda: False),
        ),
    )
    return namespace["unsloth_generic_save"], calls


@pytest.mark.parametrize("spelling", ["lora", "LoRA", " lora "])
def test_an_adapter_save_never_reaches_the_merge(monkeypatch, tmp_path, spelling):
    """The defect: "lora" matched no branch inside the merge and was merged anyway."""
    model = _PeftModel()
    generic_save, calls = _routing_environment(monkeypatch, model)
    generic_save(model, None, save_directory = str(tmp_path), save_method = spelling)
    assert calls["merge"] == [], "save_method='lora' must not call merge_and_overwrite_lora"
    assert len(calls["adapter"]) == 1
    # The canonical spelling, not the caller's: `unsloth_save_model` normalises with
    # `.lower().replace(" ", "_")` and then rejects anything that is not exactly "lora",
    # so `" lora "` forwarded verbatim becomes `"_lora_"` and raises. See
    # test_the_adapter_save_method_the_router_forwards_is_one_unsloth_save_model_accepts.
    assert calls["adapter"][0]["save_method"] == "lora"
    assert calls["adapter"][0]["save_directory"] == str(tmp_path)
    # The base model is only needed to merge against, so nothing is downloaded for it.
    assert calls["prewarm"] == []


def test_the_adapter_save_method_the_router_forwards_is_one_unsloth_save_model_accepts():
    """The two ends of the new route must agree on the spelling, or the route raises.

    `_is_adapter_save_method` is deliberately lenient: it strips and case-folds, so
    `" lora "` and `"LoRA"` select the adapter save. `unsloth_save_model` is not: it
    normalises with `.lower().replace(" ", "_")`, which turns `" lora "` into `"_lora_"`,
    and then raises RuntimeError on anything that is not exactly one of its three values.
    Forwarding the caller's spelling verbatim would therefore trade a wrong merge for a
    crash on the same input, so the router forwards the canonical value. Read out of the
    source rather than asserted about a stub, so that renaming either end fails here.
    """
    source = Path(_SAVE_PY).read_text(encoding = "utf-8")
    tree = ast.parse(source)

    def _find(name):
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == name:
                return node
        raise AssertionError(f"{name} is gone from unsloth/save.py")

    # What the router hands to unsloth_save_model on the adapter branch.
    forwarded = [
        keyword.value.value
        for node in ast.walk(_find("unsloth_generic_save"))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "unsloth_save_model"
        for keyword in node.keywords
        if keyword.arg == "save_method" and isinstance(keyword.value, ast.Constant)
    ]
    assert forwarded == ["lora"], (
        "the adapter branch must forward the canonical spelling, got " + repr(forwarded)
    )

    # What unsloth_save_model does to it, and what it then insists on.
    accepted = {
        comparator.value
        for node in ast.walk(_find("unsloth_save_model"))
        if isinstance(node, ast.Compare)
        for comparator in node.comparators
        if isinstance(comparator, ast.Constant) and isinstance(comparator.value, str)
    }
    assert "lora" in accepted, "unsloth_save_model no longer names 'lora' as a save_method"
    for value in forwarded:
        assert (
            value.lower().replace(" ", "_") in accepted
        ), f"unsloth_save_model would reject the forwarded save_method {value!r}"


@pytest.mark.parametrize("save_method", ["merged_16bit", "mxfp4", "fp8", "merged_4bit_forced"])
def test_every_other_method_still_merges(monkeypatch, tmp_path, save_method):
    """The merge is what changed for exactly one value of save_method and no other."""
    model = _PeftModel()
    generic_save, calls = _routing_environment(monkeypatch, model)
    generic_save(model, None, save_directory = str(tmp_path), save_method = save_method)
    assert calls["adapter"] == []
    assert len(calls["merge"]) == 1
    assert len(calls["prewarm"]) == 1


def test_a_model_with_no_adapter_is_unchanged(monkeypatch, tmp_path):
    """A full fine-tune asked for "lora" has no adapter, so it writes itself, as before."""
    model = _FullModel()
    generic_save, calls = _routing_environment(monkeypatch, model)
    generic_save(model, None, save_directory = str(tmp_path), save_method = "lora")
    assert calls["merge"] == [] and calls["adapter"] == []
    assert len(model.saved) == 1


def test_none_is_normalised_before_the_merge_is_reached(monkeypatch, tmp_path):
    """Whatever the writer, `None` must have become `True` by the time it is forwarded."""
    model = _PeftModel()
    generic_save, calls = _routing_environment(monkeypatch, model)
    generic_save(
        model, None, save_directory = str(tmp_path), save_method = "lora", safe_serialization = None
    )
    assert calls["adapter"][0]["safe_serialization"] is True

    model = _FullModel()
    generic_save, calls = _routing_environment(monkeypatch, model)
    generic_save(
        model,
        None,
        save_directory = str(tmp_path),
        save_method = "merged_16bit",
        safe_serialization = None,
    )
    assert model.saved[0][1]["safe_serialization"] is True


# --------------------------------------------------------- what the adapter save writes


def _adapter_save_environment(monkeypatch):
    # unsloth_save_model checks the credential with huggingface_hub.whoami before pushing,
    # and these tests never reach the network.
    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "whoami", lambda token = None: {"name": "owner"})

    # `unsloth_save_model` does `from peft import PeftModelForCausalLM` inside its body, and
    # uses it for one isinstance check that every model here answers False to. Stubbed like
    # every other dependency in this file, rather than importorskip'd, so the whole file
    # stays runnable on a bare interpreter: that is what lets it gate the cross-platform
    # runners, which ship no peft.
    if "peft" not in sys.modules:
        peft = types.ModuleType("peft")
        peft.PeftModelForCausalLM = type("PeftModelForCausalLM", (), {})
        monkeypatch.setitem(sys.modules, "peft", peft)

    uploads = []

    def upload_to_huggingface(*args, **kwargs):
        uploads.append(kwargs)
        return None

    return _load(
        "_normalize_safe_serialization",
        "_filter_push_to_hub_kwargs",
        "unsloth_save_model",
        PreTrainedTokenizerBase = type("Tokenizer", (), {}),
        ProcessorMixin = type("Processor", (), {}),
        patch_saving_functions = lambda tokenizer: tokenizer,
        get_token = lambda: "fixture-token",
        upload_to_huggingface = upload_to_huggingface,
        logger = types.SimpleNamespace(warning_once = lambda *a, **k: None),
        gc = types.SimpleNamespace(collect = lambda: None),
        psutil = types.SimpleNamespace(cpu_count = lambda logical = True: 8),
        torch = types.SimpleNamespace(
            save = lambda *args, **kwargs: None,
            cuda = types.SimpleNamespace(empty_cache = lambda: None),
        ),
        fast_save_pickle = lambda *args, **kwargs: None,
    ), uploads


class _AdapterModel:
    def __init__(self, push_signature):
        self.config = types.SimpleNamespace(_name_or_path = "base/model", model_type = "llama")
        self.saved = []
        self.pushed = []
        self.original_push_to_hub = push_signature(self.pushed)

    def add_model_tags(self, tags):
        pass

    def push_to_hub(self, **kwargs):
        # `getattr(model, "original_push_to_hub", model.push_to_hub)` evaluates its default
        # eagerly, so the attribute has to exist; the patched model always has both.
        raise AssertionError("the unpatched push_to_hub must not be the one called")

    def save_pretrained(self, **kwargs):
        self.saved.append(kwargs)


def test_the_adapter_save_forwards_a_real_safe_serialization(monkeypatch, tmp_path):
    """`None` must not reach `save_pretrained`, and the bookkeeping must not either."""
    namespace, _ = _adapter_save_environment(monkeypatch)
    model = _AdapterModel(lambda sink: (lambda **kwargs: sink.append(kwargs)))
    namespace["unsloth_save_model"](
        model,
        None,
        save_directory = str(tmp_path),
        save_method = "lora",
        safe_serialization = None,
    )
    assert len(model.saved) == 1
    settings = model.saved[0]
    assert settings["safe_serialization"] is True
    assert settings["save_directory"] == str(tmp_path)
    # A local kept for the normalisation must not be handed on as a save keyword.
    assert "_force_safe_serialization" not in settings
    assert "save_method" not in settings


def test_an_adapter_push_survives_a_transformers_that_dropped_the_keywords(monkeypatch, tmp_path):
    """transformers 5's push_to_hub has no `use_temp_dir`, and this call used to pass it.

    The adapter branch was unreachable through save_pretrained_merged, so the TypeError
    it raises on transformers 5 was invisible until the routing above was fixed.
    """
    namespace, uploads = _adapter_save_environment(monkeypatch)

    def transformers_5_signature(sink):
        def push_to_hub(
            repo_id,
            *,
            commit_message = None,
            commit_description = None,
            private = None,
            token = None,
            revision = None,
            create_pr = False,
            max_shard_size = "50GB",
            tags = None,
        ):
            sink.append(
                dict(
                    repo_id = repo_id,
                    commit_message = commit_message,
                    private = private,
                    token = token,
                    revision = revision,
                    create_pr = create_pr,
                    max_shard_size = max_shard_size,
                    tags = tags,
                )
            )

        return push_to_hub

    model = _AdapterModel(transformers_5_signature)
    namespace["unsloth_save_model"](
        model,
        None,
        save_directory = "owner/model",
        save_method = "lora",
        push_to_hub = True,
        token = "fixture-token",
    )
    assert len(model.pushed) == 1
    assert model.pushed[0]["repo_id"] == "owner/model"
    assert "unsloth" in model.pushed[0]["tags"]
    assert len(uploads) == 1
    # Nothing was written locally: an adapter push goes straight to the Hub.
    assert model.saved == []


def test_an_adapter_push_still_passes_every_keyword_a_transformers_4_accepts(monkeypatch, tmp_path):
    """The filter must subtract only what the installed signature cannot take."""
    namespace, _ = _adapter_save_environment(monkeypatch)

    def transformers_4_signature(sink):
        def push_to_hub(
            repo_id,
            use_temp_dir = None,
            commit_message = None,
            private = None,
            token = None,
            max_shard_size = "5GB",
            create_pr = False,
            safe_serialization = True,
            revision = None,
            commit_description = None,
            tags = None,
        ):
            sink.append(
                dict(
                    use_temp_dir = use_temp_dir,
                    safe_serialization = safe_serialization,
                    revision = revision,
                )
            )

        return push_to_hub

    model = _AdapterModel(transformers_4_signature)
    namespace["unsloth_save_model"](
        model,
        None,
        save_directory = "owner/model",
        save_method = "lora",
        push_to_hub = True,
        token = "fixture-token",
        safe_serialization = None,
        use_temp_dir = True,
        revision = "candidate",
    )
    assert model.pushed == [dict(use_temp_dir = True, safe_serialization = True, revision = "candidate")]


# ------------------------------------------------------- the model's own save_pretrained


def test_the_model_save_pretrained_wrapper_rewrites_only_none():
    """`model.save_pretrained(..., safe_serialization = None)` is the call in #1792."""
    namespace = _load(
        "unsloth_model_save_pretrained",
        "_normalize_safe_serialization",
    )
    wrapper = namespace["unsloth_model_save_pretrained"]

    class Model:
        def __init__(self):
            self.calls = []

        def original_model_save_pretrained(self, *args, **kwargs):
            self.calls.append((args, kwargs))
            return "result"

    model = Model()
    assert wrapper(model, "out", safe_serialization = None) == "result"
    assert model.calls[-1] == (("out",), {"safe_serialization": True})

    wrapper(model, "out", safe_serialization = False)
    assert model.calls[-1] == (("out",), {"safe_serialization": False})

    wrapper(model, "out", max_shard_size = "5GB")
    assert model.calls[-1] == (("out",), {"max_shard_size": "5GB"})


def test_the_wrapper_is_installed_on_models_and_is_idempotent():
    """A second `patch_saving_functions` must not wrap the wrapper."""
    source = _function_source("patch_saving_functions")
    assert "model.original_model_save_pretrained = model.save_pretrained" in source
    assert '!= "unsloth_model_save_pretrained"' in source
    # Its own attribute name, so the tokenizer wrapper above cannot be shadowed by it.
    assert "original_save_pretrained" in source and "original_model_save_pretrained" in source


def test_the_generated_push_to_hub_normalises_none():
    """`unsloth_push_to_hub` is built from a source template, so gate it as source."""
    source = _function_source("patch_saving_functions")
    assert 'arguments["safe_serialization"] is None' in source
    assert 'arguments["safe_serialization"] = True' in source


def test_the_documented_advice_no_longer_tells_anyone_to_pass_none_for_a_pickle():
    """The warning used to say "to force safe_serialization, set it to None"."""
    assert "To force `safe_serialization`, set it to `None` instead." not in _SOURCE
    assert "`safe_serialization` defaults to safetensors" in _SOURCE


# ----------------------------------------------------------------------------------
# What the caller is left holding: the SentenceTransformer wrapper.
# ----------------------------------------------------------------------------------


def _sentence_transformer_source():
    path = (
        Path(__file__).resolve().parent.parent.parent
        / "unsloth"
        / "models"
        / "sentence_transformer.py"
    )
    return path.read_text(encoding = "utf-8"), ast.parse(path.read_text(encoding = "utf-8"))


def _modules_branch_save_pretrained_merged(tree):
    """The second `_save_pretrained_merged`, the one that keeps `save_method`.

    The first definition refuses everything but a merge outright; this is the branch that
    forwards `save_method` on to `auto_model.save_pretrained_merged`, so it is the one
    that inherits whatever `"lora"` now means.
    """
    found = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_save_pretrained_merged"
    ]
    assert len(found) == 2, f"expected two definitions, found {len(found)}"
    keeps_save_method = [
        node
        for node in found
        if any(
            isinstance(call, ast.Call)
            and getattr(getattr(call.func, "attr", None), "__str__", lambda: "")() == "setdefault"
            for call in ast.walk(node)
        )
    ]
    assert len(keeps_save_method) == 1
    return keeps_save_method[0]


def test_sentence_transformer_merge_refuses_the_adapter_save_method():
    """An adapter-only save leaves a SentenceTransformer directory with no model in it.

    `self.save_pretrained(save_directory)` writes the scaffolding and, for a PEFT
    auto_model, an adapter; the wrapper then deletes that adapter and hands the transformer
    module to `save_pretrained_merged`. With `save_method = "lora"` that call now writes the
    adapter back and nothing else, so the directory ends up with `modules.json` and
    `adapter_config.json` but no `config.json` and no weights. `SentenceTransformer` cannot
    load it, and `_push_to_hub_merged` uploads exactly that directory.

    Before the routing fix, `"lora"` reached `merge_and_overwrite_lora`, matched no branch
    and fell through to a 16-bit merge, so this path happened to write something loadable.
    Both sibling branches in this file already refuse the method for the same reason; this
    pins the third.
    """
    _, tree = _sentence_transformer_source()
    node = _modules_branch_save_pretrained_merged(tree)
    guards = [
        call
        for call in ast.walk(node)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Name)
        and call.func.id == "_is_adapter_save_method"
    ]
    assert guards, (
        "the modules branch of _save_pretrained_merged forwards save_method = 'lora' to "
        "the adapter save, which writes no base weights into the SentenceTransformer "
        "directory it is building"
    )
    raises = [
        stmt
        for stmt in ast.walk(node)
        if isinstance(stmt, ast.Raise)
        and isinstance(stmt.exc, ast.Call)
        and getattr(stmt.exc.func, "id", "") == "NotImplementedError"
    ]
    assert len(raises) >= 2, "the adapter method has to be refused, not warned about"


def test_sentence_transformer_shares_the_router_definition_of_lora():
    """One definition of the spellings, so the two files cannot drift apart."""
    source, _ = _sentence_transformer_source()
    assert "_is_adapter_save_method" in source
    assert "from ..save import" in source


def test_the_lora_docstring_does_not_promise_an_adapter_only_directory():
    """`save_method="lora"` with a tokenizer writes tokenizer files too.

    The adapter branch of `unsloth_save_model` calls `tokenizer.save_pretrained` when the
    documented `tokenizer` argument is supplied, so "and nothing else" was false for the
    ordinary supported call. What the route really guarantees is that no base-model
    weights are written, which is the claim these docstrings now make.
    """
    import re
    from pathlib import Path

    source = (Path(__file__).resolve().parents[2] / "unsloth" / "save.py").read_text(
        encoding = "utf-8",
    )
    assert "and nothing else. Useful for HF inference." not in source, (
        "a save_method='lora' docstring still promises an adapter-only directory, which a "
        "call that passes `tokenizer` does not produce"
    )
    promises = re.findall(r"`adapter_model\.safetensors`,[^\n]*", source)
    assert promises, "the save_method list no longer names adapter_model.safetensors"
    for promise in promises:
        assert "no base-model weights" in promise, promise


@pytest.mark.parametrize("spelling", ["lora", "LoRA", " lora ", "  LORA", "lora\t", " Lora "])
def test_the_sentence_transformer_normaliser_keeps_whitespace_aliases_recognisable(spelling):
    """`_normalize_save_method` runs BEFORE the adapter guard, so it must not turn a
    spelling `_is_adapter_save_method` accepts into one it does not.

    It folded spaces to underscores without stripping first, so `" lora "` became
    `"_lora_"`, the guard returned False, and the modules-based SentenceTransformer path
    forwarded the value to `auto_model.save_pretrained_merged` instead of raising the
    NotImplementedError the two sibling branches raise. That is the merge fallthrough this
    PR exists to remove, reached through a spelling the router itself calls LoRA.
    """
    from unsloth.models.sentence_transformer import _normalize_save_method
    from unsloth.save import _is_adapter_save_method

    assert _is_adapter_save_method(spelling), "the router already calls this spelling LoRA"
    assert _is_adapter_save_method(_normalize_save_method(spelling)), (
        f"_normalize_save_method({spelling!r}) produced "
        f"{_normalize_save_method(spelling)!r}, which the adapter guard no longer accepts"
    )


@pytest.mark.parametrize(
    "spelling, expected",
    [
        ("merged_16bit", "merged_16bit"),
        (" MERGED 16BIT ", "merged_16bit"),
        ("merged 16bit", "merged_16bit"),
        # NEGATIVE CONTROL: a non-string is handed back untouched (Studio passes None
        # for whisper), and an unrelated method is not rewritten into a known one.
        (None, None),
        ("fp8", "fp8"),
    ],
)
def test_the_sentence_transformer_normaliser_is_otherwise_unchanged(spelling, expected):
    from unsloth.models.sentence_transformer import _normalize_save_method
    assert _normalize_save_method(spelling) == expected


def test_the_docstrings_describe_none_as_the_stronger_safetensors_request():
    """`None` is not a synonym for the default `True`.

    On a host with at most two physical CPUs `unsloth_save_model` downgrades a default
    `safe_serialization = True` to `fast_save_pickle`, warning that safetensors is 10x
    slower there. `None` sets `_force_safe_serialization`, which is what makes the
    branch above that downgrade fire instead. So a default merged_16bit save on a small
    box can write a pickle, and a docstring saying only an explicit `False` does would
    send that user looking for a file that is not there.
    """
    from pathlib import Path

    save_py = Path(__file__).resolve().parents[2] / "unsloth" / "save.py"
    source = save_py.read_text(encoding = "utf-8")

    assert (
        "`None` is accepted and means the same thing" not in source
    ), "a docstring still equates None with the default True"
    assert (
        source.count("`None` is stronger than the default") == 4
    ), "all four save_method docstrings have to describe None the same way"
    # The behaviour the prose describes, read from the code rather than trusted.
    assert "elif safe_serialization and (n_cpus <= 2):" in source
    assert "if _force_safe_serialization:" in source


def _wrapped_model_save_pretrained(original):
    """The shipped `unsloth_model_save_pretrained`, bound to a stub whose
    `original_model_save_pretrained` is `original`. Executed, not read."""
    import ast
    import inspect
    import textwrap
    import types as _types

    from unsloth import save as save_module

    source = inspect.getsource(save_module.patch_saving_functions)
    tree = ast.parse(textwrap.dedent(source))
    node = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "unsloth_model_save_pretrained"
    )
    module = ast.Module(body = [node], type_ignores = [])
    ast.fix_missing_locations(module)
    namespace = dict(vars(save_module))
    exec(compile(module, "<unsloth_model_save_pretrained>", "exec"), namespace)

    stub = _types.SimpleNamespace(original_model_save_pretrained = original)
    return _types.MethodType(namespace["unsloth_model_save_pretrained"], stub)


def test_a_positional_none_is_normalised_on_a_peft_style_signature():
    """`PeftModel.save_pretrained` takes safe_serialization as its SECOND positional
    parameter, so `model.save_pretrained(directory, None)` is the same request as the
    keyword form and used to reach peft as a falsy value, writing adapter_model.bin."""
    seen = {}

    def peft_like(
        save_directory,
        safe_serialization = True,
        selected_adapters = None,
        **kwargs,
    ):
        seen["safe_serialization"] = safe_serialization
        seen["save_directory"] = save_directory

    _wrapped_model_save_pretrained(peft_like)("out_dir", None)

    assert seen["save_directory"] == "out_dir"
    assert seen["safe_serialization"] is True


def test_a_positional_second_argument_that_is_not_safe_serialization_is_untouched():
    """NEGATIVE CONTROL, and the reason the position cannot be assumed:
    `PreTrainedModel.save_pretrained`'s second parameter is `is_main_process`, so
    rewriting index 1 would corrupt an ordinary transformers call."""
    seen = {}

    def transformers_like(
        save_directory,
        is_main_process = True,
        state_dict = None,
        **kwargs,
    ):
        seen["is_main_process"] = is_main_process
        seen["kwargs"] = kwargs

    _wrapped_model_save_pretrained(transformers_like)("out_dir", None)

    assert seen["is_main_process"] is None, "an unrelated positional argument was rewritten"
    assert "safe_serialization" not in seen["kwargs"]


def test_an_explicit_positional_false_still_writes_a_pickle():
    """NEGATIVE CONTROL: only None is rewritten. False is a request, not the default."""
    seen = {}

    def peft_like(
        save_directory,
        safe_serialization = True,
        **kwargs,
    ):
        seen["safe_serialization"] = safe_serialization

    _wrapped_model_save_pretrained(peft_like)("out_dir", False)
    assert seen["safe_serialization"] is False


def test_an_unreadable_signature_forwards_the_call_unchanged():
    """A builtin or C callable has no readable signature; that must not break the save."""
    seen = {}

    class _NoSignature:
        def __call__(self, *args, **kwargs):
            seen["args"] = args
            seen["kwargs"] = kwargs

    _wrapped_model_save_pretrained(_NoSignature())("out_dir", None)
    assert seen["args"] == ("out_dir", None)
