# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Load-time output a user cannot act on stays out of the default log.

* `import unsloth` on transformers 5 printed `rope_scaling is deprecated`: Unsloth's own
  rope probe writes the alias on purpose, and so do its own loader writes (a caller's
  `rope_scaling=` kwarg, automatic context extension). Those are quiet, the user's writes are not.
* A preinstalled xformers built for another torch (every Kaggle image) printed three
  lines; it is one line now, the fix and the error sit behind UNSLOTH_ENABLE_LOGGING=1.
* A multi-GPU load printed the planner's whole budget table; it is one line now.
"""

import ast
import inspect
import os
import re
import sys
import types
import warnings
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]


def _extract(
    relative_path,
    *names,
    namespace = None,
):
    """Exec just the named top-level functions / assignments of a module (no package import)."""
    tree = ast.parse((ROOT / relative_path).read_text(encoding = "utf-8"))
    nodes = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            nodes.append(node)
        elif isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id in names for t in node.targets
        ):
            nodes.append(node)
    found = {getattr(n, "name", None) or n.targets[0].id for n in nodes}
    missing = set(names) - found
    assert not missing, f"{relative_path} no longer defines {sorted(missing)}"
    namespace = {"os": os, "re": re, **(namespace or {})}
    exec(compile(ast.Module(body = nodes, type_ignores = []), str(relative_path), "exec"), namespace)
    return namespace


def _logging_helper():
    """import_fixes' own UNSLOTH_ENABLE_LOGGING parser, read at call time."""
    return _extract("unsloth/import_fixes.py", "unsloth_logging_enabled", "_UNSLOTH_LOGGING_ON")[
        "unsloth_logging_enabled"
    ]


# ---------------------------------------------------------------- rope_scaling


_ROPE_DEPRECATION = (
    "`rope_scaling` is deprecated and will be removed in a future version. Set `rope_parameters` "
    "instead, with `rope_type` and `rope_theta` included."
)


def _rope_warnings(record):
    return [
        w
        for w in record
        if issubclass(w.category, FutureWarning) and "rope_scaling" in str(w.message)
    ]


def _raw_setter_warns():
    """Does this transformers' own `rope_scaling` setter warn? (5.3 .. 5.20 have a silent one.)"""
    from transformers import LlamaConfig

    prop = inspect.getattr_static(LlamaConfig, "rope_scaling", None)
    if not isinstance(prop, property) or prop.fset is None:
        return False
    config = LlamaConfig(hidden_size = 64, num_attention_heads = 2, num_key_value_heads = 2)
    with warnings.catch_warnings(record = True) as record:
        warnings.simplefilter("always")
        prop.fset(config, {"rope_type": "linear", "factor": 2.0})
    return bool(_rope_warnings(record))


def test_rope_scaling_probe_wrapper_filters_only_its_own_warning(monkeypatch):
    from unsloth import import_fixes

    def probe():
        warnings.warn(_ROPE_DEPRECATION, FutureWarning)
        warnings.warn("an unrelated FutureWarning", FutureWarning)
        return True

    monkeypatch.setattr(import_fixes, "_probe_rope_scaling_assignment_drops_theta", probe)
    before = list(warnings.filters)
    with warnings.catch_warnings(record = True) as record:
        warnings.simplefilter("always")
        assert import_fixes._transformers_rope_scaling_assignment_drops_theta() is True
    assert _rope_warnings(record) == [], [str(w.message) for w in record]
    assert [str(w.message) for w in record] == ["an unrelated FutureWarning"]
    assert list(warnings.filters) == before  # nothing leaks past the probe


def test_rope_scaling_probe_gives_the_raw_answer():
    from unsloth import import_fixes

    with warnings.catch_warnings(record = True) as record:
        warnings.simplefilter("always")
        quiet = import_fixes._transformers_rope_scaling_assignment_drops_theta()
    assert _rope_warnings(record) == [], [str(w.message) for w in record]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert quiet == import_fixes._probe_rope_scaling_assignment_drops_theta()


def test_user_rope_scaling_write_still_warns_after_unsloth_writes():
    import transformers
    from transformers import LlamaConfig
    from unsloth import import_fixes
    from unsloth.models.vision import _apply_user_rope_scaling

    before = list(warnings.filters)
    import_fixes._transformers_rope_scaling_assignment_drops_theta()
    import_fixes.assign_rope_scaling(
        LlamaConfig(hidden_size = 64, num_attention_heads = 2, num_key_value_heads = 2),
        {"rope_type": "linear", "factor": 2.0},
    )
    _apply_user_rope_scaling(
        LlamaConfig(hidden_size = 64, num_attention_heads = 2, num_key_value_heads = 2),
        {"rope_type": "linear", "factor": 2.0},
    )
    assert list(warnings.filters) == before
    config = LlamaConfig(hidden_size = 64, num_attention_heads = 2, num_key_value_heads = 2)
    # No simplefilter("always"): a leaked "ignore" filter must be able to fail this.
    with warnings.catch_warnings(record = True) as record:
        config.rope_scaling = {"rope_type": "linear", "factor": 2.0}
    expected = _raw_setter_warns()
    assert bool(_rope_warnings(record)) is expected, (
        f"transformers {transformers.__version__}: user write warned {bool(record)}, "
        f"its own setter warns {expected}"
    )


def _rope_factor(config):
    scaling = getattr(config, "rope_parameters", None) or getattr(config, "rope_scaling", None)
    return dict(scaling or {}).get("factor")


def test_unsloth_rope_writes_are_quiet_and_applied():
    from transformers import LlamaConfig
    from unsloth import import_fixes
    from unsloth.models.vision import _apply_user_rope_scaling

    def config():
        return LlamaConfig(hidden_size = 64, num_attention_heads = 2, num_key_value_heads = 2)

    direct, vision = config(), config()
    with warnings.catch_warnings(record = True) as record:
        warnings.simplefilter("always")
        import_fixes.assign_rope_scaling(direct, {"rope_type": "linear", "factor": 2.0})
        _apply_user_rope_scaling(vision, {"rope_type": "linear", "factor": 4.0})
    assert _rope_warnings(record) == [], [str(w.message) for w in record]
    assert _rope_factor(direct) == 2.0 and _rope_factor(vision) == 4.0


def test_rope_inv_freq_config_copy_retry_is_quiet(monkeypatch):
    # Object-style rope_scaling: the first rope init raises, the retry writes the plain dict
    # on a config copy. That write is Unsloth's, so it must not print the deprecation.
    import transformers.modeling_rope_utils as rope_utils
    from transformers import LlamaConfig
    from unsloth.models import llama

    class _ObjectScaling:
        rope_type = "linear"
        factor = 2.0

        def to_dict(self):
            return {"rope_type": "linear", "factor": 2.0}

    seen = []

    def rope_init(config, device):
        scaling = config.rope_scaling  # transformers 5: an alias of rope_parameters
        if not isinstance(scaling, dict):
            raise TypeError("object-style rope_scaling")
        seen.append(dict(scaling))
        return torch.ones(4), 1.0

    monkeypatch.setitem(rope_utils.ROPE_INIT_FUNCTIONS, "linear", rope_init)
    config = LlamaConfig(hidden_size = 64, num_attention_heads = 2, num_key_value_heads = 2)
    alias = isinstance(inspect.getattr_static(LlamaConfig, "rope_scaling", None), property)
    object.__setattr__(config, "rope_parameters" if alias else "rope_scaling", _ObjectScaling())
    with warnings.catch_warnings(record = True) as record:
        warnings.simplefilter("always")
        inv_freq, scale = llama._compute_config_rope_inv_freq(config, _ObjectScaling())
    assert seen and seen[-1].get("factor") == 2.0, "the retry did not run"
    assert _rope_warnings(record) == [], [str(w.message) for w in record]
    assert scale == 1.0 and inv_freq.shape == (4,)


# ---------------------------------------------------------------- xformers


class _XformersLoadError(Exception):
    pass


_CU128_ERROR = (
    "xFormers can't load C++/CUDA extensions. xFormers was built for:\n"
    "    PyTorch 2.10.0+cu128 with CUDA 1208 (you have 2.11.0+cu130)\n"
    "    Python  3.12.12 (you have 3.12.12)"
)


def _notice():
    return _extract("unsloth/models/_utils.py", "_broken_xformers_notice")[
        "_broken_xformers_notice"
    ]


def _wrapped_register_failure():
    """The real shape: _register_extensions() failed, re-raised as ImportError ... from error."""
    try:
        try:
            raise RuntimeError(_CU128_ERROR)
        except RuntimeError as error:
            raise ImportError(
                "Unsloth: Xformers was not installed correctly.\n"
                "Please install xformers separately first.\n"
                "Longer error message:\n" + str(error)
            ) from error
    except ImportError as wrapped:
        return wrapped


def test_broken_xformers_is_one_line_by_default():
    out = _notice()(_XformersLoadError(_CU128_ERROR), "2.11.0+cu130", device_type = "cuda")
    assert "\n" not in out
    assert out.startswith("Unsloth: xformers does not load with torch 2.11.0+cu130")
    assert "PyTorch attention" in out and "UNSLOTH_ENABLE_LOGGING=1" in out
    assert "install-kernels" not in out and "xFormers can't load" not in out
    assert len(out) < 140


def test_broken_xformers_verbose_keeps_fix_and_error():
    out = _notice()(
        _XformersLoadError(_CU128_ERROR), "2.11.0+cu130", device_type = "cuda", verbose = True
    )
    assert "it was built for 2.10.0+cu128" in out
    assert "unsloth install-kernels xformers" in out
    assert "xFormers can't load C++/CUDA extensions" in out


def test_broken_xformers_still_noted_when_flash_attention_is_installed():
    # FlashAttention does not replace xformers everywhere (Mistral's uncached eval mask, packed
    # fp32), so a broken xformers keeps its one line.
    out = _notice()(
        _XformersLoadError(_CU128_ERROR),
        "2.11.0+cu130",
        device_type = "cuda",
        has_flash_attention = True,
    )
    assert "\n" not in out and out.startswith("Unsloth: xformers does not load with torch")
    assert "FlashAttention" in out


@pytest.mark.parametrize("device_type", ["hip", "xpu", "cpu"])
def test_broken_xformers_names_no_cuda_wheel_off_cuda(device_type):
    notice = _notice()
    error = _XformersLoadError(_CU128_ERROR)
    assert notice(error, "2.11.0", device_type = device_type) == ""
    verbose = notice(error, "2.11.0", device_type = device_type, verbose = True)
    assert "install-kernels" not in verbose
    assert "pip uninstall xformers" in verbose and "xFormers can't load" in verbose


def test_wrapped_xformers_failure_is_not_an_unsloth_guard():
    notice = _notice()
    wrapped = _wrapped_register_failure()
    assert wrapped.__cause__ is not None
    out = notice(wrapped, "2.11.0+cu130", device_type = "cuda")
    assert out.startswith("Unsloth: xformers does not load with torch") and "\n" not in out
    verbose = notice(wrapped, "2.11.0+cu130", device_type = "cuda", verbose = True)
    assert "unsloth install-kernels xformers" in verbose


def test_unsloth_version_guard_message_is_kept():
    guard = NotImplementedError(
        "Unsloth: Xformers 0.0.32.post2 has a broken FA3 dispatch on SM 12.0 GPUs."
    )
    notice = _notice()
    assert notice(guard, "2.11.0", device_type = "cuda") == str(guard)
    assert notice(guard, "2.11.0", device_type = "cuda", has_flash_attention = True) == str(guard)
    verbose = notice(guard, "2.11.0", device_type = "cuda", verbose = True)
    assert str(guard) in verbose and "install-kernels" not in verbose


# ---------------------------------------------------------------- placement report


_GiB = 1024**3
_LOGGING_VALUES = ["1", "True", "true"]


class _Plan:
    device_map = {"model.embed_tokens": 0, "lm_head": 1}

    def summary(self):
        return "Unsloth: placed the model over 2 GPUs: cuda:0 7.26 GiB, cuda:1 6.86 GiB (lm_head on cuda:1)."

    def describe(self):
        return "total weights      : 14.115 GiB\nhead headroom      : 1.000 GiB\nactivation reserve : 2.000 GiB"


class _OldZooPlan(_Plan):
    summary = None


class _FakeCuda:
    def is_available(self):
        return True

    def device_count(self):
        return 2

    def current_device(self):
        return 0

    def mem_get_info(self, index):
        return (14 * _GiB, 16 * _GiB)


def _loader_namespace(
    planner_module,
    monkeypatch,
    *names,
    extra = None,
):
    monkeypatch.setitem(sys.modules, "unsloth_zoo.device_map_planner", planner_module)
    return _extract(
        "unsloth/models/loader_utils.py",
        *names,
        "_placement_report",
        "_as_bytes",
        "_SIZE_UNITS",
        "_PLANNED_DEVICE_MAPS",
        "UNSLOTH_DEVICE_MAP",
        "UNSLOTH_BALANCED_DEVICE_MAP",
        namespace = {
            "inspect": inspect,
            "torch": types.SimpleNamespace(cuda = _FakeCuda(), device = torch.device),
            "DEVICE_TYPE_TORCH": "cuda",
            "is_distributed": lambda: False,
            "unsloth_logging_enabled": _logging_helper(),
            **(extra or {}),
        },
    )


def _resolver(plan, monkeypatch):
    module = types.ModuleType("unsloth_zoo.device_map_planner")
    module.plan_device_map_for_pretrained = lambda name, **kwargs: plan
    ns = _loader_namespace(module, monkeypatch, "resolve_unsloth_device_map")
    # The balanced map always splits; "unsloth" would keep a model that fits on one card.
    return lambda: ns["resolve_unsloth_device_map"](ns["UNSLOTH_BALANCED_DEVICE_MAP"], "some/model")


def test_placement_is_one_line_by_default(monkeypatch, capsys):
    monkeypatch.delenv("UNSLOTH_ENABLE_LOGGING", raising = False)
    assert _resolver(_Plan(), monkeypatch)() == _Plan.device_map
    out = capsys.readouterr().out
    assert out.strip() == _Plan().summary()
    for detail in ("head headroom", "activation reserve", "total weights"):
        assert detail not in out


@pytest.mark.parametrize("value", _LOGGING_VALUES)
def test_placement_full_table_with_logging(value, monkeypatch, capsys):
    monkeypatch.setenv("UNSLOTH_ENABLE_LOGGING", value)
    _resolver(_Plan(), monkeypatch)()
    assert capsys.readouterr().out.strip() == _Plan().describe()


def test_placement_older_zoo_keeps_its_table(monkeypatch, capsys):
    monkeypatch.delenv("UNSLOTH_ENABLE_LOGGING", raising = False)
    _resolver(_OldZooPlan(), monkeypatch)()
    assert capsys.readouterr().out.strip() == _Plan().describe()


class _BlockSwapPlan:
    layers = 4
    offload_embedding = False

    def __init__(self, device_plan):
        self.device_plan = device_plan

    def describe(self):
        return "block swap: 4 of 32 decoder layers (3.00 GiB) stay in host RAM\n" + (
            self.device_plan.describe()
        )


def _block_swap(plan, monkeypatch):
    module = types.ModuleType("unsloth_zoo.device_map_planner")
    module.plan_block_swap = lambda name, **kwargs: plan
    package = types.ModuleType("_quiet_block_swap_pkg")
    package.__path__ = []
    utils = types.ModuleType("_quiet_block_swap_pkg._utils")
    utils.usable_cuda_bytes = lambda device: 14 * _GiB
    utils.auto_plan_depth = lambda depth: depth
    monkeypatch.setitem(sys.modules, "_quiet_block_swap_pkg", package)
    monkeypatch.setitem(sys.modules, "_quiet_block_swap_pkg._utils", utils)
    ns = _loader_namespace(
        module,
        monkeypatch,
        "resolve_auto_block_swap",
        "_BLOCK_SWAP_PLANNER_KEYS",
        extra = {
            "__name__": "_quiet_block_swap_pkg.loader_utils",
            "__package__": "_quiet_block_swap_pkg",
        },
    )
    return ns["resolve_auto_block_swap"]("unsloth", "some/model", max_seq_length = 2048)


def test_block_swap_placement_is_one_line_by_default(monkeypatch, capsys):
    monkeypatch.delenv("UNSLOTH_ENABLE_LOGGING", raising = False)
    layers, device_map, _embedding = _block_swap(_BlockSwapPlan(_Plan()), monkeypatch)
    assert (layers, device_map) == (4, _Plan.device_map)
    lines = capsys.readouterr().out.strip().splitlines()
    assert lines == [
        "Unsloth: offload_layers = 'auto': 4 of 32 decoder layers (3.00 GiB) stay in host RAM",
        _Plan().summary(),
    ]


@pytest.mark.parametrize("value", _LOGGING_VALUES)
def test_block_swap_full_table_with_logging(value, monkeypatch, capsys):
    monkeypatch.setenv("UNSLOTH_ENABLE_LOGGING", value)
    _block_swap(_BlockSwapPlan(_Plan()), monkeypatch)
    out = capsys.readouterr().out
    assert _Plan().describe() in out and _Plan().summary() not in out


def test_block_swap_older_zoo_keeps_its_table(monkeypatch, capsys):
    monkeypatch.delenv("UNSLOTH_ENABLE_LOGGING", raising = False)
    _block_swap(_BlockSwapPlan(_OldZooPlan()), monkeypatch)
    assert _Plan().describe() in capsys.readouterr().out


@pytest.mark.parametrize(
    "value, enabled",
    [("1", True), ("True", True), ("true", True), ("0", False), ("yes", False), ("", False)],
)
def test_logging_helper_matches_unsloth_parser(value, enabled, monkeypatch):
    monkeypatch.setenv("UNSLOTH_ENABLE_LOGGING", value)
    assert _logging_helper()() is enabled


# ---------------------------------------------------------------- chat template without generation prompt

# Shape of Inkling 2's template: no add_generation_prompt branch, every turn closed with <|end_message|>.
_INKLING2_LIKE_TEMPLATE = (
    "{%- for m in messages -%}"
    "{%- if m.role == 'user' -%}<|message_user|><|content_text|>{{ m.content }}<|end_message|>"
    "{%- else -%}<|message_model|><|content_text|>{{ m.content }}<|end_message|>{%- endif -%}"
    "{%- endfor -%}"
)


class _Messages(__import__("logging").Handler):
    def __init__(self):
        super().__init__(0)
        self.messages = []

    def emit(self, record):
        self.messages.append(record.getMessage())


def _template_messages(
    template,
    name,
    monkeypatch,
    strict = False,
):
    import logging

    from tokenizers import Tokenizer, models
    from transformers import PreTrainedTokenizerFast

    import unsloth.tokenizer_utils as tu

    if strict:
        monkeypatch.setenv("UNSLOTH_STRICT_CHAT_TEMPLATE", "1")
    else:
        monkeypatch.delenv("UNSLOTH_STRICT_CHAT_TEMPLATE", raising = False)
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object = Tokenizer(models.WordLevel({"<unk>": 0}, unk_token = "<unk>")),
        unk_token = "<unk>",
    )
    tokenizer.chat_template = template
    tokenizer.name_or_path = name  # warning_once caches per message, so one name per case
    capture = _Messages()
    loggers = [logging.getLogger(tu.logger.name), logging.getLogger("transformers")]
    for one in loggers:
        one.addHandler(capture)
    try:
        result = tu.fix_chat_template(tokenizer)
    finally:
        for one in loggers:
            one.removeHandler(capture)
    return result, [
        m for m in capture.messages if "add_generation_prompt" in m and "by design" not in m
    ]


def test_inkling2_template_is_not_repaired_or_warned(monkeypatch):
    result, messages = _template_messages(
        _INKLING2_LIKE_TEMPLATE, "org/inkling2-case-a", monkeypatch
    )
    assert result == _INKLING2_LIKE_TEMPLATE  # no marker appended
    assert messages == []


def test_inkling2_template_passes_strict_mode(monkeypatch):
    result, messages = _template_messages(
        _INKLING2_LIKE_TEMPLATE, "org/inkling2-case-b", monkeypatch, strict = True
    )
    assert result == _INKLING2_LIKE_TEMPLATE and messages == []


def test_other_template_without_generation_prompt_still_warns(monkeypatch):
    # Same shape, but not Inkling 2's markers: a genuinely incomplete template keeps its warning.
    broken = _INKLING2_LIKE_TEMPLATE.replace("<|message_model|>", "<|assistant|>")
    _result, messages = _template_messages(broken, "org/broken-case-c", monkeypatch)
    assert messages, "the add_generation_prompt warning must still fire for unknown templates"


def test_inkling2_markers_with_a_dead_generation_block_still_warn(monkeypatch):
    # A template that has the block but ignores it is a real defect even with Inkling 2's markers.
    dead_block = _INKLING2_LIKE_TEMPLATE + "{%- if add_generation_prompt -%}{%- endif -%}"
    _result, messages = _template_messages(dead_block, "org/dead-block-case-d", monkeypatch)
    assert messages


# ---------------------------------------------------------------- is_gradient_checkpointing


def _checkpointing_helper():
    return _extract("unsloth/models/_gc_config_default.py", "is_activation_checkpointing")[
        "is_activation_checkpointing"
    ]


def test_checkpointing_flag_read_without_deprecated_alias():
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM

    model = LlamaForCausalLM(
        LlamaConfig(
            hidden_size = 16,
            intermediate_size = 32,
            num_hidden_layers = 1,
            num_attention_heads = 2,
            num_key_value_heads = 2,
            vocab_size = 32,
        )
    )
    model.gradient_checkpointing_enable()
    with warnings.catch_warnings(record = True) as record:
        warnings.simplefilter("always")
        on = _checkpointing_helper()(model)
    assert on is True
    assert not [w for w in record if "is_gradient_checkpointing" in str(w.message)]
    model.gradient_checkpointing_disable()
    assert _checkpointing_helper()(model) is False
    del torch


def test_checkpointing_flag_falls_back_to_old_name_and_default():
    import types

    helper = _checkpointing_helper()
    # TRL < 1.0 PolicyAndValueWrapper style: a plain attribute under the old name only.
    assert helper(types.SimpleNamespace(is_gradient_checkpointing = True)) is True
    assert helper(object()) is False
    assert helper(object(), default = None) is None


def test_no_unsloth_site_reads_the_deprecated_model_alias():
    offenders = []
    for path in (ROOT / "unsloth").rglob("*.py"):
        text = path.read_text(encoding = "utf-8", errors = "ignore")
        for pattern in ('"is_gradient_checkpointing", False', ".encoder.is_gradient_checkpointing"):
            if pattern in text:
                offenders.append(f"{path.relative_to(ROOT)}: {pattern}")
    assert offenders == []


# ---------------------------------------------------------------- TrainingArguments.gradient_checkpointing


def _route():
    return _extract("unsloth/models/rl.py", "_route_default_gradient_checkpointing")[
        "_route_default_gradient_checkpointing"
    ]


def _generated_config(route):
    """A config like rl.py generates from TRL <= 1.18.0 (gradient_checkpointing defaults True)."""
    import dataclasses

    from transformers import TrainingArguments

    @dataclasses.dataclass
    class TRLConfig(TrainingArguments):
        gradient_checkpointing: bool = True

    arguments = (
        "self,\n        output_dir = None,\n        activation_checkpointing = None,\n"
        "        gradient_checkpointing = True,\n        gradient_checkpointing_kwargs = None,\n"
    )
    call_args = "output_dir = output_dir, activation_checkpointing = activation_checkpointing, gradient_checkpointing = gradient_checkpointing"
    extra = ""
    if route:
        arguments, extra = _route()(arguments, call_args)
    extra = "\n".join("    " + line for line in extra.splitlines())
    source = (
        f"def __init__({arguments.rstrip().rstrip(',')}, **kwargs):\n"
        f"{extra}\n"
        "    TRLConfig.__init__(self, output_dir = output_dir, activation_checkpointing = activation_checkpointing, "
        "gradient_checkpointing = gradient_checkpointing, **kwargs)\n"
    )
    namespace = {"TRLConfig": TRLConfig}
    exec(source, namespace)
    return type("UnslothTRLConfig", (TRLConfig,), {"__init__": namespace["__init__"]})


def _resolved(cls, **kwargs):
    with warnings.catch_warnings(record = True) as record:
        warnings.simplefilter("always")
        args = cls(output_dir = "unused", use_cpu = True, report_to = "none", **kwargs)
    deprecated = [w for w in record if "`gradient_checkpointing` is deprecated" in str(w.message)]
    return (args.gradient_checkpointing, args.activation_checkpointing), bool(deprecated)


def test_gradient_checkpointing_default_does_not_warn_and_is_unchanged():
    from transformers import TrainingArguments

    if "activation_checkpointing" not in TrainingArguments.__dataclass_fields__:
        pytest.skip("transformers < 5.23 has no activation_checkpointing")
    before, after = _generated_config(route = False), _generated_config(route = True)
    # Checkpointing is on exactly when it was before, for every way of asking.
    for kwargs in (
        {},
        {"gradient_checkpointing": False},
        {"gradient_checkpointing": True},
        {"activation_checkpointing": False},
        {"activation_checkpointing": True},
    ):
        assert _resolved(after, **kwargs)[0] == _resolved(before, **kwargs)[0], kwargs
    assert _resolved(before)[1] is True  # main warned with nothing passed
    assert _resolved(after)[1] is False
    assert _resolved(after)[0] == (True, True)
    # A user passing the old name still hears about it.
    assert _resolved(after, gradient_checkpointing = False)[1] is True


def test_gradient_checkpointing_route_is_a_no_op_without_the_new_field():
    route = _route()
    arguments = "self,\n        gradient_checkpointing = True,\n"
    assert route(arguments, "gradient_checkpointing = gradient_checkpointing") == (arguments, "")
    # An already-None default (TRL >= 1.18.2) needs nothing either.
    arguments = (
        "self,\n        activation_checkpointing = None,\n        gradient_checkpointing = None,\n"
    )
    assert route(arguments, "activation_checkpointing, gradient_checkpointing") == (arguments, "")
