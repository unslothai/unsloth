# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""UNSLOTH_PARALLEL / unsloth.parallelize and the context parallel x tensor / pipeline parallel refusals, without a
process group (tests/test_parallel_cp_compose_gloo.py trains the composed layouts on gloo ranks)."""

import pytest
import torch

import unsloth  # noqa: F401
from unsloth import context_parallel as cp

_ENV = (
    "UNSLOTH_PARALLEL",
    "UNSLOTH_TP_SIZE",
    "UNSLOTH_PP_SIZE",
    "UNSLOTH_CP_SIZE",
    "UNSLOTH_CP_BACKEND",
    "UNSLOTH_FSDP2",
    "UNSLOTH_PP_SCHEDULE",
    "UNSLOTH_PP_MICROBATCHES",
    "WORLD_SIZE",
)


@pytest.fixture
def clean_env(monkeypatch):
    import os

    from unsloth import parallel_env
    from unsloth.models import tp_env

    for key in _ENV:
        monkeypatch.delenv(key, raising = False)
    yield monkeypatch
    for key in _ENV:  # what apply_parallel_env / parallelize wrote around monkeypatch
        os.environ.pop(key, None)
    parallel_env._SET_BY_PARALLELIZE[0] = None
    tp_env._reset_env_layout()


def _fake_sft(monkeypatch, inits):
    import trl

    class Trainer:
        def __init__(
            self,
            model = None,
            args = None,
            **kwargs,
        ):
            inits.append(model)
            self.model = model
            self.args = args

    monkeypatch.setattr(trl, "SFTTrainer", Trainer)
    cp.patch_sft_trainer()
    return Trainer


@pytest.mark.parametrize(
    "env, axis",
    [
        ({"UNSLOTH_TP_SIZE": "2"}, "tensor parallel (UNSLOTH_TP_SIZE=2)"),
        ({"UNSLOTH_PP_SIZE": "2"}, "pipeline parallel (UNSLOTH_PP_SIZE=2)"),
        (
            {"UNSLOTH_PP_SIZE": "2", "UNSLOTH_TP_SIZE": "2"},
            "pipeline parallel (UNSLOTH_PP_SIZE=2) and tensor",
        ),
    ],
)
@pytest.mark.parametrize("via", ["config", "kwarg"])
def test_cp_with_tp_or_pp_env_is_refused_before_the_trainer_init(clean_env, env, axis, via):
    import types

    for key, value in env.items():
        clean_env.setenv(key, value)
    inits = []
    Trainer = _fake_sft(clean_env, inits)
    model = torch.nn.Linear(2, 2)
    if via == "config":
        call = lambda: Trainer(model = model, args = types.SimpleNamespace(context_parallel_size = 2))
    else:
        call = lambda: Trainer(
            model, types.SimpleNamespace(context_parallel_size = 1), context_parallel_size = 2
        )
    with pytest.raises(NotImplementedError, match = "does not compose with") as info:
        call()
    assert axis in str(info.value), str(info.value)
    assert inits == [], "refused after the trainer __init__ ran (the model may already be split)"


def test_cp_alone_and_tp_without_cp_are_not_refused(clean_env):
    import types

    inits = []
    Trainer = _fake_sft(clean_env, inits)
    clean_env.setenv("UNSLOTH_TP_SIZE", "2")
    Trainer(model = torch.nn.Linear(2, 2), args = types.SimpleNamespace(context_parallel_size = 1))
    assert len(inits) == 1


# UNSLOTH_PARALLEL parsing and translation (parallel_env).


def test_flag_parses_every_key_and_fills_dp(clean_env):
    from unsloth import parallel_env as pe

    spec = pe.parse_parallel_spec(
        "dp=2, tp=2 ,pp=2,cp=2,cp_backend=Ulysses,fsdp=1,pp_schedule=1F1B,mb=4"
    )
    assert spec == {
        "dp": 2,
        "tp": 2,
        "pp": 2,
        "cp": 2,
        "cp_backend": "ulysses",
        "fsdp": True,
        "pp_schedule": "1f1b",
        "pp_microbatches": 4,
    }
    clean_env.setenv("WORLD_SIZE", "16")
    assert pe.translate(spec) == {
        "UNSLOTH_TP_SIZE": "2",
        "UNSLOTH_PP_SIZE": "2",
        "UNSLOTH_CP_SIZE": "2",
        "UNSLOTH_CP_BACKEND": "ulysses",
        "UNSLOTH_FSDP2": "1",
        "UNSLOTH_PP_SCHEDULE": "1f1b",
        "UNSLOTH_PP_MICROBATCHES": "4",
    }
    # dp omitted fills the world; size-1 axes write nothing.
    clean_env.setenv("WORLD_SIZE", "8")
    assert pe.translate(pe.parse_parallel_spec("tp=2")) == {"UNSLOTH_TP_SIZE": "2"}


@pytest.mark.parametrize(
    "raw, match",
    [
        ("tp=0", "must be >= 1"),
        ("tp=two", "positive integer"),
        ("tp", "not key=value"),
        ("xp=2", "unknown key"),
        ("tp=2,tp=2", "twice"),
        ("pp=2,pp_schedule=interleaved", "interleaved virtual stages are not supported yet"),
        ("cp=2,cp_backend=tree", "must be one of"),
        ("fsdp=maybe", "must be 0 or 1"),
    ],
)
def test_bad_flags_name_the_bad_part(raw, match):
    from unsloth import parallel_env as pe
    with pytest.raises(ValueError, match = match):
        pe.parse_parallel_spec(raw)


def test_world_size_and_legacy_conflicts_are_refused(clean_env):
    from unsloth import parallel_env as pe

    clean_env.setenv("WORLD_SIZE", "6")
    with pytest.raises(ValueError, match = "must divide the number of ranks"):
        pe.translate(pe.parse_parallel_spec("tp=4"))
    with pytest.raises(ValueError, match = "needs dp x pp x cp x tp = 8 ranks, but WORLD_SIZE=6"):
        pe.translate(pe.parse_parallel_spec("dp=2,tp=2,cp=2"))
    clean_env.setenv("WORLD_SIZE", "8")
    clean_env.setenv("UNSLOTH_TP_SIZE", "4")
    with pytest.raises(ValueError, match = "UNSLOTH_TP_SIZE=4 asks for something else"):
        pe.translate(pe.parse_parallel_spec("tp=2"))
    with pytest.raises(ValueError, match = "UNSLOTH_TP_SIZE=4"):
        pe.translate(pe.parse_parallel_spec("cp=2"))  # tp absent means 1
    clean_env.setenv("UNSLOTH_TP_SIZE", "2")
    assert pe.translate(pe.parse_parallel_spec("tp=2"))["UNSLOTH_TP_SIZE"] == "2"  # agreeing: fine
    clean_env.setenv("UNSLOTH_FSDP2", "1")
    with pytest.raises(ValueError, match = "UNSLOTH_FSDP2=1"):
        pe.translate(pe.parse_parallel_spec("tp=2,fsdp=0"))
    assert "UNSLOTH_FSDP2" not in pe.translate(
        pe.parse_parallel_spec("tp=2")
    )  # fsdp not given: left alone


def test_apply_is_idempotent(clean_env):
    import os

    from unsloth import parallel_env as pe

    clean_env.setenv("WORLD_SIZE", "4")
    clean_env.setenv("UNSLOTH_PARALLEL", "tp=2,cp=2,cp_backend=ulysses")
    assert pe.apply_parallel_env() == {"tp": 2, "cp": 2, "cp_backend": "ulysses"}
    assert os.environ["UNSLOTH_TP_SIZE"] == "2" and os.environ["UNSLOTH_CP_SIZE"] == "2"
    assert pe.apply_parallel_env() is not None  # sees its own values: no conflict
    assert pe.env_cp_size() == 2 and pe.env_cp_backend() == "ulysses"


def test_import_unsloth_translates_the_flag_and_refuses_a_conflict(tmp_path):
    import os
    import subprocess
    import sys

    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    code = (
        "import importlib.util, os, sys\n"
        f"sys.path.insert(0, {repo!r})\n"
        f"spec = importlib.util.spec_from_file_location('c', {os.path.join(repo, 'tests', 'conftest.py')!r})\n"
        "spec.loader.exec_module(importlib.util.module_from_spec(spec))\n"
        "import unsloth\n"
        "print(os.environ.get('UNSLOTH_TP_SIZE'), os.environ.get('UNSLOTH_CP_SIZE'), callable(unsloth.parallelize))\n"
    )
    env = {k: v for k, v in os.environ.items() if k not in _ENV}
    env.update(UNSLOTH_PARALLEL = "tp=2,cp=2", WORLD_SIZE = "4", CUDA_VISIBLE_DEVICES = "")
    out = subprocess.run(
        [sys.executable, "-c", code], env = env, capture_output = True, text = True, timeout = 600
    )
    assert out.stdout.strip().splitlines()[-1] == "2 2 True", out.stderr[-2000:]
    env["UNSLOTH_TP_SIZE"] = "4"
    out = subprocess.run(
        [sys.executable, "-c", code], env = env, capture_output = True, text = True, timeout = 600
    )
    assert (
        out.returncode != 0 and "UNSLOTH_TP_SIZE=4 asks for something else" in out.stderr
    ), out.stderr[-2000:]


def test_parallelize_sets_the_flag_and_refuses_after_a_load(clean_env):
    import os

    import unsloth as u
    from unsloth.models import tp_env

    clean_env.setenv("WORLD_SIZE", "4")
    tp_env._reset_env_layout()
    try:
        resolved = u.parallelize(tp = 2, cp = 2)
        assert resolved == {"UNSLOTH_TP_SIZE": "2", "UNSLOTH_CP_SIZE": "2"}
        assert os.environ["UNSLOTH_PARALLEL"] == "tp=2,cp=2"
        layout = tp_env.env_layout()
        assert (layout.dp, layout.cp, layout.tp, repr(layout)) == (1, 2, 2, "cp 2 x tp 2")
        with pytest.raises(RuntimeError, match = "must run before from_pretrained"):
            u.parallelize(tp = 4)
        with pytest.raises(ValueError, match = "positive integer"):
            u.parallelize(tp = 0)
    finally:
        for key in _ENV:
            os.environ.pop(key, None)
        tp_env._reset_env_layout()


def test_layout_rules_run_when_the_layout_is_read(clean_env):
    from unsloth.models import tp_env

    clean_env.setenv("WORLD_SIZE", "8")
    clean_env.setenv("UNSLOTH_PP_SIZE", "2")
    clean_env.setenv("UNSLOTH_CP_SIZE", "2")
    tp_env._reset_env_layout()
    try:
        with pytest.raises(NotImplementedError, match = r"cp x pp: .* Instead: drop cp, or drop pp"):
            tp_env.env_layout()
    finally:
        tp_env._reset_env_layout()


# UNSLOTH_CP_SIZE into the trainer arguments.


def test_env_cp_is_injected_into_the_trainer_args(clean_env):
    import types

    inits = []
    Trainer = _fake_sft(clean_env, inits)
    clean_env.setenv("UNSLOTH_CP_SIZE", "2")
    clean_env.setenv("UNSLOTH_CP_BACKEND", "ulysses")
    args = types.SimpleNamespace(context_parallel_size = 1)
    seen = {}

    def stop(self, *a, **k):
        raise RuntimeError("stop")  # past the injection; the CP runtime checks need a process group

    clean_env.setattr(cp, "_check_cp_runtime", stop)
    with pytest.raises(RuntimeError, match = "stop"):
        Trainer(model = torch.nn.Linear(2, 2), args = args)
    assert args.context_parallel_size == 2 and args.context_parallel_backend == "ulysses"
    with pytest.raises(ValueError, match = "UNSLOTH_CP_SIZE=2"):
        Trainer(model = torch.nn.Linear(2, 2), args = types.SimpleNamespace(context_parallel_size = 4))
    del seen


# context_parallel_backend = 'auto'.


@pytest.mark.parametrize(
    "heads, kv, size, tp, rl, want",
    [
        (4, 4, 2, 1, False, "ulysses"),
        (4, 2, 4, 1, False, "ulysses"),  # kv heads replicated
        (6, 6, 4, 1, False, "ring"),  # heads do not divide cp
        (4, 4, 4, 2, False, "ring"),  # 2 heads per tp rank, cp 4
        (4, 4, 2, 2, False, "ulysses"),
        (6, 6, 4, 1, True, "ulysses"),  # RL: Ulysses only (then refused on the heads)
        (None, None, 2, 1, False, "ulysses"),  # unknown model
    ],
)
def test_auto_backend(heads, kv, size, tp, rl, want):
    import types

    config = types.SimpleNamespace(num_attention_heads = heads, num_key_value_heads = kv)
    model = types.SimpleNamespace(config = config) if heads else None
    assert cp.resolve_cp_backend("auto", size, model, tp = tp, rl = rl) == want
    assert cp.resolve_cp_backend(None, size, model, tp = tp, rl = rl) == want
    assert cp.resolve_cp_backend("ring", size, model) == "ring"  # explicit: as given
    assert cp.resolve_cp_backend("ulysses", size, model) == "ulysses"


def _auto_packing(monkeypatch, config, model):
    from types import SimpleNamespace

    import unsloth.trainer as trainer_module

    class _Stub:
        def __init__(
            self,
            model = None,
            args = None,
            **kwargs,
        ):
            self.model, self.args = model, args

    for name in ("enable_padding_free_metadata", "enable_sample_packing"):
        monkeypatch.setattr(trainer_module, name, lambda model, trainer: None)
    module = SimpleNamespace(SFTTrainer = _Stub)
    trainer_module._patch_sft_trainer_auto_packing(module)
    return module.SFTTrainer(model = model, args = config)


class _Packable(torch.nn.Module):
    def __init__(self, heads):
        super().__init__()
        import types

        self.config = types.SimpleNamespace(
            num_attention_heads = heads, num_key_value_heads = heads, model_type = "llama"
        )
        self.lin = torch.nn.Linear(2, 2)

    def forward(
        self,
        input_ids = None,
        packed_seq_lengths = None,
        **kwargs,
    ):
        return input_ids


def test_auto_keeps_packing_and_explicit_ring_refuses_it(monkeypatch):
    from types import SimpleNamespace

    packed = lambda backend: SimpleNamespace(
        packing = True,
        padding_free = None,
        max_length = 64,
        context_parallel_size = 2,
        context_parallel_backend = backend,
    )
    config = packed("auto")
    _auto_packing(monkeypatch, config, _Packable(4))
    assert config.packing is True, "auto (Ulysses) must keep packing on"
    with pytest.raises(NotImplementedError, match = r"context_parallel_backend = 'ring'.*Instead"):
        _auto_packing(monkeypatch, packed("ring"), _Packable(4))
    config = packed("auto")
    config._unsloth_cp_backend_auto = True
    with pytest.raises(NotImplementedError, match = "'auto' chose ring because the attention heads"):
        _auto_packing(monkeypatch, config, _Packable(3))


def test_parallelize_refuses_a_legacy_variable_that_disagrees(clean_env):
    import unsloth as u
    clean_env.setenv("UNSLOTH_TP_SIZE", "4")
    with pytest.raises(ValueError, match = "UNSLOTH_TP_SIZE=4"):
        u.parallelize(tp = 2)


# The capability registry (parallel_capabilities).


def _q(**kw):
    from unsloth.parallel_capabilities import Query

    base = dict(trainer = "sft", family = "llama", torch = (2, 13), dp = 1)
    base.update(kw)
    return Query(**base)


def test_tensor_parallel_alone_keeps_what_tp_native_supports():
    """Tensor parallel trains full finetuning and MoE models by itself (models/tp_native.py and its gloo
    tests): the registry reports those layouts and runs them; only their compositions it has no row for refuse."""
    from unsloth import parallel_capabilities as caps

    for extra in (dict(mode = "full"), dict(moe = True, family = "qwen3_moe")):
        assert caps.cell_status(_q(tp = 2, **extra))[0] in ("single", "unverified"), extra
        caps.check(_q(tp = 2, **extra))
    with pytest.raises(NotImplementedError, match = r"^Unsloth: pp x mode"):
        caps.check(_q(pp = 2, mode = "full"))


@pytest.mark.parametrize(
    "query, axis, instead",
    [
        (dict(pp = 2, cp = 2), "cp x pp", "drop cp, or drop pp"),
        (dict(pp = 2, tp = 2, cp = 2), "cp x pp", "drop cp, or drop pp"),
        (dict(tp = 2, cp = 2, cp_backend = "ring"), "cp x tp", "cp_backend=ulysses"),
        (
            dict(tp = 2, cp = 2, cp_backend = "ulysses", trainer = "grpo"),
            "cp x tp",
            "run DPO / GRPO with cp alone",
        ),
        (dict(tp = 2, cp = 2, cp_backend = "ulysses", hybrid = True), "cp x tp", "drop tp"),
        (dict(tp = 2, cp = 2, cp_backend = "ulysses", packing = True), "cp x tp", "packing = False"),
        (dict(tp = 2, cp = 2, cp_backend = "ulysses", mode = "full"), "cp x tp", "use LoRA"),
        (dict(cp = 2, cp_backend = "ring", packing = True), "cp backend", "cp_backend=ulysses"),
        (dict(cp = 2, cp_backend = "ring", hybrid = True), "cp backend", "cp_backend=ulysses"),
        (dict(cp = 2, cp_backend = "ring", trainer = "dpo"), "cp backend", "cp_backend=ulysses"),
        (dict(cp = 2, cp_backend = "ring", sm = 7), "cp backend", "cp_backend=ulysses"),
        (dict(cp = 2, cp_backend = "ulysses", windowed = True), "cp x attention", "drop cp"),
        (dict(cp = 2, torch = (2, 6)), "cp x torch", "upgrade torch"),
        (dict(pp = 2, trainer = "grpo"), "pp x trainer", "drop pp"),
        (dict(pp = 2, packing = True), "pp x packing", "packing = False"),
        (dict(pp = 2, mode = "full"), "pp x mode", "use LoRA, or UNSLOTH_FSDP2=1"),
        (dict(tp = 2, vllm = True), "pp / tp x vLLM", "fast_inference = False"),
    ],
)
def test_every_refusal_names_its_axis_and_the_alternative(query, axis, instead):
    from unsloth import parallel_capabilities as caps

    with pytest.raises(NotImplementedError) as info:
        caps.check(_q(**query))
    message = str(info.value)
    assert message.startswith(f"Unsloth: {axis}: ") and f"Instead: {instead}" in message, message


@pytest.mark.parametrize(
    "query",
    [
        dict(tp = 2, cp = 2, cp_backend = "ulysses"),
        dict(dp = 2, tp = 2, cp = 2, cp_backend = "ulysses", fsdp2 = True),
        dict(cp = 2, cp_backend = "ulysses", packing = True, hybrid = True),
        dict(cp = 2, cp_backend = "ulysses", trainer = "grpo"),
        dict(cp = 2, cp_backend = "ring", sm = 9),
        dict(pp = 2, tp = 2, dp = 2, fsdp2 = True),
        dict(tp = 2, mode = "qlora"),
        dict(dp = 4, mode = "full", fsdp2 = True),
    ],
)
def test_supported_cells_pass(query):
    from unsloth import parallel_capabilities as caps
    caps.check(_q(**query))


def test_registry_table_and_query_from_a_model():
    import types

    from unsloth import parallel_capabilities as caps

    table = caps.markdown_table()
    assert table.count("\n") == len(caps.RULES) + 1 and "| cp x pp |" in table
    config = types.SimpleNamespace(
        model_type = "qwen3_next",
        layer_types = ["linear_attention", "full_attention"],
        sliding_window = None,
    )
    model = types.SimpleNamespace(config = config, peft_config = {"default": 1})
    layout = types.SimpleNamespace(dp = 1, tp = 2, pp = 1, cp = 2, fsdp2 = False, cp_backend = None)
    q = caps.query_from(
        layout, None, model, types.SimpleNamespace(packing = False), cp_backend = "ulysses"
    )
    assert (q.family, q.hybrid, q.windowed, q.mode, q.cp, q.tp) == (
        "qwen3_next",
        True,
        False,
        "lora",
        2,
        2,
    )
    with pytest.raises(NotImplementedError, match = "hybrid"):
        caps.check(q)


def test_cp_env_with_a_plain_trainer_is_refused(clean_env):
    import transformers

    from unsloth.models import _utils

    clean_env.setenv("UNSLOTH_CP_SIZE", "2")

    class Plain(transformers.Trainer):
        def __init__(self):
            pass

    with pytest.raises(NotImplementedError, match = r"cp x trainer: .*Plain would train every rank"):
        _utils._parallel_capabilities_pre(Plain(), Plain.__init__, (), {}, None)


def test_banner_names_the_mesh_backend_and_batch():
    import types

    from unsloth import parallel_env as pe

    trainer = types.SimpleNamespace(
        args = types.SimpleNamespace(
            per_device_train_batch_size = 2,
            gradient_accumulation_steps = 4,
            gradient_checkpointing = False,
        ),
        model = types.SimpleNamespace(config = types.SimpleNamespace(_attn_implementation = "sdpa")),
    )
    layout = pe.ParallelLayout(2, 2, 8, True, cp = 2)
    manager = types.SimpleNamespace(size = 2, backend = "ulysses")
    line = pe.announce(trainer, layout, manager)
    assert "mesh (pp 1, dp_replicate 1, dp_shard 2, cp 2, tp 2)" in line, line
    assert "effective batch 2 x 4 accumulation x 2 data parallel = 16 rows" in line
    assert (
        "cp backend ulysses" in line
        and "attention sdpa" in line
        and "gradient checkpointing off" in line
    )
    assert pe.announce(trainer, layout, manager) is None  # once per trainer


# The modularity contract: declarations, composition from them, and evidence per cell.


def test_every_contract_axis_has_declared_features():
    from unsloth import parallel_capabilities as caps

    kinds = {f.kind for f in caps.FEATURES}
    assert {
        "parallel",
        "precision",
        "compile",
        "attention",
        "packing",
        "memory",
        "kernels",
        "loss",
    } <= kinds
    names = {f.name for f in caps.FEATURES}
    assert {
        "ddp",
        "fsdp2",
        "tp",
        "pp",
        "cp.ulysses",
        "cp.ring",
        "block_compile",
        "flash_varlen",
        "flex",
    } <= names
    for f in caps.FEATURES:
        assert f.module and f.scope in ("instance", "class"), f
        assert set(f.hooks) | set(f.exclusive) <= set(caps.HOOKS), (f.name, f.hooks, f.exclusive)
        assert set(f.consumes) | set(f.uses) | set(f.produces) <= set(caps.METADATA), f.name
        assert set(f.mesh) <= {"dp", "tp", "pp", "cp"}, f.name
    table = caps.markdown_table("features")
    assert table.count("\n") == len(caps.FEATURES) + 1


@pytest.mark.parametrize(
    "query, match",
    [
        (
            dict(pp = 2, cp = 2, cp_backend = "ulysses"),
            r"pp x cp.ulysses: both run Trainer.training_step",
        ),
        (dict(attention = "flash_varlen"), r"flash_varlen consumes cu_seqlens"),
    ],
)
def test_composition_follows_from_the_declarations(query, match):
    from unsloth import parallel_capabilities as caps
    with pytest.raises(NotImplementedError, match = match):
        caps.check_features(_q(**query))


def test_declarations_accept_what_composes():
    from unsloth import parallel_capabilities as caps
    assert caps.check_features(_q(attention = "flash_varlen", packing = True)) == [
        "lora",
        "flash_varlen",
        "packing",
    ]
    assert "cp.ulysses" in caps.check_features(
        _q(tp = 2, cp = 2, cp_backend = "ulysses", compile = "block")
    )


def test_every_cell_carries_three_kinds_of_evidence_that_exist():
    import os
    import re

    from unsloth import parallel_capabilities as caps

    here = os.path.dirname(os.path.abspath(__file__))
    for cell in caps.CELLS:
        if cell.missing():
            # A cell waiting for evidence says why, and is refused (tests below).
            assert cell.note, cell.name
        for kind in ("parity", "engagement", "compile"):
            ref = getattr(cell, kind)
            if ref is None:
                continue
            if ref.startswith("eager:"):
                assert cell.match["compile"] == ("eager",), cell.name
                continue
            path, name = re.match(r"(tests/[\w.]+\.py)::(\w+)", ref).groups()
            with open(os.path.join(os.path.dirname(here), path)) as f:
                assert f"def {name}(" in f.read(), ref
    assert caps.markdown_table("cells").count("\n") == len(caps.CELLS) + 1


@pytest.mark.parametrize(
    "extra, missing",
    [
        (dict(memory = "unsloth_offload"), "memory='unsloth_offload'"),
        (dict(mode = "qlora"), "mode='qlora'"),
        (dict(kernels = True), "kernels=True"),
        (dict(family = "qwen3"), "family='qwen3'"),
        (dict(dp = 4), "dp=4"),
    ],
)
def test_a_composed_cell_without_evidence_is_refused(extra, missing):
    from unsloth import parallel_capabilities as caps

    query = _q(tp = 2, cp = 2, cp_backend = "ulysses", **extra)
    assert caps.cell_status(query)[0] == "unverified"
    with pytest.raises(
        NotImplementedError, match = "no numeric parity / engagement / compile evidence"
    ) as info:
        caps.check(query)
    assert missing in str(info.value)


def test_supported_cells_and_legacy_layouts(clean_env):
    from unsloth import parallel_capabilities as caps

    for query in (
        _q(tp = 2, cp = 2, cp_backend = "ulysses"),
        _q(tp = 2, cp = 2, dp = 2, cp_backend = "ulysses", memory = "gc"),
        _q(tp = 2, cp = 2, dp = 2, fsdp2 = True, cp_backend = "ulysses", compile = "block"),
    ):
        assert caps.cell_status(query)[0] == "supported", query
        caps.check(query)
    legacy = _q(tp = 2, dp = 2)
    assert caps.cell_status(legacy)[0] == "unverified"
    caps.check(legacy)  # shipped before the registry: runs unless strict
    assert caps.cell_status(_q(tp = 2))[0] == "single"
    clean_env.setenv("UNSLOTH_PARALLEL_STRICT", "1")
    with pytest.raises(NotImplementedError, match = "composition"):
        caps.check(legacy)
    caps.check(legacy, cells = False)  # the layout-only check at from_pretrained


@pytest.mark.parametrize(
    "query, axis",
    [
        (dict(cp = 2, cp_backend = "ulysses", attention = "flex_sparse"), "cp x attention"),
        (dict(cp = 2, cp_backend = "ulysses", mode = "fp8"), "cp x precision"),
        (dict(tp = 2, mode = "nvfp4"), "tp / pp x precision"),
        (dict(pp = 2, pp_schedule = "interleaved"), "pp x schedule"),
    ],
)
def test_new_axis_rules(query, axis):
    from unsloth import parallel_capabilities as caps
    with pytest.raises(NotImplementedError, match = f"^Unsloth: {axis}: .*Instead: "):
        caps.check(_q(**query))


def test_query_reads_compile_memory_and_attention_from_the_model():
    import types

    import torch.utils.checkpoint as checkpoint

    from unsloth import parallel_capabilities as caps

    class Layer(torch.nn.Module):
        def forward(self, x):
            return x

    model = torch.nn.Sequential(Layer())
    model.config = types.SimpleNamespace(model_type = "llama", _attn_implementation = "flex_attention")
    q = caps.query_from(None, None, model, types.SimpleNamespace(gradient_checkpointing = True))
    assert (q.compile, q.attention, q.memory) == ("eager", "flex", "gc")
    Layer.forward._unsloth_block_compiled = True
    original = checkpoint.checkpoint

    def unsloth_offloaded_gradient_checkpoint(*a, **k):
        return original(*a, **k)

    checkpoint.checkpoint = unsloth_offloaded_gradient_checkpoint
    try:
        q = caps.query_from(
            None, None, model, types.SimpleNamespace(gradient_checkpointing = True, packing = True)
        )
    finally:
        checkpoint.checkpoint = original
    assert (q.compile, q.memory, q.packing) == ("block", "unsloth_offload", True)


def test_compile_counters_count_breaks_and_recompiles():
    from unsloth.utils import compile_counters as cc

    torch._dynamo.reset()
    before = cc.snapshot()

    @torch.compile(backend = "aot_eager")
    def f(x):
        return x * 2 + 1

    f(torch.ones(3))
    f(torch.ones(3, 2))  # a rank change: one recompile
    used = cc.since(before)
    assert used["recompiles"] >= 1 and used["unique_graphs"] >= 2 and used["graph_breaks"] == 0
    assert cc.not_above(used, used) == []
    assert cc.not_above(
        {"graph_breaks": 1, "recompiles": 0}, {"graph_breaks": 0, "recompiles": 0}
    ) == ["graph_breaks 1 > 0"]


# transformers DistributedConfig names, reserved axes, HSDP.


@pytest.mark.parametrize(
    "raw, want",
    [
        ("fsdp=4,tp=2", {"tp": 2, "dp": 4, "fsdp": True}),
        (
            "fsdp_size=2,tp_size=2,cp_size=2,pp_size=2",
            {"tp": 2, "cp": 2, "pp": 2, "dp": 2, "fsdp": True},
        ),
        ("dp_shard=2,cp=2", {"cp": 2, "dp": 2, "fsdp": True}),
        ("dp_replicate=2,tp=2", {"tp": 2, "dp": 2, "fsdp": False}),
        ("fsdp=1,dp=2", {"fsdp": True, "dp": 2}),  # 0 / 1 stays the switch
        ("fsdp=0,tp=2", {"fsdp": False, "tp": 2}),
        # Failing before: the size-1 axis was compared with the other's dp and refused as a conflict.
        ("dp_shard=1,dp_replicate=2", {"dp": 2, "fsdp": False}),
        ("dp_shard=2,dp_replicate=1", {"dp": 2, "fsdp": True}),
        ("dp=2,dp_shard=1,dp_replicate=2", {"dp": 2, "fsdp": False}),
    ],
)
def test_distributed_config_names_are_accepted(raw, want):
    from unsloth import parallel_env as pe
    assert dict(pe.parse_parallel_spec(raw)) == want


@pytest.mark.parametrize(
    "raw, match",
    [
        ("ep=2", "expert parallel"),
        ("ep_size=2,tp=2", "expert parallel"),
        ("efsdp=2", "expert FSDP"),
        ("dp_shard=2,dp_replicate=2", "HSDP"),
        ("dp=4,dp_shard=2", "disagrees"),
        ("fsdp=0,dp_shard=2", "disagrees"),
    ],
)
def test_reserved_and_conflicting_dp_keys_are_refused(raw, match):
    from unsloth import parallel_env as pe
    with pytest.raises(ValueError, match = match):
        pe.parse_parallel_spec(raw)


# Muon on shards.


def test_muon_is_refused_on_fsdp2_and_tp_shards():
    from unsloth import parallel_capabilities as caps

    muon = torch.optim.Muon([torch.nn.Parameter(torch.randn(4, 4))])
    adamw = torch.optim.AdamW([torch.nn.Parameter(torch.randn(4, 4))])
    for what in ("fsdp2", "tp"):
        with pytest.raises(
            NotImplementedError, match = r"^Unsloth: fsdp2 / tp x optimizer: Muon .*Instead: adamw"
        ):
            caps.refuse_shard_unaware_optimizer(muon, what)
        caps.refuse_shard_unaware_optimizer(adamw, what)
    caps.check(_q(dp = 2, optim = "muon"))  # DDP holds whole matrices
    from unsloth.models import fsdp2_native

    with pytest.raises(NotImplementedError, match = "Muon"):
        fsdp2_native._refuse_bnb_optimizer_instance(
            muon
        )  # every native FSDP2 optimizer hook calls it


def test_muon_is_refused_before_native_fsdp2_shards(clean_env):
    import transformers

    from unsloth.models import _utils

    clean_env.setenv("UNSLOTH_FSDP2", "1")
    clean_env.setenv("WORLD_SIZE", "2")
    muon = torch.optim.Muon([torch.nn.Parameter(torch.randn(4, 4))])

    class Plain(transformers.Trainer):
        def __init__(
            self,
            model = None,
            args = None,
            optimizers = (None, None),
        ):
            pass

    with pytest.raises(NotImplementedError, match = "Muon"):
        _utils._parallel_capabilities_pre(
            Plain(), Plain.__init__, (), {"optimizers": (muon, None)}, None
        )


# MoE cells.


def test_moe_rows_cover_family_precision_aux_and_mode():
    import types

    from unsloth import parallel_capabilities as caps

    moe = [c for c in caps.CELLS if c.name.startswith("MoE ")]
    assert len(moe) == len(caps.MOE_FAMILIES) * 4 * 2 * 6
    proven = sorted(c.name for c in moe if not c.missing())
    assert "MoE qwen3_moe x lora+experts x aux off x cp" in proven
    assert "MoE qwen3_moe x full x aux off x fsdp2" in proven
    assert "MoE qwen3_moe x lora+experts x aux on x ddp" in proven
    moe_q = dict(family = "qwen3_moe", moe = True, expert_lora = True)
    assert caps.cell_status(_q(cp = 2, cp_backend = "ulysses", **moe_q))[0] == "supported"
    caps.check(_q(cp = 2, cp_backend = "ulysses", compile = "block", **moe_q))
    caps.check(_q(dp = 2, cp = 2, fsdp2 = True, cp_backend = "ulysses", **moe_q))
    caps.check(
        _q(
            cp = 2,
            cp_backend = "ulysses",
            compile = "block",
            **{**moe_q, "mode": "full", "expert_lora": False},
        )
    )
    # FSDP2 x cp block compile recompiles more than one process (measured): eager only.
    assert (
        caps.cell_status(
            _q(dp = 2, cp = 2, fsdp2 = True, cp_backend = "ulysses", compile = "block", **moe_q)
        )[0]
        != "supported"
    )
    for extra, why in (
        (dict(family = "qwen3_5_moe"), "no tiny-checkpoint gloo run"),
        (dict(mode = "qlora"), "4-bit experts need bitsandbytes on a GPU"),
        (dict(expert_lora = False), "not run yet"),
    ):
        with pytest.raises(NotImplementedError, match = why):
            caps.check(_q(cp = 2, cp_backend = "ulysses", **{**moe_q, **extra}))
    with pytest.raises(
        NotImplementedError, match = r"^Unsloth: MoE x pp / cp: .*router_aux_loss_coef = 0.0"
    ):
        caps.check(_q(cp = 2, cp_backend = "ulysses", router_aux_loss = True, **moe_q))
    with pytest.raises(
        NotImplementedError, match = r"x pp' lacks parity, engagement, compile evidence; pipeline"
    ):
        caps.check(_q(pp = 2, **moe_q))
    # Tensor parallel MoE runs (main's tp_native support): reported, not refused.
    assert caps.cell_status(_q(tp = 2, **moe_q))[0] == "unverified"
    caps.check(_q(tp = 2, **moe_q))
    assert caps.cell_status(_q(dp = 2, fsdp2 = True, router_aux_loss = True, **moe_q))[0] == "supported"
    assert (
        caps.cell_status(_q(dp = 2, router_aux_loss = True, **{**moe_q, "family": "gpt_oss"}))[0]
        == "unverified"
    )
    caps.check(
        _q(dp = 2, router_aux_loss = True, **{**moe_q, "family": "gpt_oss"})
    )  # dp alone: reported, runs


def test_aux_loss_is_read_the_way_trl_turns_it_on():
    import types

    from unsloth import parallel_capabilities as caps

    class SFTTrainer:
        pass

    config = types.SimpleNamespace(
        model_type = "qwen3_moe",
        num_experts = 4,
        output_router_logits = False,
        router_aux_loss_coef = 0.001,
    )
    model = types.SimpleNamespace(config = config)
    trl_default = types.SimpleNamespace(router_aux_loss_coef = None)
    assert caps.query_from(None, SFTTrainer, model, trl_default).router_aux_loss is True
    assert (
        caps.query_from(
            None, SFTTrainer, model, types.SimpleNamespace(router_aux_loss_coef = 0.0)
        ).router_aux_loss
        is False
    )
    assert (
        caps.query_from(None, None, model, trl_default).router_aux_loss is False
    )  # a plain Trainer
    config.output_router_logits = True
    assert caps.query_from(None, None, model, None).router_aux_loss is True
    dense = types.SimpleNamespace(
        model_type = "qwen3_next",
        num_experts = 2,
        num_hidden_layers = 2,
        mlp_only_layers = [0, 1],
        output_router_logits = True,
    )
    assert (
        caps.query_from(None, SFTTrainer, types.SimpleNamespace(config = dense), trl_default).moe
        is False
    )


def test_cp_with_native_fsdp2_is_a_supported_cell():
    from unsloth import parallel_capabilities as caps
    status, cell = caps.cell_status(_q(dp = 2, cp = 2, fsdp2 = True, cp_backend = "ulysses"))
    assert status == "supported" and "native FSDP2" in cell.name


# The notebooks launcher (fsdp2_scripts, M4) writes every axis and an agreeing UNSLOTH_FSDP2.

LAUNCHER_KEYS = ("dp", "tp", "pp", "cp", "fsdp", "cp_backend", "pp_schedule")


@pytest.mark.parametrize(
    "flag, world, legacy",
    [
        ("dp=2,tp=1,pp=1,cp=1,fsdp=1", "2", {"UNSLOTH_FSDP2": "1"}),
        ("dp=2,tp=1,pp=1,cp=1,fsdp=0", "2", {"UNSLOTH_FSDP2": "0"}),
        ("dp=1,tp=2,pp=1,cp=1,fsdp=0", "2", {"UNSLOTH_FSDP2": "0", "UNSLOTH_TP_SIZE": "2"}),
        (
            "dp=1,tp=1,pp=2,cp=1,fsdp=0,pp_schedule=1f1b",
            "2",
            {"UNSLOTH_FSDP2": "0", "UNSLOTH_PP_SIZE": "2", "UNSLOTH_PP_SCHEDULE": "1F1B"},
        ),
        (
            "dp=1,tp=1,pp=1,cp=2,fsdp=1,cp_backend=ulysses",
            "2",
            {"UNSLOTH_FSDP2": "true", "UNSLOTH_CP_SIZE": "2", "UNSLOTH_CP_BACKEND": "Ulysses"},
        ),
        (
            "dp=2,tp=2,pp=2,cp=2,fsdp=1",
            "16",
            {
                "UNSLOTH_FSDP2": "1",
                "UNSLOTH_TP_SIZE": "2",
                "UNSLOTH_PP_SIZE": "2",
                "UNSLOTH_CP_SIZE": "2",
            },
        ),
        (
            "dp=2,tp=1,pp=1,cp=1,fsdp=0,cp_backend=auto",
            "2",
            {"UNSLOTH_TP_SIZE": "1", "UNSLOTH_CP_SIZE": "1"},
        ),
    ],
)
def test_launcher_flag_with_agreeing_legacy_variables_is_accepted(clean_env, flag, world, legacy):
    import os

    from unsloth import parallel_env as pe

    clean_env.setenv("WORLD_SIZE", world)
    clean_env.setenv("UNSLOTH_PARALLEL", flag)
    for key, value in legacy.items():
        clean_env.setenv(key, value)
    spec = pe.apply_parallel_env()
    assert set(spec) <= set(LAUNCHER_KEYS)
    assert (
        pe.apply_parallel_env() == spec
    )  # idempotent: the second read agrees with what the first wrote
    fsdp = os.environ.get("UNSLOTH_FSDP2")
    assert fsdp in ("1", "0", "true") and (fsdp in ("1", "true")) == spec["fsdp"]


@pytest.mark.parametrize(
    "flag, legacy, name",
    [
        ("dp=2,tp=1,pp=1,cp=1,fsdp=1", {"UNSLOTH_FSDP2": "0"}, "UNSLOTH_FSDP2"),
        ("dp=1,tp=2,pp=1,cp=1,fsdp=0", {"UNSLOTH_TP_SIZE": "4"}, "UNSLOTH_TP_SIZE"),
        ("dp=1,tp=1,pp=1,cp=2,fsdp=0", {"UNSLOTH_CP_SIZE": "4"}, "UNSLOTH_CP_SIZE"),
        (
            "dp=1,tp=1,pp=1,cp=2,fsdp=0,cp_backend=ring",
            {"UNSLOTH_CP_BACKEND": "ulysses"},
            "UNSLOTH_CP_BACKEND",
        ),
        ("dp=2,tp=1,pp=1,cp=1,fsdp=0", {"UNSLOTH_PP_SIZE": "2"}, "UNSLOTH_PP_SIZE"),
    ],
)
def test_launcher_flag_with_a_disagreeing_legacy_variable_is_refused(clean_env, flag, legacy, name):
    from unsloth import parallel_env as pe

    clean_env.setenv("WORLD_SIZE", "2")
    clean_env.setenv("UNSLOTH_PARALLEL", flag)
    for key, value in legacy.items():
        clean_env.setenv(key, value)
    with pytest.raises(ValueError, match = f"{name}=.*asks for something else"):
        pe.apply_parallel_env()


def test_the_launcher_keys_are_accepted_and_ep_is_reserved():
    from unsloth import parallel_env as pe
    assert set(LAUNCHER_KEYS) <= set(pe._KEYS)
    assert {"ep", "efsdp"} <= set(pe._RESERVED)


@pytest.mark.parametrize("kind", ["sft", "dpo", "grpo"])
def test_router_aux_loss_after_init_is_refused_under_cp_for_each_trl_trainer(kind):
    """SFTTrainer writes output_router_logits into the config; DPOTrainer / GRPOTrainer only set aux_loss_enabled
    and pass the flag per forward. The post-init guard reads both; router_aux_loss_coef = 0.0 runs."""
    import types

    from unsloth import context_parallel as cpm
    from unsloth import parallel_capabilities as caps

    def trainer(aux, coef):
        config = types.SimpleNamespace(
            model_type = "qwen3_moe",
            num_experts = 4,
            router_aux_loss_coef = 0.001,
            output_router_logits = aux and kind == "sft",
        )
        model = types.SimpleNamespace(config = config)
        args = types.SimpleNamespace(router_aux_loss_coef = coef)
        return types.SimpleNamespace(model = model, args = args, aux_loss_enabled = aux), model, args

    on, model, args = trainer(True, None)
    with pytest.raises(
        NotImplementedError, match = r"^Unsloth: MoE x pp / cp: .*router_aux_loss_coef = 0.0"
    ):
        cpm._refuse_router_aux_loss_after_init(on, 2)
    assert (
        caps.query_from(None, type(f"{kind.upper()}Trainer", (), {}), model, args).router_aux_loss
        is True
    )
    off, model, args = trainer(False, 0.0)
    cpm._refuse_router_aux_loss_after_init(off, 2)
    assert (
        caps.query_from(None, type(f"{kind.upper()}Trainer", (), {}), model, args).router_aux_loss
        is False
    )
