# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""unsloth/utils/padding_choice.py: the padded vs padding-free rule, on CPU.

The measured table is real data (Colab A100 / L4, Kaggle T4, Qwen3 0.6B / 4B LoRA r16 16 bit, 3 reps,
torch 2.14.1; 4 bit agreed within the A/A band): padded + block compile throughput relative to padding-free. The rule must
never pick padded where padded lost.
"""

import importlib.util
import pathlib
import sys
from types import SimpleNamespace

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location(
        "_unsloth_padding_choice", ROOT / "unsloth" / "utils" / "padding_choice.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


pc = _load()

# Qwen3 LoRA linear FLOPs per token (gradient checkpointing) and attention FLOPs per pair, from mfu.py.
Q06 = (3.26e9, 9.18e5)
Q4 = (2.34e10, 2.36e6)
PEAK = {"A100": 312.0, "L4": 121.0, "T4": 65.0}
# (gpu, model, padded tokens, padded pairs per micro batch, measured padded/padding-free - 1 in %)
# Alpaca (short) at batch 2 / 8 / 16 and 1-4k token documents at batch 2 / 8.
ALP2, ALP8, ALP16 = (472, 6.4e4), (2883, 6.6e5), (6665, 1.6e6)
LONG2, LONG8 = (6176, 9.9e6), (30299, 5.8e7)
MEASURED = [
    ("A100", Q06, ALP2, +15.3), ("A100", Q06, ALP8, +11.0), ("A100", Q06, ALP16, +9.7),
    ("A100", Q06, LONG2, -1.3), ("A100", Q06, LONG8, -28.2),
    ("A100", Q4, ALP2, +16.2), ("A100", Q4, ALP8, +0.7), ("A100", Q4, ALP16, -42.5),
    ("A100", Q4, LONG2, -12.8), ("A100", Q4, LONG8, -33.0),
    ("L4", Q06, ALP2, +17.0), ("L4", Q06, ALP8, -1.3), ("L4", Q06, ALP16, -55.7),
    ("L4", Q06, LONG2, -15.8), ("L4", Q06, LONG8, -34.2),
    ("L4", Q4, ALP2, +3.0), ("L4", Q4, ALP8, -44.4), ("L4", Q4, ALP16, -57.3),
    ("L4", Q4, LONG2, -14.4), ("L4", Q4, LONG8, -32.5),
    ("T4", Q06, ALP2, +31.2), ("T4", Q06, ALP8, -15.8), ("T4", Q06, ALP16, -44.3),
    ("T4", Q06, LONG2, -12.6), ("T4", Q06, LONG8, -31.0),
    ("T4", Q4, ALP2, -20.8), ("T4", Q4, ALP8, -38.4), ("T4", Q4, ALP16, -53.8),
    ("T4", Q4, LONG2, -13.9), ("T4", Q4, LONG8, -32.1),
]  # fmt: skip


@pytest.mark.parametrize("gpu, flops, tokens, gain", MEASURED)
def test_rule_never_picks_padded_where_it_lost(gpu, flops, tokens, gain):
    use_pf, reason, gpu_s = pc.choose_padding(
        flops[0], flops[1], tokens[0], tokens[1], PEAK[gpu], total_microbatches = 10**9
    )
    if gain < 5.0:
        assert use_pf, (gpu, tokens, gain, reason, gpu_s)


def test_rule_picks_padded_in_clear_host_bound_cells():
    wins = [
        (gpu, f, t)
        for gpu, f, t, gain in MEASURED
        if gain >= 14.0 and not (gpu == "L4" and f is Q4)
    ]
    assert wins
    for gpu, f, t in wins:
        use_pf, reason, _ = pc.choose_padding(f[0], f[1], t[0], t[1], PEAK[gpu], 10**9)
        assert not use_pf, (gpu, t, reason)


# Colab G4 = RTX PRO 6000 Blackwell Server Edition (sm120, 500 dense TFLOPs), 16 bit, 3 reps; 4 bit
# agreed within the A/A band. (model, padded tokens / pairs, gain torch 2.14.1 %, gain torch 2.10.0 %)
SM120 = (12, 0)
MEASURED_SM120 = [
    (Q06, ALP2, +5.4, +23.6), (Q06, ALP8, +5.4, +64.5), (Q06, ALP16, -29.2, +59.6),
    (Q06, LONG2, -14.5, -4.9), (Q06, LONG8, -37.0, -16.5),
    (Q4, ALP2, +5.2, +24.1), (Q4, ALP8, -39.6, -19.3), (Q4, ALP16, -53.6, -40.3),
    (Q4, LONG2, -15.2, -12.5), (Q4, LONG8, -35.2, -25.4),
]  # fmt: skip


@pytest.mark.parametrize("flops, tokens, gain_214, gain_210", MEASURED_SM120)
@pytest.mark.parametrize("peak", [500.0, 503.8, 438.9])  # Server, Workstation, Max-Q
def test_sm120_never_picks_padded_where_it_lost(flops, tokens, gain_214, gain_210, peak):
    use_pf, reason, _ = pc.choose_padding(
        flops[0], flops[1], tokens[0], tokens[1], peak, 10**9, capability = SM120
    )
    if min(gain_214, gain_210) < 0:
        assert use_pf, (tokens, gain_214, gain_210, reason)
    if tokens is ALP2:  # the measured wins on both torch versions
        assert not use_pf, (tokens, reason)


def test_sm120_with_the_old_threshold_would_pick_a_losing_cell():
    # Why sm120 has its own threshold: at 0.10 s the 0.6B Alpaca batch 16 cell (-29% on torch
    # 2.14.1) is picked.
    assert pc.choose_padding(*Q06, *ALP16, 500.0, 10**9)[0] is False
    assert pc.choose_padding(*Q06, *ALP16, 500.0, 10**9, capability = SM120)[0] is True


@pytest.mark.parametrize("gpu, cap", [("A100", (8, 0)), ("L4", (8, 9)), ("T4", (7, 5))])
def test_measured_capabilities_keep_their_decisions(gpu, cap):
    for _, flops, tokens, _ in [m for m in MEASURED if m[0] == gpu]:
        args = (flops[0], flops[1], tokens[0], tokens[1], PEAK[gpu], 10**9)
        assert pc.choose_padding(*args)[0] == pc.choose_padding(*args, capability = cap)[0]


@pytest.mark.parametrize("cap", [(9, 0), (10, 0), (10, 3), (8, 6), (12, 1)])
def test_unmeasured_capability_keeps_padding_free(cap):
    # H100 / B200 / B300 / consumer Ampere: the datasheet peak alone would pick padded here.
    use_pf, reason, _ = pc.choose_padding(*Q06, *ALP2, 989.0, 10**9, capability = cap)
    assert use_pf and "not measured" in reason


def test_short_runs_keep_padding_free():
    # 60 steps x grad accum 4 cannot repay the extra block compile.
    use_pf, reason, _ = pc.choose_padding(*Q06, *ALP2, PEAK["A100"], total_microbatches = 240)
    assert use_pf and "too short" in reason
    use_pf, _, _ = pc.choose_padding(*Q06, *ALP2, PEAK["A100"], total_microbatches = 2000)
    assert not use_pf
    # 4 bit compiles far longer, so needs a longer run.
    use_pf, _, _ = pc.choose_padding(
        *Q06, *ALP2, PEAK["A100"], total_microbatches = 2000, load_in_4bit = True
    )
    assert use_pf
    use_pf, _, _ = pc.choose_padding(
        *Q06, *ALP2, PEAK["A100"], total_microbatches = 6000, load_in_4bit = True
    )
    assert not use_pf
    use_pf, _, _ = pc.choose_padding(*Q06, *ALP2, PEAK["A100"], total_microbatches = None)
    assert use_pf


@pytest.mark.parametrize("peak", [None, 0, -1.0])
def test_unknown_gpu_keeps_padding_free(peak):
    assert pc.choose_padding(*Q06, *ALP2, peak, 10**9)[0] is True


def test_effective_max_length_follows_trainer_precedence():
    import dataclasses

    @dataclasses.dataclass
    class Cfg:
        max_length: int = 1024

    model = SimpleNamespace(max_seq_length = 4096)
    # SFTConfig's default is replaced by the model length, as the generated SFTTrainer init does.
    assert pc._effective_max_length(model, Cfg()) == 4096
    # A caller-set cap below the model length wins; above it, the model length wins.
    assert pc._effective_max_length(model, Cfg(max_length = 512)) == 512
    assert pc._effective_max_length(model, Cfg(max_length = 8192)) == 4096
    # No model length: the config value is used as is.
    assert pc._effective_max_length(SimpleNamespace(), Cfg()) == 1024
    assert pc._effective_max_length(SimpleNamespace(), Cfg(max_length = None)) is None


def test_default_max_length_does_not_clip_long_rows():
    # Codex P2 case: default max_length 1024, model loaded at 4096, rows of 2000 / 4000 tokens on A100.
    import dataclasses

    @dataclasses.dataclass
    class Cfg:
        max_length: int = 1024

    cap = pc._effective_max_length(SimpleNamespace(max_seq_length = 4096), Cfg())
    _, padded, pairs = pc.estimate_microbatch_tokens([2000, 4000] * 50, 2, cap)
    assert pc.choose_padding(*Q06, padded, pairs, PEAK["A100"], 10**9)[0] is True
    _, padded, pairs = pc.estimate_microbatch_tokens([2000, 4000] * 50, 2, 1024)
    assert (
        pc.choose_padding(*Q06, padded, pairs, PEAK["A100"], 10**9)[0] is False
    )  # the old clipped estimate


def test_estimate_microbatch_tokens():
    real, padded, pairs = pc.estimate_microbatch_tokens([100] * 50, 4)
    assert real == padded == 400 and pairs == 4 * 100 * 101 / 2
    real, padded, _ = pc.estimate_microbatch_tokens([10, 1000], 8, max_length = 500)
    assert padded == pytest.approx(
        8 * 500, rel = 0.02
    )  # 8 draws of 2 lengths: max is almost always 500
    assert real < padded
    assert pc.estimate_microbatch_tokens([], 2) is None
    assert pc.estimate_microbatch_tokens([5, 6], 0) is None
    # deterministic
    assert pc.estimate_microbatch_tokens(list(range(1, 300)), 3) == pc.estimate_microbatch_tokens(
        list(range(1, 300)), 3
    )


@pytest.mark.parametrize(
    "value, mode",
    [
        (None, "auto"), ("auto", "auto"), ("", "auto"), ("padding_free", "padding_free"),
        ("0", "padding_free"), ("padded", "padded"), ("PADDED", "padded"), ("junk", "auto"),
    ],
)  # fmt: skip
def test_mode(monkeypatch, value, mode):
    if value is None:
        monkeypatch.delenv("UNSLOTH_PADDING_CHOICE", raising = False)
    else:
        monkeypatch.setenv("UNSLOTH_PADDING_CHOICE", value)
    assert pc.padding_choice_mode() == mode


def test_env_overrides_and_failures(monkeypatch):
    monkeypatch.setenv("UNSLOTH_PADDING_CHOICE", "padded")
    assert pc.decide_auto_padding_free(None, None, None, None)[0] is False
    monkeypatch.setenv("UNSLOTH_PADDING_CHOICE", "padding_free")
    assert pc.decide_auto_padding_free(None, None, None, None)[0] is True
    monkeypatch.setenv("UNSLOTH_PADDING_CHOICE", "auto")
    assert pc.decide_auto_padding_free(None, None, None, None)[0] is True
    assert pc.decide_auto_padding_free("org/model", None, None, None)[0] is True

    def boom(*a, **k):
        raise RuntimeError("x")

    monkeypatch.setattr(pc, "_decide", boom)
    use_pf, reason = pc.decide_auto_padding_free(object(), None, None, None)
    assert use_pf is True and "RuntimeError" in reason


class _Tok:
    chat_template = "x"

    def __call__(
        self,
        texts,
        add_special_tokens = False,
    ):
        return {"input_ids": [t.split() for t in texts]}

    def apply_chat_template(
        self,
        messages,
        tokenize = False,
    ):
        return " ".join(m["content"] for m in messages)


def test_row_texts_formats():
    tok = _Tok()
    cfg = SimpleNamespace(dataset_text_field = "text")
    rows = [{"text": "a b c"}, {"text": "d"}]
    assert pc._row_texts(rows, ["text"], cfg, None, tok) == ["a b c", "d"]
    rows = [
        {
            "messages": [
                {"role": "user", "content": "hi there"},
                {"role": "assistant", "content": "yo"},
            ]
        }
    ]
    assert pc._row_texts(rows, ["messages"], cfg, None, tok) == ["hi there yo"]
    rows = [{"prompt": "p q", "completion": " r"}]
    assert pc._row_texts(rows, ["prompt", "completion"], cfg, None, tok) == ["p q r"]
    rows = [{"instruction": "x", "output": "y z"}]
    assert pc._row_texts(rows, ["instruction", "output"], cfg, None, tok) is None
    fmt = lambda row: row["instruction"] + " " + row["output"]
    assert pc._row_texts(rows, ["instruction", "output"], cfg, fmt, tok) == ["x y z"]
    assert pc._row_texts(rows, ["instruction", "output"], cfg, lambda r: [fmt(r)], tok) == ["x y z"]
    assert pc._row_texts(rows, ["instruction", "output"], cfg, lambda r: 3, tok) is None


def test_sample_lengths():
    datasets = pytest.importorskip("datasets")
    tok = _Tok()
    cfg = SimpleNamespace(dataset_text_field = "text")
    ds = datasets.Dataset.from_dict({"text": ["a b c", "d", "e f"]})
    assert sorted(pc._sample_lengths(ds, cfg, tok, None)) == [1, 2, 3]
    ds = datasets.Dataset.from_dict({"input_ids": [[1, 2], [3], [4, 5, 6, 7]]})
    assert sorted(pc._sample_lengths(ds, cfg, None, None)) == [1, 2, 4]
    big = datasets.Dataset.from_dict({"text": ["w " * (i % 7 + 1) for i in range(5000)]})
    lengths = pc._sample_lengths(big, cfg, tok, None)
    assert len(lengths) == pc._SAMPLE_ROWS
    assert pc._sample_lengths(None, cfg, tok, None) is None
    assert pc._sample_lengths(datasets.Dataset.from_dict({"text": []}), cfg, tok, None) is None
    iterable = datasets.Dataset.from_dict({"text": ["a"]}).to_iterable_dataset()
    assert pc._sample_lengths(iterable, cfg, tok, None) is None


def test_total_microbatches():
    cfg = SimpleNamespace(
        per_device_train_batch_size = 2, gradient_accumulation_steps = 4, max_steps = 60
    )
    assert pc._total_microbatches(cfg, 52000) == 240
    cfg = SimpleNamespace(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        max_steps = -1,
        num_train_epochs = 1,
    )
    assert pc._total_microbatches(cfg, 52001) == 26001
    assert pc._total_microbatches(cfg, None) is None


def test_moe_detection():
    assert pc._is_moe(SimpleNamespace(num_experts = 8))
    assert pc._is_moe(SimpleNamespace(n_routed_experts = 64))
    assert not pc._is_moe(SimpleNamespace(num_experts = 0, hidden_size = 8))
    assert not pc._is_moe(SimpleNamespace(hidden_size = 8))


def test_trainer_wires_the_choice():
    src = (ROOT / "unsloth" / "trainer.py").read_text(encoding = "utf-8")
    assert "_auto_padding_choice(" in src
    assert "from unsloth.utils.padding_choice import decide_auto_padding_free" in src


@pytest.mark.parametrize("platform, hip, cuda, available, reason", [
    ("linux", "6.2.41133", None, True, "ROCm"),
    ("win32", None, "13.0", True, "Windows"),
    ("win32", "6.2", None, True, "ROCm"),
    ("linux", None, None, False, "not an NVIDIA CUDA device"),
    ("darwin", None, None, False, "not an NVIDIA CUDA device"),
])  # fmt: skip
def test_non_cuda_linux_keeps_padding_free(monkeypatch, platform, hip, cuda, available, reason):
    torch = pytest.importorskip("torch")
    monkeypatch.setenv("UNSLOTH_PADDING_CHOICE", "auto")
    monkeypatch.setattr(pc.sys, "platform", platform)
    monkeypatch.setattr(torch.version, "hip", hip, raising = False)
    monkeypatch.setattr(torch.version, "cuda", cuda, raising = False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: available)
    use_pf, why = pc.decide_auto_padding_free(torch.nn.Linear(2, 2), SimpleNamespace(), None, None)
    assert use_pf is True and reason in why, why
