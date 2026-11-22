# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""When CUDA-graph decode runs (decided per call): UNSLOTH_DECODE_CUDA_GRAPHS unset = on inside
Unsloth's generate() (generate_scope()) and an RL trainer's generation (rl_generation(), entered by
rl.py's unwrap_model_for_generation), off for a hand-written decode loop outside both; =1 on
everywhere, =0 off everywhere. Returning to training (or for_training) releases the graphs and every
decode buffer."""

import pytest

import unsloth  # noqa: F401  (must precede transformers)
from unsloth import FastLanguageModel
import torch

import unsloth.models._decode_graphs as G

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason = "CUDA only")
# The unset default turns graphs on for NVIDIA CUDA and gfx1151 ROCm; other ROCm archs need the flag at 1.
rl_default = pytest.mark.skipif(
    not G._DEFAULT_OK, reason = "RL default is NVIDIA CUDA / gfx1151 only"
)
MODEL = "trl-internal-testing/tiny-Qwen3ForCausalLM"
PROMPTS = ["Once upon a time there was", "def f(x):"]
ENV = "UNSLOTH_DECODE_CUDA_GRAPHS"


@pytest.fixture
def runtime(monkeypatch):
    # The runtime decision, not a programmatic override.
    monkeypatch.setattr(G, "ENABLED", None)
    monkeypatch.delenv(ENV, raising = False)
    return monkeypatch


def test_enabled_decision(runtime):
    runtime.setattr(G, "_DEFAULT_OK", True)
    assert G.enabled() is False
    with G.rl_generation():
        assert G.enabled() is True
        with G.rl_generation():
            assert G.enabled() is True
        assert G.enabled() is True
    assert G.enabled() is False
    runtime.setenv(ENV, "0")
    with G.rl_generation():
        assert G.enabled() is False
    runtime.setenv(ENV, "1")
    assert G.enabled() is True
    runtime.setattr(G, "ENABLED", False)
    assert G.enabled() is False


def test_rl_default_is_nvidia_cuda_only(runtime):
    runtime.setattr(G, "_DEFAULT_OK", False)  # ROCm / HIP, or no CUDA device
    with G.rl_generation():
        assert G.enabled() is False
        runtime.setenv(ENV, "1")
        assert G.enabled() is True
    runtime.setattr(torch.version, "hip", "6.4.0", raising = False)
    runtime.setattr(G, "_rocm_default_ok", lambda: False)
    assert G._default_ok() is False


def test_rl_default_rocm_only_gfx1151_with_skinny_linear(runtime):
    """ROCm turns the RL default on only for gfx1151 with the skinny decode linear enabled (measured exact
    and faster there); other HIP archs, or the kill switch, keep it off."""
    import types

    from unsloth.kernels import rocm_skinny_linear as S

    runtime.delenv("UNSLOTH_ROCM_SKINNY_LINEAR", raising = False)
    runtime.setattr(torch.version, "hip", "7.1.0", raising = False)
    runtime.setattr(torch.cuda, "is_available", lambda: True)
    runtime.setattr(torch.cuda, "current_device", lambda: 0)
    runtime.setattr(torch.cuda, "device_count", lambda: 1)
    runtime.setattr(S, "triton", types.SimpleNamespace())
    for arch, want in (
        ("gfx1151", True),
        ("gfx1100", False),
        ("gfx942", False),
        ("gfx1201", False),
    ):
        runtime.setattr(
            torch.cuda,
            "get_device_properties",
            lambda i, a = arch: types.SimpleNamespace(gcnArchName = a + ":xnack-"),
        )
        assert G._default_ok() is want, arch
        runtime.setenv("UNSLOTH_ROCM_SKINNY_LINEAR", "0")
        assert G._default_ok() is False
        runtime.delenv("UNSLOTH_ROCM_SKINNY_LINEAR")
    runtime.setattr(S, "triton", None)
    runtime.setattr(
        torch.cuda, "get_device_properties", lambda i: types.SimpleNamespace(gcnArchName = "gfx1151")
    )
    assert G._default_ok() is False
    runtime.setattr(torch.version, "hip", None, raising = False)
    assert G._default_ok() is True


def test_rl_default_rocm_needs_every_visible_gpu_measured(runtime):
    """Decode can run on any visible GPU, not only the one current at import: a gfx1151 beside another arch (either
    order) keeps the ROCm default off, two gfx1151 keep it on."""
    import types

    from unsloth.kernels import rocm_skinny_linear as S

    runtime.delenv("UNSLOTH_ROCM_SKINNY_LINEAR", raising = False)
    runtime.setattr(torch.version, "hip", "7.1.0", raising = False)
    runtime.setattr(torch.cuda, "is_available", lambda: True)
    runtime.setattr(torch.cuda, "current_device", lambda: 0)
    runtime.setattr(torch.cuda, "device_count", lambda: 2)
    runtime.setattr(S, "triton", types.SimpleNamespace())
    for archs, want in (
        (("gfx1151", "gfx1151"), True),
        (("gfx1151", "gfx1100"), False),
        (("gfx1100", "gfx1151"), False),
    ):
        runtime.setattr(
            torch.cuda,
            "get_device_properties",
            lambda i, a = archs: types.SimpleNamespace(gcnArchName = a[i] + ":xnack-"),
        )
        assert G._default_ok() is want, archs


def test_rl_scope_is_per_thread(runtime):
    import threading

    runtime.setattr(G, "_DEFAULT_OK", True)
    seen = []
    with G.rl_generation():
        t = threading.Thread(target = lambda: seen.append(G.enabled()))
        t.start()
        t.join()
        assert G.enabled() is True
    assert seen == [False]


def test_enabled_is_trace_safe(runtime):
    def f(x):
        return x + 1 if G.enabled() or G.static() is not None else x - 1

    compiled = torch.compile(f, fullgraph = True, backend = "eager")
    with G.rl_generation():
        assert compiled(torch.zeros(2)).tolist() == [-1.0, -1.0]


def _load():
    model, tok = FastLanguageModel.from_pretrained(MODEL, max_seq_length = 2048, load_in_4bit = False)
    model = FastLanguageModel.get_peft_model(
        model, r = 8, lora_alpha = 16, random_state = 0,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )  # fmt: skip
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return model, tok


class _Stats:
    def __init__(self, model):
        self.model = model

    def __call__(self):
        for m in self.model.modules():
            if "_unsloth_decode_graphs" in m.__dict__:
                return G.stats(m)
        return {}


def _causal(model):
    return model


def _graph_buffers(model):
    return sum("_unsloth_graph_buffers" in m.__dict__ for m in model.modules())


def _decode_buffers(model):
    return sum("paged_attention" in m.__dict__ for m in model.modules())


def _resident_graph_buffers(model):
    """Graph KV caches whose pages are mapped (a VMM cache asleep through training holds none)."""
    from unsloth.models import _decode_vmm as V
    return sum(
        V.buffer_of(bufs[0]) is None or V.buffer_of(bufs[0]).awake
        for m in model.modules()
        for bufs in (m.__dict__.get("_unsloth_graph_buffers") or {}).values()
    )


def _vmm(runtime, vmm):
    from unsloth.models import _decode_vmm as V
    runtime.setenv("UNSLOTH_DECODE_GRAPHS_VMM", vmm)
    if vmm == "1" and not V.available(torch.device("cuda", torch.cuda.current_device())):
        pytest.skip("CUDA virtual memory management")


def _assert_training_got_the_memory(model, vmm):
    # VMM: the caches sleep (pages freed, addresses and graphs kept); else everything is released.
    assert _resident_graph_buffers(model) == 0 and _decode_buffers(model) == 0
    if vmm == "1":
        assert _graph_buffers(model) > 0 and _Stats(model)()["graphs"] > 0
    else:
        assert _graph_buffers(model) == 0 and _Stats(model)()["graphs"] == 0


def _gen(
    model,
    tok,
    new = 24,
):
    enc = tok(PROMPTS, return_tensors = "pt", padding = True).to("cuda")
    out = model.generate(
        **enc, max_new_tokens = new, min_new_tokens = new, do_sample = False,
        pad_token_id = tok.pad_token_id, output_logits = True, return_dict_in_generate = True,
    )  # fmt: skip
    return out.sequences.cpu(), torch.stack(out.logits).cpu()


@rl_default
@pytest.mark.parametrize("vmm", ["0", "1"])
def test_plain_generate_on_by_default_bitwise_and_released(runtime, vmm):
    if not G._STUDIO_DEFAULT_OK:
        pytest.skip("the plain generate() default stays eager on native Windows")
    _vmm(runtime, vmm)
    model, tok = _load()
    FastLanguageModel.for_inference(model)
    runtime.setenv(ENV, "0")
    eager = _gen(model, tok)
    assert _Stats(model)().get("replays", 0) == 0
    assert _graph_buffers(model) == 0

    runtime.delenv(ENV)
    graph = _gen(model, tok)
    stats = _Stats(model)()
    assert stats["disabled"] is None and stats["replays"] == 24 - 2, stats
    assert torch.equal(eager[0], graph[0]) and torch.equal(eager[1], graph[1])
    with G.rl_generation():
        assert torch.equal(_gen(model, tok)[1], eager[1])

    runtime.setenv(ENV, "0")
    replays = _Stats(model)()["replays"]
    with G.rl_generation():
        assert torch.equal(_gen(model, tok)[1], eager[1])
    assert _Stats(model)()["replays"] == replays

    # for_training frees what an inference-mode generate() kept (a returned cache would release them).
    runtime.delenv(ENV)
    enc = tok(PROMPTS, return_tensors = "pt", padding = True).to("cuda")
    model.generate(**enc, max_new_tokens = 8, do_sample = False, pad_token_id = tok.pad_token_id)
    assert _graph_buffers(model) > 0
    FastLanguageModel.for_training(model)
    _assert_training_got_the_memory(model, vmm)

    # The returned cache views the decode buffers: everything is released, VMM or not.
    _gen(model, tok)
    assert model.training
    assert _graph_buffers(model) == 0 and _decode_buffers(model) == 0
    assert _Stats(model)()["graphs"] == 0


def test_release_rearms_a_disabled_model():
    """A capture failure (e.g. an OOM mid-GRPO) must not keep the model eager for life."""
    model = torch.nn.Linear(2, 2)
    state = G._state(model)
    for drop in (G.release, G.invalidate):
        state.disabled = "capture failed: OutOfMemoryError"
        drop(model)
        assert state.disabled is None


def test_generate_scope_turns_graphs_on(runtime):
    runtime.setattr(G, "_DEFAULT_OK", True)
    runtime.setattr(G, "_STUDIO_DEFAULT_OK", True)
    assert G.enabled() is False
    with G.generate_scope():
        assert G.enabled() is True
        runtime.setenv(ENV, "0")
        assert G.enabled() is False
    runtime.delenv(ENV)
    assert G.enabled() is False
    runtime.setattr(G, "_STUDIO_DEFAULT_OK", False)  # native Windows
    with G.generate_scope():
        assert G.enabled() is False


@rl_default
@pytest.mark.parametrize("keep", ["0", "1"])
def test_grpo_generation_uses_graphs_and_releases(runtime, tmp_path, keep):
    """Real GRPOTrainer step: generate() runs graphed with the variable unset and matches the eager
    rollout; keep=0 frees decode buffers before the loss, keep=1 keeps graphs and their store
    (only attention attributes go), so the second step replays without capture."""
    from datasets import Dataset
    from trl import GRPOConfig, GRPOTrainer

    runtime.setenv("UNSLOTH_DECODE_GRAPHS_KEEP_FOR_TRAINING", keep)
    model, tok = _load()
    rows = [{"prompt": p} for p in PROMPTS * 4]
    seen = {"replays": [], "buffers_at_loss": [], "captures": []}

    def reward(completions, **kw):
        return [float(len(c)) for c in completions]

    cfg = GRPOConfig(
        output_dir = str(tmp_path), per_device_train_batch_size = 4, num_generations = 4,
        max_completion_length = 16, max_steps = 2, logging_steps = 1, save_strategy = "no",
        report_to = "none", seed = 0, bf16 = torch.cuda.is_bf16_supported(), learning_rate = 1e-4,
    )  # fmt: skip
    trainer = GRPOTrainer(model = model, reward_funcs = [reward], args = cfg,
                          train_dataset = Dataset.from_list(rows), processing_class = tok)  # fmt: skip
    # TRL >= 0.24 generates in _generate_single_turn; older TRL only has _generate_and_score_completions.
    hook = (
        "_generate_single_turn"
        if hasattr(trainer, "_generate_single_turn")
        else "_generate_and_score_completions"
    )
    gen = getattr(trainer, hook)

    def recorded(*a, **k):
        before = _Stats(model)()
        out = gen(*a, **k)
        after = _Stats(model)()
        seen["replays"].append(after.get("replays", 0) - before.get("replays", 0))
        seen["captures"].append(after.get("captures", 0) - before.get("captures", 0))
        seen["buffers_at_loss"].append((_graph_buffers(model) > 0, _decode_buffers(model)))
        return out

    setattr(trainer, hook, recorded)
    trainer.train()
    assert len(seen["replays"]) == 2 and all(r > 0 for r in seen["replays"]), seen
    if keep == "0":
        assert seen["buffers_at_loss"] == [(False, 0), (False, 0)], seen
        assert seen["captures"] == [1, 1], seen
    else:
        assert seen["buffers_at_loss"] == [(True, 0), (True, 0)], seen
        assert seen["captures"] == [1, 0], seen
    assert _Stats(model)()["disabled"] is None


@rl_default
def test_out_of_memory_in_the_graph_path_redoes_the_call_eagerly(runtime):
    """A real CUDA OOM only with graphs on (first graph buffer asks for 1 PiB): generate() frees
    graphs, redoes it eagerly once, counts it, warns once, returns eager tokens and logits."""
    if not G._STUDIO_DEFAULT_OK:
        pytest.skip("the plain generate() default stays eager on native Windows")
    model, tok = _load()
    FastLanguageModel.for_inference(model)
    runtime.setenv(ENV, "0")
    eager = _gen(model, tok)
    runtime.delenv(ENV)
    original = G.alloc_decode_buffers

    def allocate(*args, **kwargs):
        if G.enabled():
            torch.empty(1 << 50, dtype = torch.uint8, device = "cuda")
        return original(*args, **kwargs)

    runtime.setattr(G, "alloc_decode_buffers", allocate)
    runtime.setattr(G, "_OOM_WARNED", False)
    redos = G.EAGER_REDOS
    with pytest.warns(UserWarning, match = "ran out of GPU memory"):
        redone = _gen(model, tok)
    assert G.EAGER_REDOS == redos + 1
    assert torch.equal(eager[0], redone[0]) and torch.equal(eager[1], redone[1])
    assert _graph_buffers(model) == 0
    # Without the forced OOM the next call is graphed again.
    runtime.setattr(G, "alloc_decode_buffers", original)
    replays = _Stats(model)().get("replays", 0)
    assert torch.equal(_gen(model, tok)[1], eager[1])
    assert _Stats(model)()["replays"] > replays and G.EAGER_REDOS == redos + 1


@rl_default
@pytest.mark.parametrize("vmm", ["0", "1"])
def test_training_forward_without_for_training_frees_generate_memory(runtime, vmm):
    """Sampling in a trainer callback, then model.train() alone: the first training forward frees
    decode caches (and graphs without VMM); with VMM graphs replay on the next generate()."""
    if not G._STUDIO_DEFAULT_OK:
        pytest.skip("the plain generate() default stays eager on native Windows")
    _vmm(runtime, vmm)
    model, tok = _load()
    FastLanguageModel.for_inference(model)
    enc = tok(PROMPTS, return_tensors = "pt", padding = True).to("cuda")
    model.generate(**enc, max_new_tokens = 8, do_sample = False, pad_token_id = tok.pad_token_id)
    assert _graph_buffers(model) > 0
    model.train()
    out = model(**enc, labels = enc.input_ids)
    out.loss.backward()
    _assert_training_got_the_memory(model, vmm)
    model.zero_grad(set_to_none = True)
    FastLanguageModel.for_inference(model)
    captures = _Stats(model)()["captures"]
    graphed = _gen_tokens(model, tok, PROMPTS, 8)
    if vmm == "1":
        assert _Stats(model)()["captures"] == captures  # woken caches, same addresses: replayed
    runtime.setenv(ENV, "0")
    assert torch.equal(graphed, _gen_tokens(model, tok, PROMPTS, 8))


def _gen_tokens(model, tok, prompts, new):
    enc = tok(prompts, return_tensors = "pt", padding = True).to("cuda")
    out = model.generate(
        **enc, max_new_tokens = new, min_new_tokens = new, do_sample = False,
        pad_token_id = tok.pad_token_id,
    )  # fmt: skip
    return out.cpu()


@rl_default
def test_short_calls_decode_eagerly_and_keep_the_kept_graph(runtime):
    """Short calls decode eagerly unless their batch is kept; tokens equal eager."""
    if not G._STUDIO_DEFAULT_OK:
        pytest.skip("the plain generate() default stays eager on native Windows")
    model, tok = _load()
    FastLanguageModel.for_inference(model)
    one, two = PROMPTS[:1], PROMPTS
    runtime.setenv(ENV, "0")
    eager = {
        (p, n): _gen_tokens(model, tok, ps, n)
        for p, ps in (("one", one), ("two", two))
        for n in (4, 24)
    }
    runtime.delenv(ENV)

    def check(p, ps, n):
        assert torch.equal(_gen_tokens(model, tok, ps, n), eager[(p, n)]), (p, n)

    def batches():
        return {k[0] for m in model.modules() for k in m.__dict__.get("_unsloth_graph_buffers", {})}

    check("two", two, 24)
    before = _Stats(model)()
    assert before["captures"] > 0 and batches() == {2}, (before, batches())
    check("one", one, 4)
    after = _Stats(model)()
    assert after["captures"] == before["captures"] and after["replays"] == before["replays"], (
        before,
        after,
    )
    assert batches() == {2}, batches()
    check("two", two, 4)
    short = _Stats(model)()
    assert short["captures"] == before["captures"] and short["replays"] > before["replays"], (
        before,
        short,
    )
    check("two", two, 24)
    assert _Stats(model)()["captures"] == before["captures"], (before, _Stats(model)())
