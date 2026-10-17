# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team.
"""ORPO / CPO concatenated_forward scores from hidden states through the chunked log-prob head and matches TRL."""

from __future__ import annotations

import ast
import functools
import importlib.util
import os
import textwrap
import types

import pytest
import torch

os.environ.setdefault("UNSLOTH_COMPILE_DISABLE", "1")
import unsloth.models.rl_replacements as rl

trl = pytest.importorskip("trl")

if rl.unsloth_chunked_logprobs is None:
    # Skip because the kernel ships in unsloth_zoo (unslothai/unsloth-zoo#1803).
    pytest.skip(
        reason = "installed unsloth_zoo predates unsloth_chunked_logprobs", allow_module_level = True
    )


def _trl_class(kind):
    """TRL's own class and the source of its concatenated_forward, read from the file (importing unsloth may
    already have rebound the class to the patched one)."""
    for name in (f"trl.experimental.{kind}.{kind}_trainer", f"trl.trainer.{kind}_trainer"):
        try:
            spec = importlib.util.find_spec(name)
        except Exception:
            spec = None
        if spec is None or spec.origin is None:
            continue
        text = open(spec.origin, encoding = "utf-8").read()
        node = next(
            (
                n
                for n in ast.walk(ast.parse(text))
                if isinstance(n, ast.FunctionDef) and n.name == "concatenated_forward"
            ),
            None,
        )
        if node is None:
            continue
        module = importlib.import_module(name)
        cls = getattr(module, "ORPOTrainer" if kind == "orpo" else "CPOTrainer")
        return cls, ast.get_source_segment(text, node, padded = True)
    # Skip because this TRL ships no such trainer (CPO / ORPO left trl.trainer in 0.29).
    pytest.skip(reason = f"no {kind} trainer in the installed TRL")


def _compile(source, rows_per_chunk = None):
    ns = dict(vars(rl))
    ns["nn"] = torch.nn
    if rows_per_chunk is not None:
        ns["unsloth_chunked_logprobs"] = functools.partial(
            rl.unsloth_chunked_logprobs, rows_per_chunk = rows_per_chunk
        )
    exec(textwrap.dedent(source), ns)
    return ns["concatenated_forward"]


class _HiddenStatesLM(torch.nn.Module):
    """Wraps a causal LM the way Unsloth's forwards behave: with UNSLOTH_RETURN_HIDDEN_STATES=1 the final-norm hidden
    states come back as ``.logits``."""

    __UNSLOTH_SUPPORTS_RETURN_HIDDEN_STATES__ = True

    def __init__(
        self,
        lm,
        honour = True,
    ):
        super().__init__()
        self.lm = lm
        self.honour = honour
        self.config = lm.config

    def get_output_embeddings(self):
        return self.lm.get_output_embeddings()

    def forward(
        self,
        input_ids,
        attention_mask = None,
        use_cache = False,
    ):
        if self.honour and os.environ.get("UNSLOTH_RETURN_HIDDEN_STATES", "0") == "1":
            return types.SimpleNamespace(logits = self.lm.hidden(input_ids, attention_mask))
        return types.SimpleNamespace(logits = self.lm.logits(input_ids, attention_mask))


class _QwenLM(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.config = model.config

    def get_output_embeddings(self):
        return self.model.get_output_embeddings()

    def hidden(self, input_ids, attention_mask):
        body = self.model.get_base_model().model
        return body(input_ids = input_ids, attention_mask = attention_mask).last_hidden_state

    def logits(self, input_ids, attention_mask):
        return self.model(
            input_ids = input_ids, attention_mask = attention_mask, use_cache = False
        ).logits


class _ToyLM(torch.nn.Module):
    """Head with a bias, a logit scale and a soft cap, applied in the model's own order (scale, then cap)."""

    def __init__(
        self,
        vocab = 61,
        hidden = 12,
    ):
        super().__init__()
        g = torch.Generator().manual_seed(1)
        self.embed = torch.nn.Parameter(torch.randn(vocab, hidden, generator = g))
        self.mix = torch.nn.Parameter(torch.randn(hidden, hidden, generator = g) / hidden**0.5)
        self.lm_head = torch.nn.Linear(hidden, vocab, bias = True)
        with torch.no_grad():
            self.lm_head.weight.copy_(torch.randn(vocab, hidden, generator = g))
            self.lm_head.bias.copy_(torch.randn(vocab, generator = g))
        self.config = types.SimpleNamespace(
            model_type = "cohere", logit_scale = 0.7, final_logit_softcapping = 3.0
        )

    def get_output_embeddings(self):
        return self.lm_head

    def hidden(self, input_ids, attention_mask):
        return torch.tanh(self.embed[input_ids] @ self.mix)

    def logits(self, input_ids, attention_mask):
        z = self.lm_head(self.hidden(input_ids, attention_mask)) * self.config.logit_scale
        cap = self.config.final_logit_softcapping
        return cap * torch.tanh(z / cap)


def _qwen_lora():
    transformers = pytest.importorskip("transformers")
    peft = pytest.importorskip("peft")
    try:
        model = transformers.AutoModelForCausalLM.from_pretrained(
            "trl-internal-testing/tiny-Qwen3ForCausalLM", torch_dtype = torch.float32
        )
    except Exception as error:
        # Skip because the offline checkpoint is missing, not a code failure.
        pytest.skip(reason = f"tiny Qwen3 checkpoint unavailable: {error}")
    torch.manual_seed(0)
    model = peft.get_peft_model(
        model, peft.LoraConfig(r = 4, lora_alpha = 8, target_modules = ["q_proj", "v_proj", "down_proj"])
    )
    # Non-zero B so LoRA A also gets a gradient.
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "lora_B" in name:
                p.normal_(0, 0.05)
    return _QwenLM(model)


def _batch(vocab, seed = 0):
    """ORPO/CPO collator layout: right-padded rows, prompt labels -100, chosen and rejected of different widths."""
    g = torch.Generator().manual_seed(seed)
    B, prompt = 3, 4
    out = {}
    for side, lengths, width in (("chosen", (9, 7, 11), 11), ("rejected", (12, 6, 10), 12)):
        ids = torch.zeros(B, width, dtype = torch.long)
        mask = torch.zeros(B, width, dtype = torch.long)
        labels = torch.full((B, width), -100, dtype = torch.long)
        for i, n in enumerate(lengths):
            ids[i, :n] = torch.randint(1, vocab, (n,), generator = g)
            mask[i, :n] = 1
            labels[i, prompt:n] = ids[i, prompt:n]
        out[f"{side}_input_ids"], out[f"{side}_attention_mask"], out[f"{side}_labels"] = (
            ids,
            mask,
            labels,
        )
    return out


def _trainer(
    cls,
    kind,
    loss_type = "sigmoid",
    cpo_alpha = 1.0,
    alpha = 0.0,
):
    trainer = cls.__new__(cls)
    trainer.__dict__.update(
        is_encoder_decoder = False,
        label_pad_token_id = -100,
        padding_value = 0,
        pad_token_id = 0,
        aux_loss_enabled = False,
        aux_loss_coef = 0.0,
        beta = 0.2,
        loss_type = loss_type,
        cpo_alpha = cpo_alpha,
        alpha = alpha,
        simpo_gamma = 0.5,
        label_smoothing = 0.1 if loss_type == "sigmoid" else 0.0,
        accelerator = types.SimpleNamespace(
            device = torch.device("cpu"),
            unwrap_model = lambda m: m,
            gather_for_metrics = lambda x: x,
        ),
    )
    return trainer


def _run(trainer, forward, model, batch, params):
    trainer.concatenated_forward = types.MethodType(forward, trainer)
    loss, metrics = trainer.get_batch_loss_metrics(model, batch)
    grads = torch.autograd.grad(loss, params)
    return loss.detach(), metrics, grads


CASES = [
    ("orpo", {}),
    ("cpo", {"loss_type": "sigmoid"}),
    ("cpo", {"loss_type": "hinge"}),
    ("cpo", {"loss_type": "ipo"}),
    ("cpo", {"loss_type": "simpo"}),
    ("cpo", {"loss_type": "simpo", "cpo_alpha": 0.0, "alpha": 0.25}),  # AlphaPO
    ("cpo", {"loss_type": "sigmoid", "cpo_alpha": 0.0}),
]


@pytest.mark.parametrize("model_name", ["qwen3_lora", "toy_bias_scale_softcap"])
@pytest.mark.parametrize("kind,kwargs", CASES)
def test_matches_trl(kind, kwargs, model_name, monkeypatch):
    monkeypatch.delenv("UNSLOTH_RETURN_HIDDEN_STATES", raising = False)
    cls, source = _trl_class(kind)
    patched_source = getattr(rl, f"{kind}_trainer_concatenated_forward")(
        "concatenated_forward", source
    )
    assert (
        patched_source != source
    ), "rewrite did not recognise the installed TRL concatenated_forward"
    stock, patched = _compile(source), _compile(patched_source, rows_per_chunk = 5)

    lm = _qwen_lora() if model_name == "qwen3_lora" else _ToyLM()
    model = _HiddenStatesLM(lm)
    params = [p for p in model.parameters() if p.requires_grad]
    vocab = model.get_output_embeddings().weight.shape[0]
    batch = _batch(min(vocab, 1000))

    trainer = _trainer(cls, kind, **kwargs)
    want_loss, want_metrics, want_grads = _run(trainer, stock, model, batch, params)
    assert getattr(trainer, "_unsloth_pref_chunked_calls", 0) == 0
    got_loss, got_metrics, got_grads = _run(trainer, patched, model, batch, params)
    assert trainer._unsloth_pref_chunked_calls == 1, "chunked path did not run"
    assert "UNSLOTH_RETURN_HIDDEN_STATES" not in os.environ, "flag not restored"

    torch.testing.assert_close(got_loss, want_loss, rtol = 1e-5, atol = 1e-6)
    assert got_metrics.keys() == want_metrics.keys()
    for key in want_metrics:
        torch.testing.assert_close(
            torch.tensor(float(got_metrics[key])),
            torch.tensor(float(want_metrics[key])),
            # AlphaPO margins subtract two rewards of magnitude ~5: a few fp32 ulps.
            rtol = 1e-5,
            atol = 1e-5,
            msg = lambda m: f"{key}: {m}",
        )
    for g, w in zip(got_grads, want_grads):
        torch.testing.assert_close(g, w, rtol = 1e-4, atol = 1e-6)


def test_forward_ignoring_the_flag_runs_trl_math(monkeypatch):
    monkeypatch.setenv("UNSLOTH_RETURN_HIDDEN_STATES", "0")
    cls, source = _trl_class("orpo")
    stock = _compile(source)
    patched = _compile(rl.orpo_trainer_concatenated_forward("concatenated_forward", source))
    model = _HiddenStatesLM(_ToyLM(), honour = False)
    params = list(model.parameters())
    batch = _batch(61)
    trainer = _trainer(cls, "orpo")
    want = _run(trainer, stock, model, batch, params)
    got = _run(trainer, patched, model, batch, params)
    assert getattr(trainer, "_unsloth_pref_chunked_calls", 0) == 0
    assert os.environ["UNSLOTH_RETURN_HIDDEN_STATES"] == "0", "prior flag value not restored"
    assert torch.equal(got[0], want[0]) and got[1] == want[1]
    for g, w in zip(got[2], want[2]):
        assert torch.equal(g, w)


def test_labels_differing_from_input_ids_take_a_second_nll_pass(monkeypatch):
    """ORPO's NLL reads input_ids, the log-probs read labels; a row where they disagree must score both."""
    monkeypatch.delenv("UNSLOTH_RETURN_HIDDEN_STATES", raising = False)
    cls, source = _trl_class("orpo")
    stock = _compile(source)
    patched = _compile(rl.orpo_trainer_concatenated_forward("concatenated_forward", source))
    model = _HiddenStatesLM(_ToyLM())
    params = list(model.parameters())
    batch = _batch(61)
    batch["chosen_labels"][0, 6] = (batch["chosen_input_ids"][0, 6] + 1) % 61
    trainer = _trainer(cls, "orpo")
    want = _run(trainer, stock, model, batch, params)
    got = _run(trainer, patched, model, batch, params)
    assert trainer._unsloth_pref_chunked_calls == 1
    torch.testing.assert_close(got[0], want[0], rtol = 1e-5, atol = 1e-6)
    for key in want[1]:
        assert abs(got[1][key] - want[1][key]) <= 1e-5 * max(1.0, abs(want[1][key])), key
    for g, w in zip(got[2], want[2]):
        torch.testing.assert_close(g, w, rtol = 1e-4, atol = 1e-6)


def test_declines_without_kernel_or_on_unsupported_setups(monkeypatch):
    cls, _ = _trl_class("orpo")
    model = _HiddenStatesLM(_ToyLM())
    assert rl._unsloth_pref_chunked_head(_trainer(cls, "orpo"), model) is not None
    for attr, value in (
        ("is_encoder_decoder", True),
        ("aux_loss_enabled", True),
        ("label_pad_token_id", -1),
        ("is_deepspeed_enabled", True),
    ):
        trainer = _trainer(cls, "orpo")
        setattr(trainer, attr, value)
        assert rl._unsloth_pref_chunked_head(trainer, model) is None, attr
    # A subclass reading the full chosen / rejected logits, or scoring its own log-probs, keeps TRL's path.
    for hook in ("get_batch_loss_metrics", "get_batch_logps", "concatenated_forward"):
        sub = type("Sub", (cls,), {hook: lambda self, *a, **k: None})
        assert rl._unsloth_pref_chunked_head(_trainer(sub, "orpo"), model) is None, hook
    monkeypatch.setenv("UNSLOTH_PREFERENCE_CHUNKED", "0")
    assert rl._unsloth_pref_chunked_head(_trainer(cls, "orpo"), model) is None
    monkeypatch.delenv("UNSLOTH_PREFERENCE_CHUNKED")
    monkeypatch.setattr(rl, "unsloth_chunked_logprobs", None)
    assert rl._unsloth_pref_chunked_head(_trainer(cls, "orpo"), model) is None


@pytest.mark.parametrize("kind", ["orpo", "cpo"])
def test_rewrite_is_idempotent_and_leaves_other_layouts(kind):
    _, source = _trl_class(kind)
    rewrite = getattr(rl, f"{kind}_trainer_concatenated_forward")
    once = rewrite("concatenated_forward", source)
    assert rewrite("concatenated_forward", once) == once
    assert rewrite("get_batch_loss_metrics", source) == source
    changed = source.replace("average_log_prob=", "average_log_prob=not ", 1)
    assert rewrite("concatenated_forward", changed) == changed


def test_generated_items_are_self_contained():
    for trainer in ("orpo_trainer", "cpo_trainer"):
        pre = "\n".join(rl.RL_PRE_ITEMS[trainer])
        for name in (
            "_unsloth_pref_chunked_head",
            "_unsloth_pref_chunked_forward",
            "_unsloth_pref_logit_transforms",
            "_unsloth_gkd_dense_head",
            "_unsloth_grpo_returns_hidden_states",
            "_unsloth_grpo_hidden_states_signal",
            "_unsloth_get_model_config",
            "_unsloth_text_configs",
        ):
            assert f"def {name}(" in pre, (trainer, name)
        assert (
            "from unsloth_zoo.fused_losses.chunked_logprob import unsloth_chunked_logprobs" in pre
        )
