# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

"""On-policy distillation helpers (unsloth/models/_on_policy_distillation.py) and the AsyncGRPO StableReinforce
sampler KL rewrite (unsloth/models/rl.py), on CPU against float64 oracles of the tinker-cookbook formulas."""

import ast
import contextlib
import functools
import importlib
import importlib.util
import inspect
import os
import textwrap
import types
import warnings

import pytest
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
OPD_PATH = os.path.join(REPO_ROOT, "unsloth", "models", "_on_policy_distillation.py")
RL_SOURCE_PATH = os.path.join(REPO_ROOT, "unsloth", "models", "rl.py")


def _load_opd():
    spec = importlib.util.spec_from_file_location("_unsloth_opd_under_test", OPD_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


opd = _load_opd()


# A causal LM that, like Unsloth's training forward, ignores the 2D attention mask.
class ToyCausalLM(torch.nn.Module):
    def __init__(
        self,
        vocab = 11,
        hidden = 8,
        seed = 0,
    ):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.embed = torch.nn.Embedding(vocab, hidden)
        self.lm_head = torch.nn.Linear(hidden, vocab, bias = False)
        with torch.no_grad():
            self.embed.weight.copy_(torch.randn(vocab, hidden, generator = g))
            self.lm_head.weight.copy_(torch.randn(vocab, hidden, generator = g))
        self.config = types.SimpleNamespace(vocab_size = vocab)
        self.calls = 0

    def get_output_embeddings(self):
        return self.lm_head

    def forward(
        self,
        input_ids,
        attention_mask = None,
        use_cache = None,
    ):
        self.calls += 1
        h = self.embed(input_ids).double().cumsum(dim = 1) / torch.arange(
            1, input_ids.shape[1] + 1
        ).view(1, -1, 1)
        return types.SimpleNamespace(logits = self.lm_head(h.to(self.lm_head.weight.dtype)))


class FakeLora(torch.nn.Module):
    """A PEFT-like wrapper: the adapter adds a bias to the logits unless disable_adapter() is active."""

    def __init__(
        self,
        inner,
        shift = 0.7,
    ):
        super().__init__()
        self.base_model = types.SimpleNamespace(model = inner)
        self.inner = inner
        self.peft_config = {"default": object()}
        self.shift = shift
        self.enabled = True
        self.disable_calls = 0

    @contextlib.contextmanager
    def disable_adapter(self):
        self.disable_calls += 1
        self.enabled = False
        try:
            yield
        finally:
            self.enabled = True

    def forward(
        self,
        input_ids,
        attention_mask = None,
        use_cache = None,
    ):
        out = self.inner(input_ids, attention_mask = attention_mask)
        if self.enabled:
            out = types.SimpleNamespace(
                logits = out.logits + self.shift * torch.arange(out.logits.shape[-1])
            )
        return out


def _left_padded_batch(seed = 0, vocab = 11):
    g = torch.Generator().manual_seed(seed)
    prompt_ids = torch.randint(1, vocab, (3, 5), generator = g)
    prompt_mask = torch.ones(3, 5, dtype = torch.long)
    prompt_mask[0, :3] = 0
    prompt_mask[1, :1] = 0
    prompt_ids = prompt_ids * prompt_mask  # pad id 0 on the left
    completion_ids = torch.randint(1, vocab, (3, 4), generator = g)
    completion_mask = torch.ones(3, 4, dtype = torch.long)
    completion_mask[0, 2:] = 0
    completion_mask[2, 3:] = 0
    completion_ids = completion_ids * completion_mask
    return prompt_ids, prompt_mask, completion_ids, completion_mask


def _oracle_logps(
    model,
    prompt_ids,
    prompt_mask,
    completion_ids,
    completion_mask,
    temperature = 1.0,
):
    """Row by row with no padding at all, float64."""
    out = torch.zeros(completion_ids.shape, dtype = torch.float64)
    for b in range(completion_ids.shape[0]):
        p = prompt_ids[b][prompt_mask[b].bool()]
        c = completion_ids[b][completion_mask[b].bool()]
        ids = torch.cat([p, c]).unsqueeze(0)
        with torch.no_grad():
            logits = model(ids).logits[0].double()
        logits = logits[p.numel() - 1 : p.numel() - 1 + c.numel()] / temperature
        out[b, : c.numel()] = torch.log_softmax(logits, -1).gather(-1, c.unsqueeze(-1)).squeeze(-1)
    return out


@pytest.mark.parametrize("temperature", (1.0, 0.7))
@pytest.mark.parametrize("rows_per_chunk", (None, 1, 2))
def test_score_completion_logps_matches_unpadded_oracle(temperature, rows_per_chunk):
    model = ToyCausalLM().double()
    batch = _left_padded_batch()
    got = opd.score_completion_logps(
        model, *batch, temperature = temperature, rows_per_chunk = rows_per_chunk
    )
    want = _oracle_logps(model, *batch, temperature = temperature)
    torch.testing.assert_close(got.double(), want, rtol = 1e-6, atol = 1e-6)
    assert torch.all(got[batch[3] == 0] == 0)


def test_score_completion_logps_needs_the_roll():
    # Scoring the left-padded rows as they are lets the pads into a mask-blind model's context.
    model = ToyCausalLM().double()
    prompt_ids, prompt_mask, completion_ids, completion_mask = _left_padded_batch()
    ids = torch.cat([prompt_ids, completion_ids], 1)
    with torch.no_grad():
        logits = model(ids).logits[:, prompt_ids.shape[1] - 1 : -1].double()
    naive = (
        torch.log_softmax(logits, -1).gather(-1, completion_ids.unsqueeze(-1)).squeeze(-1)
        * completion_mask
    )
    want = _oracle_logps(model, prompt_ids, prompt_mask, completion_ids, completion_mask)
    assert not torch.allclose(naive[0], want[0])


def test_score_completion_logps_empty():
    model = ToyCausalLM()
    out = opd.score_completion_logps(
        model,
        torch.zeros(0, 3, dtype = torch.long),
        torch.zeros(0, 3, dtype = torch.long),
        torch.zeros(0, 2, dtype = torch.long),
        torch.zeros(0, 2, dtype = torch.long),
    )
    assert out.shape == (0, 2) and model.calls == 0


def test_score_completion_logps_scores_real_logits(monkeypatch):
    # UNSLOTH_RETURN_HIDDEN_STATES would skip the model's logit scaling / softcapping: off while scoring, restored after.
    model = ToyCausalLM().double()
    seen = []

    class Probe(torch.nn.Module):
        def get_output_embeddings(self):
            return model.lm_head

        def forward(
            self,
            input_ids,
            attention_mask = None,
            use_cache = None,
        ):
            seen.append(os.environ.get("UNSLOTH_RETURN_HIDDEN_STATES"))
            return model(input_ids)

    monkeypatch.setenv("UNSLOTH_RETURN_HIDDEN_STATES", "1")
    batch = _left_padded_batch()
    torch.testing.assert_close(
        opd.score_completion_logps(Probe(), *batch).double(),
        _oracle_logps(model, *batch),
        rtol = 1e-6,
        atol = 1e-6,
    )
    assert seen and set(seen) == {"0"} and os.environ["UNSLOTH_RETURN_HIDDEN_STATES"] == "1"

    class Hidden(Probe):
        def forward(
            self,
            input_ids,
            attention_mask = None,
            use_cache = None,
        ):
            return types.SimpleNamespace(logits = model.embed(input_ids))

    with pytest.raises(RuntimeError, match = "expected logits"):
        opd.score_completion_logps(Hidden(), *batch)
    assert os.environ["UNSLOTH_RETURN_HIDDEN_STATES"] == "1"


class KeepCausalLM(ToyCausalLM):
    """ToyCausalLM whose forward honours logits_to_keep like transformers / Unsloth (an int keeps the last columns)."""

    def __init__(
        self,
        *args,
        ignore_keep = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.ignore_keep = ignore_keep
        self.widths = []

    def forward(
        self,
        input_ids,
        attention_mask = None,
        use_cache = None,
        logits_to_keep = 0,
    ):
        logits = super().forward(input_ids, attention_mask, use_cache).logits
        if logits_to_keep and not self.ignore_keep:
            logits = logits[:, -logits_to_keep:]
        self.widths.append((input_ids.shape[1], logits.shape[1]))
        return types.SimpleNamespace(logits = logits)


@pytest.mark.parametrize("ignore_keep", (False, True))
@pytest.mark.parametrize("rows_per_chunk", (None, 1, 2))
def test_score_completion_logps_keeps_only_completion_columns(ignore_keep, rows_per_chunk):
    # The head sees only the columns from the earliest predicting position on, with bitwise-identical log-probs.
    model = KeepCausalLM(ignore_keep = ignore_keep).double()
    batch = _left_padded_batch()
    got = opd.score_completion_logps(model, *batch, rows_per_chunk = rows_per_chunk)
    full = opd.score_completion_logps(ToyCausalLM().double(), *batch, rows_per_chunk = rows_per_chunk)
    assert torch.equal(got, full)
    torch.testing.assert_close(
        got.double(), _oracle_logps(ToyCausalLM().double(), *batch), rtol = 1e-6, atol = 1e-6
    )
    P, C = batch[0].shape[1], batch[2].shape[1]
    max_left_pad = int((batch[1] == 0).sum(1).max())
    for L, width in model.widths:
        assert L == P + C
        if ignore_keep:
            assert width == L
        else:
            assert C + 1 <= width <= C + 1 + max_left_pad < L
    if not ignore_keep and rows_per_chunk is None:
        assert model.widths == [(P + C, C + 1 + max_left_pad)]


def test_score_completion_logps_rejects_a_wrong_logits_width():
    class Wrong(KeepCausalLM):
        def forward(
            self,
            input_ids,
            attention_mask = None,
            use_cache = None,
            logits_to_keep = 0,
        ):
            logits = ToyCausalLM.forward(self, input_ids).logits[:, -2:]
            return types.SimpleNamespace(logits = logits)

    with pytest.raises(RuntimeError, match = "logit columns"):
        opd.score_completion_logps(Wrong().double(), *_left_padded_batch())


def test_distillation_advantages_match_tinker():
    g = torch.Generator().manual_seed(0)
    sampled = -torch.rand(4, 6, generator = g, dtype = torch.float64) * 3
    teacher = -torch.rand(4, 6, generator = g, dtype = torch.float64) * 3
    mask = (torch.rand(4, 6, generator = g) > 0.3).long()
    for coef in (1.0, 0.5):
        got = opd.on_policy_distillation_advantages(sampled, teacher, mask, coef)
        # tinker-cookbook incorporate_kl_penalty, kl_discount_factor = 0.
        reverse_kl = (sampled - teacher) * mask
        want = -coef * mask * reverse_kl
        torch.testing.assert_close(got.double(), want, rtol = 1e-6, atol = 1e-7)
        assert got.dtype == torch.float32


def test_distillation_advantages_drop_unscored_tokens():
    sampled = torch.tensor([[-1.0, float("nan"), -2.0]])
    teacher = torch.tensor([[-1.5, -1.0, -float("inf")]])
    got = opd.on_policy_distillation_advantages(sampled, teacher, torch.ones(1, 3), 1.0)
    assert got.tolist() == [[-0.5, 0.0, 0.0]]


def test_adapter_off_teacher_scores_the_base():
    inner = ToyCausalLM()
    student = FakeLora(inner)
    teacher = opd.AdapterOffTeacher(student)
    ids = torch.randint(1, 11, (2, 5))
    with torch.no_grad():
        base = inner(ids).logits
        with_adapter = student(ids).logits
    got = teacher(input_ids = ids).logits
    assert torch.equal(got, base) and not torch.equal(got, with_adapter)
    assert student.enabled and not got.requires_grad
    # Owns nothing: accelerate's prepare_model, .eval() and .to() never reach the student.
    assert list(teacher.parameters()) == [] and list(teacher.children()) == []
    student.train()
    teacher.eval()
    assert student.training
    # Reads go to the causal LM TRL reaches through PEFT's base_model.model.
    assert teacher.config is inner.config and teacher.get_output_embeddings() is inner.lm_head
    assert teacher._unsloth_adapter_off_student_model is inner


def test_adapter_off_teacher_exposes_the_base_head():
    # lm_head trained through modules_to_save: DistillationTrainer projects with teacher.get_output_embeddings().
    inner = ToyCausalLM()
    base_head = inner.lm_head

    class ModulesToSave(torch.nn.Module):
        def __init__(self, original):
            super().__init__()
            self.original_module = original
            self.trained = torch.nn.Linear(original.in_features, original.out_features, bias = False)

    wrapped = ModulesToSave(base_head)
    inner.get_output_embeddings = lambda: wrapped
    teacher = opd.AdapterOffTeacher(FakeLora(inner))
    assert teacher.get_output_embeddings() is base_head
    inner.get_output_embeddings = lambda: base_head
    assert teacher.get_output_embeddings() is base_head


def test_distillation_refuses_generation_overrides(monkeypatch):
    monkeypatch.setattr(opd, "_require_zoo", lambda flag, option: None)
    for overrides, word in (
        ({"top_p": 0.5}, "top_p"),
        ({"temperature": 0.7}, "temperature override"),
    ):
        trainer = _grpo_trainer(
            unsloth_distillation_coef = 1.0,
            use_vllm = False,
            temperature = 1.0,
            generation_kwargs = overrides,
        )
        with pytest.raises(ValueError, match = word):
            opd.grpo_setup_distillation(trainer)
    same = _grpo_trainer(
        unsloth_distillation_coef = 1.0,
        use_vllm = False,
        temperature = 0.7,
        generation_kwargs = {"temperature": 0.7, "max_new_tokens": 9},
    )
    assert opd.grpo_setup_distillation(same) == 1.0


def test_adapter_off_teacher_generates_with_the_adapter_off():
    # GKD seq_kd calls teacher.generate, which must not resolve to the student's own generate with LoRA on.
    student = FakeLora(ToyCausalLM())
    seen = []
    student.generate = (
        lambda *a, **k: seen.append((student.enabled, torch.is_grad_enabled())) or "ids"
    )
    teacher = opd.AdapterOffTeacher(student)
    assert teacher.generate(torch.ones(1, 2, dtype = torch.long)) == "ids"
    assert seen == [(False, False)] and student.enabled


def test_adapter_off_teacher_nested_disable_keeps_adapter_off():
    student = FakeLora(ToyCausalLM())
    teacher = opd.AdapterOffTeacher(student)
    ids = torch.randint(1, 11, (1, 4))
    with opd._adapter_off(student):
        teacher(input_ids = ids)
        # The inner call must not re-enable the adapter on its way out.
        assert not student.enabled
    assert student.enabled and student.disable_calls == 1


def test_adapter_off_teacher_through_forward_redirection():
    # TRL's DistillationTrainer reroutes the teacher backbone call through _ForwardRedirection.
    inner = ToyCausalLM()
    student = FakeLora(inner)
    teacher = opd.AdapterOffTeacher(student)
    seen = {}

    def method(module, ids):
        seen["enabled"] = student.enabled
        return module._unsloth_student(ids).logits

    original_forward = teacher.forward

    def wrapped_forward(*a, **k):
        teacher.forward = original_forward
        return method(*a, **k)

    teacher.forward = wrapped_forward
    ids = torch.randint(1, 11, (1, 4))
    out = teacher(teacher, ids)
    assert seen["enabled"] is False and torch.equal(out, inner(ids).logits.detach())


def test_resolve_teacher_model():
    student = FakeLora(ToyCausalLM())
    for request in ("base", " BASE ", "Base"):
        assert isinstance(opd.resolve_teacher_model(request, student), opd.AdapterOffTeacher)
    other = ToyCausalLM()
    assert opd.resolve_teacher_model(other, student) is other
    assert opd.resolve_teacher_model("Qwen/Qwen3-8B", student) == "Qwen/Qwen3-8B"
    assert opd.resolve_teacher_model("./base", student) == "./base"
    mapping = opd.resolve_teacher_model({"a": "base", "b": other}, student)
    assert isinstance(mapping["a"], opd.AdapterOffTeacher) and mapping["b"] is other
    with pytest.raises(ValueError, match = "LoRA"):
        opd.resolve_teacher_model("base", ToyCausalLM())
    with pytest.raises(ValueError, match = "DeepSpeed and FSDP"):
        opd.resolve_teacher_model(
            "base", student, types.SimpleNamespace(deepspeed = "ds.json", fsdp = "")
        )
    with pytest.raises(ValueError, match = "DeepSpeed and FSDP"):
        opd.resolve_teacher_model(
            "base", student, types.SimpleNamespace(deepspeed = None, fsdp = "full_shard")
        )


def _grpo_trainer(**args):
    model = FakeLora(ToyCausalLM())
    return types.SimpleNamespace(
        args = types.SimpleNamespace(**args),
        model = model,
        accelerator = types.SimpleNamespace(num_processes = 1, unwrap_model = lambda m: m),
        _metrics = {"train": __import__("collections").defaultdict(list)},
    )


def test_grpo_setup_validates(monkeypatch):
    monkeypatch.setattr(opd, "_require_zoo", lambda flag, option: None)
    assert opd.grpo_setup_distillation(_grpo_trainer(unsloth_distillation_coef = 0.0)) == 0.0
    with pytest.raises(ValueError, match = ">= 0"):
        opd.grpo_setup_distillation(_grpo_trainer(unsloth_distillation_coef = -1.0))
    with pytest.raises(ValueError, match = "coef is 0"):
        opd.grpo_setup_distillation(_grpo_trainer(unsloth_distillation_coef = 0.0), ToyCausalLM())
    with pytest.raises(ValueError, match = "DeepSpeed and FSDP"):
        opd.grpo_setup_distillation(
            _grpo_trainer(unsloth_distillation_coef = 1.0, deepspeed = "ds.json")
        )
    with pytest.raises(ValueError, match = "takes 'base'"):
        opd.grpo_setup_distillation(
            _grpo_trainer(
                unsloth_distillation_coef = 1.0, unsloth_distillation_teacher = "Qwen/Qwen3-4B"
            )
        )
    trainer = _grpo_trainer(unsloth_distillation_coef = 1.0, unsloth_distillation_teacher = "base")
    assert opd.grpo_setup_distillation(trainer) == 1.0
    assert isinstance(trainer._unsloth_distillation_teacher, opd.AdapterOffTeacher)
    small_vocab = ToyCausalLM(vocab = 7)
    with pytest.raises(ValueError, match = "tokenizer"):
        opd.grpo_setup_distillation(_grpo_trainer(unsloth_distillation_coef = 1.0), small_vocab)
    separate = ToyCausalLM(seed = 3)
    trainer = _grpo_trainer(unsloth_distillation_coef = 1.0)
    opd.grpo_setup_distillation(trainer, separate)
    assert trainer._unsloth_distillation_teacher is separate
    assert not separate.training and not any(p.requires_grad for p in separate.parameters())


def test_grpo_setup_refuses_filters_it_cannot_reproduce(monkeypatch):
    monkeypatch.setattr(opd, "_require_zoo", lambda flag, option: None)
    for field, value in (
        ("top_k", 1),
        ("top_p", 0.9),
        ("min_p", 0.05),
        ("repetition_penalty", 1.1),
    ):
        trainer = _grpo_trainer(unsloth_distillation_coef = 1.0, use_vllm = False, **{field: value})
        with pytest.raises(ValueError, match = field):
            opd.grpo_setup_distillation(trainer)
        # vLLM's sampling logprobs are the sampler's, filters included.
        trainer = _grpo_trainer(unsloth_distillation_coef = 1.0, use_vllm = True, **{field: value})
        assert opd.grpo_setup_distillation(trainer) == 1.0
    defaults = _grpo_trainer(
        unsloth_distillation_coef = 1.0,
        use_vllm = False,
        top_k = 0,
        top_p = 1.0,
        min_p = None,
        repetition_penalty = 1.0,
    )
    assert opd.grpo_setup_distillation(defaults) == 1.0
    # vLLM with filters but no sampling logprobs in the rollout (older TRL, IS correction off): refused there.
    trainer = _grpo_trainer(
        unsloth_distillation_coef = 1.0,
        use_vllm = True,
        temperature = 1.0,
        vllm_sampling_params = types.SimpleNamespace(min_p = 0.1),
    )
    assert opd.grpo_setup_distillation(trainer) == 1.0
    batch = _left_padded_batch()
    output = dict(
        prompt_ids = batch[0],
        prompt_mask = batch[1],
        completion_ids = batch[2],
        completion_mask = batch[3],
        advantages = torch.zeros(3),
    )
    with pytest.raises(ValueError, match = "min_p"):
        opd.grpo_add_distillation_advantages(trainer, output)


def test_distillation_refuses_untruncated_filtered_vllm_logps(monkeypatch):
    import unsloth_zoo.rl_replacements as zoo_rl

    monkeypatch.setattr(opd, "_require_zoo", lambda flag, option: None)
    trainer = _grpo_trainer(unsloth_distillation_coef = 1.0, temperature = 1.0)
    opd.grpo_setup_distillation(trainer)
    batch = _left_padded_batch()
    output = dict(
        prompt_ids = batch[0],
        prompt_mask = batch[1],
        completion_ids = batch[2],
        completion_mask = batch[3],
        advantages = torch.zeros(3),
        sampling_per_token_logps = -torch.rand(batch[2].shape),
    )
    for untruncated, support, refused in (
        (True, (1, 1.0, 0.0), True),
        (True, None, False),
        (False, (0, 0.9, 0.0), False),
    ):
        monkeypatch.setattr(
            zoo_rl, "grpo_rollout_logprobs_untruncated", lambda t, v = untruncated: v, raising = False
        )
        monkeypatch.setattr(zoo_rl, "grpo_sampling_support", lambda t, v = support: v, raising = False)
        if refused:
            with pytest.raises(ValueError, match = "UNSLOTH_VLLM_UNTRUNCATED_LOGPROBS=0"):
                opd.grpo_add_distillation_advantages(trainer, dict(output))
        else:
            assert "unsloth_distillation_advantages" in opd.grpo_add_distillation_advantages(
                trainer, dict(output)
            )


def test_grpo_setup_refuses_a_subclass_objective(monkeypatch):
    monkeypatch.setattr(opd, "_require_zoo", lambda flag, option: None)
    import sys

    fake = types.ModuleType("unsloth.models._grpo_subclass_loss")
    fake.grpo_subclass_compute_loss = lambda trainer: getattr(trainer, "own_objective", None)
    monkeypatch.setitem(sys.modules, "unsloth.models._grpo_subclass_loss", fake)
    trainer = _grpo_trainer(unsloth_distillation_coef = 1.0)
    trainer.own_objective = object()
    with pytest.raises(ValueError, match = "_compute_loss"):
        opd.grpo_setup_distillation(trainer)
    trainer = _grpo_trainer(unsloth_sampler_kl_coef = 0.01, unsloth_sampler_kl_estimator = "k1")
    trainer.own_objective = object()
    with pytest.raises(ValueError, match = "_compute_loss"):
        opd.grpo_check_sampler_kl(trainer)
    assert opd.grpo_setup_distillation(_grpo_trainer(unsloth_distillation_coef = 1.0)) == 1.0


def test_grpo_setup_refuses_an_old_zoo(monkeypatch):
    import unsloth_zoo.rl_replacements as zoo_rl

    monkeypatch.delattr(zoo_rl, "GRPO_TOKEN_ADVANTAGES", raising = False)
    with pytest.raises(RuntimeError, match = "newer unsloth_zoo"):
        opd.grpo_setup_distillation(_grpo_trainer(unsloth_distillation_coef = 1.0))
    monkeypatch.delattr(zoo_rl, "GRPO_SAMPLER_KL", raising = False)
    with pytest.raises(RuntimeError, match = "newer unsloth_zoo"):
        opd.grpo_check_sampler_kl(
            _grpo_trainer(unsloth_sampler_kl_coef = 0.01, unsloth_sampler_kl_estimator = "k1")
        )
    assert opd.grpo_check_sampler_kl(_grpo_trainer(unsloth_sampler_kl_coef = 0.0)) == 0.0
    with pytest.raises(ValueError, match = "'k1' or 'k3'"):
        opd.grpo_check_sampler_kl(
            _grpo_trainer(unsloth_sampler_kl_coef = 0.0, unsloth_sampler_kl_estimator = "k2")
        )


def test_grpo_add_distillation_advantages_end_to_end(monkeypatch):
    monkeypatch.setattr(opd, "_require_zoo", lambda flag, option: None)
    trainer = _grpo_trainer(unsloth_distillation_coef = 0.5, temperature = 1.0)
    opd.grpo_setup_distillation(trainer)
    prompt_ids, prompt_mask, completion_ids, completion_mask = _left_padded_batch()
    output = dict(
        prompt_ids = prompt_ids,
        prompt_mask = prompt_mask,
        completion_ids = completion_ids,
        completion_mask = completion_mask,
        advantages = torch.zeros(3),
    )
    opd.grpo_add_distillation_advantages(trainer, output)
    student = _oracle_logps(trainer.model, prompt_ids, prompt_mask, completion_ids, completion_mask)
    teacher = _oracle_logps(
        trainer.model.inner, prompt_ids, prompt_mask, completion_ids, completion_mask
    )
    want = -0.5 * (student - teacher) * completion_mask
    torch.testing.assert_close(
        output["unsloth_distillation_advantages"].double(), want, rtol = 1e-5, atol = 1e-5
    )
    # vLLM's logprobs, when present, are the sampler's: they replace the student scoring pass.
    sampled = -torch.rand(completion_ids.shape, dtype = torch.float32)
    output["sampling_per_token_logps"] = sampled
    calls = trainer.model.inner.calls
    opd.grpo_add_distillation_advantages(trainer, output)
    assert trainer.model.inner.calls == calls + 1  # the teacher only
    want = -0.5 * (sampled.double() - teacher) * completion_mask
    torch.testing.assert_close(
        output["unsloth_distillation_advantages"].double(), want, rtol = 1e-5, atol = 1e-5
    )
    kl = trainer._metrics["train"]["distillation/teacher_kl"]
    assert len(kl) == 2
    with pytest.raises(NotImplementedError):
        opd.grpo_add_distillation_advantages(trainer, dict(output, pixel_values = torch.zeros(1)))


def test_wrap_grpo_on_policy_distillation_is_idempotent_and_off_by_default():
    class T:
        def _generate_and_score_completions(self, inputs):
            return {"x": inputs}

    opd.wrap_grpo_on_policy_distillation(T)
    first = T._generate_and_score_completions
    opd.wrap_grpo_on_policy_distillation(T)
    assert T._generate_and_score_completions is first
    assert T()._generate_and_score_completions(3) == {"x": 3}


# AsyncGRPO StableReinforce sampler KL (rl.py source rewrite of TRL's compute_loss).


def _load_rl(names):
    src = open(RL_SOURCE_PATH, encoding = "utf-8").read()
    ns = {
        "functools": functools,
        "inspect": inspect,
        "textwrap": textwrap,
        "torch": torch,
        "contextvars": __import__("contextvars"),
        "weakref": __import__("weakref"),
        "types": types,
    }
    for node in ast.parse(src).body:
        is_constant = isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id.startswith("_ASYNC_GRPO_") for t in node.targets
        )
        if is_constant or (isinstance(node, ast.FunctionDef) and node.name in names):
            exec(compile(ast.Module([node], []), RL_SOURCE_PATH, "exec"), ns)
    assert all(n in ns for n in names)
    return ns


RL = _load_rl(("_async_grpo_sampler_kl_compute_loss", "_async_grpo_masked_compute_loss"))


def _loss_118(
    self,
    model,
    inputs,
    return_outputs = False,
    num_items_in_batch = None,
):
    # The policy-loss lines of TRL >= 1.16 AsyncGRPOTrainer.compute_loss (score_centering off), verbatim.
    log_probs = model(inputs)
    old_log_probs = inputs["old_log_probs"]
    advantages = inputs["advantages"]
    log_ratio = log_probs - old_log_probs
    coef_1 = torch.exp(log_ratio)
    low, high = 0.0, 1e9
    is_masked = (coef_1 < low) | (coef_1 > high)
    coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)
    per_token_loss1 = coef_1 * advantages
    per_token_loss2 = coef_2 * advantages
    per_token_loss = -torch.min(per_token_loss1, per_token_loss2) * ~is_masked

    loss = per_token_loss.sum()
    return loss


def _loss_115(
    self,
    model,
    inputs,
    return_outputs = False,
    num_items_in_batch = None,
):
    # TRL 1.13 - 1.15: masked reduction.
    log_probs = model(inputs)
    old_log_probs = inputs["old_log_probs"]
    advantages = inputs["advantages"]
    completion_mask = inputs["completion_mask"]
    log_ratio = log_probs - old_log_probs
    coef_1 = torch.exp(log_ratio)
    coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)
    per_token_loss1 = coef_1 * advantages
    per_token_loss2 = coef_2 * advantages
    per_token_loss = -torch.min(per_token_loss1, per_token_loss2)

    loss = (per_token_loss * completion_mask).sum()
    return loss


@pytest.mark.parametrize("estimator", ("k1", "k3"))
@pytest.mark.parametrize("source", (_loss_118, _loss_115))
@pytest.mark.parametrize("d", (0.0, 1e-6, -1e-6, 0.1, -0.1, 1.0, -1.0, 10.0, -10.0))
def test_async_sampler_kl_matches_fp64_oracle(estimator, source, d):
    coef = 0.01
    patched = RL["_async_grpo_sampler_kl_compute_loss"](source, False)
    trainer = types.SimpleNamespace(
        epsilon_low = 0.2,
        epsilon_high = 0.28,
        _unsloth_sampler_kl_coef = coef,
        _unsloth_sampler_kl_estimator = estimator,
    )
    g = torch.Generator().manual_seed(0)
    old = -torch.rand(7, generator = g, dtype = torch.float64) * 3
    signs = torch.where(torch.rand(7, generator = g) > 0.5, 1.0, -1.0).double()
    log_probs = (old + d * signs).requires_grad_(True)
    adv = torch.randn(7, generator = g, dtype = torch.float64)
    mask = torch.tensor([1, 1, 1, 0, 1, 1, 1], dtype = torch.float64)
    inputs = {"old_log_probs": old, "advantages": adv, "completion_mask": mask}
    loss = patched(trainer, lambda _: log_probs, inputs)
    (grad,) = torch.autograd.grad(loss, log_probs)
    lp = log_probs.detach().clone().requires_grad_(True)
    base = source(trainer, lambda _: lp, inputs)
    log_r = lp - old
    kl = -log_r if estimator == "k1" else torch.exp(log_r) - 1 - log_r
    weight = mask if source is _loss_115 else torch.ones_like(mask)
    want = base + coef * (kl * weight).sum()
    (want_grad,) = torch.autograd.grad(want, lp)
    torch.testing.assert_close(loss, want, rtol = 1e-12, atol = 1e-12)
    torch.testing.assert_close(grad, want_grad, rtol = 1e-12, atol = 1e-12)


def test_async_sampler_kl_is_cached_and_composes_with_the_mask():
    fn = RL["_async_grpo_sampler_kl_compute_loss"]
    assert fn(_loss_118, False) is fn(_loss_118, False)
    both = fn(_loss_118, True)
    trainer = types.SimpleNamespace(
        epsilon_low = 0.2,
        epsilon_high = 0.28,
        _unsloth_sampler_kl_coef = 0.5,
        _unsloth_sampler_kl_estimator = "k3",
    )
    old = torch.tensor([-1.0, -1.0], dtype = torch.float64)
    lp = torch.tensor(
        [-1.0 + 0.9, -1.0 + 0.01], dtype = torch.float64, requires_grad = True
    )  # ratio e^0.9 is masked
    inputs = {"old_log_probs": old, "advantages": torch.tensor([1.0, 1.0], dtype = torch.float64)}
    (grad,) = torch.autograd.grad(both(trainer, lambda _: lp, inputs), lp)
    r = torch.exp(lp.detach() - old)
    # Masked token: only the KL gradient 0.5 * (r - 1); kept token: -r * A plus the KL.
    torch.testing.assert_close(grad, torch.stack([0.5 * (r[0] - 1), -r[1] + 0.5 * (r[1] - 1)]))


def test_async_sampler_kl_refuses_an_unknown_loss():
    def compute_loss(
        self,
        model,
        inputs,
        return_outputs = False,
        num_items_in_batch = None,
    ):
        return model(inputs).sum()

    with pytest.raises(ValueError, match = "sampler KL cannot be added"):
        RL["_async_grpo_sampler_kl_compute_loss"](compute_loss, False)


def test_async_sampler_kl_rewrites_the_installed_trl_loss():
    AsyncGRPOTrainer = None
    error = None
    for name in (
        "trl.trainer.async_grpo_trainer",
        "trl.experimental.async_grpo.async_grpo_trainer",
    ):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                AsyncGRPOTrainer = importlib.import_module(name).AsyncGRPOTrainer
            break
        except Exception as e:
            error = e
    if AsyncGRPOTrainer is None:
        pytest.skip(reason = f"needs TRL >= 1.13, which ships AsyncGRPOTrainer: {error}")
    for masked in (False, True):
        patched = RL["_async_grpo_sampler_kl_compute_loss"](AsyncGRPOTrainer.compute_loss, masked)
        assert patched.__globals__ is AsyncGRPOTrainer.compute_loss.__globals__
        assert "_unsloth_sampler_kl_coef" in patched.__code__.co_names
