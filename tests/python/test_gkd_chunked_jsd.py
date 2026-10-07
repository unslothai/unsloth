# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team.
"""GKD compute_loss routes through the chunked generalized JSD (#11554) and matches TRL."""

from __future__ import annotations

import ast
import inspect
import os
import textwrap
import types

import pytest
from real_accelerator import has_real_cuda
import torch

os.environ.setdefault("UNSLOTH_COMPILE_DISABLE", "1")
import unsloth.models.rl_replacements as rl

trl = pytest.importorskip("trl")
try:
    from trl.experimental.gkd.gkd_trainer import GKDTrainer
except Exception:
    from trl.trainer.gkd_trainer import GKDTrainer

if rl.distillation_chunked_jsd is None:
    pytest.skip("installed unsloth_zoo predates distillation_chunked_jsd", allow_module_level = True)

# The two compute_loss layouts TRL shipped between 0.22.2 and 1.14.0, non-Liger branch only.
PROMPT_LAYOUT = """
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        student_outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
        )
        self.teacher_model.eval()
        with torch.no_grad():
            teacher_outputs = self.teacher_model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
            )
        prompt_lengths = inputs["prompts"].shape[1]
        shifted_student_logits = student_outputs.logits[:, prompt_lengths - 1 : -1, :]
        shifted_teacher_logits = teacher_outputs.logits[:, prompt_lengths - 1 : -1, :]
        shifted_labels = inputs["labels"][:, prompt_lengths:]
        loss = self.generalized_jsd_loss(
            student_logits=shifted_student_logits,
            teacher_logits=shifted_teacher_logits,
            labels=shifted_labels,
            beta=self.beta,
        )
        empty_cache()
        return (loss, student_outputs) if return_outputs else loss
"""
SHIFT_LAYOUT = (
    PROMPT_LAYOUT.replace('        prompt_lengths = inputs["prompts"].shape[1]\n', "")
    .replace("[:, prompt_lengths - 1 : -1, :]", "[:, :-1, :]")
    .replace('inputs["labels"][:, prompt_lengths:]', 'inputs["labels"][:, 1:]')
    .replace(
        "beta=self.beta,\n", "beta=self.beta,\n            num_items_in_batch=num_items_in_batch,\n"
    )
)


def test_both_layouts_are_recognised():
    assert rl._unsloth_gkd_layout(PROMPT_LAYOUT) == {"shift": "prompt", "num_items_in_batch": False}
    assert rl._unsloth_gkd_layout(SHIFT_LAYOUT) == {"shift": "shift", "num_items_in_batch": True}


def test_installed_trl_layout_is_recognised():
    """Read TRL's file, not the class: importing unsloth has already rebound the class to the patched one."""
    import importlib.util

    for name in ("trl.trainer.gkd_trainer", "trl.experimental.gkd.gkd_trainer"):
        try:
            spec = importlib.util.find_spec(name)
        except Exception:
            spec = None
        if spec is None or spec.origin is None:
            continue
        text = open(spec.origin, encoding = "utf-8").read()
        nodes = [
            n
            for n in ast.walk(ast.parse(text))
            if isinstance(n, ast.FunctionDef) and n.name == "compute_loss"
        ]
        if not nodes:
            continue
        source = ast.get_source_segment(text, nodes[0], padded = True)
        assert rl._unsloth_gkd_layout(source) is not None, f"{name}.compute_loss is not recognised"
        return
    pytest.skip("no GKD trainer in the installed TRL")


def test_patched_trainer_uses_the_chunked_loss():
    patched = GKDTrainer
    try:
        import trl.trainer.gkd_trainer as mod
        patched = mod.GKDTrainer
    except Exception:
        pass
    assert hasattr(patched, "_unsloth_trl_compute_loss"), patched
    assert rl._unsloth_gkd_jsd_supported(patched)


@pytest.mark.parametrize("source", [PROMPT_LAYOUT, SHIFT_LAYOUT])
def test_rewrite_parses_and_keeps_trl_body(source):
    new = rl.gkd_trainer_compute_loss("compute_loss", source)
    ast.parse("class X:\n" + new)
    assert new.count("def compute_loss(") == 1
    assert "def _unsloth_trl_compute_loss(" in new
    assert "_unsloth_gkd_chunked_loss(" in new
    assert rl.gkd_trainer_compute_loss("training_step", source) == source


@pytest.mark.parametrize(
    "mutate",
    [
        lambda s: s.replace("beta=self.beta,", "beta=self.beta, reduction='sum',"),
        lambda s: s.replace("beta=self.beta,", "beta=0.3,"),
        lambda s: s.replace(
            'attention_mask=inputs["attention_mask"],\n        )',
            'attention_mask=inputs["attention_mask"], pixel_values=inputs["pixel_values"],\n        )',
            1,
        ),
        lambda s: s.replace(
            'inputs["labels"][:, prompt_lengths:]', 'inputs["labels"][:, prompt_lengths + 1:]'
        ),
        lambda s: s.replace("self.generalized_jsd_loss(", "self.other_loss("),
        lambda s: s.replace("beta=self.beta,", "beta=self.beta, temperature=self.temperature,"),
        lambda s: s.replace(
            "        empty_cache()\n",
            "        loss = loss + 0.1 * self.generalized_jsd_loss(student_logits=shifted_student_logits, teacher_logits=shifted_teacher_logits, labels=shifted_labels, beta=self.beta)\n        empty_cache()\n",
        ),
        lambda s: s.replace(
            "return (loss, student_outputs) if return_outputs else loss",
            "return (loss * 2, student_outputs) if return_outputs else loss * 2",
        ),
    ],
)
def test_unrecognised_layouts_are_left_to_trl(mutate):
    mutated = mutate(PROMPT_LAYOUT)
    assert mutated != PROMPT_LAYOUT
    assert rl._unsloth_gkd_layout(mutated) is None
    assert rl.gkd_trainer_compute_loss("compute_loss", mutated) == mutated


class _Head(torch.nn.Module):
    def __init__(self, head):
        super().__init__()
        self.head = head

    def get_output_embeddings(self):
        return self.head


def test_dense_head_guard():
    assert rl._unsloth_gkd_dense_head(_Head(torch.nn.Linear(8, 32, bias = False))) is not None
    assert rl._unsloth_gkd_dense_head(_Head(torch.nn.Embedding(32, 8))) is None
    assert rl._unsloth_gkd_dense_head(_Head(None)) is None
    int_head = torch.nn.Linear(8, 32, bias = False)
    int_head.weight = torch.nn.Parameter(torch.zeros(32, 8, dtype = torch.uint8), requires_grad = False)
    assert rl._unsloth_gkd_dense_head(_Head(int_head)) is None
    empty = torch.nn.Linear(8, 32, bias = False)
    empty.weight = torch.nn.Parameter(torch.empty(0))  # ZeRO-3 partitioned
    assert rl._unsloth_gkd_dense_head(_Head(empty)) is None


def test_quantized_and_adapted_heads_fall_back():
    peft = pytest.importorskip("peft")

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = torch.nn.Linear(8, 8)
            self.lm_head = torch.nn.Linear(8, 32, bias = False)

        def get_output_embeddings(self):
            return self.lm_head

    lora = peft.get_peft_model(Tiny(), peft.LoraConfig(target_modules = ["lm_head"], r = 2))
    assert rl._unsloth_gkd_dense_head(lora.base_model.model) is None
    saved = peft.get_peft_model(
        Tiny(), peft.LoraConfig(target_modules = ["proj"], modules_to_save = ["lm_head"], r = 2)
    )
    assert rl._unsloth_gkd_dense_head(saved.base_model.model) is None
    try:
        import bitsandbytes as bnb
    except Exception:
        return
    assert rl._unsloth_gkd_dense_head(_Head(bnb.nn.Linear8bitLt(8, 32, bias = False))) is None


class _TinyLM(torch.nn.Module):
    """Honours UNSLOTH_RETURN_HIDDEN_STATES like Unsloth's compiled forwards: hidden states come back as ``.logits``."""

    __UNSLOTH_SUPPORTS_RETURN_HIDDEN_STATES__ = True

    def __init__(self, vocab, hidden, softcap, seed):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.embed = torch.nn.Parameter(
            torch.randn(vocab, hidden, generator = g, dtype = torch.float64)
        )
        self.mix = torch.nn.Parameter(
            torch.randn(hidden, hidden, generator = g, dtype = torch.float64) / hidden**0.5
        )
        self.lm_head = torch.nn.Linear(hidden, vocab, bias = False, dtype = torch.float64)
        with torch.no_grad():
            self.lm_head.weight.copy_(torch.randn(vocab, hidden, generator = g, dtype = torch.float64))
        self.config = types.SimpleNamespace(final_logit_softcapping = softcap, model_type = "tiny")

    def get_output_embeddings(self):
        return self.lm_head

    def forward(self, input_ids, attention_mask):
        hidden = torch.tanh(self.embed[input_ids] @ self.mix)
        if os.environ.get("UNSLOTH_RETURN_HIDDEN_STATES", "0") == "1":
            return types.SimpleNamespace(logits = hidden)
        logits = self.lm_head(hidden)
        if self.config.final_logit_softcapping:
            cap = self.config.final_logit_softcapping
            logits = cap * torch.tanh(logits / cap)
        return types.SimpleNamespace(logits = logits)


def _trainer(beta, student, teacher):
    trainer = GKDTrainer.__new__(GKDTrainer)
    trainer.__dict__.update(
        beta = beta,
        temperature = 0.9,
        use_liger_gkd_loss = False,
        teacher_model = teacher,
        accelerator = types.SimpleNamespace(unwrap_model = lambda m: m),
    )
    return trainer


def _inputs(vocab):
    g = torch.Generator().manual_seed(0)
    input_ids = torch.randint(0, vocab, (3, 21), generator = g)
    labels = input_ids.clone()
    labels[:, :7] = -100  # prompt
    labels[1, 17:] = -100  # right padding
    return {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
        "labels": labels,
        "prompts": input_ids[:, :7],
    }


@pytest.mark.parametrize("beta", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("shift", ["prompt", "shift"])
def test_chunked_loss_matches_trl(beta, shift, monkeypatch):
    monkeypatch.setenv("UNSLOTH_GKD_CHUNK_SIZE", "8")
    vocab = 97
    student, teacher = _TinyLM(vocab, 16, 0.0, 1), _TinyLM(vocab, 24, 5.0, 2)
    trainer, inputs = _trainer(beta, student, teacher), _inputs(vocab)
    layout = {"shift": shift, "num_items_in_batch": shift == "shift"}

    monkeypatch.setenv("UNSLOTH_RETURN_HIDDEN_STATES", "0")
    s, t = student(**{k: inputs[k] for k in ("input_ids", "attention_mask")}).logits, None
    with torch.no_grad():
        t = teacher(**{k: inputs[k] for k in ("input_ids", "attention_mask")}).logits
    if shift == "prompt":
        s, t, labels = s[:, 6:-1], t[:, 6:-1], inputs["labels"][:, 7:]
    else:
        s, t, labels = s[:, :-1], t[:, :-1], inputs["labels"][:, 1:]
    want = GKDTrainer.generalized_jsd_loss(s, t, labels = labels, beta = beta)
    want_grads = torch.autograd.grad(want, (student.embed, student.mix))

    got = rl._unsloth_gkd_chunked_loss(trainer, student, inputs, None, layout)
    assert got is not None, "chunked path declined a model that honours the flag"
    assert os.environ.get("UNSLOTH_RETURN_HIDDEN_STATES") == "0", "flag not restored"
    got_grads = torch.autograd.grad(got, (student.embed, student.mix))
    torch.testing.assert_close(got.double(), want, rtol = 1e-5, atol = 1e-7)
    for g, w in zip(got_grads, want_grads):
        torch.testing.assert_close(g, w, rtol = 1e-4, atol = 1e-7)


def test_num_items_in_batch_only_where_trl_uses_it(monkeypatch):
    vocab = 97
    student, teacher = _TinyLM(vocab, 16, 0.0, 1), _TinyLM(vocab, 24, 0.0, 2)
    trainer, inputs = _trainer(0.5, student, teacher), _inputs(vocab)
    local = rl._unsloth_gkd_chunked_loss(
        trainer,
        student,
        inputs,
        torch.tensor(1000),
        {"shift": "shift", "num_items_in_batch": False},
    )
    scaled = rl._unsloth_gkd_chunked_loss(
        trainer, student, inputs, torch.tensor(1000), {"shift": "shift", "num_items_in_batch": True}
    )
    n = int((inputs["labels"][:, 1:] != -100).sum())
    torch.testing.assert_close(scaled * 1000, local * n, rtol = 1e-5, atol = 1e-8)


def test_fallbacks_return_none(monkeypatch):
    vocab = 97
    student, teacher = _TinyLM(vocab, 16, 0.0, 1), _TinyLM(vocab, 24, 0.0, 2)
    inputs = _inputs(vocab)
    layout = {"shift": "shift", "num_items_in_batch": False}
    monkeypatch.setenv("UNSLOTH_GKD_CHUNKED", "0")
    assert (
        rl._unsloth_gkd_chunked_loss(_trainer(0.5, student, teacher), student, inputs, None, layout)
        is None
    )
    monkeypatch.delenv("UNSLOTH_GKD_CHUNKED")
    liger = _trainer(0.5, student, teacher)
    liger.use_liger_gkd_loss = True
    assert rl._unsloth_gkd_chunked_loss(liger, student, inputs, None, layout) is None
    assert (
        rl._unsloth_gkd_chunked_loss(_trainer(0.5, student, None), student, inputs, None, layout)
        is None
    )
    assert (
        rl._unsloth_gkd_chunked_loss(
            _trainer(0.5, student, _TinyLM(98, 24, 0.0, 2)), student, inputs, None, layout
        )
        is None
    )

    class Custom(GKDTrainer):
        @staticmethod
        def generalized_jsd_loss(*args, **kwargs):
            return 0

    custom = Custom.__new__(Custom)
    custom.__dict__.update(_trainer(0.5, student, teacher).__dict__)
    assert rl._unsloth_gkd_chunked_loss(custom, student, inputs, None, layout) is None


def test_forward_without_hidden_states_finishes_on_trl_math(monkeypatch):
    """A teacher that ignores the flag returns real logits; the call must still match TRL, densely."""
    vocab = 97
    student, teacher = _TinyLM(vocab, 16, 0.0, 1), _TinyLM(vocab, 24, 5.0, 2)
    type(teacher)
    real_forward = teacher.forward

    def logits_only(input_ids, attention_mask):
        prior = os.environ.get("UNSLOTH_RETURN_HIDDEN_STATES")
        os.environ["UNSLOTH_RETURN_HIDDEN_STATES"] = "0"
        try:
            return real_forward(input_ids, attention_mask)
        finally:
            os.environ["UNSLOTH_RETURN_HIDDEN_STATES"] = prior

    teacher.forward = logits_only
    trainer, inputs = _trainer(0.5, student, teacher), _inputs(vocab)
    layout = {"shift": "shift", "num_items_in_batch": False}
    got = rl._unsloth_gkd_chunked_loss(trainer, student, inputs, None, layout)
    monkeypatch.setenv("UNSLOTH_RETURN_HIDDEN_STATES", "0")
    s = student(inputs["input_ids"], inputs["attention_mask"]).logits[:, :-1]
    with torch.no_grad():
        t = real_forward(inputs["input_ids"], inputs["attention_mask"]).logits[:, :-1]
    want = GKDTrainer.generalized_jsd_loss(s, t, labels = inputs["labels"][:, 1:], beta = 0.5)
    torch.testing.assert_close(got, want, rtol = 1e-6, atol = 1e-9)


def test_generated_items_are_self_contained():
    """RL_PRE_ITEMS carries every helper the rewritten compute_loss calls into the generated module."""
    pre = "\n".join(rl.RL_PRE_ITEMS["gkd_trainer"])
    for name in (
        "_unsloth_gkd_chunked_loss",
        "_unsloth_gkd_dense_head",
        "_unsloth_gkd_logit_transforms",
        "_unsloth_gkd_project",
        "_unsloth_gkd_canonical",
        "_unsloth_gkd_jsd_supported",
        "_unsloth_gkd_note_fallback",
        "_unsloth_gkd_chunk_size",
        "_unsloth_grpo_returns_hidden_states",
        "_unsloth_grpo_hidden_states_signal",
        "_unsloth_get_model_config",
    ):
        assert f"def {name}(" in pre, name
    assert "from unsloth_zoo.rl_replacements import distillation_chunked_jsd" in pre
    assert rl.gkd_trainer_compute_loss in rl.RL_FUNCTIONS["gkd_trainer"]


def _jsd_source():
    """TRL's own ``generalized_jsd_loss`` from the installed file (the class may already be the generated one)."""
    import importlib.util

    for name in ("trl.experimental.gkd.gkd_trainer", "trl.trainer.gkd_trainer"):
        try:
            spec = importlib.util.find_spec(name)
        except Exception:
            spec = None
        if spec is None or spec.origin is None:
            continue
        text = open(spec.origin, encoding = "utf-8").read()
        for node in ast.walk(ast.parse(text)):
            if isinstance(node, ast.FunctionDef) and node.name == "generalized_jsd_loss":
                return textwrap.dedent(ast.get_source_segment(text, node, padded = True))
    pytest.skip("no GKD trainer in the installed TRL")


def _jsd_class(source, tmp_path, name):
    import importlib.util

    body = (
        "import torch\nimport torch.nn.functional as F\nclass GKDTrainer:\n    @staticmethod\n"
        + textwrap.indent(source, "    ")
    )
    path = tmp_path / f"{name}.py"
    path.write_text(body)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.GKDTrainer


def test_installed_generalized_jsd_is_accepted(tmp_path):
    assert rl._unsloth_gkd_jsd_supported(_jsd_class(_jsd_source(), tmp_path, "gkd_jsd_ok"))


@pytest.mark.parametrize(
    "old, new",
    [
        (
            "jsd = beta * kl_teacher + (1 - beta) * kl_student",
            "jsd = (beta * kl_teacher + (1 - beta) * kl_student) * temperature ** 2",
        ),
        ("jsd = jsd[mask]", "jsd = jsd[mask] + 0.0"),
        (
            "student_log_probs = F.log_softmax(student_logits, dim=-1)",
            "student_log_probs = F.log_softmax(student_logits.float(), dim=-1)",
        ),
        ("jsd_sum / num_items_in_batch", "jsd_sum / num_items_in_batch / jsd.size(-1)"),
        ("return jsd_sum / num_items_in_batch", "return jsd_sum"),
    ],
)
def test_changed_generalized_jsd_is_left_to_trl(old, new, tmp_path):
    source = _jsd_source()
    if old not in source:
        pytest.skip("line not in this TRL release")
    mutated = source.replace(old, new)
    name = f"gkd_jsd_mut_{abs(hash((old, new)))}"
    assert not rl._unsloth_gkd_jsd_supported(_jsd_class(mutated, tmp_path, name))


def test_fsdp_deepspeed_and_missing_inputs_fall_back():
    vocab = 97
    student, teacher = _TinyLM(vocab, 16, 0.0, 1), _TinyLM(vocab, 24, 0.0, 2)
    layout = {"shift": "prompt", "num_items_in_batch": False}
    for flag in ("is_fsdp_enabled", "is_deepspeed_enabled"):
        trainer = _trainer(0.5, student, teacher)
        setattr(trainer, flag, True)
        assert rl._unsloth_gkd_chunked_loss(trainer, student, _inputs(vocab), None, layout) is None
        assert trainer._unsloth_gkd_chunked_fallbacks == {"FSDP / DeepSpeed": 1}
    inputs = _inputs(vocab)
    inputs.pop("prompts")
    assert (
        rl._unsloth_gkd_chunked_loss(_trainer(0.5, student, teacher), student, inputs, None, layout)
        is None
    )


def test_batch_encoding_inputs_take_the_chunked_path():
    from transformers import BatchEncoding

    vocab = 97
    student, teacher = _TinyLM(vocab, 16, 0.0, 1), _TinyLM(vocab, 24, 0.0, 2)
    layout = {"shift": "prompt", "num_items_in_batch": False}
    got = rl._unsloth_gkd_chunked_loss(
        _trainer(0.5, student, teacher), student, BatchEncoding(_inputs(vocab)), None, layout
    )
    want = rl._unsloth_gkd_chunked_loss(
        _trainer(0.5, student, teacher), student, _inputs(vocab), None, layout
    )
    assert got is not None
    torch.testing.assert_close(got, want)


def test_dense_fallback_projects_on_the_head_device():
    """A dispatched model can leave its final hidden states on another device than a tied lm_head."""
    head = torch.nn.Linear(4, 5).to("meta")
    logits = rl._unsloth_gkd_project(torch.randn(2, 3, 4), head, 0.5, 30.0)
    assert logits.device == head.weight.device and logits.shape == (2, 3, 5)


@pytest.mark.skipif(
    not has_real_cuda(),
    reason = "needs a second device to split hidden states from the head",
)
def test_chunked_loss_with_heads_on_another_device():
    """A dispatched model can return hidden states from one device while a tied lm_head sits on another."""
    vocab = 97
    student, teacher = _TinyLM(vocab, 16, 0.0, 1), _TinyLM(vocab, 24, 5.0, 2)
    trainer, inputs = _trainer(0.5, student, teacher), _inputs(vocab)
    layout = {"shift": "shift", "num_items_in_batch": False}
    expected = rl._unsloth_gkd_chunked_loss(trainer, student, inputs, None, layout)
    student.lm_head.cuda()
    teacher.lm_head.cuda()
    loss = rl._unsloth_gkd_chunked_loss(trainer, student, inputs, None, layout)
    loss.backward()
    torch.testing.assert_close(loss.cpu().double(), expected.double(), rtol = 1e-5, atol = 1e-7)
    assert student.mix.grad is not None and student.mix.grad.device.type == "cpu"


def test_offloaded_meta_heads_fall_back():
    """Accelerate CPU / disk offload keeps weights on meta between forwards; only the hooked forward can read them."""
    model = _TinyLM(97, 16, 0.0, 1)
    assert rl._unsloth_gkd_dense_head(model) is model.lm_head
    model.lm_head.to("meta")
    assert rl._unsloth_gkd_dense_head(model) is None


def test_ddp_find_unused_with_trainable_head_falls_back():
    """DDP(find_unused_parameters=True) marks a head skipped in forward as unused, then its grad hook fires twice."""
    student, teacher = _TinyLM(97, 16, 0.0, 1), _TinyLM(97, 24, 5.0, 2)
    ddp = torch.nn.parallel.DistributedDataParallel.__new__(
        torch.nn.parallel.DistributedDataParallel
    )
    torch.nn.Module.__init__(ddp)
    ddp.module, ddp.find_unused_parameters = student, True
    ddp.forward = student.forward
    trainer = _trainer(0.5, student, teacher)
    trainer.accelerator = types.SimpleNamespace(unwrap_model = lambda m: getattr(m, "module", m))
    layout = {"shift": "shift", "num_items_in_batch": False}
    assert rl._unsloth_gkd_chunked_loss(trainer, ddp, _inputs(97), None, layout) is None
    assert any(
        "find_unused_parameters" in reason for reason in trainer._unsloth_gkd_chunked_fallbacks
    )
    student.lm_head.weight.requires_grad_(False)
    assert rl._unsloth_gkd_chunked_loss(trainer, ddp, _inputs(97), None, layout) is not None


def test_minicpm3_scales_hidden_states_before_the_head_so_falls_back():
    """MiniCPM3 divides hidden states by logits_scaling before lm_head; hidden_states[-1] predates that."""
    student, teacher = _TinyLM(97, 16, 0.0, 1), _TinyLM(97, 24, 5.0, 2)
    teacher.config.model_type = "minicpm3"
    trainer = _trainer(0.5, student, teacher)
    layout = {"shift": "shift", "num_items_in_batch": False}
    assert rl._unsloth_gkd_chunked_loss(trainer, student, _inputs(97), None, layout) is None
    assert any("minicpm3" in reason for reason in trainer._unsloth_gkd_chunked_fallbacks)


@pytest.mark.skipif(
    not has_real_cuda(),
    reason = "needs a second device to split hidden states from the head",
)
def test_dense_fallback_colocates_with_the_labels():
    """One forward returns logits on the input device, the other hidden states projected on its head's device."""
    vocab = 97
    student, teacher = _TinyLM(vocab, 16, 0.0, 1), _TinyLM(vocab, 24, 5.0, 2)
    real_forward = teacher.forward

    def logits_only(input_ids, attention_mask):
        prior = os.environ.get("UNSLOTH_RETURN_HIDDEN_STATES")
        os.environ["UNSLOTH_RETURN_HIDDEN_STATES"] = "0"
        try:
            return real_forward(input_ids, attention_mask)
        finally:
            os.environ["UNSLOTH_RETURN_HIDDEN_STATES"] = prior

    teacher.forward = logits_only
    trainer, inputs = _trainer(0.5, student, teacher), _inputs(vocab)
    layout = {"shift": "shift", "num_items_in_batch": False}
    want = rl._unsloth_gkd_chunked_loss(trainer, student, inputs, None, layout)
    student.lm_head.cuda()
    got = rl._unsloth_gkd_chunked_loss(trainer, student, inputs, None, layout)
    assert got.device == inputs["labels"].device
    torch.testing.assert_close(got, want, rtol = 1e-6, atol = 1e-9)


def test_prediction_head_before_the_decoder_falls_back():
    """ModernBERT-decoder / RoBERTa-style: logits = decoder(lm_head(hidden)), so hidden states alone miss lm_head."""
    model = _TinyLM(97, 16, 0.0, 1)
    decoder = model.lm_head
    model.lm_head = torch.nn.Sequential(
        torch.nn.Linear(16, 16, dtype = torch.float64), torch.nn.GELU()
    )
    model.decoder = decoder
    model.get_output_embeddings = lambda: model.decoder
    assert rl._unsloth_gkd_dense_head(model) is None


class _MaskBlindLM(_TinyLM):
    """Causal and position aware, and like Unsloth's training forward it ignores ``attention_mask``."""

    def forward(self, input_ids, attention_mask):
        x = self.embed[input_ids]
        steps = torch.arange(1, x.shape[1] + 1, dtype = x.dtype).view(1, -1, 1)
        hidden = torch.tanh((x.cumsum(dim = 1) / steps + 0.1 * steps) @ self.mix)
        if os.environ.get("UNSLOTH_RETURN_HIDDEN_STATES", "0") == "1":
            return types.SimpleNamespace(logits = hidden)
        return types.SimpleNamespace(logits = self.lm_head(hidden))


@pytest.mark.parametrize("shift", ["prompt", "shift"])
def test_left_padded_rows_score_as_if_unpadded(shift, monkeypatch):
    """TRL's DataCollatorForChatML left-pads; a mask-blind forward must not see the pads."""
    monkeypatch.setenv("UNSLOTH_GKD_CHUNK_SIZE", "8")
    vocab, pad = 97, 0
    student, teacher = _MaskBlindLM(vocab, 16, 0.0, 1), _MaskBlindLM(vocab, 24, 0.0, 2)
    g = torch.Generator().manual_seed(0)
    rows = [(5, 7), (3, 4), (2, 9)]
    width, prompt_width = max(p + c for p, c in rows), max(p for p, _ in rows)
    input_ids = torch.full((3, width), pad)
    attention_mask = torch.zeros_like(input_ids)
    labels = torch.full_like(input_ids, -100)
    for i, (p, c) in enumerate(rows):
        ids = torch.randint(1, vocab, (p + c,), generator = g)
        input_ids[i, width - p - c :], attention_mask[i, width - p - c :] = ids, 1
        labels[i, width - c :] = ids[p:]
    inputs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "prompts": input_ids[:, :prompt_width],
    }
    scored = labels.clone()
    if shift == "prompt":
        scored[:, :prompt_width] = -100

    monkeypatch.setenv("UNSLOTH_RETURN_HIDDEN_STATES", "0")
    total, count = 0.0, 0
    for i in range(3):
        keep = attention_mask[i].bool()
        ids, lab = input_ids[i, keep][None], scored[i, keep][None]
        s = student(ids, None).logits[:, :-1]
        with torch.no_grad():
            t = teacher(ids, None).logits[:, :-1]
        total = total + GKDTrainer.generalized_jsd_loss(
            s, t, labels = lab[:, 1:], beta = 0.5, reduction = "sum"
        )
        count += int((lab[:, 1:] != -100).sum())
    want = total / count

    namespace = dict(vars(rl), GKDTrainer = GKDTrainer, empty_cache = lambda: None)
    source = PROMPT_LAYOUT if shift == "prompt" else SHIFT_LAYOUT
    exec(
        "class _Generated(GKDTrainer):\n" + rl.gkd_trainer_compute_loss("compute_loss", source),
        namespace,
    )
    trainer = namespace["_Generated"].__new__(namespace["_Generated"])
    trainer.__dict__.update(_trainer(0.5, student, teacher).__dict__)
    monkeypatch.delenv("UNSLOTH_RETURN_HIDDEN_STATES")
    torch.testing.assert_close(
        trainer.compute_loss(student, inputs).double(), want, rtol = 1e-5, atol = 1e-7
    )
    if shift == "prompt":
        monkeypatch.setenv("UNSLOTH_GKD_CHUNKED", "0")
        torch.testing.assert_close(
            trainer.compute_loss(student, inputs), want, rtol = 1e-5, atol = 1e-7
        )
    # TRL's Liger branch has no prompt slice: its rows are rolled, every label kept.
    liger = rl._unsloth_gkd_right_align(
        inputs, {"shift": shift, "num_items_in_batch": False}, liger = True
    )
    assert liger["prompts"] is inputs["prompts"]
    for i in range(3):
        keep = attention_mask[i].bool()
        width = int(keep.sum())
        assert torch.equal(liger["labels"][i, :width], labels[i, keep])
        assert torch.equal(liger["input_ids"][i, :width], input_ids[i, keep])
