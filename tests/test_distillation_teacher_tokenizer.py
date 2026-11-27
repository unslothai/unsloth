# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

"""External distillation teacher tokenizer identity (unsloth/models/_on_policy_distillation.py check_teacher_tokenizer).

vocab_size alone passes Qwen2.5 vs Qwen3 (both 151936 embedding rows); Qwen3 adds <tool_response> and <think> tokens at
ids 151665-151668, so the teacher would score other tokens than the student sampled. CPU, tokenizer files only."""

import importlib.util
import os
import types
import warnings

import pytest
import torch

pytest.importorskip("transformers")

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
OPD_PATH = os.path.join(REPO_ROOT, "unsloth", "models", "_on_policy_distillation.py")
RL_REPLACEMENTS_PATH = os.path.join(REPO_ROOT, "unsloth", "models", "rl_replacements.py")
TOKENIZER_FILES = ["tokenizer*", "*.json", "vocab*", "merges*"]


def _load_opd():
    spec = importlib.util.spec_from_file_location("_unsloth_opd_tokenizer_under_test", OPD_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


opd = _load_opd()


def _tokenizer_dir(repo):
    hub = pytest.importorskip("huggingface_hub")
    try:
        return hub.snapshot_download(repo, allow_patterns = TOKENIZER_FILES)
    except Exception as error:
        pytest.skip(f"tokenizer files of {repo} unavailable: {type(error).__name__}")


def _fast_tokenizer(
    tokens,
    added = (),
    eos = None,
):
    from tokenizers import Tokenizer, models
    from transformers import PreTrainedTokenizerFast

    vocab = {t: i for i, t in enumerate(tokens)}
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object = Tokenizer(models.WordLevel(vocab, unk_token = tokens[0])),
        unk_token = tokens[0],
    )
    if added:
        tokenizer.add_tokens(list(added), special_tokens = True)
    if eos is not None:
        tokenizer.eos_token = eos
    return tokenizer


class _Teacher(torch.nn.Module):
    def __init__(
        self,
        source,
        vocab = 151936,
    ):
        super().__init__()
        self.lm_head = torch.nn.Linear(2, 2)
        self.config = types.SimpleNamespace(vocab_size = vocab, _name_or_path = source)


def _grpo_trainer(student_tokenizer):
    inner = types.SimpleNamespace(config = types.SimpleNamespace(vocab_size = 151936))
    return types.SimpleNamespace(
        args = types.SimpleNamespace(unsloth_distillation_coef = 1.0),
        model = inner,
        processing_class = student_tokenizer,
        accelerator = types.SimpleNamespace(num_processes = 1, unwrap_model = lambda m: m),
    )


def test_qwen25_student_qwen3_teacher_is_refused(monkeypatch):
    from transformers import AutoTokenizer

    monkeypatch.delenv("UNSLOTH_DISTILLATION_SKIP_TOKENIZER_CHECK", raising = False)
    monkeypatch.setattr(opd, "_require_zoo", lambda flag, option: None)
    qwen25, qwen3 = _tokenizer_dir("Qwen/Qwen2.5-0.5B"), _tokenizer_dir("Qwen/Qwen3-0.6B")
    student = AutoTokenizer.from_pretrained(qwen25)
    # Same embedding rows: the vocab_size check passes, the tokenizer check must not.
    with pytest.raises(ValueError, match = "<think>"):
        opd.grpo_setup_distillation(_grpo_trainer(student), _Teacher(qwen3))
    with pytest.raises(ValueError, match = "UNSLOTH_DISTILLATION_SKIP_TOKENIZER_CHECK"):
        opd.check_teacher_tokenizer(qwen3, student, "GKDTrainer")
    # The identical tokenizer (another folder holding the same files) passes.
    trainer = _grpo_trainer(student)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        opd.grpo_setup_distillation(trainer, _Teacher(qwen25 + os.sep))
    assert trainer._unsloth_distillation_teacher is not None
    # The escape hatch skips it.
    monkeypatch.setenv("UNSLOTH_DISTILLATION_SKIP_TOKENIZER_CHECK", "1")
    opd.grpo_setup_distillation(_grpo_trainer(student), _Teacher(qwen3))


def test_token_maps_added_tokens_and_specials():
    base = ["<unk>", "a", "b", "c"]
    same = _fast_tokenizer(base, added = ["<eos>"])
    assert opd.tokenizer_mismatches(same, _fast_tokenizer(base, added = ["<eos>"])) == ([], [])
    errors, _ = opd.tokenizer_mismatches(same, _fast_tokenizer(base, added = ["<think>"]))
    assert errors and "<think>" in errors[0] and "<eos>" in errors[0]
    errors, _ = opd.tokenizer_mismatches(
        _fast_tokenizer(["<unk>", "a", "c", "b"]), _fast_tokenizer(base)
    )
    assert errors and "2 token ids" in errors[0]
    # Same token map, other eos (a Base student of an Instruct teacher): warned, not refused.
    errors, notes = opd.tokenizer_mismatches(
        _fast_tokenizer(base, added = ["<e1>", "<e2>"], eos = "<e1>"),
        _fast_tokenizer(base, added = ["<e1>", "<e2>"], eos = "<e2>"),
    )
    assert errors == [] and notes and "eos_token_id" in notes[0]
    # Registering an existing token as special on one side only keeps the token -> id map: warned, not refused.
    registered = _fast_tokenizer(base)
    registered.add_special_tokens({"additional_special_tokens": ["c"]})
    errors, notes = opd.tokenizer_mismatches(registered, _fast_tokenizer(base))
    assert errors == [] and any("registrations" in n for n in notes)
    # A pad token only the student has (Unsloth may add one) is not a mismatch.
    padded = _fast_tokenizer(base, added = ["<pad>"])
    padded.pad_token = "<pad>"
    assert opd.tokenizer_mismatches(padded, _fast_tokenizer(base))[0] == []


def test_skips_without_a_loadable_source(tmp_path):
    student = _fast_tokenizer(["<unk>", "a"])
    opd.check_teacher_tokenizer(opd.AdapterOffTeacher.__new__(opd.AdapterOffTeacher), student)
    opd.check_teacher_tokenizer(_Teacher(None), student)
    with pytest.warns(UserWarning, match = "could not load the teacher's tokenizer"):
        opd.check_teacher_tokenizer(_Teacher(str(tmp_path / "missing")), student)


def test_gkd_and_distillation_trainers_run_the_check(monkeypatch):
    import ast
    import sys

    source = open(RL_REPLACEMENTS_PATH, encoding = "utf-8").read()
    for trainer in ("gkd_trainer", "distillation_trainer"):
        assert (
            f'RL_EXTRA_ARGS["{trainer}"].append(distillation_trainer_teacher_tokenizer)' in source
        )
    node = next(
        n
        for n in ast.parse(source).body
        if isinstance(n, ast.FunctionDef) and n.name == "distillation_trainer_teacher_tokenizer"
    )
    namespace = {}
    exec(compile(ast.Module([node], []), RL_REPLACEMENTS_PATH, "exec"), namespace)
    hook = namespace["distillation_trainer_teacher_tokenizer"]
    assert hook("model, args", "") == ""
    code = hook("model, teacher_model, args, processing_class", "")
    # The generated __init__ lines run the check on the trainer's own teacher_model / processing_class.
    monkeypatch.setitem(sys.modules, "unsloth.models._on_policy_distillation", opd)
    seen = []
    monkeypatch.setattr(opd, "check_teacher_tokenizer", lambda *a: seen.append(a))
    self = type("GKDTrainer", (), {})()
    exec(code, {}, {"teacher_model": "teacher", "processing_class": "tok", "self": self})
    assert seen == [("teacher", "tok", "GKDTrainer")]
    # The `tokenizer =` alias is what the superclass receives, so it is the one checked.
    seen.clear()
    exec(
        code,
        {},
        {"teacher_model": "teacher", "processing_class": None, "tokenizer": "alias", "self": self},
    )
    exec(
        code,
        {},
        {"teacher_model": "teacher", "processing_class": "tok", "tokenizer": None, "self": self},
    )
    assert seen == [("teacher", "alias", "GKDTrainer"), ("teacher", "tok", "GKDTrainer")]
