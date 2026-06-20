"""ORPO should use a processor's tokenizer for text-only row tokenization."""

import ast
import os
import re


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RL_PATH = os.path.join(REPO_ROOT, "unsloth", "models", "rl_replacements.py")


def _load_orpo_rewriter(name = "orpo_trainer_text_tokenizer"):
    src = open(RL_PATH).read()
    tree = ast.parse(src)
    ns = {"re": re}
    # Materialise sibling module-level _-prefixed assignments the rewriter may reference.
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id.startswith("_"):
                    exec(ast.get_source_segment(src, node), ns)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            exec(ast.get_source_segment(src, node), ns)
            return ns[name]
    raise AssertionError(f"{name} not found")


class _Tokenizer:
    bos_token_id = 1
    eos_token_id = 2

    def __init__(self):
        self.calls = []

    def __call__(
        self,
        text,
        add_special_tokens = False,
        **kwargs,
    ):
        self.calls.append((text, add_special_tokens, kwargs))
        ids = [ord(c) % 31 + 3 for c in text]
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}


class _Processor:
    def __init__(self):
        self.tokenizer = _Tokenizer()

    def __call__(self, *args, **kwargs):
        raise AssertionError("text-only ORPO tokenization should not call processor")


class _Trainer:
    def __init__(self):
        self.processing_class = _Processor()
        self.is_encoder_decoder = False
        self.max_length = 2048
        self.max_prompt_length = 1024
        self.max_completion_length = 1024
        self.truncation_mode = "keep_end"
        self.label_pad_token_id = -100
        self.padding_value = 0


def _exec_rewritten(
    function_name,
    source,
    extra_ns = None,
):
    rewriter = _load_orpo_rewriter()
    rewritten = rewriter(function_name, source)
    ns = {} if extra_ns is None else dict(extra_ns)
    exec(rewritten, ns)
    return ns[function_name]


def test_orpo_tokenize_row_returns_original_when_tokenizer_anchor_missing():
    rewriter = _load_orpo_rewriter()
    source = """
def tokenize_row(self, feature, model=None):
    output = {}
    output["prompt_input_ids"] = self.processing_class(feature["prompt"], add_special_tokens=False)["input_ids"]
    return output
"""

    rewritten = rewriter("tokenize_row", source)

    assert rewritten == source
    assert "tokenizer(" not in rewritten


def test_orpo_build_tokenized_answer_uses_processor_tokenizer():
    source = """
def build_tokenized_answer(self, prompt, answer):
    full_tokenized = self.processing_class(prompt + answer, add_special_tokens=False)
    prompt_input_ids = self.processing_class(prompt, add_special_tokens=False)["input_ids"]
    return full_tokenized["input_ids"][len(prompt_input_ids):]
"""
    fn = _exec_rewritten("build_tokenized_answer", source)
    trainer = _Trainer()

    assert fn(trainer, "a", "b")
    assert [call[0] for call in trainer.processing_class.tokenizer.calls] == ["ab", "a"]


def test_orpo_tokenize_row_uses_processor_tokenizer():
    source = """
def tokenize_row(self, feature, model=None):
    batch = {}
    prompt = feature["prompt"]
    chosen = feature["chosen"]
    rejected = feature["rejected"]
    if not self.is_encoder_decoder:
        prompt_tokens = self.processing_class(prompt, add_special_tokens=False)
        prompt_tokens = {f"prompt_{k}": v for k, v in prompt_tokens.items()}
        chosen_tokens = self.build_tokenized_answer(prompt, chosen)
        rejected_tokens = self.build_tokenized_answer(prompt, rejected)
        prompt_len_input_ids = len(prompt_tokens["prompt_input_ids"])
        chosen_prompt_len_input_ids = len(chosen_tokens["prompt_input_ids"])
        rejected_prompt_len_input_ids = len(rejected_tokens["prompt_input_ids"])
        prompt_tokens, chosen_tokens, rejected_tokens = add_bos_token_if_needed(
            self.processing_class.bos_token_id,
            prompt_len_input_ids,
            prompt_tokens,
            chosen_prompt_len_input_ids,
            chosen_tokens,
            rejected_prompt_len_input_ids,
            rejected_tokens,
        )
        chosen_tokens, rejected_tokens = add_eos_token_if_needed(
            self.processing_class.eos_token_id, chosen_tokens, rejected_tokens
        )
        batch["prompt_input_ids"] = prompt_tokens["prompt_input_ids"]
        batch["chosen_input_ids"] = chosen_tokens["input_ids"]
        batch["rejected_input_ids"] = rejected_tokens["input_ids"]
    return batch
"""

    def add_bos_token_if_needed(*args):
        return args[2], args[4], args[6]

    def add_eos_token_if_needed(eos_token_id, chosen_tokens, rejected_tokens):
        chosen_tokens["input_ids"] = chosen_tokens["input_ids"] + [eos_token_id]
        rejected_tokens["input_ids"] = rejected_tokens["input_ids"] + [eos_token_id]
        return chosen_tokens, rejected_tokens

    trainer = _Trainer()
    trainer.build_tokenized_answer = lambda prompt, answer: {
        "prompt_input_ids": trainer.processing_class.tokenizer(prompt)["input_ids"],
        "input_ids": trainer.processing_class.tokenizer(answer)["input_ids"],
    }
    fn = _exec_rewritten(
        "tokenize_row",
        source,
        {
            "add_bos_token_if_needed": add_bos_token_if_needed,
            "add_eos_token_if_needed": add_eos_token_if_needed,
        },
    )

    output = fn(trainer, {"prompt": "p", "chosen": "c", "rejected": "r"})

    assert output["chosen_input_ids"][-1] == _Tokenizer.eos_token_id
    assert [call[0] for call in trainer.processing_class.tokenizer.calls] == [
        "p",
        "p",
        "c",
        "p",
        "r",
    ]


def test_orpo_init_pad_token_id_falls_back_to_tokenizer():
    rewriter = _load_orpo_rewriter("orpo_trainer_processor_pad_token")
    source = """
def __init__(self, processing_class):
    data_collator = DPODataCollatorWithPadding(
        pad_token_id=processing_class.pad_token_id,
    )
    self.padding_value = processing_class.pad_token_id
"""

    rewritten = rewriter("__init__", source)

    assert "processing_class.pad_token_id" not in rewritten
    assert "getattr(processing_class, 'pad_token_id'" in rewritten

    class _Processor:
        # No pad_token_id at the processor level; only on the inner tokenizer.
        class tokenizer:
            pad_token_id = 17

    captured = {}

    def DPODataCollatorWithPadding(**kwargs):
        captured["pad_token_id"] = kwargs["pad_token_id"]
        return object()

    ns = {"DPODataCollatorWithPadding": DPODataCollatorWithPadding}
    exec(rewritten, ns)

    class _Trainer:
        pass

    trainer = _Trainer()
    ns["__init__"](trainer, _Processor())

    assert captured["pad_token_id"] == 17
    assert trainer.padding_value == 17


def test_orpo_init_pad_token_id_uses_processor_when_present():
    rewriter = _load_orpo_rewriter("orpo_trainer_processor_pad_token")
    source = """
def __init__(self, processing_class):
    self.padding_value = processing_class.pad_token_id
"""

    rewritten = rewriter("__init__", source)

    class _Tokenizer:
        # Inner tokenizer must NOT be consulted when the processor exposes
        # pad_token_id itself.
        pad_token_id = 999

    class _Processor:
        pad_token_id = 42
        tokenizer = _Tokenizer()

    ns = {}
    exec(rewritten, ns)

    class _Trainer:
        pass

    trainer = _Trainer()
    ns["__init__"](trainer, _Processor())
    assert trainer.padding_value == 42


def test_orpo_init_pad_token_id_noop_on_non_init():
    rewriter = _load_orpo_rewriter("orpo_trainer_processor_pad_token")
    source = "def tokenize_row(self):\n    return processing_class.pad_token_id\n"
    assert rewriter("tokenize_row", source) == source
