import ast
from pathlib import Path
import re
import subprocess

import pytest


def _load_chat_template_functions(*function_names):
    source_path = Path(__file__).resolve().parents[1] / "unsloth" / "chat_templates.py"
    tree = ast.parse(source_path.read_text(encoding = "utf-8"), filename = str(source_path))
    selected = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in function_names
    ]
    loaded_names = {node.name for node in selected}
    missing = [name for name in function_names if name not in loaded_names]
    if missing:
        raise RuntimeError(f"Missing expected functions in chat_templates.py: {missing}")

    module_ast = ast.Module(body = selected, type_ignores = [])
    namespace = {"re": re}
    exec(compile(module_ast, str(source_path), "exec"), namespace)
    return [namespace[name] for name in function_names]


def test_create_formatter_preserves_optional_prompt_behavior():
    parse_combined_prompt, create_formatter = _load_chat_template_functions(
        "_parse_combined_prompt",
        "_create_formatter",
    )

    class DummyDataset:
        column_names = ["instruction", "input", "output"]

    merged_prompt = "{instruction}[[ - {input}: {output}]]"
    possible_columns, optional_prompts = parse_combined_prompt(merged_prompt, DummyDataset())
    formatter = create_formatter(possible_columns, optional_prompts, "merged")

    examples = {
        "instruction": ["first", "second", "third"],
        "input": ["x", "", None],
        "output": ["o1", "o2", "o3"],
    }
    assert formatter(examples) == {
        "merged": [
            "first - x: o1",
            "second",
            "third",
        ]
    }


def test_create_formatter_keeps_optional_block_for_zero_value():
    parse_combined_prompt, create_formatter = _load_chat_template_functions(
        "_parse_combined_prompt",
        "_create_formatter",
    )

    class DummyDataset:
        column_names = ["instruction", "step", "comment"]

    merged_prompt = "{instruction}[[ Step {step}: {comment}]]"
    possible_columns, optional_prompts = parse_combined_prompt(merged_prompt, DummyDataset())
    formatter = create_formatter(possible_columns, optional_prompts, "merged")

    examples = {
        "instruction": ["run", "skip"],
        "step": [0, ""],
        "comment": ["warmup", "n/a"],
    }
    assert formatter(examples) == {
        "merged": [
            "run Step 0: warmup",
            "skip",
        ]
    }


def test_create_formatter_rejects_optional_without_column():
    parse_combined_prompt, create_formatter = _load_chat_template_functions(
        "_parse_combined_prompt",
        "_create_formatter",
    )

    class DummyDataset:
        column_names = ["instruction"]

    possible_columns, optional_prompts = parse_combined_prompt("[[constant]]", DummyDataset())
    with pytest.raises(IndexError, match = "Optional"):
        create_formatter(possible_columns, optional_prompts, "merged")


def test_hf_gguf_equivalence_uses_shell_false_and_preserves_quotes(monkeypatch):
    remove_special_tokens, test_hf_gguf_equivalence = _load_chat_template_functions(
        "remove_special_tokens",
        "test_hf_gguf_equivalence",
    )
    test_hf_gguf_equivalence.__globals__["remove_special_tokens"] = remove_special_tokens

    captured_commands = []

    class FakePopen:
        def __init__(self, command, shell, stdout, stderr, bufsize):
            captured_commands.append((list(command), shell))
            self.stdout = [
                b"1 -> 'tok1'\n",
                b"2 -> 'tok2'\n",
            ]

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(subprocess, "Popen", FakePopen)

    class DummyTokenizer:
        bos_token = "<s>"
        chat_template = "dummy"

        def apply_chat_template(self, messages, tokenize = False, add_generation_prompt = True):
            del messages, tokenize, add_generation_prompt
            return "<s>I can't be shell-escaped"

        def __call__(self, prompt):
            del prompt

            class Tokenized:
                input_ids = [1, 2]

            return Tokenized()

        def batch_decode(self, input_ids):
            del input_ids
            return ["tok1", "tok2"]

    assert test_hf_gguf_equivalence(DummyTokenizer(), gguf_model = "./fake.gguf") is True
    assert len(captured_commands) == 2
    assert all(shell is False for _, shell in captured_commands)
    assert all(isinstance(command, list) for command, _ in captured_commands)

    second_command = captured_commands[1][0]
    prompt_index = second_command.index("-p")
    assert second_command[prompt_index + 1] == "I can't be shell-escaped"


def test_to_sharegpt_conversation_extension_two():
    datasets = pytest.importorskip("datasets")
    _, _, to_sharegpt = _load_chat_template_functions(
        "_parse_combined_prompt",
        "_create_formatter",
        "to_sharegpt",
    )
    dataset = datasets.Dataset.from_dict(
        {
            "instruction": ["i1", "i2"],
            "output": ["o1", "o2"],
        }
    )

    converted = to_sharegpt(
        dataset,
        merged_prompt = "{instruction}",
        output_column_name = "output",
        conversation_extension = 2,
        random_state = 3407,
    )
    assert converted.column_names == ["conversations"]
    assert len(converted) == 2
    for conversation in converted["conversations"]:
        assert len(conversation) == 4
        assert [turn["from"] for turn in conversation] == ["human", "gpt", "human", "gpt"]
