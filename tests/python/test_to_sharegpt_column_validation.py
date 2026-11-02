import ast
from pathlib import Path

import pytest


def _load_to_sharegpt():
    # Extract to_sharegpt without importing unsloth (it needs unsloth_zoo and a GPU).
    source = Path(__file__).parents[2] / "unsloth" / "chat_templates.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    funcs = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "to_sharegpt"]
    assert funcs, "to_sharegpt not found in unsloth/chat_templates.py"
    module = ast.Module(body=funcs, type_ignores=[])
    namespace = {"__builtins__": __builtins__}
    exec(compile(module, str(source), "exec"), namespace)
    return namespace["to_sharegpt"]


class FakeDataset:
    """Minimal stand-in for a datasets.Dataset, enough for the guard clauses."""

    def __init__(self, columns, rows=None):
        self.column_names = list(columns)
        self._rows = rows or []

    def __len__(self):
        return len(self._rows)

    def __getitem__(self, index):
        return self._rows[index]

    def map(self, *args, **kwargs):
        raise AssertionError("map() should not be reached in these tests")


def test_missing_output_column_raises_domain_error():
    to_sharegpt = _load_to_sharegpt()
    dataset = FakeDataset(["instruction"], rows=[{"instruction": "hi"}])

    with pytest.raises(KeyError) as excinfo:
        to_sharegpt(dataset)

    message = str(excinfo.value)
    assert "output" in message
    assert "output_column_name" in message


def test_missing_output_column_honours_custom_name():
    to_sharegpt = _load_to_sharegpt()
    dataset = FakeDataset(["instruction", "answer"], rows=[{"instruction": "hi", "answer": "there"}])

    with pytest.raises(KeyError) as excinfo:
        to_sharegpt(dataset, output_column_name="answer", merged_column_name="missing")

    assert "missing" in str(excinfo.value)


def test_empty_dataset_with_conversations_column_does_not_raise_index_error():
    to_sharegpt = _load_to_sharegpt()
    dataset = FakeDataset(["conversations", "output"])

    # The empty-dataset probe must not blow up with IndexError; the missing input column is
    # the honest error here.
    with pytest.raises(KeyError) as excinfo:
        to_sharegpt(dataset)

    assert "instruction" in str(excinfo.value)


def test_sharegpt_dataset_still_rejected():
    to_sharegpt = _load_to_sharegpt()
    dataset = FakeDataset(["conversations", "output"], rows=[{"conversations": [{"from": "human"}], "output": "x"}])

    with pytest.raises(TypeError) as excinfo:
        to_sharegpt(dataset)

    assert "ShareGPT" in str(excinfo.value)
