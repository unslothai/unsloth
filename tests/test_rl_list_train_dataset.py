# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""TRL 1.10+ rejects a plain list train_dataset, which every vision notebook passes
(list of PIL conversations + skip_prepare_dataset=True). The compiled SFT / GRPO / RLOO
trainers must accept it again, while DPO / KTO / Reward keep TRL's check."""

import ast
import importlib.util
import inspect
import re
import sys
import textwrap
from pathlib import Path

import pytest
import torch

MODELS = Path(__file__).resolve().parents[1] / "unsloth" / "models"


def _lift(path, names, namespace):
    tree = ast.parse(path.read_text(encoding = "utf-8"))
    nodes = [
        node
        for node in tree.body
        if (isinstance(node, ast.FunctionDef) and node.name in names)
        or (
            isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id in names for t in node.targets)
        )
    ]
    exec(compile(ast.Module(body = nodes, type_ignores = []), str(path), "exec"), namespace)
    return namespace


_rl = _lift(
    MODELS / "rl.py",
    ("_LIST_TRAIN_DATASET_TRAINERS", "_TRL_TRAIN_DATASET_TYPE_CHECK", "_allow_list_train_dataset"),
    {"re": re},
)
allow_list = _rl["_allow_list_train_dataset"]
column_names = _lift(MODELS / "_utils.py", ("_unsloth_dataset_column_names",), {})[
    "_unsloth_dataset_column_names"
]


class Dataset:
    pass


class IterableDataset:
    pass


class TorchDataset(torch.utils.data.Dataset):
    def __len__(self):
        return 1

    def __getitem__(self, i):
        if i >= 1:
            raise IndexError(i)
        return {"messages": []}


class TorchStream(torch.utils.data.IterableDataset):
    def __iter__(self):
        return iter([{"prompt": "x"}])


SFT_INIT = """
def __init__(self, train_dataset):
    if train_dataset is None:
        raise ValueError("`train_dataset` is required")
    elif isinstance(train_dataset, IterableDataset):
        pass
    elif not isinstance(train_dataset, Dataset):
        raise TypeError("`train_dataset` must be a `Dataset` or `IterableDataset`")
    return "ok"
"""

GRPO_INIT = """
def __init__(self, train_dataset):
    if train_dataset is None:
        raise ValueError("`train_dataset` is required")
    elif not isinstance(train_dataset, (Dataset, IterableDataset)):
        raise TypeError("`train_dataset` must be a `Dataset` or `IterableDataset`")
    return "ok"
"""


def _build(source):
    namespace = {"Dataset": Dataset, "IterableDataset": IterableDataset, "torch": torch}
    exec(textwrap.dedent(source), namespace)
    return namespace["__init__"]


LIST_ROWS = ([{"messages": []}], ({"messages": []},))
BAD = ({"train": []}, "text", 3)  # a DatasetDict is a dict


@pytest.mark.parametrize(
    "trainer_file, init",
    [("sft_trainer", SFT_INIT), ("grpo_trainer", GRPO_INIT), ("rloo_trainer", GRPO_INIT)],
    ids = ["sft", "grpo", "rloo"],
)
def test_list_like_datasets_are_accepted(trainer_file, init):
    with pytest.raises(TypeError):
        _build(init)(None, LIST_ROWS[0])
    patched = allow_list("__init__", init, trainer_file)
    assert allow_list("__init__", patched, trainer_file) == patched
    fn = _build(patched)
    for good in (*LIST_ROWS, Dataset(), IterableDataset()):
        assert fn(None, good) == "ok"
    for bad in (*BAD, TorchDataset(), TorchStream()):
        with pytest.raises(TypeError, match = "must be a `Dataset`"):
            fn(None, bad)


@pytest.mark.parametrize("trainer_file", ["dpo_trainer", "kto_trainer", "reward_trainer"])
def test_trainers_that_always_map_keep_the_check(trainer_file):
    assert allow_list("__init__", SFT_INIT, trainer_file) == SFT_INIT


def test_other_functions_untouched():
    assert allow_list("compute_loss", SFT_INIT, "sft_trainer") == SFT_INIT


def test_skip_prepare_label_guard_uses_the_helper():
    source = "def f(dataset):\n    cols = get_dataset_column_names(dataset)\n    return cols\n"
    patched = allow_list("_reject_skip_prepare_without_labels", source, "sft_trainer")
    assert "cols = _unsloth_dataset_column_names(dataset)" in patched


class _WithColumns:
    column_names = ["input_ids", "labels"]


@pytest.mark.parametrize(
    "dataset, expected",
    [
        (_WithColumns(), ["input_ids", "labels"]),
        ([{"input_ids": [1], "completion_mask": [1]}], ["input_ids", "completion_mask"]),
        (({"messages": []},), ["messages"]),
        (TorchDataset(), ["messages"]),
        (TorchStream(), ["prompt"]),
        ([], []),
        ((), []),
        ([("a", "b")], []),
    ],
    ids = [
        "column_names",
        "list",
        "tuple",
        "torch",
        "torch_stream",
        "empty_list",
        "empty_tuple",
        "non_mapping",
    ],
)
def test_dataset_column_names(dataset, expected):
    assert column_names(dataset) == expected


def test_matches_installed_trl():
    trl = pytest.importorskip("trl")
    checked = 0
    for trainer_file, name in (
        ("sft_trainer", "SFTTrainer"),
        ("grpo_trainer", "GRPOTrainer"),
        ("rloo_trainer", "RLOOTrainer"),
    ):
        spec = importlib.util.find_spec(f"trl.trainer.{trainer_file}")
        if spec is None or spec.origin is None:
            continue
        text = Path(spec.origin).read_text(encoding = "utf-8")
        if "must be a `Dataset` or `IterableDataset`" not in text:
            continue
        widened = "list, tuple)"
        assert widened in allow_list("__init__", text, trainer_file), name
        checked += 1
        # tests/conftest.py imports unsloth, which swaps in the compiled trainers.
        if "unsloth" in sys.modules:
            compiled = [c for c in getattr(trl, name).__mro__ if c.__name__ == f"_Unsloth{name}"]
            assert compiled, name
            assert widened in inspect.getsource(compiled[0].__init__), name
    if checked == 0:
        pytest.skip(
            reason = f"trl {trl.__version__} predates the 1.10 train_dataset type check, nothing to widen"
        )
