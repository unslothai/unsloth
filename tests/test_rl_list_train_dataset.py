# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""TRL 1.10+ rejects a plain list train_dataset, which every vision notebook passes
(list of PIL conversations + skip_prepare_dataset=True). The compiled SFT / GRPO / RLOO
trainers must accept it again, while DPO / KTO / Reward keep TRL's check."""

import ast
import inspect
import re
import textwrap
from pathlib import Path

import pytest
import torch

SOURCE_PATH = Path(__file__).resolve().parents[1] / "unsloth" / "models" / "rl.py"
NAMES = (
    "_LIST_TRAIN_DATASET_TRAINERS",
    "_TRL_TRAIN_DATASET_TYPE_CHECK",
    "_allow_list_train_dataset",
)


def _load():
    text = SOURCE_PATH.read_text(encoding = "utf-8")
    tree = ast.parse(text)
    nodes = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in NAMES:
            nodes.append(node)
        elif isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id in NAMES for t in node.targets
        ):
            nodes.append(node)
    namespace = {"re": re}
    exec(compile(ast.Module(body = nodes, type_ignores = []), str(SOURCE_PATH), "exec"), namespace)
    return namespace["_allow_list_train_dataset"]


allow_list = _load()


class Dataset:
    pass


class IterableDataset:
    pass


class TorchDataset(torch.utils.data.Dataset):
    def __len__(self):
        return 1

    def __getitem__(self, i):
        return {"messages": []}


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


@pytest.mark.parametrize(
    "trainer_file, init",
    [("sft_trainer", SFT_INIT), ("grpo_trainer", GRPO_INIT), ("rloo_trainer", GRPO_INIT)],
    ids = ["sft", "grpo", "rloo"],
)
def test_list_like_datasets_are_accepted(trainer_file, init):
    patched = allow_list("__init__", init, trainer_file)
    assert patched != init
    fn = _build(patched)
    for good in (
        [{"messages": []}],
        ({"messages": []},),
        TorchDataset(),
        Dataset(),
        IterableDataset(),
    ):
        assert fn(None, good) == "ok"
    # Anything else (a DatasetDict is a dict) still gets TRL's error.
    for bad in ({"train": []}, "text", 3):
        with pytest.raises(TypeError, match = "must be a `Dataset`"):
            fn(None, bad)


@pytest.mark.parametrize("trainer_file", ["dpo_trainer", "kto_trainer", "reward_trainer"])
def test_trainers_that_always_map_keep_the_check(trainer_file):
    assert allow_list("__init__", SFT_INIT, trainer_file) == SFT_INIT


def test_other_functions_untouched():
    assert allow_list("compute_loss", SFT_INIT, "sft_trainer") == SFT_INIT


def test_skip_prepare_label_guard_reads_list_columns():
    source = "def f(dataset):\n    cols = get_dataset_column_names(dataset)\n    return cols\n"
    patched = allow_list("_reject_skip_prepare_without_labels", source, "sft_trainer")
    namespace = {"get_dataset_column_names": lambda d: d.column_names}
    exec(patched.replace("def f", "def _reject_skip_prepare_without_labels"), namespace)
    fn = namespace["_reject_skip_prepare_without_labels"]
    assert fn([{"input_ids": [1], "completion_mask": [1]}]) == ["input_ids", "completion_mask"]
    assert fn([]) == []


def test_matches_installed_trl():
    trl = pytest.importorskip("trl")
    import importlib.util

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
            continue  # TRL < 1.10 has no check: nothing to widen.
        assert "list, tuple, torch.utils.data.Dataset)" in allow_list(
            "__init__", text, trainer_file
        ), name
        checked += 1
        # When unsloth already patched trl (tests/conftest.py imports it), the compiled parent must carry it too.
        compiled = [c for c in getattr(trl, name).__mro__ if c.__name__ == f"_Unsloth{name}"]
        if compiled:
            assert "list, tuple, torch.utils.data.Dataset)" in inspect.getsource(
                compiled[0].__init__
            ), name
    if checked == 0:
        pytest.skip(
            reason = f"trl {trl.__version__} predates the 1.10 train_dataset type check, nothing to widen"
        )
