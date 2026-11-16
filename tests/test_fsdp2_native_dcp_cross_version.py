# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team.

"""A sharded optimizer checkpoint (UNSLOTH_FSDP2_DCP_OPTIM=1) written by torch 2.15 stores tensor
strides in its .metadata (TensorProperties pickles 6 fields), which torch <= 2.14 cannot unpickle.
Resuming it there names the cause instead of "too many values to unpack (expected 5)"."""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_fsdp2_native_units import _load_native  # noqa: E402

dcp = pytest.importorskip("torch.distributed.checkpoint")
from torch.distributed.checkpoint.metadata import TensorProperties  # noqa: E402


def _reader_has_strides():
    return "strides" in getattr(TensorProperties, "__dataclass_fields__", {})


def _save_as_torch_2_15(directory, monkeypatch):
    """dcp.save in the torch 2.15 metadata format: native there, else the 6-field pickle 2.15 writes."""
    if not _reader_has_strides():
        old = TensorProperties.__getstate__
        monkeypatch.setattr(TensorProperties, "__getstate__", lambda self: (*old(self), (4, 1)))
    state = {"state": {"w.exp_avg": torch.arange(12.0).reshape(3, 4)}, "step": torch.tensor(3.0)}
    dcp.save(state, checkpoint_id = os.fspath(directory), no_dist = True)
    monkeypatch.undo()


def test_newer_dcp_metadata_names_the_torch_version(tmp_path, monkeypatch):
    native = _load_native()
    _save_as_torch_2_15(tmp_path, monkeypatch)
    if _reader_has_strides():
        # 2.15+ reads its own format (and the older 5-field one).
        assert native._drop_unsaved_group_keys({"param_groups": []}, tmp_path) == []
        assert "state.w.exp_avg" in native._read_dcp_metadata(tmp_path)
        return
    with pytest.raises(RuntimeError, match = "newer torch"):
        native._drop_unsaved_group_keys({"param_groups": []}, tmp_path)


def test_same_version_metadata_still_reads(tmp_path):
    native = _load_native()
    state = {"state": {"w.exp_avg": torch.ones(2, 2)}}
    dcp.save(state, checkpoint_id = os.fspath(tmp_path), no_dist = True)
    assert "state.w.exp_avg" in native._read_dcp_metadata(tmp_path)


def test_other_value_errors_are_not_rewritten(tmp_path, monkeypatch):
    native = _load_native()

    def boom(self):
        raise ValueError("some other problem")

    monkeypatch.setattr(dcp.FileSystemReader, "read_metadata", boom)
    with pytest.raises(ValueError, match = "some other problem"):
        native._read_dcp_metadata(tmp_path)
