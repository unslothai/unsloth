# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""``_preprocess_snac_dataset`` copies each SNAC level to the host once instead of reading seven
scalars per frame. Pin that rows, warnings, progress and stop boundaries match the per-scalar loop,
which still serves any codes the fast path does not accept."""

import ast
import sys
import textwrap
import types
from pathlib import Path

import numpy as np
import pytest
import torch
from datasets import Dataset

_TRAINER = Path(__file__).resolve().parents[1] / "core" / "training" / "trainer.py"


def _load_method():
    source = _TRAINER.read_text(encoding = "utf-8")
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.FunctionDef) and node.name == "_preprocess_snac_dataset":
            return textwrap.dedent(ast.get_source_segment(source, node))
    raise AssertionError("_preprocess_snac_dataset not found")


class _Levels:
    """A sequence that is not a list or tuple, so the method takes its per-scalar loop."""

    def __init__(self, levels):
        self._levels = levels

    def __getitem__(self, index):
        return self._levels[index]

    def __len__(self):
        return len(self._levels)


def _codes_for(row, dtype):
    if row % 11 == 5:
        frames = 0
    else:
        frames = 3 + row % 7
    levels = [
        (torch.arange(frames * rate, dtype = torch.int64) * (row + 1) % 4096).to(dtype).reshape(1, -1)
        for rate in (1, 2, 4)
    ]
    if frames > 2 and row % 3 == 0:
        levels[0][0, 1] = levels[0][0, 0]  # a repeated frame the dedupe drops
        levels[1][0, 2:4] = levels[1][0, 0:2]
        levels[2][0, 4:8] = levels[2][0, 0:4]
    if row % 13 == 6:
        levels[1] = levels[1][:, :1]  # short middle level: the existing catch-and-skip
    if row % 17 == 7:
        levels[2] = levels[2][:, : 4 * frames - 1]  # short fine level
    if row % 19 == 8:
        levels[2] = torch.cat([levels[2], levels[2]], dim = 1)  # longer than needed: ignored
    if row % 23 == 9:
        levels[0] = levels[0].reshape(1, 1, -1)  # 3-D: .item() raises inside the loop
    if row % 29 == 10:
        levels = [level[:, ::1].t().contiguous().t() for level in levels]  # non-contiguous
    if row % 31 == 11:
        levels[1] = levels[1][:0]  # empty batch
    return levels


def _run(method, rows, *, stop_after, plain, dtype, events, monkeypatch):
    class Codec:
        @classmethod
        def from_pretrained(cls, name):
            events.append(("codec_load", name))
            return cls()

        def to(self, device):
            events.append(("codec_to", str(device)))
            return self

        def eval(self):
            return self

        def encode(self, waveform):
            row = int(waveform[0, 0, 0].item())
            if dtype == "functional":
                levels = [
                    torch._to_functional_tensor(level) for level in _codes_for(row, torch.int64)
                ]
            else:
                levels = _codes_for(row, dtype)
            return levels if plain else _Levels(levels)

    class Logger:
        def info(self, message):
            events.append(("info", message))

        def warning(self, message):
            events.append(("warning", message))

    class Tokenizer:
        def encode(
            self,
            text,
            add_special_tokens = True,
        ):
            return [1] + [ord(char) for char in text]

    class Owner:
        max_seq_length = 96
        tokenizer = Tokenizer()

        def __init__(self):
            self.checks = 0

        def _resolve_audio_columns(self, dataset, custom_format_mapping):
            return {"audio_col": "audio", "text_col": "text", "speaker_col": "speaker"}

        @property
        def should_stop(self):
            self.checks += 1
            events.append(("stop_check", self.checks))
            return stop_after is not None and self.checks > stop_after

        def _update_progress(self, **kwargs):
            events.append(("progress", kwargs))

    class Rows:
        column_names = ["audio", "text", "speaker"]

        def __init__(self, count):
            self.rows = [
                {
                    "audio": {"array": np.array([i], dtype = np.float32), "sampling_rate": 24000},
                    "text": "say %d" % i,
                    "speaker": "alice" if i % 2 else None,
                }
                for i in range(count)
            ]
            if count > 4:
                self.rows[3]["text"] = ""
                self.rows[4]["audio"] = None

        def cast_column(self, column, feature):
            return self

        def __getitem__(self, index):
            return self.rows[index]

        def __len__(self):
            return len(self.rows)

    torchaudio = types.ModuleType("torchaudio")
    transforms = types.ModuleType("torchaudio.transforms")
    torchaudio.transforms = transforms
    snac = types.ModuleType("snac")
    snac.SNAC = Codec
    monkeypatch.setitem(sys.modules, "torchaudio", torchaudio)
    monkeypatch.setitem(sys.modules, "torchaudio.transforms", transforms)
    monkeypatch.setitem(sys.modules, "snac", snac)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    namespace = {
        "logger": Logger(),
        "Dataset": Dataset,
        "gc": types.SimpleNamespace(collect = lambda: None),
        "clear_gpu_cache": lambda: None,
    }
    exec(method, namespace)
    owner = Owner()
    try:
        dataset = namespace["_preprocess_snac_dataset"](owner, Rows(rows))
    except ValueError as error:
        return ("raised", str(error)), events
    return (dataset.to_dict(), dataset._fingerprint), events


_FAST_DTYPES = [torch.int64, torch.int32, torch.int16, torch.uint8]
# Floats and functional tensors (no storage, so .tolist() raises where .item() works) must take
# the scalar loop in full.
_SCALAR_LOOP_DTYPES = [torch.float32, "functional"]


@pytest.mark.parametrize("dtype", _FAST_DTYPES + _SCALAR_LOOP_DTYPES)
@pytest.mark.parametrize("stop_after", [None, 57])
def test_fast_path_matches_scalar_loop(monkeypatch, dtype, stop_after):
    if dtype == "functional" and not hasattr(torch, "_to_functional_tensor"):
        pytest.skip("torch has no functional tensors")
    method = _load_method()
    calls = {"tolist": 0, "item": 0}
    real_tolist, real_item = torch.Tensor.tolist, torch.Tensor.item

    def counting_tolist(self):
        calls["tolist"] += 1
        return real_tolist(self)

    def counting_item(self):
        calls["item"] += 1
        return real_item(self)

    monkeypatch.setattr(torch.Tensor, "tolist", counting_tolist)
    monkeypatch.setattr(torch.Tensor, "item", counting_item)

    fast_events, slow_events = [], []
    fast = _run(
        method,
        120,
        stop_after = stop_after,
        plain = True,
        dtype = dtype,
        events = fast_events,
        monkeypatch = monkeypatch,
    )
    fast_calls = dict(calls)
    calls.update(tolist = 0, item = 0)
    slow = _run(
        method,
        120,
        stop_after = stop_after,
        plain = False,
        dtype = dtype,
        events = slow_events,
        monkeypatch = monkeypatch,
    )

    assert fast == slow
    assert fast_events == slow_events
    assert any(event[0] == "warning" for event in fast_events)  # the error rows were exercised
    assert calls["tolist"] == 0
    if dtype in _SCALAR_LOOP_DTYPES:
        assert fast_calls["item"] == calls["item"]
        return
    # The fast path ran: whole-level copies, and only the waveform reads and malformed rows use .item().
    assert fast_calls["tolist"] > 0
    assert fast_calls["item"] < calls["item"] // 5


def test_stop_before_any_row_raises_the_same_error(monkeypatch):
    method = _load_method()
    outcomes = []
    for plain in (True, False):
        events = []
        result = _run(
            method,
            3,
            stop_after = 0,
            plain = plain,
            dtype = torch.int64,
            events = events,
            monkeypatch = monkeypatch,
        )
        outcomes.append((result, events))
    assert outcomes[0] == outcomes[1]
