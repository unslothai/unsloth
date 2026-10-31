# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""``_preprocess_snac_dataset`` copies each SNAC level to the host once instead of seven ``.item()``
reads per frame: rows, skips, progress and stop boundaries must match the old per-scalar loop."""

import ast
import re
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


_OLD_LOOP = """\
for i in range(codes[0].shape[1]):
    all_codes.append(codes[0][0][i].item() + AUDIO_OFFSET)
    all_codes.append(codes[1][0][2 * i].item() + AUDIO_OFFSET + 4096)
    all_codes.append(codes[2][0][4 * i].item() + AUDIO_OFFSET + (2 * 4096))
    all_codes.append(codes[2][0][(4 * i) + 1].item() + AUDIO_OFFSET + (3 * 4096))
    all_codes.append(codes[1][0][(2 * i) + 1].item() + AUDIO_OFFSET + (4 * 4096))
    all_codes.append(codes[2][0][(4 * i) + 2].item() + AUDIO_OFFSET + (5 * 4096))
    all_codes.append(codes[2][0][(4 * i) + 3].item() + AUDIO_OFFSET + (6 * 4096))
"""


def _old_method(method):
    match = re.search(r"\n( +)all_codes = \[\]\n(.*?)\n\1if len\(all_codes\) == 0:", method, re.S)
    assert match, "SNAC interleave block not found"
    loop = textwrap.indent(_OLD_LOOP, match.group(1))
    return method[: match.start(2)] + loop + method[match.end(2) :]


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
        levels[2] = levels[2][:, : 4 * frames - 1]
    if row % 19 == 8:
        levels[2] = torch.cat([levels[2], levels[2]], dim = 1)  # longer than needed: ignored
    if row % 23 == 9:
        levels[0] = levels[0].reshape(1, 1, -1)  # 3-D: .item() raises inside the loop
    if row % 29 == 10:
        levels = [level[:, ::1].t().contiguous().t() for level in levels]
    if frames and row % 31 == 11:
        levels[1] = levels[1][:0]  # empty batch (encode always gets batch 1)
    return levels


def _run(method, rows, *, stop_after, dtype, events, monkeypatch):
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
            return _codes_for(int(waveform[0, 0, 0].item()), dtype)

    class Logger:
        def info(self, message):
            events.append(("info", message))

        def warning(self, message):
            # Malformed codes raise on both sides, with different exception text.
            events.append(("warning", message.split(":")[0]))

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


@pytest.mark.parametrize("dtype", [torch.int64, torch.int32, torch.int16, torch.uint8])
@pytest.mark.parametrize("stop_after", [None, 57])
def test_host_copy_matches_scalar_loop(monkeypatch, dtype, stop_after):
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

    results, counts = [], []
    for source in (method, _old_method(method)):
        events = []
        result = _run(
            source,
            120,
            stop_after = stop_after,
            dtype = dtype,
            events = events,
            monkeypatch = monkeypatch,
        )
        results.append((result, events))
        counts.append(dict(calls))
        calls.update(tolist = 0, item = 0)

    assert results[0] == results[1]
    assert any(event[0] == "warning" for event in results[0][1])
    new, old = counts
    assert new["tolist"] > 0 and old["tolist"] == 0
    assert new["item"] < old["item"] // 5


def test_stop_before_any_row_raises_the_same_error(monkeypatch):
    method = _load_method()
    outcomes = []
    for source in (method, _old_method(method)):
        events = []
        result = _run(
            source,
            3,
            stop_after = 0,
            dtype = torch.int64,
            events = events,
            monkeypatch = monkeypatch,
        )
        outcomes.append((result, events))
    assert outcomes[0] == outcomes[1]
