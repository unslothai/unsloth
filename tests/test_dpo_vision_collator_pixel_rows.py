# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Grid vision DPO collator reuses TRL's padded pixel rows: bitwise identical to per-row conversion, else falls back."""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("trl")
from trl.trainer import dpo_trainer as _dpo  # noqa: E402

RL_PATH = Path(__file__).resolve().parents[1] / "unsloth" / "models" / "rl_replacements.py"

if 'example["pixel_values"]' not in inspect.getsource(_dpo.DataCollatorForPreference):
    pytest.skip(
        reason = "TRL >= 0.29 DataCollatorForPreference does not collate pixel_values; nothing to reuse",
        allow_module_level = True,
    )


def _patched_collator_class(base_cls = None):
    src = RL_PATH.read_text(encoding = "utf-8")
    nodes = [
        n
        for n in ast.parse(src).body
        if (isinstance(n, ast.FunctionDef) and n.name == "dpo_trainer_data_collator_vision_keys")
        or (
            isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "_DPO_VISION_KEYS" for t in n.targets)
        )
    ]
    ns = {}
    exec(compile(ast.Module(body = nodes, type_ignores = []), str(RL_PATH), "exec"), ns)
    code = ns["dpo_trainer_data_collator_vision_keys"]("data_collator", "")
    imp = "from trl.trainer.dpo_trainer import DataCollatorForPreference\n"
    assert imp in code
    cls = type("PrivateCollator", (base_cls or _dpo.DataCollatorForPreference,), {})
    g = {"DataCollatorForPreference": cls}
    exec(compile(code.replace(imp, ""), "<patched collator>", "exec"), g)
    first = cls.torch_call
    exec(compile(code.replace(imp, ""), "<patched collator>", "exec"), g)
    assert cls.torch_call is first
    return cls


def _reference(examples):
    """The previous behaviour: TRL's batch, then every row converted again and concatenated."""
    out = _dpo.DataCollatorForPreference(pad_token_id = 0).torch_call(examples)
    out["pixel_values"] = torch.cat([torch.as_tensor(e["pixel_values"]) for e in examples], dim = 0)
    out["image_grid_thw"] = torch.cat(
        [torch.as_tensor(e["image_grid_thw"]).reshape(-1, 3) for e in examples], dim = 0
    )
    return out


def _rows(
    patches = (6, 3, 9),
    dim = 12,
    arrow = True,
):
    g = torch.Generator().manual_seed(0)
    rows = []
    for i, n in enumerate(patches):
        rows.append(
            {
                "prompt_input_ids": list(range(1, 4 + i)),
                "chosen_input_ids": [7] * (2 + i),
                "rejected_input_ids": [8] * (3 - i % 2),
                "pixel_values": torch.randn(n, dim, generator = g).tolist(),
                "image_grid_thw": [[1, 1, n]],
            }
        )
    if not arrow:
        return rows
    from datasets import Dataset

    ds = Dataset.from_list(rows)
    return ds.__getitems__(list(range(len(rows))))


def _assert_same(got, want):
    assert list(got.keys()) == list(want.keys())
    for k in want:
        assert got[k].dtype == want[k].dtype, k
        assert got[k].shape == want[k].shape, k
        assert torch.equal(got[k], want[k]), k


def _count_full_row_conversions(monkeypatch, examples):
    sizes = {len(e["pixel_values"]) for e in examples}
    calls = []
    real = torch.as_tensor

    def spy(x, *a, **k):
        if isinstance(x, list) and len(x) in sizes and len(x) > 1:
            calls.append(len(x))
        return real(x, *a, **k)

    monkeypatch.setattr(torch, "as_tensor", spy)
    return calls


@pytest.mark.parametrize("patches", [(6, 3, 9), (4,), (5, 5)])
def test_rows_come_from_trl_padded_tensor(monkeypatch, patches):
    cls = _patched_collator_class()
    examples = _rows(patches)
    want = _reference(examples)
    calls = _count_full_row_conversions(monkeypatch, examples)
    got = cls(pad_token_id = 0).torch_call(examples)
    _assert_same(got, want)
    assert calls == []


def test_left_padded_trl_tensor_falls_back(monkeypatch):
    class LeftPad(_dpo.DataCollatorForPreference):
        def torch_call(self, examples):
            out = super().torch_call(examples)
            out["pixel_values"] = _dpo.pad(
                [torch.tensor(e["pixel_values"]) for e in examples],
                padding_value = 0.0,
                padding_side = "left",
            )
            return out

    cls = _patched_collator_class(LeftPad)
    examples = _rows((6, 3, 9))
    want = _reference(examples)
    calls = _count_full_row_conversions(monkeypatch, examples)
    _assert_same(cls(pad_token_id = 0).torch_call(examples), want)
    assert calls


@pytest.mark.parametrize("case", ["dtype", "missing", "flat"])
def test_unusable_trl_tensor_falls_back(case):
    base = _dpo.DataCollatorForPreference

    class Odd(base):
        def torch_call(self, examples):
            out = super().torch_call(examples)
            if case == "missing":
                out.pop("pixel_values")
            elif case == "flat":
                out["pixel_values"] = torch.cat(
                    [torch.as_tensor(e["pixel_values"]) for e in examples]
                )
            return out

    cls = _patched_collator_class(Odd)
    examples = _rows((6, 3, 9), arrow = False)
    if case == "dtype":
        # TRL pads to the first row's dtype; a row of another dtype must keep its own conversion.
        examples[1]["pixel_values"] = torch.tensor(examples[1]["pixel_values"], dtype = torch.float16)
    want = _reference(examples)
    got = cls(pad_token_id = 0).torch_call(examples)
    if case == "missing":
        assert list(got.keys())[-2:] == ["pixel_values", "image_grid_thw"]
        got = {k: got[k] for k in want}
    _assert_same(got, want)


def test_non_grid_batch_untouched():
    cls = _patched_collator_class()
    examples = _rows((6, 3), arrow = False)
    for e in examples:
        e.pop("image_grid_thw")
    want = _dpo.DataCollatorForPreference(pad_token_id = 0).torch_call(examples)
    _assert_same(cls(pad_token_id = 0).torch_call(examples), want)
