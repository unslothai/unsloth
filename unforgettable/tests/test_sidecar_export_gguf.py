# Copyright 2026-present the Unforgettable contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import json
from pathlib import Path

import pytest

from unforgettable.cli import main
from unforgettable.sidecar.adapters import get_adapter, insert_adapter
from unforgettable.sidecar.export_gguf import export_adapter_gguf
from unforgettable.sidecar.train import FakeTrainBackend, train_pack
from unforgettable.store.records import insert_record, insert_retrieve_use, insert_rollout
from unforgettable.sidecar.pack import pack_from_admitted_b


def _peft_dir(root: Path) -> Path:
    dest = root / "peft"
    dest.mkdir()
    (dest / "adapter_config.json").write_text(
        json.dumps({"peft_type": "LORA", "base_model_name_or_path": "toy"}),
        encoding = "utf-8",
    )
    return dest


def test_fake_train_does_not_export_gguf(db_path):
    for i in range(4):
        rec = insert_record(
            kind = "procedure",
            title = f"Playbook {i}",
            body = f"steps {i}",
            provenance = "world",
            db_path = db_path,
        )
        insert_retrieve_use(
            episode_id = f"ep-{i}",
            record_id = rec["id"],
            contact = "world",
            db_path = db_path,
        )
        insert_rollout(
            episode_id = f"ep-{i}",
            contact = "world",
            outcome = "pass",
            summary = "ok",
            db_path = db_path,
        )
    report = pack_from_admitted_b(db_path = db_path)
    result = train_pack(
        report.pack_id,
        backend = FakeTrainBackend(),
        base_model = "fake",
        db_path = db_path,
    )
    assert result.gguf_path is None
    row = get_adapter(result.adapter_id, db_path = db_path)
    assert row is not None
    assert row.get("gguf_path") in (None, "")


def test_export_adapter_gguf_writes_with_stub_converter(tmp_path, monkeypatch):
    peft = _peft_dir(tmp_path)
    converter = tmp_path / "convert_lora_to_gguf.py"
    converter.write_text(
        "import argparse, pathlib\n"
        "p = argparse.ArgumentParser()\n"
        "p.add_argument('src')\n"
        "p.add_argument('--outfile')\n"
        "p.add_argument('--outtype')\n"
        "p.add_argument('--base', default=None)\n"
        "a = p.parse_args()\n"
        "pathlib.Path(a.outfile).write_text('gguf')\n",
        encoding = "utf-8",
    )
    monkeypatch.setenv("UNFORGETTABLE_CONVERT_LORA_TO_GGUF", str(converter))
    out = export_adapter_gguf(peft, base_model = "toy-base")
    assert Path(out).is_file()
    assert Path(out).name == "toy-base-lora-f16.gguf"


def test_export_adapter_gguf_missing_converter(tmp_path, monkeypatch):
    peft = _peft_dir(tmp_path)
    monkeypatch.delenv("UNFORGETTABLE_CONVERT_LORA_TO_GGUF", raising = False)
    monkeypatch.setattr(
        "unforgettable.sidecar.export_gguf.find_converter",
        lambda: None,
    )
    with pytest.raises(FileNotFoundError, match = "convert_lora_to_gguf"):
        export_adapter_gguf(peft, base_model = "toy")


def test_cli_export_gguf(tmp_path, db_path, monkeypatch, capsys):
    peft = _peft_dir(tmp_path)
    converter = tmp_path / "convert_lora_to_gguf.py"
    converter.write_text(
        "import argparse, pathlib\n"
        "p = argparse.ArgumentParser()\n"
        "p.add_argument('src')\n"
        "p.add_argument('--outfile')\n"
        "p.add_argument('--outtype')\n"
        "p.add_argument('--base', default=None)\n"
        "a = p.parse_args()\n"
        "pathlib.Path(a.outfile).write_text('gguf')\n",
        encoding = "utf-8",
    )
    monkeypatch.setenv("UNFORGETTABLE_CONVERT_LORA_TO_GGUF", str(converter))
    rec = insert_record(
        kind = "procedure",
        title = "Playbook 0",
        body = "steps 0",
        provenance = "world",
        db_path = db_path,
    )
    insert_retrieve_use(
        episode_id = "ep-0",
        record_id = rec["id"],
        contact = "world",
        db_path = db_path,
    )
    insert_rollout(
        episode_id = "ep-0",
        contact = "world",
        outcome = "pass",
        summary = "ok",
        db_path = db_path,
    )
    pack = pack_from_admitted_b(db_path = db_path)
    row = insert_adapter(
        pack_id = pack.pack_id,
        backend = "unsloth",
        base_model = "toy-base",
        recipe = "sft",
        path = str(peft),
        db_path = db_path,
    )
    assert main(["export-gguf", row["id"], "--db", str(db_path)]) == 0
    out = capsys.readouterr().out
    assert "gguf_path" in out
    stored = get_adapter(row["id"], db_path = db_path)
    assert stored is not None
    assert stored["gguf_path"]
    assert Path(stored["gguf_path"]).is_file()


def test_export_refuses_fake_dir(tmp_path):
    fake = tmp_path / "fake"
    fake.mkdir()
    (fake / "adapter_config.json").write_text(json.dumps({"fake": True}), encoding = "utf-8")
    with pytest.raises(ValueError, match = "not a PEFT"):
        export_adapter_gguf(fake)
