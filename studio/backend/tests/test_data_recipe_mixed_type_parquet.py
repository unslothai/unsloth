# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

pytest.importorskip("pyarrow")

import pandas as pd
import pyarrow
import pyarrow.parquet as pq

TEXT_ROW = {"instruction": "Explain caching", "output": "Caches keep responses."}
OBJECT_ROW = {"instruction": "Give a config", "output": {"debug": True, "level": "info"}}


@pytest.fixture(name = "provider")
def fixture_provider():
    lock = threading.Lock()
    calls = [0]

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            with lock:
                calls[0] += 1
                answer = TEXT_ROW if calls[0] % 2 else OBJECT_ROW
            content = "```json\n" + json.dumps(answer) + "\n```"
            reply = json.dumps(
                {
                    "id": "c",
                    "object": "chat.completion",
                    "created": 0,
                    "model": "x",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": content},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(reply)))
            self.end_headers()
            self.wfile.write(reply)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target = server.serve_forever, daemon = True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}/v1"
    server.shutdown()


def test_full_run_keeps_a_structured_field_whose_type_varies_by_row(
    tmp_path: Path, provider
) -> None:
    # unslothai/unsloth#5848: the Output Format box takes a JSON shape, so nothing pins a field's type; one row
    # answering with text and another with an object failed the whole run after every LLM call was paid for.
    pytest.importorskip("data_designer")
    from core.data_recipe.service import build_config_builder, create_data_designer

    recipe = {
        "model_providers": [{"name": "p", "endpoint": provider, "provider_type": "openai"}],
        "model_configs": [
            {
                "alias": "m",
                "model": "x",
                "provider": "p",
                "inference_parameters": {"max_parallel_requests": 1},
            }
        ],
        "columns": [
            {
                "column_type": "sampler",
                "name": "topic",
                "sampler_type": "category",
                "params": {"values": ["web", "db"]},
            },
            {
                "column_type": "llm-structured",
                "name": "dataset_row",
                "model_alias": "m",
                "prompt": "One pair about {{ topic }}.",
                "output_format": {"instruction": "string", "output": "string"},
            },
        ],
    }
    designer = create_data_designer(recipe, artifact_path = str(tmp_path / "artifacts"))
    results = designer.create(build_config_builder(recipe), num_records = 4, dataset_name = "run")

    rows = pq.read_table(results.artifact_storage.final_dataset_path).to_pylist()
    outputs = sorted(row["dataset_row"]["output"] for row in rows)
    assert len(rows) == 4
    assert {row["dataset_row"]["instruction"] for row in rows} == {
        TEXT_ROW["instruction"],
        OBJECT_ROW["instruction"],
    }
    assert TEXT_ROW["output"] in outputs
    assert {"debug": True, "level": "info"} in [json.loads(o) for o in outputs if o.startswith("{")]


def test_unify_only_rewrites_the_disagreeing_paths() -> None:
    from core.data_recipe.service import _unify_cell_types

    values = [
        {"messages": [{"role": "user", "content": "hi"}], "score": 1},
        {
            "messages": [{"role": "user", "content": [{"type": "text", "text": "yo"}]}],
            "score": "high",
        },
        None,
    ]
    unified = _unify_cell_types(values)
    pyarrow.array(unified, from_pandas = True)
    assert unified[0] == {"messages": [{"role": "user", "content": "hi"}], "score": "1"}
    assert unified[1]["messages"][0]["content"] == '[{"type": "text", "text": "yo"}]'
    assert unified[1]["score"] == "high"
    assert unified[2] is None


def test_writable_frames_are_written_untouched(tmp_path: Path) -> None:
    from core.data_recipe.service import write_parquet_with_unified_types

    frame = pd.DataFrame({"row": [TEXT_ROW, dict(TEXT_ROW)]})
    seen = []
    write_parquet_with_unified_types(frame, tmp_path / "x.parquet", seen.append)
    assert len(seen) == 1 and seen[0] is frame


def test_write_that_still_fails_raises_the_original_error(tmp_path: Path) -> None:
    from core.data_recipe.service import write_parquet_with_unified_types

    frame = pd.DataFrame({"row": [TEXT_ROW, OBJECT_ROW]})

    def write(_frame):
        raise pyarrow.ArrowInvalid("still broken" if _frame is not frame else "original")

    with pytest.raises(pyarrow.ArrowInvalid, match = "original"):
        write_parquet_with_unified_types(frame, tmp_path / "x.parquet", write)


def _write_shards(base: Path) -> Path:
    parquet_dir = base / "parquet-files"
    parquet_dir.mkdir(parents = True)
    pq.write_table(
        pyarrow.Table.from_pylist([{"row": TEXT_ROW}]), parquet_dir / "batch_00000.parquet"
    )
    pq.write_table(
        pyarrow.Table.from_pylist([{"row": OBJECT_ROW}]), parquet_dir / "batch_00001.parquet"
    )
    return parquet_dir


def test_merge_joins_batches_whose_field_types_differ(tmp_path: Path) -> None:
    pytest.importorskip("data_designer")
    from core.data_recipe.jobs.worker import _merge_batches_to_single_parquet

    parquet_dir = _write_shards(tmp_path)
    _merge_batches_to_single_parquet(tmp_path)

    assert sorted(p.name for p in parquet_dir.iterdir()) == ["batch_00000.parquet"]
    rows = pq.read_table(parquet_dir / "batch_00000.parquet").to_pylist()
    assert [row["row"]["output"] for row in rows] == [
        TEXT_ROW["output"],
        '{"debug": true, "level": "info"}',
    ]


def test_failed_merge_keeps_the_generated_batches(tmp_path: Path, monkeypatch) -> None:
    pytest.importorskip("data_designer")
    from core.data_recipe.jobs import worker

    parquet_dir = _write_shards(tmp_path)

    def fail(*_args, **_kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(worker, "write_parquet_with_unified_types", fail)
    with pytest.raises(OSError):
        worker._merge_batches_to_single_parquet(tmp_path)

    assert sorted(p.name for p in parquet_dir.iterdir()) == [
        "batch_00000.parquet",
        "batch_00001.parquet",
    ]
    assert not list(tmp_path.glob("*.merging"))


@pytest.mark.parametrize(
    "first, second, expected",
    [
        ({"o": {"d": True}}, {"o": "x"}, [{"o": '{"d": true}'}, {"o": "x"}]),
        (1, 1.5, [1.0, 1.5]),
        (None, "s", [None, "s"]),
        ({"a": "x"}, {"a": "y", "b": "z"}, [{"a": "x", "b": None}, {"a": "y", "b": "z"}]),
    ],
)
def test_batches_with_different_schemas_load_as_one_dataset(
    tmp_path: Path, first, second, expected
) -> None:
    from core.data_recipe.jobs.worker import _unify_batch_schemas

    parquet_dir = tmp_path / "parquet-files"
    parquet_dir.mkdir()
    pd.DataFrame({"r": [first], "n": [0]}).to_parquet(
        parquet_dir / "batch_00000.parquet", index = False
    )
    pd.DataFrame({"r": [second], "n": [1]}).to_parquet(
        parquet_dir / "batch_00001.parquet", index = False
    )

    _unify_batch_schemas(tmp_path)

    assert sorted(p.name for p in parquet_dir.iterdir()) == [
        "batch_00000.parquet",
        "batch_00001.parquet",
    ]
    table = pq.read_table(parquet_dir)
    assert table.column("r").to_pylist() == expected
    assert table.column("n").to_pylist() == [0, 1]


def test_batches_that_already_agree_are_not_rewritten(tmp_path: Path) -> None:
    from core.data_recipe.jobs.worker import _unify_batch_schemas

    parquet_dir = _write_shards(tmp_path)
    for path in parquet_dir.iterdir():
        path.unlink()
    pd.DataFrame({"r": ["a"]}).to_parquet(parquet_dir / "batch_00000.parquet", index = False)
    pd.DataFrame({"r": ["b"]}).to_parquet(parquet_dir / "batch_00001.parquet", index = False)
    before = {p.name: p.read_bytes() for p in parquet_dir.iterdir()}

    _unify_batch_schemas(tmp_path)

    assert {p.name: p.read_bytes() for p in parquet_dir.iterdir()} == before


def test_pandas_missing_values_do_not_flatten_the_struct() -> None:
    from core.data_recipe.service import _unify_cell_types

    unified = _unify_cell_types([{"o": {"d": True}}, {"o": "x"}, pd.NA])
    pyarrow.array(unified, from_pandas = True)
    assert unified[:2] == [{"o": '{"d": true}'}, {"o": "x"}]


def test_batch_unify_leaves_agreeing_columns_and_promotes_with_arrow(tmp_path: Path) -> None:
    from core.data_recipe.jobs.worker import _unify_batch_schemas

    parquet_dir = tmp_path / "parquet-files"
    parquet_dir.mkdir()
    stamp = pd.Timestamp("2026-01-01 12:34:56.123456789")
    for index, (row, when) in enumerate([({"o": {"d": True}}, None), ({"o": "x"}, stamp)]):
        table = pyarrow.table(
            {
                "r": pyarrow.array([row]),
                "ts": pyarrow.array([stamp], pyarrow.timestamp("ns")),
                "maybe": pyarrow.array([when], pyarrow.timestamp("ns") if when else pyarrow.null()),
            }
        )
        pq.write_table(table, parquet_dir / f"batch_{index:05d}.parquet")

    _unify_batch_schemas(tmp_path)

    table = pq.read_table(parquet_dir)
    assert table.schema.field("ts").type == pyarrow.timestamp("ns")
    assert table.schema.field("maybe").type == pyarrow.timestamp("ns")
    assert table.column("ts").to_pylist() == [stamp, stamp]
    assert table.column("maybe").to_pylist() == [None, stamp]
    assert table.column("r").to_pylist() == [{"o": '{"d": true}'}, {"o": "x"}]


def test_dates_and_datetimes_on_one_path_keep_their_time() -> None:
    import datetime

    from core.data_recipe.service import _unify_cell_types

    values = [datetime.date(2026, 1, 2), datetime.datetime(2026, 1, 2, 12, 34, 56, 123456), None]
    assert _unify_cell_types(values) == ["2026-01-02", "2026-01-02T12:34:56.123456", None]
    assert _unify_cell_types([1, 2.5]) == [1, 2.5]
