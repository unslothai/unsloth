# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""One local JSON preview, including native parsing and Arrow conversion, in a spawn child."""

from __future__ import annotations

import json
from itertools import islice
from pathlib import Path


def read_json_preview(*, path: str, preview_size: int, load_kwargs: dict, output_dir: str):
    output = Path(output_dir)
    try:
        from utils.security.hub_data_files import install as guard_hub_data_files
        from utils.security.dataset_media import install as guard_dataset_media

        guard_hub_data_files()
        guard_dataset_media()
        if load_kwargs.get("features") is not None:
            from utils.datasets.audio_decode import ensure_audio_decoding
            ensure_audio_decoding()

        import pyarrow as pa
        from datasets import Dataset, load_dataset

        streamed = load_dataset(
            "json", data_files = path, split = "train", streaming = True, **load_kwargs
        )
        rows = list(islice(streamed, preview_size))
        if rows:
            dataset = Dataset.from_list(rows, features = load_kwargs.get("features"))
            table = dataset.data.table
            table.validate(full = True)
            with pa.OSFile(str(output / "preview.arrow"), "wb") as sink:
                with pa.ipc.new_stream(sink, table.schema) as writer:
                    writer.write_table(table)
        reply = {"ok": True, "empty": not rows}
    except Exception as exc:
        reply = {"ok": False, "error": str(exc)}
    (output / "result.json").write_text(json.dumps(reply), encoding = "utf-8")
