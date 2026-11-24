# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""An unstructured seed upload waiting for text extraction must not hold its bytes in memory."""

import asyncio
import gc
import weakref

from routes.data_recipe import seed


class _Bytes:
    """Stands in for the uploaded bytes: weak-referenceable, sized like a real upload."""

    def __len__(self):
        return 500 * 1024 * 1024


def test_the_upload_bytes_are_released_before_extraction(tmp_path, monkeypatch):
    monkeypatch.setattr(seed, "UNSTRUCTURED_UPLOAD_ROOT", tmp_path)
    raw_path = tmp_path / "block" / "f.txt"
    raw_path.parent.mkdir()
    raw_path.write_text("hello", encoding = "utf-8")
    held = {}

    async def store(file, block_dir, native_path_lease):
        blob = _Bytes()
        held["ref"] = weakref.ref(blob)
        return "doc.txt", blob, raw_path, "f"

    async def extract(path, ext):
        gc.collect()
        held["alive_during_extraction"] = held["ref"]() is not None
        return "hello"

    monkeypatch.setattr(seed, "_store_unstructured_raw", store)
    monkeypatch.setattr(seed, "_extract_text_from_file_async", extract)

    response = asyncio.run(seed.upload_unstructured_file(None, "block", None))
    assert response.status == "ok" and response.size_bytes == 500 * 1024 * 1024
    assert held["alive_during_extraction"] is False, "the upload's bytes stayed in memory"
