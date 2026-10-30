# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import os

_MORE_AUDIO_EXTENSIONS = (".amr", ".aif", ".aiff", ".aifc", ".caf")


def media_columns_hold_media(
    dataset,
    detected,
    *,
    sample_rows = 16,
) -> tuple[bool, bool]:
    # The detector also flags columns by name alone, so a text column called `filename` or `speech` must hold media.
    from itertools import islice

    from datasets import Audio, Image
    from hub.utils.dataset_format import (
        _AUDIO_EXTENSIONS,
        _has_image_header,
        _holds_images,
        _is_audio_value,
        _is_image_value,
    )

    features = getattr(dataset, "features", None) or {}
    rows = []

    def values(column):
        if not rows:
            rows.extend(row or {} for row in islice(iter(dataset), sample_rows))
        return [row.get(column) for row in rows]

    def bare(value):
        return value.strip().split("?")[0] if isinstance(value, str) else ""

    def sniff(head, kind):
        if kind == "image":
            return _has_image_header(head) or head[:4] in (b"II*\x00", b"MM\x00*")
        return (
            head[:4] in (b"fLaC", b"OggS", b"\x1aE\xdf\xa3", b"0&\xb2u", b"caff")
            or head[:3] == b"ID3"
            or head[:5] == b"#!AMR"
            or (head[:4] == b"FORM" and head[8:12] in (b"AIFF", b"AIFC"))
            or head[4:8] == b"ftyp"
            or (head[:4] == b"RIFF" and head[8:12] == b"WAVE")
            or (len(head) > 1 and head[0] == 0xFF and head[1] & 0xE0 == 0xE0)
        )

    def reference(value, kind):
        if isinstance(value, dict):
            value = value.get("bytes") or value.get("path")
        if isinstance(value, (bytes, bytearray)):
            return sniff(bytes(value[:12]), kind)
        path = bare(value)
        if not path:
            return False
        if path.lower().startswith(("http://", "https://")):
            from urllib.request import Request, urlopen
            try:
                request = Request(value.strip(), headers = {"Range": "bytes=0-11"})
                with urlopen(request, timeout = 5) as response:
                    if response.headers.get_content_type().startswith(f"{kind}/"):
                        return True
                    return sniff(response.read(12), kind)
            except Exception:  # noqa: BLE001
                return False
        if not os.path.isfile(path):
            return False
        with open(path, "rb") as file:
            return sniff(file.read(12), kind)

    def image(column):
        return isinstance(features.get(column), Image) or any(
            _is_image_value(value) or _holds_images(value) or reference(value, "image")
            for value in values(column)
        )

    def audio(column):
        return isinstance(features.get(column), Audio) or any(
            _is_audio_value(value)
            or bare(value).lower().endswith(_AUDIO_EXTENSIONS + _MORE_AUDIO_EXTENSIONS)
            or reference(value, "audio")
            for value in values(column)
        )

    return (
        any(image(column) for column in detected.get("multimodal_columns", [])),
        any(audio(column) for column in detected.get("audio_columns", [])),
    )
