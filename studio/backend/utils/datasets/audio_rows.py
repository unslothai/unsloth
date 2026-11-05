# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Audio dataset helpers shared by the CUDA and MLX trainers; torch-free."""

from typing import Any


def resolve_audio_columns(dataset, custom_format_mapping: dict = None) -> dict:
    """Resolve audio/text/speaker columns from user mapping or fallback. Returns dict with keys
    audio_col, text_col, speaker_col (may be None)."""
    cols = dataset.column_names

    if custom_format_mapping:
        audio_col = None
        text_col = None
        speaker_col = None
        for col, role in custom_format_mapping.items():
            if role == "audio":
                audio_col = col
            elif role == "text":
                text_col = col
            elif role == "speaker_id":
                speaker_col = col
        if audio_col and audio_col in cols and text_col and text_col in cols:
            return {
                "audio_col": audio_col,
                "text_col": text_col,
                "speaker_col": speaker_col,
            }

    audio_col = next((c for c in cols if c.lower() in ("audio", "speech")), None)
    text_col = next(
        (c for c in cols if c.lower() in ("text", "sentence", "transcript", "transcription")),
        None,
    )

    speaker_col = None
    if "source" in cols:
        speaker_col = "source"
    elif "speaker_id" in cols:
        speaker_col = "speaker_id"

    if audio_col is None or text_col is None or speaker_col is None:
        from hub.utils.dataset_format import detect_multimodal_dataset

        detected = detect_multimodal_dataset(dataset)
        audio_col = audio_col or detected["detected_audio_column"]
        text_col = text_col or detected["detected_text_column"]
        speaker_col = speaker_col or detected["detected_speaker_column"]

    return {
        "audio_col": audio_col,
        "text_col": text_col,
        "speaker_col": speaker_col,
    }


def image_slot_count(row_images: Any) -> int:
    if isinstance(row_images, list):
        return sum(1 for image in row_images if image is not None)
    return 0 if row_images is None else 1


def audio_transcription_messages(
    audio: Any,
    label: str,
    n_images: int = 0,
) -> list:
    """System/user/assistant chat turns that train an audio-input model to transcribe ``audio``,
    with one image slot per image of the row ahead of the audio."""
    return [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "You are an assistant that transcribes speech accurately.",
                }
            ],
        },
        {
            "role": "user",
            "content": [
                *({"type": "image"} for _ in range(n_images)),
                {"type": "audio", "audio": audio},
                {"type": "text", "text": "Please transcribe this audio."},
            ],
        },
        {"role": "assistant", "content": [{"type": "text", "text": label}]},
    ]


def resolve_audio_vlm_image_column(
    dataset,
    custom_format_mapping,
    audio_col,
    text_col = None,
    speaker_col = None,
):
    cols = dataset.column_names
    reserved = {audio_col, text_col}
    reserved.update(
        col for col, role in (custom_format_mapping or {}).items() if role and role != "image"
    )
    for col, role in (custom_format_mapping or {}).items():
        if role == "image" and col in cols and col not in reserved:
            return col
    from hub.utils.dataset_format import (
        _first_row,
        _is_image_list_item,
        _is_image_value,
        detect_multimodal_dataset,
    )

    from datasets import Image

    def image_typed(col):
        feature = (getattr(dataset, "features", None) or {}).get(col)
        inner = feature[0] if isinstance(feature, list) and feature else feature
        return isinstance(getattr(inner, "feature", inner), Image)

    if speaker_col and not image_typed(speaker_col):
        reserved.add(speaker_col)

    def image_list(value):
        if not isinstance(value, list):
            return False
        items = [
            item for item in value if item is not None and not (isinstance(item, str) and not item)
        ]
        return bool(items) and all(map(_is_image_list_item, items))

    sample = _first_row(dataset) or {}

    def empty(value):
        # Not `in (None, "", [])`: that compares a pixel ndarray elementwise and raises.
        return (
            value is None
            or (isinstance(value, str) and not value)
            or (
                isinstance(value, list)
                and all(item is None or (isinstance(item, str) and not item) for item in value)
            )
        )

    ordered = list(dict.fromkeys(detect_multimodal_dataset(dataset)["multimodal_columns"] + cols))
    ordered = [col for col in ordered if col not in reserved]
    typed = {col: image_typed(col) for col in ordered}
    values = {col: sample.get(col) for col in ordered}
    first_candidate = next(
        (
            i
            for i, col in enumerate(ordered)
            if typed[col] or _is_image_value(values[col]) or image_list(values[col])
        ),
        len(ordered),
    )
    unresolved = {col for col in ordered[:first_candidate] if not typed[col] and empty(values[col])}
    if unresolved:
        for i, row in enumerate(dataset):
            if i >= 100 or not unresolved:
                break
            for col in tuple(unresolved):
                if not empty(row.get(col)):
                    values[col] = row[col]
                    unresolved.remove(col)
    return next(
        (
            col
            for col in ordered
            if typed[col] or _is_image_value(values[col]) or image_list(values[col])
        ),
        None,
    )


def prepare_audio_vlm_image_column(dataset, image_col):
    from datasets import Image, Sequence

    feature = dataset.features.get(image_col)
    inner = feature[0] if isinstance(feature, list) else getattr(feature, "feature", None)

    pixel_feature, dimensions = feature, 0
    while getattr(pixel_feature, "feature", None) is not None:
        pixel_feature = pixel_feature.feature
        dimensions += 1
    if dimensions <= 1 and getattr(pixel_feature, "dtype", "") in (
        "string",
        "large_string",
    ):

        def normalize_paths(values):
            return {
                image_col: [
                    [None if item == "" else item for item in value]
                    if isinstance(value, list)
                    else None
                    if value == ""
                    else value
                    for value in values
                ]
            }

        dataset = dataset.map(normalize_paths, batched = True, input_columns = image_col)
    if dimensions in (2, 3) and getattr(pixel_feature, "dtype", "").startswith(
        ("int", "uint", "float")
    ):
        from itertools import islice

        import numpy as np

        value = next(
            (row[image_col] for row in islice(dataset, 100) if row[image_col] is not None),
            None,
        )
        shape = np.asarray(value).shape if value is not None else ()
        if value is None or len(shape) == 2 or (len(shape) == 3 and shape[-1] in (1, 3, 4)):
            inner = None

    # Image(decode = False) yields {"path", "bytes"} dicts the processor rejects.
    def decodes(f):
        return isinstance(f, Image) and f.decode

    if inner is not None:
        if not decodes(inner):
            dataset = dataset.cast_column(image_col, Sequence(Image()))
    elif not decodes(feature):
        dataset = dataset.cast_column(image_col, Image())
    return dataset
