# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A decoded Audio cell must be summarised, not serialised sample by sample.

The dataset preview compressed the undecoded {"bytes", "path"} shape only. When
torchcodec cannot load its FFmpeg libraries the soundfile fallback decodes instead,
and the dataset formatter hands back {"path", "array", "sampling_rate"} with the
waveform as a plain list. Ten preview rows of a few seconds each then serialised to
tens of MB of floats, and the client died with "Maximum call stack size exceeded"
before it could POST /api/train/start.
"""

from __future__ import annotations

import pytest

from hub.services.datasets.formatting import _serialize_preview_value


def test_a_decoded_audio_cell_collapses_to_a_summary():
    cell = {"path": "a.wav", "array": [0.0] * 240_000, "sampling_rate": 24_000}
    out = _serialize_preview_value({"audio": cell})
    assert out == {"audio": "<audio, 240000 samples @ 24000 Hz, 10.0s>"}
    assert len(str(out)) < 200


def test_the_undecoded_shape_still_reports_bytes():
    out = _serialize_preview_value({"bytes": b"RIFFxxxx", "path": "a.wav"})
    assert out == "<binary data, 8 bytes>"


def test_a_decoder_object_is_left_to_repr():
    class _Decoder:
        def __repr__(self):
            return "<AudioDecoder>"

    assert _serialize_preview_value({"audio": _Decoder()}) == {"audio": "<AudioDecoder>"}


def test_an_ordinary_list_column_is_untouched():
    assert _serialize_preview_value({"ids": [1, 2, 3]}) == {"ids": [1, 2, 3]}


def test_a_missing_rate_does_not_raise():
    out = _serialize_preview_value({"array": [0.0, 0.1], "sampling_rate": None})
    assert out.startswith("<audio, 2 samples")


def test_a_numpy_array_cell_is_summarised_too():
    # The decoder hands back a numpy array, not a list; the raw repr was landing in the table.
    np = pytest.importorskip("numpy")
    cell = {"path": "a.m4a", "array": np.zeros(22050, dtype = "float32"), "sampling_rate": 22050}
    assert _serialize_preview_value({"audio": cell}) == {
        "audio": "<audio, 22050 samples @ 22050 Hz, 1.0s>"
    }
