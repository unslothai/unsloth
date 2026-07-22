"""Regression test for unsloth#4202: detect missing torchvision so the loader can
surface the real cause instead of a misleading collator error."""

import importlib.util
from unittest import mock

from unsloth.models.vision import _missing_torchvision_error


def test_error_text_mentions_torchvision_is_detected():
    err = ImportError("Qwen3VLVideoProcessor requires the Torchvision library but ...")
    assert _missing_torchvision_error(err) is True


def test_torchvision_missing_is_detected_without_error():
    with mock.patch.object(importlib.util, "find_spec", return_value = None):
        assert _missing_torchvision_error(None) is True


def test_torchvision_present_unrelated_error_is_not_flagged():
    sentinel = object()
    with mock.patch.object(importlib.util, "find_spec", return_value = sentinel):
        assert _missing_torchvision_error(ValueError("unrelated")) is False
        assert _missing_torchvision_error(None) is False


def test_matches_real_environment():
    # find_spec is the source of truth when no error is supplied.
    expected = importlib.util.find_spec("torchvision") is None
    assert _missing_torchvision_error(None) is expected
