# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""An Unsloth save must record a `tokenizer_class` transformers can resolve: it is
written from the live class name, and Unsloth once leaked `_Unsloth_Patched_` into
`processor_class` that way (unsloth#4085). An invariant, not a fix."""

import json
import os
from pathlib import Path

import pytest

MODEL_DIR = Path(os.environ.get("UNSLOTH_TEST_LOCAL_MODEL", "unsloth/gemma-3-270m-it"))

# Must be MULTIMODAL: a text-only repo's AutoProcessor returns a plain tokenizer.
PROCESSOR_DIR = os.environ.get("UNSLOTH_TEST_LOCAL_PROCESSOR")


def _resolver():
    tokenization_auto = pytest.importorskip(
        "transformers.models.auto.tokenization_auto",
        reason="transformers is unavailable",
    )
    resolve = getattr(tokenization_auto, "tokenizer_class_from_name", None)
    if resolve is None:
        pytest.skip("this transformers has no tokenizer_class_from_name")
    return resolve


def _recorded_class(directory, filename_prefix=None):
    name = (
        f"{filename_prefix}-tokenizer_config.json" if filename_prefix else "tokenizer_config.json"
    )
    path = Path(directory) / name
    if not path.is_file():
        pytest.skip(f"the save wrote no {name}, so there is nothing to check")
    with path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    if "tokenizer_class" not in config:
        # Not a failure: a config may omit the key so AutoTokenizer resolves by model_type.
        pytest.skip(f"{name} records no tokenizer_class")
    return config["tokenizer_class"]


def _assert_resolvable(recorded):
    resolve = _resolver()
    assert (
        isinstance(recorded, str) and recorded
    ), f"tokenizer_class must be a non-empty string, got {recorded!r}"
    assert not recorded.startswith("_Unsloth_Patched_"), (
        f"an Unsloth wrapper class name leaked into the export as {recorded!r}; "
        f"this is the shape of unsloth#4085"
    )
    assert resolve(recorded) is not None, (
        f"tokenizer_config.json records tokenizer_class={recorded!r}, which the "
        f"installed transformers cannot resolve, so AutoTokenizer.from_pretrained "
        f"on this export fails"
    )


@pytest.fixture(scope="module")
def local_model():
    """A directory or hub id `AutoTokenizer` can load: only an absolute path must exist."""
    if MODEL_DIR.is_absolute() and not (MODEL_DIR / "config.json").is_file():
        pytest.skip(f"no local model at {MODEL_DIR}")
    return MODEL_DIR


def test_a_plain_tokenizer_save_records_a_resolvable_class(local_model, tmp_path):
    transformers = pytest.importorskip("transformers")
    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained(str(local_model))
    except Exception as error:
        pytest.skip(f"could not load a tokenizer from {local_model}: {error}")
    tokenizer.save_pretrained(str(tmp_path))
    _assert_resolvable(_recorded_class(tmp_path))


def test_an_unsloth_patched_tokenizer_save_records_a_resolvable_class(local_model, tmp_path):
    transformers = pytest.importorskip("transformers")
    save = pytest.importorskip("unsloth.save")
    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained(str(local_model))
    except Exception as error:
        pytest.skip(f"could not load a tokenizer from {local_model}: {error}")

    save.patch_saving_functions(tokenizer)
    assert tokenizer.save_pretrained.__name__ == "unsloth_tokenizer_save_pretrained", (
        "patch_saving_functions did not wrap the tokenizer's save_pretrained, so "
        "this test is not exercising Unsloth's save path"
    )

    tokenizer.save_pretrained(str(tmp_path))
    _assert_resolvable(_recorded_class(tmp_path))


def test_a_processor_save_records_a_resolvable_tokenizer_class(tmp_path):
    """Save gets a Processor, so `type(obj).__name__` is the wrong source for this key."""
    if not PROCESSOR_DIR:
        pytest.skip("set UNSLOTH_TEST_LOCAL_PROCESSOR to a multimodal checkpoint")
    transformers = pytest.importorskip("transformers")
    try:
        processor = transformers.AutoProcessor.from_pretrained(PROCESSOR_DIR)
    except Exception as error:
        pytest.skip(f"no processor for {PROCESSOR_DIR}: {error}")
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        pytest.skip(f"{type(processor).__name__} has no tokenizer to record")

    processor.save_pretrained(str(tmp_path))
    recorded = _recorded_class(tmp_path)
    _assert_resolvable(recorded)
    assert not recorded.endswith("Processor"), (
        f"tokenizer_class was taken from the Processor ({recorded!r}) rather than "
        f"from its tokenizer ({type(tokenizer).__name__!r})"
    )


def test_the_filename_prefix_variant_is_covered_too(local_model, tmp_path):
    """`filename_prefix` writes `<prefix>-tokenizer_config.json`, which a repair must find."""
    transformers = pytest.importorskip("transformers")
    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained(str(local_model))
    except Exception as error:
        pytest.skip(f"could not load a tokenizer from {local_model}: {error}")
    tokenizer.save_pretrained(str(tmp_path), filename_prefix="unsloth")
    _assert_resolvable(_recorded_class(tmp_path, filename_prefix="unsloth"))
