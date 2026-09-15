# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""An Unsloth save must record a `tokenizer_class` transformers can resolve.

This pins an invariant rather than fixing a defect. `PreTrainedTokenizerBase.save_pretrained`
writes `tokenizer_class` from the live class name, so anything that hands the
save a wrapper class, or a generic fallback where the checkpoint has its own
tokenizer, lands a name in `tokenizer_config.json` that
`AutoTokenizer.from_pretrained` then cannot resolve, or resolves to the wrong
family. Unsloth has shipped exactly that: the `_Unsloth_Patched_` prefix leaked
into `processor_class` (unsloth#4085), and the read-side workarounds for it are
still live in `unsloth_zoo/hf_utils.py` and `unsloth/models/vision.py`.

Measured on the save paths reachable today the value is correct, which is why
there is no repair in `unsloth/save.py` to go with this file: #7681 named
`tokenizer_class` as unhandled, and it turned out to be handled by transformers
itself. These tests exist so that if a future patch reintroduces the #4085 shape
it fails here instead of in an export someone tries to serve.

CPU only, no network: everything loads from the local model directory.
"""

import json
import os
from pathlib import Path

import pytest

# A local checkpoint to save from. `UNSLOTH_TEST_LOCAL_MODEL` overrides it, and
# the default is the small instruct model the rest of the suite already uses.
MODEL_DIR = Path(os.environ.get("UNSLOTH_TEST_LOCAL_MODEL", "unsloth/gemma-3-270m-it"))

# A multimodal checkpoint, so the Processor case is a real Processor and not a
# tokenizer wearing the name. Text-only repos hand back a plain tokenizer from
# `AutoProcessor`, which has no `.tokenizer` and so proves nothing about the
# `type(obj).__name__` trap. Point `UNSLOTH_TEST_LOCAL_PROCESSOR` at a local
# multimodal checkpoint to exercise it; without one that single test skips.
PROCESSOR_DIR = os.environ.get("UNSLOTH_TEST_LOCAL_PROCESSOR")


def _resolver():
    """transformers' own name to class lookup, which is what a consumer uses."""
    tokenization_auto = pytest.importorskip(
        "transformers.models.auto.tokenization_auto",
        reason = "transformers is unavailable",
    )
    resolve = getattr(tokenization_auto, "tokenizer_class_from_name", None)
    if resolve is None:
        pytest.skip("this transformers has no tokenizer_class_from_name")
    return resolve


def _recorded_class(directory, filename_prefix = None):
    """The `tokenizer_class` a save actually wrote, or a skip saying why not."""
    name = (
        f"{filename_prefix}-tokenizer_config.json" if filename_prefix else "tokenizer_config.json"
    )
    path = Path(directory) / name
    if not path.is_file():
        pytest.skip(f"the save wrote no {name}, so there is nothing to check")
    with path.open("r", encoding = "utf-8") as handle:
        config = json.load(handle)
    if "tokenizer_class" not in config:
        # Not an assertion failure: some configs omit the key deliberately so
        # that AutoTokenizer resolves by `model_type`. The invariant here is
        # about a name that is present being resolvable.
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


@pytest.fixture(scope = "module")
def local_model():
    """A directory or hub id `AutoTokenizer` can load, or a skip.

    An absolute path must exist; a bare `owner/name` is a hub id, left to the
    cache, and the individual tests skip with the loader's own error if it is
    not there. Written this way so the default is a hub id that upstream CI can
    resolve rather than a path that only exists on one machine.
    """
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
    """The path that matters. `patch_saving_functions` replaces the tokenizer's
    `save_pretrained` with `unsloth_tokenizer_save_pretrained`, and it is that
    wrapper, and the post-save repairs it runs, which decide what ends up in
    `tokenizer_config.json`."""
    transformers = pytest.importorskip("transformers")
    save = pytest.importorskip("unsloth.save")
    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained(str(local_model))
    except Exception as error:
        pytest.skip(f"could not load a tokenizer from {local_model}: {error}")

    save.patch_saving_functions(tokenizer)
    # Assert the patch really took, else this test would pass while measuring
    # stock transformers behaviour.
    assert tokenizer.save_pretrained.__name__ == "unsloth_tokenizer_save_pretrained", (
        "patch_saving_functions did not wrap the tokenizer's save_pretrained, so "
        "this test is not exercising Unsloth's save path"
    )

    tokenizer.save_pretrained(str(tmp_path))
    _assert_resolvable(_recorded_class(tmp_path))


def test_a_processor_save_records_a_resolvable_tokenizer_class(tmp_path):
    """For a vision model the object handed to save is a Processor, not a
    tokenizer, so `type(obj).__name__` is the wrong source for this key: it
    would write `Qwen3VLProcessor`, which is not a tokenizer class at all. The
    value has to come from the Processor's `.tokenizer`."""
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
    """`save_pretrained(filename_prefix = ...)` writes `<prefix>-tokenizer_config.json`,
    which any post-save repair has to look for under that name."""
    transformers = pytest.importorskip("transformers")
    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained(str(local_model))
    except Exception as error:
        pytest.skip(f"could not load a tokenizer from {local_model}: {error}")
    tokenizer.save_pretrained(str(tmp_path), filename_prefix = "unsloth")
    _assert_resolvable(_recorded_class(tmp_path, filename_prefix = "unsloth"))
