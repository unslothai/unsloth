# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The VLM / audio detection sets are read from transformers' registry source, so the first
model check in a process that has not loaded torch does not import it. Same sets as the import."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

import utils.models.model_config as mc

_BACKEND_DIR = Path(__file__).resolve().parent.parent
_HAS_TRANSFORMERS = importlib.util.find_spec("transformers") is not None


@pytest.mark.skipif(not _HAS_TRANSFORMERS, reason = "transformers not installed")
def test_building_the_sets_does_not_import_torch_or_the_registry():
    code = textwrap.dedent(
        """
        import sys
        import utils.models.model_config as mc
        sets = mc._detection_sets()
        assert "llava" in sets[0] and "whisper" in sets[2], sets
        heavy = sorted(m for m in ("torch", "transformers.models.auto.modeling_auto") if m in sys.modules)
        print("HEAVY", heavy)
        """
    )
    out = subprocess.run(
        [sys.executable, "-c", code], cwd = _BACKEND_DIR, capture_output = True, text = True, timeout = 600
    )
    assert out.returncode == 0, out.stderr[-2000:]
    assert "HEAVY []" in out.stdout, out.stdout[-500:]


def _registry_importable() -> bool:
    try:
        from transformers.models.auto import modeling_auto  # noqa: F401
    except Exception:  # noqa: BLE001 -- e.g. a transformers whose dependency check refuses this env
        return False
    return True


def _forget_the_imported_registry(monkeypatch):
    # Another test in this process may have imported the registry, which makes the build skip the source.
    monkeypatch.delitem(sys.modules, "transformers.models.auto.modeling_auto", raising = False)


def test_sets_from_source_equal_sets_from_the_import(monkeypatch):
    _forget_the_imported_registry(monkeypatch)
    read = []
    real = mc._mapping_names_from_source
    monkeypatch.setattr(
        mc, "_mapping_names_from_source", lambda path: read.append(real(path)) or read[-1]
    )
    from_source = mc._build_detection_sets()
    if not _HAS_TRANSFORMERS or not _registry_importable():
        pytest.skip("the transformers registry does not import here")
    assert read and read[0] is not None
    monkeypatch.setattr(mc, "_mapping_names_from_source", lambda path: None)
    from_import = mc._build_detection_sets()
    assert from_source == from_import


def test_an_imported_registry_is_used_instead_of_the_source(monkeypatch):
    import types

    fake = types.ModuleType("transformers.models.auto.modeling_auto")
    fake.MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES = {"fake_vlm": "FakeForImageTextToText"}
    parents = {}
    for name in ("transformers", "transformers.models", "transformers.models.auto"):
        parents[name] = types.ModuleType(name)
        parents[name].__path__ = []
        monkeypatch.setitem(sys.modules, name, parents[name])
    parents["transformers.models.auto"].modeling_auto = fake
    monkeypatch.setitem(sys.modules, "transformers.models.auto.modeling_auto", fake)
    monkeypatch.setattr(
        mc, "_mapping_names_from_source", lambda path: pytest.fail("read the source")
    )
    vlm_types, vlm_classes, _ = mc._build_detection_sets()
    assert "fake_vlm" in vlm_types and "FakeForImageTextToText" in vlm_classes


_GOOD = """
from collections import OrderedDict
MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES = OrderedDict([("llava", "LlavaForConditionalGeneration")])
MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES = OrderedDict([("blip", ("BlipA", "BlipB"))])
MODEL_FOR_CTC_MAPPING_NAMES = OrderedDict([("wav2vec2", "Wav2Vec2ForCTC")])
MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING_NAMES = {"whisper": "WhisperForConditionalGeneration"}
MODEL_FOR_AUDIO_CLASSIFICATION_MAPPING_NAMES = OrderedDict([])
MODEL_FOR_TEXT_TO_WAVEFORM_MAPPING_NAMES = OrderedDict([("bark", "BarkModel")])
MODEL_FOR_TEXT_TO_SPECTROGRAM_MAPPING_NAMES = OrderedDict([("speecht5", "SpeechT5ForTextToSpeech")])
MODEL_FOR_AUDIO_XVECTOR_MAPPING_NAMES = OrderedDict([("wavlm", "WavLMForXVector")])
MULTIMODAL = OrderedDict([*list(MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES.items()), ("x", "Y")])
LAZY = _LazyAutoMapping(CONFIG_MAPPING_NAMES, MODEL_FOR_CTC_MAPPING_NAMES)
FIRST = MODEL_FOR_CTC_MAPPING_NAMES["wav2vec2"]
"""


def _read(tmp_path, source: str):
    path = tmp_path / "modeling_auto.py"
    path.write_text(source, encoding = "utf-8")
    return mc._mapping_names_from_source(str(path))


def test_reader_takes_plain_literals(tmp_path):
    found = _read(tmp_path, _GOOD)
    assert found["MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES"] == {"blip": ("BlipA", "BlipB")}
    assert found["MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING_NAMES"] == {
        "whisper": "WhisperForConditionalGeneration"
    }
    assert found["MODEL_FOR_AUDIO_CLASSIFICATION_MAPPING_NAMES"] == {}


def test_a_name_the_module_never_mentions_reads_as_empty(tmp_path):
    source = "\n".join(l for l in _GOOD.splitlines() if "VISION_2_SEQ" not in l)
    assert _read(tmp_path, source)["MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES"] == {}


@pytest.mark.parametrize(
    "extra",
    [
        'MODEL_FOR_CTC_MAPPING_NAMES.update({"hubert": "HubertForCTC"})',
        'MODEL_FOR_CTC_MAPPING_NAMES["hubert"] = "HubertForCTC"',
        'del MODEL_FOR_CTC_MAPPING_NAMES["wav2vec2"]',
        'MODEL_FOR_CTC_MAPPING_NAMES.pop("wav2vec2")',
        'MODEL_FOR_CTC_MAPPING_NAMES = OrderedDict([("a", "B")])',
        'MODEL_FOR_CTC_MAPPING_NAMES |= {"a": "B"}',
        "def f():\n    global MODEL_FOR_CTC_MAPPING_NAMES",
        "from .other import MODEL_FOR_CTC_MAPPING_NAMES",
        "for MODEL_FOR_CTC_MAPPING_NAMES in []:\n    pass",
    ],
)
def test_reader_declines_when_the_runtime_dict_could_differ(tmp_path, extra):
    assert _read(tmp_path, _GOOD + "\n" + extra + "\n") is None


@pytest.mark.parametrize(
    "source",
    [
        _GOOD.replace('OrderedDict([("wav2vec2", "Wav2Vec2ForCTC")])', "_build()"),
        _GOOD.replace('("wav2vec2", "Wav2Vec2ForCTC")', '("wav2vec2", NAME)'),
        _GOOD.replace('("wav2vec2", "Wav2Vec2ForCTC")', '(1, "Wav2Vec2ForCTC")'),
        "\n".join(l for l in _GOOD.splitlines() if "VISION_2_SEQ" not in l)
        + "\ndef __getattr__(name):\n    return {}\n",
        "\n".join(l for l in _GOOD.splitlines() if "VISION_2_SEQ" not in l)
        + "\nfrom .other import *\n",
        "\n".join(l for l in _GOOD.splitlines() if l.startswith("MODEL_FOR_VISION_2_SEQ") is False)
        + "\nprint(MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES)\n",
        "def broken(:\n",
        "x = 1\n",
    ],
)
def test_reader_declines_what_it_cannot_prove(tmp_path, source):
    assert _read(tmp_path, source) is None


def test_a_declined_source_falls_back_to_the_import(monkeypatch):
    _forget_the_imported_registry(monkeypatch)
    calls = []
    monkeypatch.setattr(mc, "_mapping_names_from_source", lambda path: calls.append(path))
    sets = mc._build_detection_sets()
    if _HAS_TRANSFORMERS:
        assert calls and "llava" in sets[0]
