# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""An unreadable tokenizer_config.json must not read as "not an audio model".

`detect_audio_type` returns None for both "definitely not audio" and "could not tell", and
the trainer used it to choose a preprocessing path. A TTS run whose repo could not be read
therefore took the text path and failed much later with "Could not auto-detect format
mapping", which names a column-mapping problem rather than the read that failed.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.training.trainer import _AUDIO_SNIFF_ROWS as _SNIFF_ROWS
from core.training.trainer import UnslothTrainer

_BACKEND = Path(__file__).resolve().parents[1]
_load_and_format_dataset = UnslothTrainer.load_and_format_dataset


class _Dataset:
    column_names = ["audio", "text"]

    def __len__(self):
        return 4


def _trainer(*, audio_type, known, dataset_audio):
    """A stand-in carrying only what load_and_format_dataset reads."""
    errors: list[str] = []
    trainer = SimpleNamespace(
        should_stop = False,
        _audio_type = audio_type,
        _audio_type_known = known,
        _is_dataset_audio = dataset_audio,
        is_audio_vlm = False,
        is_vlm = False,
        model_name = "org/model",
        tokenizer = None,
        errors = errors,
        _update_progress = lambda **kw: errors.append(kw.get("error")) if kw.get("error") else None,
        _resolve_eval_split_from_dataset = lambda dataset: None,
    )
    trainer.load_and_format_dataset = _load_and_format_dataset.__get__(trainer)
    return trainer


def _run(
    trainer,
    monkeypatch,
    dataset = None,
    format_type = "auto",
):
    from hub.utils import dataset_cache

    monkeypatch.setattr(
        dataset_cache, "load_cached_hf_dataset", lambda *a, **k: dataset or _Dataset()
    )
    monkeypatch.setattr(
        "core.training.trainer.format_and_template_dataset",
        lambda dataset, **kw: {"dataset": dataset, "detected_format": "test", "warnings": []},
    )
    monkeypatch.setattr(
        "core.training.trainer.load_dataset",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("no remote access")),
    )
    monkeypatch.setattr(
        sys.modules["datasets"],
        "get_dataset_split_names",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("no remote access")),
    )
    return trainer.load_and_format_dataset(
        "org/dataset",
        format_type = format_type,
        dataset_local_files_only = True,
        dataset_local_path = "/cache/snapshot",
    )


def test_an_inconclusive_probe_on_an_audio_dataset_is_refused(monkeypatch):
    trainer = _trainer(audio_type = None, known = False, dataset_audio = True)
    assert _run(trainer, monkeypatch) is None
    assert trainer.errors, "the run failed without reporting why"
    reported = trainer.errors[0]
    assert "tokenizer_config.json" in reported
    assert "Could not auto-detect format mapping" not in reported


def test_a_definitive_non_audio_model_still_takes_the_text_path(monkeypatch):
    # The common case, unaffected.
    trainer = _trainer(audio_type = None, known = True, dataset_audio = True)
    assert _run(trainer, monkeypatch) is not None
    assert not trainer.errors


def test_an_inconclusive_probe_without_audio_data_is_left_alone(monkeypatch):
    # A text run whose repo happens to be gated must not start failing.
    trainer = _trainer(audio_type = None, known = False, dataset_audio = False)
    assert _run(trainer, monkeypatch) is not None
    assert not trainer.errors


def test_the_trainer_probes_with_the_checked_variant():
    # detect_audio_type collapses "not audio" and "unreadable" into None.
    text = (_BACKEND / "core/training/trainer.py").read_text(encoding = "utf-8")
    assert "detect_audio_type_checked(" in text
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("self._audio_type") and "detect_audio_type" in stripped:
            assert "detect_audio_type_checked(" in stripped, stripped


def test_every_probe_site_records_whether_it_was_definitive():
    # A site that reassigns _audio_type without its flag leaves the flag describing the
    # previous probe, which is worse than not having it.
    lines = (_BACKEND / "core/training/trainer.py").read_text(encoding = "utf-8").splitlines()
    sites = [
        i for i, l in enumerate(lines) if "detect_audio_type_checked(" in l and "import" not in l
    ]
    assert sites, "no probe sites found"
    for i in sites:
        # A window, because the retry site unpacks into locals first so a still-inconclusive
        # answer cannot overwrite a good earlier one.
        window = "\n".join(lines[i : i + 25])
        assert "self._audio_type_known" in window, lines[i].strip()


class _TypedDataset(_Dataset):
    """A dataset with a real `features` schema and rows, as a loaded HF dataset has."""

    def __init__(
        self,
        features,
        rows = None,
    ):
        self.features = features
        self._rows = rows if rows is not None else [{"audio": "hello", "text": "hi"}]

    def __iter__(self):
        return iter(self._rows)


def _audio_features():
    from datasets import Audio, Value
    return {"audio": Audio(sampling_rate = 16000), "text": Value("string")}


def _text_features():
    from datasets import Value
    return {"audio": Value("string"), "text": Value("string")}


def test_a_text_dataset_with_a_column_named_audio_is_not_refused(monkeypatch):
    # is_dataset_audio is true on a column-NAME keyword match alone, never looking at the
    # value, so a text dataset with an `audio` column arrives here flagged as audio.
    # Refusing it would take away a text path that works.
    trainer = _trainer(audio_type = None, known = False, dataset_audio = True)
    result = _run(trainer, monkeypatch, dataset = _TypedDataset(_text_features()))
    assert result is not None, "a text dataset was refused for having an audio-named column"
    assert not trainer.errors


def test_a_real_audio_column_is_still_refused(monkeypatch):
    # The case the guard exists for.
    trainer = _trainer(audio_type = None, known = False, dataset_audio = True)
    assert _run(trainer, monkeypatch, dataset = _TypedDataset(_audio_features())) is None
    assert trainer.errors and "tokenizer_config.json" in trainer.errors[0]


def test_a_dataset_that_cannot_report_a_schema_is_still_refused(monkeypatch):
    # A DatasetDict or iterable dataset has no usable `features`. Unknown is not "no audio
    # here", and an unreadable model probe remains the likelier explanation.
    trainer = _trainer(audio_type = None, known = False, dataset_audio = True)
    assert _run(trainer, monkeypatch, dataset = _Dataset()) is None
    assert trainer.errors and "tokenizer_config.json" in trainer.errors[0]


def test_a_path_backed_audio_column_is_still_refused(monkeypatch):
    # A JSON/CSV audio dataset carries paths as a Value("string") until the preprocessor
    # casts it (_preprocess_snac_dataset does exactly that), so the schema alone would call
    # it textual and hand it to the text path this guard prevents.
    trainer = _trainer(audio_type = None, known = False, dataset_audio = True)
    dataset = _TypedDataset(
        _text_features(), rows = [{"audio": "clips/utt_0001.wav", "text": "hello"}]
    )
    assert _run(trainer, monkeypatch, dataset = dataset) is None
    assert trainer.errors and "tokenizer_config.json" in trainer.errors[0]


def test_a_raw_text_run_is_never_refused(monkeypatch):
    # raw/CPT reads the text column and ignores any audio one by design, so it is not the
    # unsafe audio-to-text fallback this guard targets.
    trainer = _trainer(audio_type = None, known = False, dataset_audio = True)
    _run(
        trainer,
        monkeypatch,
        dataset = _TypedDataset(_audio_features()),
        format_type = "raw",
    )
    # The stub cannot drive the whole raw-text branch, so assert only that the guard did not
    # fire; anything reported past this point is the stub.
    reported = " ".join(trainer.errors)
    assert "tokenizer_config.json" not in reported, f"raw-text run was refused: {reported}"


def test_a_transient_probe_failure_is_rechecked_after_the_tokenizer_loads(monkeypatch):
    # The probe and the tokenizer load are two reads of the same repo, so a timeout or 5xx
    # can leave the probe inconclusive while the download right after it succeeds. Without a
    # recheck the run is refused later claiming an unreadable file that has just been read.
    import core.training.trainer as trainer_mod

    answers = [(None, False), ("whisper", True)]
    calls = []

    def fake_probe(*a, **kw):
        calls.append(a[0] if a else None)
        return answers[min(len(calls) - 1, len(answers) - 1)]

    monkeypatch.setattr(trainer_mod, "detect_audio_type_checked", fake_probe)
    monkeypatch.setattr(trainer_mod, "is_vision_model", lambda *a, **kw: False)

    class _Proc:
        @classmethod
        def from_pretrained(cls, *a, **kw):
            return object()

    import transformers

    monkeypatch.setattr(transformers, "AutoProcessor", _Proc, raising = False)
    monkeypatch.setattr(transformers, "AutoTokenizer", _Proc, raising = False)

    trainer = UnslothTrainer()
    trainer.pre_detect_and_load_tokenizer("org/model", is_dataset_audio = True)

    assert len(calls) == 2, "the probe was not retried after the tokenizer load"
    assert trainer._audio_type == "whisper"
    assert trainer._audio_type_known is True
    assert trainer.is_audio is True


def test_a_probe_that_stays_inconclusive_is_not_overwritten(monkeypatch):
    # A second inconclusive read must leave the flag False so the guard still protects the run.
    import core.training.trainer as trainer_mod

    monkeypatch.setattr(trainer_mod, "detect_audio_type_checked", lambda *a, **kw: (None, False))
    monkeypatch.setattr(trainer_mod, "is_vision_model", lambda *a, **kw: False)

    class _Proc:
        @classmethod
        def from_pretrained(cls, *a, **kw):
            return object()

    import transformers

    monkeypatch.setattr(transformers, "AutoTokenizer", _Proc, raising = False)

    trainer = UnslothTrainer()
    trainer.pre_detect_and_load_tokenizer("org/model", is_dataset_audio = True)
    assert trainer._audio_type_known is False
    assert trainer._audio_type is None


def test_a_null_first_audio_value_does_not_decide_the_dataset(monkeypatch):
    # A JSON/CSV audio dataset can carry a null first value and real paths after it, and the
    # preprocessors skip such a row rather than fail, so row 0 alone would call it textual.
    trainer = _trainer(audio_type = None, known = False, dataset_audio = True)
    dataset = _TypedDataset(
        _text_features(),
        rows = [
            {"audio": None, "text": "hello"},
            {"audio": "", "text": "there"},
            {"audio": "clips/utt_0003.wav", "text": "world"},
        ],
    )
    assert _run(trainer, monkeypatch, dataset = dataset) is None
    assert trainer.errors and "tokenizer_config.json" in trainer.errors[0]


def test_rows_that_offer_no_usable_value_are_unknown_not_textual(monkeypatch):
    # All-null candidate values answer neither way, and unknown must keep refusing.
    trainer = _trainer(audio_type = None, known = False, dataset_audio = True)
    dataset = _TypedDataset(
        _text_features(), rows = [{"audio": None, "text": None}, {"audio": None, "text": None}]
    )
    assert _run(trainer, monkeypatch, dataset = dataset) is None
    assert trainer.errors and "tokenizer_config.json" in trainer.errors[0]


def test_the_retry_reloads_the_processor_when_it_discovers_whisper(monkeypatch):
    # The loader is chosen from the audio type, so a retry that flips the answer to whisper
    # has already stored an AutoTokenizer. _preprocess_whisper_dataset reads
    # .feature_extractor and .tokenizer off the processor, so every sample would be skipped.
    import core.training.trainer as trainer_mod

    answers = [(None, False), ("whisper", True)]
    calls = []

    def fake_probe(*a, **kw):
        calls.append(1)
        return answers[min(len(calls) - 1, len(answers) - 1)]

    monkeypatch.setattr(trainer_mod, "detect_audio_type_checked", fake_probe)
    monkeypatch.setattr(trainer_mod, "is_vision_model", lambda *a, **kw: False)

    loaded: list[str] = []

    class _Processor:
        @classmethod
        def from_pretrained(cls, *a, **kw):
            loaded.append("processor")
            return object()

    class _Tokenizer:
        @classmethod
        def from_pretrained(cls, *a, **kw):
            loaded.append("tokenizer")
            return object()

    import transformers

    monkeypatch.setattr(transformers, "AutoProcessor", _Processor, raising = False)
    monkeypatch.setattr(transformers, "AutoTokenizer", _Tokenizer, raising = False)

    trainer = UnslothTrainer()
    trainer.pre_detect_and_load_tokenizer("org/model", is_dataset_audio = True)

    assert trainer._audio_type == "whisper"
    assert loaded == [
        "tokenizer",
        "processor",
    ], f"expected a processor reload once the retry found whisper, got {loaded}"


def test_the_retry_does_not_reload_when_the_answer_is_unchanged(monkeypatch):
    # A retry that confirms the same type must not pay for a second load.
    import core.training.trainer as trainer_mod

    monkeypatch.setattr(trainer_mod, "detect_audio_type_checked", lambda *a, **kw: (None, True))
    monkeypatch.setattr(trainer_mod, "is_vision_model", lambda *a, **kw: False)

    loaded: list[str] = []

    class _Tokenizer:
        @classmethod
        def from_pretrained(cls, *a, **kw):
            loaded.append("tokenizer")
            return object()

    import transformers

    monkeypatch.setattr(transformers, "AutoTokenizer", _Tokenizer, raising = False)

    trainer = UnslothTrainer()
    trainer.pre_detect_and_load_tokenizer("org/model", is_dataset_audio = True)
    assert loaded == ["tokenizer"], f"tokenizer loaded more than once: {loaded}"


def test_transcripts_are_not_evidence_that_the_audio_column_is_empty(monkeypatch):
    # A path-backed audio dataset has a populated transcript in every row, so all-null audio
    # values in the sniff window answer "unknown", not "textual" on the strength of the text:
    # later rows may hold real paths, and the preprocessors skip the leading bad ones.
    trainer = _trainer(audio_type = None, known = False, dataset_audio = True)
    rows = [{"audio": None, "text": f"transcript {i}"} for i in range(_SNIFF_ROWS + 4)]
    dataset = _TypedDataset(_text_features(), rows = rows)
    assert _run(trainer, monkeypatch, dataset = dataset) is None
    assert trainer.errors and "tokenizer_config.json" in trainer.errors[0]


def test_a_populated_audio_named_column_of_prose_still_answers_textual(monkeypatch):
    # The other half: an audio-NAMED column that really holds prose is the false positive
    # this veto exists for, and it must keep answering False.
    trainer = _trainer(audio_type = None, known = False, dataset_audio = True)
    rows = [{"audio": f"a sentence {i}", "text": "hello"} for i in range(4)]
    dataset = _TypedDataset(_text_features(), rows = rows)
    _run(trainer, monkeypatch, dataset = dataset)
    assert not any(
        "tokenizer_config.json" in e for e in trainer.errors
    ), f"a prose column named audio should not be refused: {trainer.errors}"


def test_the_first_tokenizer_load_failing_still_reaches_the_retry(monkeypatch):
    # Spark-TTS keeps its tokenizer under LLM/ and the subfolder kwarg is chosen from the
    # audio type, so an inconclusive first probe reads the root and raises: a retry placed
    # only after a SUCCESSFUL load can never run for the models whose layout the type decides.
    import core.training.trainer as trainer_mod

    answers = [(None, False), ("bicodec", True)]
    calls = []

    def fake_probe(*a, **kw):
        calls.append(1)
        return answers[min(len(calls) - 1, len(answers) - 1)]

    monkeypatch.setattr(trainer_mod, "detect_audio_type_checked", fake_probe)
    monkeypatch.setattr(trainer_mod, "is_vision_model", lambda *a, **kw: False)
    monkeypatch.setattr(
        trainer_mod,
        "_spark_tts_tokenizer_kwargs",
        lambda audio_type, name: {"subfolder": "LLM"} if audio_type == "bicodec" else {},
    )

    attempts = []

    class _Tokenizer:
        @classmethod
        def from_pretrained(cls, *a, **kw):
            attempts.append(kw.get("subfolder"))
            if kw.get("subfolder") != "LLM":
                raise OSError("Can't load tokenizer for 'org/spark'")
            return object()

    import transformers

    monkeypatch.setattr(transformers, "AutoTokenizer", _Tokenizer, raising = False)

    trainer = UnslothTrainer()
    trainer.pre_detect_and_load_tokenizer("org/spark", is_dataset_audio = True)

    assert attempts == [None, "LLM"], f"expected a retry under LLM/, got {attempts}"
    assert trainer.tokenizer is not None
    assert trainer._audio_type == "bicodec"


def test_a_load_failure_with_a_definitive_type_is_not_swallowed(monkeypatch):
    # The retry must not turn a genuine "this repo has no tokenizer" into a silent pass.
    import core.training.trainer as trainer_mod

    monkeypatch.setattr(
        trainer_mod, "detect_audio_type_checked", lambda *a, **kw: ("bicodec", True)
    )
    monkeypatch.setattr(trainer_mod, "is_vision_model", lambda *a, **kw: False)

    class _Tokenizer:
        @classmethod
        def from_pretrained(cls, *a, **kw):
            raise OSError("Can't load tokenizer for 'org/broken'")

    import transformers

    monkeypatch.setattr(transformers, "AutoTokenizer", _Tokenizer, raising = False)

    trainer = UnslothTrainer()
    with pytest.raises(OSError):
        trainer.pre_detect_and_load_tokenizer("org/broken", is_dataset_audio = True)
