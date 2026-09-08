# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Quieting third-party tqdm bars must not take anything real with it.

The bars themselves carry no signal in a log with no terminal, but three things
ride along with them and have to survive: the export dialog's live Hub upload
progress, the "Applying chat template ... 42%" status the UI derives from the
datasets bar's counter, and an operator's explicit choice.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from loggers import config as log_config  # noqa: E402

_HUB = "HF_HUB_DISABLE_PROGRESS_BARS"


def test_the_default_is_installed_and_marked(monkeypatch):
    monkeypatch.setattr(log_config, "_BARS_RESTORED", False)
    monkeypatch.delenv(_HUB, raising = False)
    monkeypatch.delenv(log_config._PROGRESS_BARS_DEFAULTED, raising = False)
    monkeypatch.delenv("UNSLOTH_STUDIO_ACCESS_LOG_DEDUP_MS", raising = False)
    monkeypatch.delenv("UNSLOTH_STUDIO_ACCESS_LOG_POLL_DEDUP_MS", raising = False)

    log_config.quiet_third_party_progress_bars()

    import os

    assert os.environ[_HUB] == "1"
    assert os.environ[log_config._PROGRESS_BARS_DEFAULTED] == "1"


def test_verbose_leaves_the_bars_alone(monkeypatch):
    # --verbose zeroes both access-log windows and promises everything back; the flag
    # is inherited by the workers, so setting it here would keep them quiet anyway.
    monkeypatch.setattr(log_config, "_BARS_RESTORED", False)
    monkeypatch.delenv(_HUB, raising = False)
    monkeypatch.setenv("UNSLOTH_STUDIO_ACCESS_LOG_DEDUP_MS", "0")
    monkeypatch.setenv("UNSLOTH_STUDIO_ACCESS_LOG_POLL_DEDUP_MS", "0")

    log_config.quiet_third_party_progress_bars()

    import os

    assert _HUB not in os.environ


def test_hugging_face_false_spellings_are_honored(monkeypatch):
    # The Hub reads only 1/ON/YES/TRUE as true, so "off" and "no" ask to keep bars.
    monkeypatch.setattr(log_config, "_BARS_RESTORED", False)
    monkeypatch.delenv("UNSLOTH_STUDIO_ACCESS_LOG_DEDUP_MS", raising = False)
    monkeypatch.delenv("UNSLOTH_STUDIO_ACCESS_LOG_POLL_DEDUP_MS", raising = False)
    for value in ("off", "no", "0", "false", ""):
        monkeypatch.setenv(_HUB, value)
        called = []
        monkeypatch.setattr(log_config, "_silence_datasets_bar_output", lambda: called.append(1))
        log_config.quiet_third_party_progress_bars()
        assert called == [], value


def test_the_hub_is_not_imported_just_to_quiet_it():
    # A worker calls setup_logging BEFORE prepending its transformers sidecar to
    # sys.path; importing the Hub here would cache the base environment's copy.
    code = (
        "import sys; sys.path.insert(0, %r)\n"
        "import os\n"
        "os.environ.pop('HF_HUB_DISABLE_PROGRESS_BARS', None)\n"
        "os.environ.pop('UNSLOTH_STUDIO_ACCESS_LOG_DEDUP_MS', None)\n"
        "os.environ.pop('UNSLOTH_STUDIO_ACCESS_LOG_POLL_DEDUP_MS', None)\n"
        "from loggers.config import quiet_third_party_progress_bars\n"
        "quiet_third_party_progress_bars()\n"
        "print('HUB_IMPORTED' if 'huggingface_hub' in sys.modules else 'HUB_ABSENT')\n"
    ) % str(_BACKEND)
    out = subprocess.run([sys.executable, "-c", code], capture_output = True, text = True, timeout = 300)
    assert "HUB_ABSENT" in out.stdout, out.stdout + out.stderr


def test_allow_progress_bars_only_undoes_our_own_default(monkeypatch):
    monkeypatch.setattr(log_config, "_BARS_RESTORED", False)
    monkeypatch.setenv(_HUB, "1")
    monkeypatch.setenv(log_config._PROGRESS_BARS_DEFAULTED, "1")
    log_config.allow_progress_bars()
    import os

    assert _HUB not in os.environ

    # An operator who set it themselves keeps it.
    monkeypatch.setenv(_HUB, "1")
    monkeypatch.delenv(log_config._PROGRESS_BARS_DEFAULTED, raising = False)
    log_config.allow_progress_bars()
    assert os.environ[_HUB] == "1"


def test_the_export_worker_keeps_its_progress_bars():
    text = (_BACKEND / "core/export/worker.py").read_text(encoding = "utf-8")
    assert "allow_progress_bars()" in text
    assert "quiet_progress_bars = False" in text


def test_the_datasets_bar_keeps_counting_but_writes_nothing(capfd):
    # chat_templates.py polls tqdm._instances for the formatting status, and
    # datasets' own disable_progress_bar() forces tqdm(disable = True), which never
    # registers the bar at all.
    import datasets  # noqa: F401
    from datasets.utils.tqdm import tqdm as ds_bar
    from tqdm.auto import tqdm as base_tqdm

    log_config._silence_datasets_bar_output()
    bar = ds_bar(total = 10, desc = "Applying chat template")
    try:
        bar.update(4)
        instances = [b for b in list(getattr(base_tqdm, "_instances", set())) if b is bar]
        assert instances, "the bar must stay registered for the UI status poller"
        assert instances[0].n == 4
    finally:
        bar.close()
    captured = capfd.readouterr()
    assert "Applying chat template" not in captured.out + captured.err


def test_silencing_the_datasets_bar_twice_is_harmless():
    from datasets.utils.tqdm import tqdm as ds_bar

    log_config._silence_datasets_bar_output()
    first = ds_bar.__init__
    log_config._silence_datasets_bar_output()
    assert ds_bar.__init__ is first


def test_trainer_summary_metrics_are_republished():
    text = (_BACKEND / "core/training/trainer.py").read_text(encoding = "utf-8")
    assert "trainer summary" in text
    for key in ("train_samples_per_second", "train_steps_per_second", "total_flos"):
        assert key in text, key


def test_setup_time_is_never_reported_as_throughput():
    # elapsed_seconds covers imports, the model load and the dataset build, and on a
    # resume the counters predate this process, so the first line reports no rate at
    # all and the second one measures a real in-training interval.
    text = (_BACKEND / "core/training/training.py").read_text(encoding = "utf-8")
    assert "The first logged line reports no throughput on purpose" in text
    assert "_progress_run_resumed" not in text


def test_the_early_dataset_branches_are_covered():
    # The raw-text and audio-VLM branches run their own filter/map and return before
    # the chat-template path, so the suppression has to come before them.
    text = (_BACKEND / "core/training/trainer.py").read_text(encoding = "utf-8")
    quiet_at = text.index("quiet_third_party_progress_bars()")
    assert quiet_at < text.index("# ========== AUDIO MODELS: custom preprocessing ==========")
    assert quiet_at < text.index("# ========== FORMAT FIRST ==========")


def test_the_dataset_load_itself_is_covered():
    # load_dataset() draws "Generating train split" and download/extract bars of its
    # own, on both the local-file and the Hub branch, so the suppression has to come
    # before the first load and not just before the map/filter work that follows it.
    text = (_BACKEND / "core/training/trainer.py").read_text(encoding = "utf-8")
    body = text[text.index("    def load_and_format_dataset(") :]
    assert body.index("quiet_third_party_progress_bars()") < body.index("= load_dataset(")
    # The class-level patch needs datasets in sys.modules, which the module-level
    # import guarantees for every caller of this method.
    assert "\nfrom datasets import Dataset\n" in text


def test_the_diffusion_trainers_quiet_diffusers_once_it_is_imported():
    # diffusers is imported inside the two training entrypoints, not at module level,
    # so the child-process call runs while it is still absent from sys.modules and
    # cannot reach it. The pipeline load that draws "Loading pipeline components..."
    # happens further down the same function.
    entrypoints = {
        "diffusion_lora_trainer.py": "def run_diffusion_lora_training(",
        "diffusion_dit_trainer.py": "def _train_dit(",
    }
    for name, entrypoint in entrypoints.items():
        text = (_BACKEND / "core/training" / name).read_text(encoding = "utf-8")
        body = text[text.index(entrypoint) :]
        assert body.index("from diffusers") < body.index("quiet_third_party_progress_bars()"), name


def test_the_precache_helper_restores_rather_than_enables():
    text = (_BACKEND / "utils/datasets/llm_assist.py").read_text(encoding = "utf-8")
    assert "if not _bars_were_off:" in text
    assert "_bars_were_off = bool(are_progress_bars_disabled())" in text


def test_the_video_loader_quiets_diffusers_too():
    text = (_BACKEND / "core/inference/video.py").read_text(encoding = "utf-8")
    assert "quiet_third_party_progress_bars()" in text


def test_our_own_conversion_bars_are_redirected(monkeypatch):
    monkeypatch.setenv(_HUB, "1")
    assert "file" in log_config.quiet_bar_kwargs()
    monkeypatch.setenv(_HUB, "off")
    assert log_config.quiet_bar_kwargs() == {}
    monkeypatch.delenv(_HUB, raising = False)
    assert log_config.quiet_bar_kwargs() == {}

    text = (_BACKEND / "utils/datasets/format_conversion.py").read_text(encoding = "utf-8")
    assert text.count("**_quiet_bar_kwargs(),") == 2


def test_the_embedding_trainer_is_quiet_too():
    # _run_embedding_training bypasses UnslothTrainer entirely.
    text = (_BACKEND / "core/training/worker.py").read_text(encoding = "utf-8")
    assert '"disable_tqdm": _hf_stdout_progress_disabled(),' in text
    assert "_drop_hf_stdout_callbacks(trainer)" in text


def test_the_diffusion_training_child_quiets_diffusers():
    # That child never runs setup_logging, and diffusers honours no env var.
    text = (_BACKEND / "core/training/diffusion_training_service.py").read_text(encoding = "utf-8")
    assert "quiet_third_party_progress_bars()" in text


def test_embedding_runs_republish_the_trainer_summary():
    text = (_BACKEND / "core/training/worker.py").read_text(encoding = "utf-8")
    body = text[text.index("class _EmbeddingProgressCallback") :]
    assert "trainer summary" in body


def test_evaluation_progress_survives_the_dropped_bar():
    # ProgressCallback's per-batch eval bar was the only sign a long evaluation was
    # moving; the replacement has to publish it as status and a structured line.
    text = (_BACKEND / "core/training/trainer.py").read_text(encoding = "utf-8")
    assert "def on_prediction_step(" in text
    assert '"evaluating"' in text
    assert "Evaluating..." in text


def test_evaluation_progress_is_throttled_and_counts():
    """The throttle from _ProgressCallback.on_prediction_step, in isolation.

    Importing the trainer module pulls in unsloth and torch, so the rule is checked
    the same way the throughput one is.
    """

    def report(
        seen,
        last_report,
        now,
        window = 15.0,
    ):
        return not (last_report and (now - last_report) < window)

    assert report(1, 0.0, 100.0) is True  # first batch always reports
    assert report(2, 100.0, 101.0) is False  # a second later, still quiet
    assert report(900, 100.0, 116.0) is True  # 16s later, one more line


def test_the_embedding_worker_quiets_dataset_bars():
    text = (_BACKEND / "core/training/worker.py").read_text(encoding = "utf-8")
    body = text[text.index("def _run_embedding_training") :]
    assert "quiet_third_party_progress_bars()" in body


def test_evaluation_hands_the_status_back_to_training():
    # An empty status is ignored downstream, so the UI would sit on "Evaluating..."
    # for the rest of the run.
    text = (_BACKEND / "core/training/trainer.py").read_text(encoding = "utf-8")
    on_evaluate = text[text.index("def on_evaluate(") : text.index("def on_prediction_step(")]
    assert "Training in progress..." in on_evaluate


def test_the_training_worker_keeps_its_bars_countable():
    # It polls tqdm._instances to turn the Hub download and "Loading checkpoint shards"
    # bars into the UI status, and a disabled bar is never registered there. The call
    # must also precede setup_logging: huggingface_hub reads the env var once, into a
    # module constant, and refuses to re-enable afterwards.
    text = (_BACKEND / "core/training/worker.py").read_text(encoding = "utf-8")
    assert text.index("keep_progress_bars_countable()") < text.index(
        'service_name = "unsloth-studio-training-worker"'
    )


def test_bars_stay_registered_once_the_worker_takes_them_back(monkeypatch, capfd):
    monkeypatch.setattr(log_config, "_BARS_RESTORED", False)
    monkeypatch.setenv(_HUB, "1")
    monkeypatch.setenv(log_config._PROGRESS_BARS_DEFAULTED, "1")
    from tqdm.auto import tqdm as base_tqdm
    from tqdm.std import tqdm as std_tqdm

    # The redirect patches the shared tqdm class, so put it back for the other tests.
    monkeypatch.setattr(std_tqdm, "__init__", std_tqdm.__init__)
    monkeypatch.setattr(std_tqdm, "_unsloth_every_output_silenced", False, raising = False)

    log_config.keep_progress_bars_countable()
    import os

    assert _HUB not in os.environ
    # A later quieting call must not undo it; the poller would go blind.
    log_config.quiet_third_party_progress_bars()
    assert _HUB not in os.environ

    bar = base_tqdm(total = 10, desc = "model-00002-of-00004.safetensors")
    try:
        bar.update(6)
        assert bar in list(getattr(base_tqdm, "_instances", set()))
        assert bar.n == 6
    finally:
        bar.close()
    captured = capfd.readouterr()
    assert "model-00002-of-00004.safetensors" not in captured.out + captured.err


def test_an_operator_who_turned_bars_off_keeps_them_off(monkeypatch):
    # Only Unsloth's own default is ever taken back.
    monkeypatch.setattr(log_config, "_BARS_RESTORED", False)
    monkeypatch.setenv(_HUB, "1")
    monkeypatch.delenv(log_config._PROGRESS_BARS_DEFAULTED, raising = False)

    log_config.keep_progress_bars_countable()

    import os

    assert os.environ[_HUB] == "1"
    assert log_config._BARS_RESTORED is False


def test_the_shared_dataset_loader_quiets_the_bars_it_just_imported():
    # The server never imports datasets at boot, so setup_logging cannot patch the bar
    # class; this shared entry point is the first place it exists.
    text = (_BACKEND / "utils/datasets/cache_safe.py").read_text(encoding = "utf-8")
    body = text[text.index("def load_dataset_cache_safe") :]
    assert body.index("from datasets import load_dataset") < body.index(
        "quiet_third_party_progress_bars()"
    )
