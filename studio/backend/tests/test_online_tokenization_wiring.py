# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""`UnslothTrainer._configure_online_tokenization`: what it changes, and when.

The gate itself is covered by ``test_online_tokenization.py``; this is the
wiring. The method must apply all four parts of the mechanism or leave
``config_args`` and the dataset wrapper exactly as it found them: half-applied is
the dangerous state, since ``skip_prepare_dataset`` without the lazy transform
trains on raw strings. Every degradation path gets a case, driven through the
real method, because "silently takes the old path" is a claim about side effects.
"""

import contextlib
import json
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, "studio/backend")

datasets = pytest.importorskip("datasets")
# Importing the trainer imports torch; a runner without it skips the module
# rather than failing collection.
pytest.importorskip("torch")

from utils.datasets.online_tokenization import MIN_ROWS_FOR_ONLINE  # noqa: E402


_STUBBED: list = []


def _stub_if_missing(name, attrs):
    """Stand in for a dep the CPU-only test job does not install.

    The real one wins whenever it imports, same rule and same ``__spec__ = None``
    (which quiets the trainer's namespace-shadow guard) as
    ``test_training_preflight.py``.
    """
    if name in sys.modules:
        return
    import importlib
    import types
    from unittest.mock import MagicMock

    try:
        importlib.import_module(name)
        return
    except Exception:  # noqa: BLE001
        pass
    module = types.ModuleType(name)
    module.__spec__ = None
    for attr in attrs:
        setattr(module, attr, MagicMock())
    sys.modules[name] = module
    _STUBBED.append(name)


@contextlib.contextmanager
def _stubbed():
    for name, attrs in (
        ("unsloth", ("FastLanguageModel", "FastVisionModel", "is_bfloat16_supported")),
        ("unsloth.chat_templates", ("get_chat_template",)),
        ("trl", ("SFTTrainer", "SFTConfig")),
    ):
        _stub_if_missing(name, attrs)
    try:
        yield
    finally:
        while _STUBBED:
            sys.modules.pop(_STUBBED.pop(), None)


with _stubbed():
    from core.training.trainer import UnslothTrainer  # noqa: E402

_configure = UnslothTrainer._configure_online_tokenization


ROWS = MIN_ROWS_FOR_ONLINE + 5


def _single_process_launch(monkeypatch):
    """Clear every launcher variable, so a run reads as Unsloth's own launch.

    Same helper and same constant tuples as ``test_training_preflight.py``: the
    two must not disagree about what counts as a launcher, or one file starts
    passing on a set of variables the other never clears.
    """
    from core.training.dataset_bounds import WORLD_SIZE_ENV_FILES, WORLD_SIZE_ENV_VARS
    for name in WORLD_SIZE_ENV_VARS + WORLD_SIZE_ENV_FILES:
        monkeypatch.delenv(name, raising = False)


@pytest.fixture(autouse = True)
def _no_ambient_launcher(monkeypatch):
    """Every case in this file starts from a single-process launch.

    The pass count is read out of the environment, so without this a case's
    result depends on whatever the runner's shell happens to export, and on
    whichever earlier test last set one of these. Both were live here: the file
    already sets ``WORLD_SIZE`` in one test, and pytest's monkeypatch undo only
    covers variables a test itself touched.
    """
    _single_process_launch(monkeypatch)


class _Tokenizer:
    bos_token = "<s>"
    chat_template = "{{ messages }}"

    def __call__(
        self,
        texts,
        truncation = True,
        max_length = 8,
        add_special_tokens = True,
    ):
        if isinstance(texts, str):
            texts = [texts]
        return {"input_ids": [[7] * min(len(t), max_length) for t in texts]}


def _dataset(n = ROWS, columns = None):
    data = {"text": [f"row {i}" for i in range(n)]}
    data.update(columns or {})
    return datasets.Dataset.from_dict(data)


def _fake_self(**overrides):
    trainer = SimpleNamespace(
        tokenizer = _Tokenizer(),
        model = SimpleNamespace(),
        is_vlm = False,
        is_audio = False,
        is_audio_vlm = False,
        _cuda_audio_used = False,
        _online_prewarm_batches = 0,
        _online_eval_dataset = None,
    )
    for key, value in overrides.items():
        setattr(trainer, key, value)
    trainer._configure_online_tokenization = _configure.__get__(trainer)
    return trainer


def _config_args(**overrides):
    args = {
        "dataset_text_field": "text",
        "max_seq_length": 2048,
        "packing": False,
        "num_train_epochs": 1,
        "per_device_train_batch_size": 2,
        "gradient_accumulation_steps": 4,
        "dataset_num_proc": 8,
    }
    args.update(overrides)
    return args


def _run(
    monkeypatch,
    *,
    self_overrides = None,
    config_overrides = None,
    wrapper = None,
    eval_dataset = None,
    **call_overrides,
):
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.delenv("UNSLOTH_STUDIO_ONLINE_TOKENIZATION", raising = False)
    # Pin the TRL hook: these cover the wiring, not the runner's TRL version.
    monkeypatch.setattr(
        "utils.datasets.online_tokenization.trl_supports_skip_prepare_dataset",
        lambda: True,
    )
    # Pin the worker count for the same reason: resolve_worker_count sizes itself from
    # CPU affinity and the cgroup quota and returns 0 below MIN_ONLINE_WORKERS, vetoing
    # before the pass count is compared. On a two-core runner every case here asserted
    # about the runner, on the wrong veto reason. The worker gate is covered by
    # test_online_tokenization.py.
    monkeypatch.setattr(
        "utils.datasets.online_tokenization.resolve_worker_count",
        lambda desired = None: 4,
    )
    trainer = _fake_self(**(self_overrides or {}))
    config_args = _config_args(**(config_overrides or {}))
    wrapper = {"dataset": _dataset()} if wrapper is None else wrapper
    kwargs = dict(
        config_args = config_args,
        dataset = wrapper,
        eval_dataset = eval_dataset,
        training_args = {},
        data_collator = None,
        raw_text_mode = False,
        is_deepseek_ocr = False,
    )
    kwargs.update(call_overrides)
    decision = trainer._configure_online_tokenization(**kwargs)
    return decision, config_args, wrapper, trainer


# ------------------------------------------------------------------ applied fully


def test_a_qualifying_run_gets_all_four_parts_of_the_mechanism(monkeypatch):
    decision, config_args, wrapper, trainer = _run(monkeypatch)
    assert decision.enabled, decision.reason
    # 1. lazy view in place of the eager split
    assert wrapper["dataset"].format["type"] == "custom"
    assert "input_ids" in wrapper["dataset"][0]
    # 2. TRL told not to run its own tokenizing map
    assert config_args["dataset_kwargs"] == {"skip_prepare_dataset": True}
    # 3. workers, overlapped with the GPU
    assert config_args["dataloader_num_workers"] >= 2
    assert config_args["dataloader_persistent_workers"] is True
    assert config_args["dataloader_prefetch_factor"] > 0
    # 4. a prewarm depth for _preflight_first_batch to drain
    assert trainer._online_prewarm_batches == decision.prewarm_batches


def test_the_lazy_view_yields_what_the_eager_map_would_have(monkeypatch):
    """Same tokenizer, same truncation, same `add_special_tokens`: the rows the
    collator sees must be identical, or the loss moves."""
    _, config_args, wrapper, trainer = _run(monkeypatch)
    expected = _Tokenizer()(["row 3"], max_length = 2048)["input_ids"][0]
    assert wrapper["dataset"][3]["input_ids"] == expected


def test_an_eval_split_is_transformed_with_the_same_settings(monkeypatch):
    """`skip_prepare_dataset` skips TRL's EVAL preparation too, so an untouched
    eval split would reach the model as raw strings."""
    eval_split = _dataset(64)
    decision, _, _, trainer = _run(monkeypatch, eval_dataset = eval_split)
    assert decision.enabled, decision.reason
    assert trainer._online_eval_dataset is not eval_split
    assert "input_ids" in trainer._online_eval_dataset[0]


def test_the_eval_split_gets_its_own_double_bos_probe(monkeypatch):
    """TRL runs `_prepare_dataset` once per split, so `add_special_tokens` comes
    from each split's own first row; reusing the train answer would shift every
    eval sequence by a token whenever the splits disagree about a leading BOS."""

    class _Recording(_Tokenizer):
        def __init__(self):
            self.seen = []

        def __call__(
            self,
            texts,
            truncation = True,
            max_length = 8,
            add_special_tokens = True,
        ):
            self.seen.append(add_special_tokens)
            return super().__call__(
                texts,
                truncation = truncation,
                max_length = max_length,
                add_special_tokens = add_special_tokens,
            )

    # train rows are plain; the eval split already carries the BOS token.
    eval_split = datasets.Dataset.from_dict(
        {"text": [f"{_Tokenizer.bos_token}row {i}" for i in range(64)]}
    )
    tokenizer = _Recording()
    decision, _, wrapper, trainer = _run(
        monkeypatch,
        self_overrides = {"tokenizer": tokenizer},
        eval_dataset = eval_split,
    )
    assert decision.enabled, decision.reason

    tokenizer.seen.clear()
    wrapper["dataset"][0]
    assert tokenizer.seen == [True], "plain train rows keep the tokenizer's specials"

    tokenizer.seen.clear()
    trainer._online_eval_dataset[0]
    assert tokenizer.seen == [False], "an eval split that already has BOS must not get a second"


# ---------------------------------------------------- degradation: nothing touched


def _assert_untouched(config_args, wrapper, trainer, original):
    assert wrapper["dataset"] is original
    for key in (
        "dataset_kwargs",
        "remove_unused_columns",
        "dataloader_num_workers",
        "dataloader_prefetch_factor",
        "dataloader_persistent_workers",
    ):
        assert key not in config_args, f"{key} leaked onto the eager path"
    assert trainer._online_prewarm_batches == 0


def test_packing_on_takes_the_old_path(monkeypatch):
    original = _dataset()
    decision, config_args, wrapper, trainer = _run(
        monkeypatch, wrapper = {"dataset": original}, config_overrides = {"packing": True}
    )
    assert not decision.enabled and "packing" in decision.reason
    _assert_untouched(config_args, wrapper, trainer, original)


def test_a_streaming_split_takes_the_old_path(monkeypatch):
    stream = _dataset(64).to_iterable_dataset()
    decision, config_args, wrapper, trainer = _run(monkeypatch, wrapper = {"dataset": stream})
    assert not decision.enabled
    _assert_untouched(config_args, wrapper, trainer, stream)


def test_a_vlm_takes_the_old_path(monkeypatch):
    original = _dataset()
    decision, config_args, wrapper, trainer = _run(
        monkeypatch, wrapper = {"dataset": original}, self_overrides = {"is_vlm": True}
    )
    assert not decision.enabled and "multimodal" in decision.reason
    _assert_untouched(config_args, wrapper, trainer, original)


def test_an_already_tokenized_split_takes_the_old_path(monkeypatch):
    original = _dataset(columns = {"input_ids": [[1, 2]] * ROWS})
    decision, config_args, wrapper, trainer = _run(monkeypatch, wrapper = {"dataset": original})
    assert not decision.enabled and "input_ids" in decision.reason
    _assert_untouched(config_args, wrapper, trainer, original)


@pytest.mark.parametrize("platform", ["win32", "darwin"])
def test_windows_and_macos_take_the_old_path(monkeypatch, platform):
    original = _dataset()
    monkeypatch.setattr(sys, "platform", platform)
    monkeypatch.delenv("UNSLOTH_STUDIO_ONLINE_TOKENIZATION", raising = False)
    trainer = _fake_self()
    config_args = _config_args()
    wrapper = {"dataset": original}
    decision = trainer._configure_online_tokenization(
        config_args = config_args,
        dataset = wrapper,
        eval_dataset = None,
        training_args = {},
        data_collator = None,
        raw_text_mode = False,
        is_deepseek_ocr = False,
    )
    assert not decision.enabled and platform in decision.reason
    _assert_untouched(config_args, wrapper, trainer, original)


def test_a_custom_collator_takes_the_old_path(monkeypatch):
    original = _dataset()
    decision, config_args, wrapper, trainer = _run(
        monkeypatch, wrapper = {"dataset": original}, data_collator = object()
    )
    assert not decision.enabled and "collator" in decision.reason
    _assert_untouched(config_args, wrapper, trainer, original)


def test_completion_masking_takes_the_old_path(monkeypatch):
    """`train_on_responses_only` maps and filters the trainer's split, which on
    a lazy view would materialise the whole thing and can drop rows."""
    original = _dataset()
    decision, config_args, wrapper, trainer = _run(
        monkeypatch,
        wrapper = {"dataset": original},
        training_args = {"train_on_completions": True},
    )
    assert not decision.enabled and "completions" in decision.reason
    _assert_untouched(config_args, wrapper, trainer, original)


def test_raw_text_and_cpt_take_the_old_path(monkeypatch):
    original = _dataset()
    decision, *_ = _run(monkeypatch, wrapper = {"dataset": original}, raw_text_mode = True)
    assert not decision.enabled and "raw-text" in decision.reason

    original = _dataset()
    decision, config_args, wrapper, trainer = _run(
        monkeypatch, wrapper = {"dataset": original}, training_args = {"is_cpt": True}
    )
    assert not decision.enabled and "pretraining" in decision.reason
    _assert_untouched(config_args, wrapper, trainer, original)


def test_a_broken_gate_degrades_instead_of_failing_the_run(monkeypatch):
    """The feature is an optimisation. Any unexpected failure in it must cost
    the user speed, never the run."""
    original = _dataset()

    def _explode(**kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr("utils.datasets.online_tokenization.decide_online_tokenization", _explode)
    decision, config_args, wrapper, trainer = _run(monkeypatch, wrapper = {"dataset": original})
    assert not decision.enabled and "boom" in decision.reason
    _assert_untouched(config_args, wrapper, trainer, original)


def test_a_failure_while_attaching_rolls_the_dataset_back(monkeypatch):
    original = _dataset()

    def _explode(dataset, **kwargs):
        raise RuntimeError("attach failed")

    monkeypatch.setattr("utils.datasets.online_tokenization.attach_online_tokenization", _explode)
    decision, config_args, wrapper, trainer = _run(monkeypatch, wrapper = {"dataset": original})
    assert not decision.enabled and "attach failed" in decision.reason
    _assert_untouched(config_args, wrapper, trainer, original)


# ---------------------------------------------------------------- step-capped runs


def test_a_step_cap_is_resolved_into_passes_rather_than_guessed(monkeypatch):
    """`max_steps` alone reads as "unknown length" in the gate; Unsloth knows the
    row count and the microbatch size, so it answers the question here."""
    decision, _, _, _ = _run(monkeypatch, config_overrides = {"max_steps": 30, "num_train_epochs": 1})
    assert decision.enabled, decision.reason


def test_a_step_cap_that_exceeds_one_pass_takes_the_old_path(monkeypatch):
    original = _dataset()
    # 100_000 steps x 2 x 4 = 800k rows over a 10_005-row split: 80 passes.
    decision, config_args, wrapper, trainer = _run(
        monkeypatch,
        wrapper = {"dataset": original},
        config_overrides = {"max_steps": 100_000},
    )
    assert not decision.enabled and "one pass" in decision.reason
    _assert_untouched(config_args, wrapper, trainer, original)


def test_world_size_scales_the_rows_a_step_consumes(monkeypatch):
    """DDP consumes `batch x accum x world_size` rows per step, so ignoring the
    rank count would call a multi-pass run single-pass."""
    monkeypatch.setenv("WORLD_SIZE", "8")
    original = _dataset()
    decision, config_args, wrapper, trainer = _run(
        monkeypatch,
        wrapper = {"dataset": original},
        config_overrides = {"max_steps": 200},
    )
    assert not decision.enabled and "one pass" in decision.reason
    _assert_untouched(config_args, wrapper, trainer, original)


# 200 steps x 2 x 4 = 1600 rows per replica over a 10_005-row split: 0.16 passes on
# one process, 1.28 on eight. Every launcher below advertises the same eight, so these
# cases differ from the control at the bottom only in whether the variable is read.
_EIGHT_RANK_STEPS = 200


def _eight_ranks(
    monkeypatch,
    expect_enabled = False,
    **env,
):
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    original = _dataset()
    decision, config_args, wrapper, trainer = _run(
        monkeypatch,
        wrapper = {"dataset": original},
        config_overrides = {"max_steps": _EIGHT_RANK_STEPS},
    )
    if expect_enabled:
        assert decision.enabled, decision.reason
        return decision
    assert not decision.enabled and "one pass" in decision.reason, decision.reason
    _assert_untouched(config_args, wrapper, trainer, original)
    return decision


def test_an_mpirun_launch_scales_the_rows_a_step_consumes(monkeypatch):
    """mpirun never sets WORLD_SIZE. Reading that one variable alone calls an
    eight-rank run single-process and engages a view that re-tokenizes on every
    extra pass."""
    _eight_ranks(monkeypatch, OMPI_COMM_WORLD_SIZE = "8")


def test_a_per_node_torchrun_scales_the_rows_a_step_consumes(monkeypatch):
    """torchrun sets WORLD_SIZE and LOCAL_WORLD_SIZE both, so this is the defensive
    case: an environment that kept the per-node count and lost the global one still
    has to be counted rather than read as a single process."""
    _eight_ranks(monkeypatch, LOCAL_WORLD_SIZE = "8")


def test_an_mlx_hostfile_scales_the_rows_a_step_consumes(monkeypatch, tmp_path):
    """mlx.launch's ring backend advertises its ranks as a JSON file rather than a
    number; its NCCL backend is CUDA-only, so this path is reachable.

    Written in the shape the ring backend really uses: the outer list has one entry
    per rank, and each entry is that rank's own list of addresses, because a pair of
    peers may hold several connections."""
    hostfile = tmp_path / "hosts.json"
    hostfile.write_text(
        json.dumps([[f"10.0.0.{i}:9000", f"10.0.0.{i}:9001"] for i in range(8)]),
        encoding = "utf-8",
    )
    _eight_ranks(monkeypatch, MLX_HOSTFILE = str(hostfile))


def test_an_inline_hosts_payload_scales_the_rows_a_step_consumes(monkeypatch):
    """The same variable also carries the payload inline, in the {"hosts": [...]}
    object form `unsloth_cli/_inference.py` accepts."""
    payload = json.dumps({"hosts": [f"10.0.0.{i}:9000" for i in range(8)]})
    _eight_ranks(monkeypatch, MLX_HOSTFILE = payload)


# Only values that RAISE: the old max(1, int(...)) already answered 1 for "0" and
# "-4", so those would pass against the bug and belong in dataset_bounds' own tests.
@pytest.mark.parametrize("junk", ["auto", "", "eight"])
def test_a_junk_world_size_no_longer_disables_online_tokenization(monkeypatch, junk):
    """The direction this used to fail in was not the obvious one.

    `int("auto")` raises, the enclosing `except` leaves the pass count unresolved,
    and an unresolved step-capped run reads as infinite passes, so a launcher that
    exported a non-numeric WORLD_SIZE silently turned the feature OFF on a run that
    qualifies. Unusable values are a single process, which is what this host is.
    """
    monkeypatch.setenv("WORLD_SIZE", junk)
    original = _dataset()
    decision, config_args, wrapper, trainer = _run(
        monkeypatch,
        wrapper = {"dataset": original},
        config_overrides = {"max_steps": 30},
    )
    assert decision.enabled, decision.reason
    assert wrapper["dataset"] is not original
    assert config_args["dataset_kwargs"] == {"skip_prepare_dataset": True}


def test_the_resolved_pass_count_handed_to_the_gate_is_the_arithmetic(monkeypatch):
    """The veto only sees a number, so assert the number rather than its verdict:
    a wrong world size that still lands on the same side of 1.0 is a bug that has
    not surfaced yet."""
    monkeypatch.setenv("OMPI_COMM_WORLD_SIZE", "8")
    seen: dict = {}
    import utils.datasets.online_tokenization as online_mod

    real = online_mod.decide_online_tokenization

    def _record(**kwargs):
        seen.update(kwargs)
        return real(**kwargs)

    monkeypatch.setattr(online_mod, "decide_online_tokenization", _record)
    _run(
        monkeypatch,
        config_overrides = {"max_steps": _EIGHT_RANK_STEPS},
    )
    expected = (_EIGHT_RANK_STEPS * 2 * 4 * 8) / ROWS
    assert seen["resolved_max_steps_epochs"] == pytest.approx(expected)
    assert seen["resolved_max_steps_epochs"] > 1.0


def _captured_logger(monkeypatch):
    """Record what the trainer logs, without depending on the logging config."""
    lines: list = []

    class _Recorder:
        def info(self, message, *args, **kwargs):
            lines.append(str(message))

        warning = error = debug = info

    import core.training.trainer as trainer_mod

    monkeypatch.setattr(trainer_mod, "logger", _Recorder())
    return lines


def test_a_multi_rank_launch_names_the_variable_that_claimed_the_ranks(monkeypatch):
    """A size variable left behind by an earlier mpirun, or inherited from an
    interactive srun, reads here as a multi-rank launch on a machine running one
    process, and its whole visible effect is this run being told it makes several
    passes. Name the variable so that verdict is not silent. The merged row bound
    reads the same variables, so the environment is trusted either way."""
    lines = _captured_logger(monkeypatch)
    _eight_ranks(monkeypatch, OMPI_COMM_WORLD_SIZE = "8")
    reported = [line for line in lines if "data-parallel processes" in line]
    assert len(reported) == 1, lines
    assert "8 data-parallel processes" in reported[0]
    assert "OMPI_COMM_WORLD_SIZE=8" in reported[0]


def test_a_single_process_launch_says_nothing_about_launchers(monkeypatch):
    """The report is for the surprising case only; a normal run must not grow a
    line about a world size of one."""
    lines = _captured_logger(monkeypatch)
    _eight_ranks(monkeypatch, expect_enabled = True)
    assert not [line for line in lines if "data-parallel processes" in line], lines


def test_a_single_process_launch_still_qualifies(monkeypatch):
    """The control for every case above: same steps, same split, no launcher
    variable at all. Counting a rank that is not there would veto this run."""
    decision = _eight_ranks(monkeypatch, expect_enabled = True)
    assert decision.prewarm_batches > 0


# ------------------------------------------------------------- the prewarm barrier


def _preflight_self(loader_calls, batches):
    from utils.datasets.online_tokenization import memoize_train_dataloader  # noqa: F401

    class _Loader:
        def __init__(self):
            self.iterations = 0

        def __iter__(self):
            self.iterations += 1
            return iter(batches)

    class _Inner:
        def __init__(self):
            self.loader = _Loader()

        def get_train_dataloader(self):
            loader_calls.append(1)
            return self.loader

    trainer = SimpleNamespace(
        trainer = _Inner(),
        model_name = "org/model",
        tokenizer = None,
        _online_prewarm_batches = 0,
    )
    trainer._preflight_first_batch = UnslothTrainer._preflight_first_batch.__get__(trainer)
    trainer._chat_template_renders_empty = UnslothTrainer._chat_template_renders_empty.__get__(
        trainer
    )
    return trainer


def test_the_eager_path_still_pulls_exactly_one_batch():
    """No prewarm depth means today's behaviour, unchanged."""
    import torch

    batch = {"input_ids": torch.ones(1, 4, dtype = torch.long)}
    calls: list = []
    trainer = _preflight_self(calls, [batch, batch, batch])
    assert trainer._preflight_first_batch() is None
    assert len(calls) == 1
    assert not getattr(trainer.trainer, "_unsloth_online_memoized", False)


def test_the_prewarm_drains_the_requested_depth_and_keeps_the_loader():
    import torch

    batch = {"input_ids": torch.ones(1, 4, dtype = torch.long)}
    calls: list = []
    trainer = _preflight_self(calls, [batch] * 32)
    trainer._online_prewarm_batches = 16
    assert trainer._preflight_first_batch() is None
    # The memo is what makes the barrier mean anything: transformers rebuilds the
    # train loader every call, so without it train() forks a second worker set.
    assert trainer.trainer._unsloth_online_memoized is True
    assert trainer.trainer.get_train_dataloader() is trainer.trainer.loader
    assert len(calls) == 1


def test_a_short_split_prewarms_fewer_batches_rather_than_failing():
    import torch

    batch = {"input_ids": torch.ones(1, 4, dtype = torch.long)}
    trainer = _preflight_self([], [batch, batch])
    trainer._online_prewarm_batches = 16
    assert trainer._preflight_first_batch() is None


def test_an_empty_split_still_reports_the_no_rows_error():
    trainer = _preflight_self([], [])
    trainer._online_prewarm_batches = 16
    error = trainer._preflight_first_batch()
    assert error and "no training rows" in error
