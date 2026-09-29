# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""What the mechanism does once it is running, not what the gate decided.

Three claims a gate test cannot reach, each a property of real DataLoader worker
processes pulling real rows through the lazy view:

1. the prewarm barrier does not consume rows training then never sees,
2. the loader the barrier filled is the one ``train()`` uses,
3. those workers are gone once training is over.

Asserted against a real ``DataLoader`` with forked workers over a real
``datasets.Dataset``. No model, no GPU, and a stand-in tokenizer: none of these
claims is about tokenization.
"""

import multiprocessing
import sys

import pytest

sys.path.insert(0, "studio/backend")

from utils.datasets.online_tokenization import (  # noqa: E402
    attach_online_tokenization,
    memoize_train_dataloader,
    release_train_dataloader,
)

datasets = pytest.importorskip("datasets")
torch = pytest.importorskip("torch")

from torch.utils.data import DataLoader, RandomSampler  # noqa: E402

WORKERS = 2
PREFETCH = 2
PREWARM = WORKERS * PREFETCH
BATCH = 4
ROWS = 400


class _Tokenizer:
    """Deterministic and module-level, so a forked worker inherits it intact."""

    bos_token = None
    chat_template = ""

    def __call__(
        self,
        texts,
        truncation = True,
        max_length = 8,
        add_special_tokens = True,
    ):
        if isinstance(texts, str):
            texts = [texts]
        return {"input_ids": [[len(t)] * min(len(t), max_length) for t in texts]}


def _collate(rows):
    """Keep the rows as they arrive: the assertions are about WHICH rows."""
    return [tuple(row["input_ids"]) for row in rows]


def _view():
    dataset = datasets.Dataset.from_dict({"text": [f"row {i}" for i in range(ROWS)]})
    return attach_online_tokenization(
        dataset,
        tokenizer = _Tokenizer(),
        text_field = "text",
        max_length = 8,
        add_special_tokens = True,
    )


class _FakeTrainer:
    """Only the surface the mechanism touches: one loader factory, counted.

    Each call builds a new loader, as ``Trainer.get_train_dataloader`` does:
    transformers rebuilds the train loader every time, which is why
    ``memoize_train_dataloader`` exists.
    """

    def __init__(
        self,
        dataset,
        shuffle = False,
    ):
        self.dataset = dataset
        self.shuffle = shuffle
        self.calls = 0

    def get_train_dataloader(self):
        self.calls += 1
        return DataLoader(
            self.dataset,
            batch_size = BATCH,
            sampler = RandomSampler(self.dataset) if self.shuffle else None,
            shuffle = False,
            num_workers = WORKERS,
            prefetch_factor = PREFETCH,
            persistent_workers = True,
            collate_fn = _collate,
        )


def _prewarm(trainer, batches):
    """The barrier from ``UnslothTrainer._preflight_first_batch``, verbatim:
    memoize, pull ``batches`` microbatches, drop the local names. The memo keeps
    the filled workers alive past this function."""
    memoize_train_dataloader(trainer)
    loader = trainer.get_train_dataloader()
    iterator = iter(loader)
    next(iterator)
    for _ in range(max(0, batches - 1)):
        try:
            next(iterator)
        except StopIteration:
            break
    del iterator, loader


def _expected_rows():
    """Every row the view yields, in backing order, as `_collate` renders them."""
    return [tuple([len(f"row {i}")] * min(len(f"row {i}"), 8)) for i in range(ROWS)]


def _take(loader, count):
    taken = []
    for batch in loader:
        taken.append(batch)
        if len(taken) == count:
            break
    return taken


@pytest.fixture(autouse = True)
def _no_leaked_workers():
    """A failing assertion must not leave worker processes behind for the next test."""
    before = set(multiprocessing.active_children())
    yield
    for child in set(multiprocessing.active_children()) - before:
        child.terminate()
        child.join(timeout = 5)


def test_the_prewarm_re_iterates_from_the_start_rather_than_continuing():
    """The barrier pulls microbatches; training must not begin where it stopped.

    A sequential sampler makes it exact: had the prewarm left the iterator where
    it finished, training would start at row 16 and come up ``PREWARM * BATCH``
    rows short.
    """
    trainer = _FakeTrainer(_view())
    _prewarm(trainer, PREWARM)

    pass_batches = list(trainer.get_train_dataloader())
    rows = [row for batch in pass_batches for row in batch]

    assert rows == _expected_rows(), "training did not start from the first row"
    assert len(rows) == ROWS, f"the prewarm swallowed {ROWS - len(rows)} rows"
    release_train_dataloader(trainer)


def test_a_shuffled_pass_after_prewarming_still_covers_every_row():
    """Same claim with the sampler a real run uses: nothing is missing, and
    nothing is served twice to make up the count."""
    torch.manual_seed(0)
    trainer = _FakeTrainer(_view(), shuffle = True)
    _prewarm(trainer, PREWARM)

    rows = [row for batch in trainer.get_train_dataloader() for row in batch]

    assert len(rows) == ROWS
    assert sorted(rows) == sorted(_expected_rows())
    release_train_dataloader(trainer)


def test_train_uses_the_loader_the_barrier_filled():
    """Without the memo the barrier forks workers, fills them, and train()
    throws them away and forks a second set."""
    trainer = _FakeTrainer(_view())
    memoize_train_dataloader(trainer)
    first = trainer.get_train_dataloader()
    _take(first, 1)
    second = trainer.get_train_dataloader()

    assert second is first
    assert trainer.calls == 1, "the underlying factory ran more than once"
    release_train_dataloader(trainer)


def test_the_workers_are_gone_once_training_is_over():
    """Persistent workers survive train() by design, so something has to end
    them; otherwise Unsloth merges, quantizes and exports alongside them."""
    before = len(multiprocessing.active_children())
    trainer = _FakeTrainer(_view())
    _prewarm(trainer, PREWARM)
    _take(trainer.get_train_dataloader(), 3)

    during = len(multiprocessing.active_children())
    assert during == before + WORKERS, "the barrier did not fork the workers"

    released = release_train_dataloader(trainer)

    assert released == WORKERS
    assert len(multiprocessing.active_children()) == before


def test_releasing_puts_the_real_getter_back_and_is_idempotent():
    """It is called from a finally that two paths reach twice, and a trainer
    reused afterwards must rebuild rather than be handed a dead loader."""
    trainer = _FakeTrainer(_view())
    _prewarm(trainer, PREWARM)

    assert release_train_dataloader(trainer) == WORKERS
    assert release_train_dataloader(trainer) == 0
    assert "get_train_dataloader" not in trainer.__dict__
    assert trainer._unsloth_online_memoized is False

    rebuilt = trainer.get_train_dataloader()
    assert trainer.calls == 2
    del rebuilt


def test_a_wrapped_loader_reports_its_workers_once():
    """`accelerator.prepare` returns a wrapper that shares the inner loader's
    iterator, so a walk over both sees one worker set twice. Observed on a real
    run as a count of 8 for 2 workers."""

    class _Wrapper:
        def __init__(self, inner):
            self.base_dataloader = inner
            self._iterator = None

    trainer = _FakeTrainer(_view())
    memoize_train_dataloader(trainer)
    inner = trainer.get_train_dataloader()
    _take(inner, 1)
    wrapper = _Wrapper(inner)
    wrapper._iterator = inner._iterator
    trainer._unsloth_online_loader_cache["loader"] = wrapper

    assert release_train_dataloader(trainer) == WORKERS
    assert inner._iterator is None and wrapper._iterator is None


def test_the_memoized_eval_workers_are_released_too():
    """`dataloader_num_workers` is a TrainingArguments setting, so the eval loader
    forks the same workers and transformers parks it in `_eval_dataloaders`; torch
    keeps its `_iterator` alive after the eval loop drains it, so those workers
    outlive train() just as the train ones do."""
    before = len(multiprocessing.active_children())
    trainer = _FakeTrainer(_view())
    _prewarm(trainer, PREWARM)

    eval_loader = DataLoader(
        _view(),
        batch_size = BATCH,
        num_workers = WORKERS,
        prefetch_factor = PREFETCH,
        persistent_workers = True,
        collate_fn = _collate,
    )
    list(eval_loader)  # the eval loop drains it; torch retains the iterator
    trainer._eval_dataloaders = {"eval": eval_loader}

    assert eval_loader._iterator is not None, "torch dropped the persistent iterator"
    assert len(multiprocessing.active_children()) == before + 2 * WORKERS

    released = release_train_dataloader(trainer)

    assert released == 2 * WORKERS, "the eval loader's workers were left running"
    assert eval_loader._iterator is None
    assert trainer._eval_dataloaders == {}, "the dead loader is still memoized"
    assert len(multiprocessing.active_children()) == before


def test_releasing_a_trainer_that_never_went_online_does_nothing():
    trainer = _FakeTrainer(_view())
    assert release_train_dataloader(trainer) == 0
    assert trainer.calls == 0
