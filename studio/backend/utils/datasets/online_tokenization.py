# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Online (overlapped) dataset tokenization for the plain-text SFT path.

Studio's default text pipeline tokenizes the whole split up front: TRL's
``_prepare_dataset`` runs a ``.map()`` over every row before ``train()`` is
allowed to begin.  That map is the single largest fixed cost of starting a run
-- 97s of the 106s of preparation measured on 100k rows of OpenMathReasoning
with ``dataset_num_proc = 8`` -- and it buys nothing that could not be done
while the GPU is already busy.

This module moves it into the DataLoader workers.  Four pieces, all of which
are needed together:

1. ``datasets.Dataset.with_transform`` attaches a per-batch tokenizer that runs
   on ``__getitem__``.  It returns an immutable *view*; ``set_transform`` would
   mutate the caller's object, which the preview/eval code also holds.
2. TRL is handed ``dataset_kwargs = {"skip_prepare_dataset": True}`` so it does
   not run its own tokenizing map over the view (which would materialise
   exactly the pass we are trying to avoid).  Studio already uses that hook for
   the VLM branch.
3. ``dataloader_num_workers`` > 0 with a prefetch factor and persistent workers,
   so the tokenizer runs in worker processes overlapped with the GPU.
4. A prewarm barrier pulls ``max(grad_accum, workers * prefetch)`` microbatches
   through the pipeline before ``train()``, because plain prefetch does not
   promise that the first ``__next__`` is ready.

The transform reproduces ``unsloth_zoo.dataset_utils.sft_prepare_dataset``'s
tokenize step exactly -- same truncation, same ``max_length``, same double-BOS
rule -- so the rows the model sees are byte-identical to the eager path.  Any
configuration where that equivalence is not provable takes the eager path
unchanged; see :func:`decide_online_tokenization`.

Two costs are worth stating rather than discovering.  The pass gate counts
passes over the TRAIN split only: an eval split is re-tokenized on every
evaluation, because a lazy view tokenizes on each ``__getitem__``, where the
eager map tokenized it once.  Studio's eval splits are small enough that this
has not been worth a gate of its own, but it is a real cost that scales with
``eval_steps``.  And the worker processes are persistent by design -- the
barrier's workers have to survive into ``train()`` -- so they must be shut down
explicitly when training ends; see :func:`release_train_dataloader`.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Any, Optional

from loggers import get_logger

logger = get_logger(__name__)


# Below this many rows the eager tokenize map costs a few seconds and the win
# does not pay for four worker processes.  10k is the smallest size the A/B was
# measured at (time to first step 23.1s -> 12.1s), so it is the smallest size
# the win is established for.
MIN_ROWS_FOR_ONLINE = 10_000

# Measured ceiling, not a guess: four workers stayed ahead of a B200 training a
# 0.6B model, and more workers only add fork cost and RAM.
MAX_ONLINE_WORKERS = 4

# Fewer than this and the tokenizer cannot stay ahead of the GPU, so the lazy
# view would show up as slower steps instead of a faster start.
MIN_ONLINE_WORKERS = 2

DEFAULT_PREFETCH_FACTOR = 4

ENV_FLAG = "UNSLOTH_STUDIO_ONLINE_TOKENIZATION"

# Columns whose presence means the split is already tokenized (or is a
# prompt/completion split, which the zoo tokenizes with a different function).
_PRETOKENIZED_COLUMNS = ("input_ids", "labels", "prompt", "completion")

# Set by :func:`attach_online_tokenization` on the view it returns, and read by
# unsloth's `max_length` enforcement scan as an attestation that every row is
# already truncated to that width.  Without it the scan reads every row of a
# lazy split, which is the eager pass again.
TRUNCATION_ATTESTATION_ATTR = "_unsloth_truncated_to"


@dataclass(frozen = True)
class OnlineTokenizationDecision:
    """Whether this run takes the online path, and with what settings.

    ``enabled`` False always means "behave exactly as before"; ``reason`` names
    the single gate that decided it, for the training log.
    """

    enabled: bool
    reason: str
    workers: int = 0
    prefetch_factor: int = 0
    prewarm_batches: int = 0
    checks: tuple = field(default = ())

    def as_log_line(self) -> str:
        if not self.enabled:
            return f"Online tokenization: off ({self.reason})"
        return (
            f"Online tokenization: on ({self.reason}); "
            f"workers={self.workers}, prefetch={self.prefetch_factor}, "
            f"prewarm={self.prewarm_batches} microbatches"
        )


def env_override() -> Optional[bool]:
    """``UNSLOTH_STUDIO_ONLINE_TOKENIZATION``: 0/false forces off, 1/true forces on.

    Unset (the normal case) returns None and the gates below decide.  "Forces
    on" only removes the *heuristic* gates (row count, epoch count); the
    correctness gates are never overridden, because taking the lazy path on a
    VLM or a pre-tokenized split does not train differently, it fails.
    """
    raw = os.environ.get(ENV_FLAG)
    if raw is None:
        return None
    raw = raw.strip().lower()
    if raw in ("0", "false", "no", "off"):
        return False
    if raw in ("1", "true", "yes", "on"):
        return True
    return None


def dataloader_worker_start_method() -> Optional[str]:
    """How DataLoader workers will actually start, read without fixing it.

    ``multiprocessing.get_start_method()`` with no argument RESOLVES the default
    and pins the context, after which a later ``set_start_method()`` raises --
    unsloth's own ``dataset_num_proc`` avoids it for that reason.  So: the
    explicitly set method if there is one, else the platform default, which is
    the first entry of ``get_all_start_methods()`` and costs nothing to read.
    """
    try:
        import multiprocessing

        explicit = multiprocessing.get_start_method(allow_none = True)
        if explicit:
            return explicit
        methods = multiprocessing.get_all_start_methods()
        return methods[0] if methods else None
    except Exception:  # noqa: BLE001 - unreadable reads as "not fork"
        return None


def platform_supports_dataloader_workers() -> bool:
    """Fork, and only fork.

    The hazard is not the operating system, it is ``spawn``: a spawned worker
    re-imports the entry point against a fresh ``sys.path``, and Studio's is
    modified in-process, so the import fails (the same reason ``trainer.py``
    already forces ``dataloader_num_workers = 0`` on Windows and macOS).  Those
    two platforms default to spawn, which is why they named this gate, but a
    Linux process whose start method has been set to ``spawn`` or
    ``forkserver`` is the identical hazard and a platform check cannot see it.
    """
    if sys.platform in ("win32", "darwin"):
        return False
    return dataloader_worker_start_method() == "fork"


def trl_supports_skip_prepare_dataset() -> bool:
    """Feature-detect the ``skip_prepare_dataset`` hook rather than assume it.

    Two independent signals: ``SFTConfig`` must carry a ``dataset_kwargs``
    field, and ``SFTTrainer.__init__`` must actually read the key.  If the
    source cannot be read (a compiled or patched build), the field alone
    decides -- Studio's VLM branch has depended on this hook across every
    supported TRL, so a missing source is not evidence of a missing hook.
    """
    try:
        import dataclasses

        from trl import SFTConfig, SFTTrainer
    except Exception:  # noqa: BLE001 - no TRL means no SFT run at all
        return False

    try:
        names = {f.name for f in dataclasses.fields(SFTConfig)}
    except Exception:  # noqa: BLE001
        names = set(getattr(SFTConfig, "__annotations__", {}) or {})
    if "dataset_kwargs" not in names:
        return False

    try:
        import inspect
        source = inspect.getsource(SFTTrainer.__init__)
    except Exception:  # noqa: BLE001
        return True
    return "skip_prepare_dataset" in source


def dataset_supports_with_transform(dataset: Any) -> bool:
    """A map-style ``datasets.Dataset`` with the lazy-view API.

    Explicitly not a duck-typed ``hasattr`` check: ``IterableDataset`` also has
    ``with_transform`` in recent ``datasets``, and a stream is exactly the case
    this feature must not touch.
    """
    try:
        from datasets import Dataset as HfDataset
        from datasets import IterableDataset as HfIterableDataset
    except Exception:  # noqa: BLE001
        return False
    if isinstance(dataset, HfIterableDataset):
        return False
    if not isinstance(dataset, HfDataset):
        return False
    return callable(getattr(dataset, "with_transform", None))


def is_processor(processing_class: Any) -> bool:
    """True for a multimodal processor rather than a plain tokenizer.

    ``hasattr(x, "tokenizer")`` is the same test ``sft_prepare_dataset`` uses,
    with the ``ProcessorMixin`` isinstance as the primary.
    """
    try:
        from transformers import ProcessorMixin
        if isinstance(processing_class, ProcessorMixin):
            return True
    except Exception:  # noqa: BLE001
        pass
    return hasattr(processing_class, "tokenizer")


def model_needs_token_type_ids(model: Any, processing_class: Any) -> bool:
    """Mirror of the zoo's ``_needs_token_type_ids`` probe.

    Gemma-family modelling modules build their causal mask from
    ``token_type_ids``, so the zoo's tokenize call asks for them.  Rather than
    reproduce that column lazily and hope the two agree, the online path simply
    declines those models and they keep today's eager behaviour.
    """
    marker = "create_" + "causal_mask_mapping"
    try:
        candidates = [model, getattr(model, "model", None)]
        for candidate in candidates:
            if candidate is None:
                continue
            module = sys.modules.get(type(candidate).__module__)
            if module is not None and hasattr(module, marker):
                return True
    except Exception:  # noqa: BLE001
        return True  # unprobeable reads as "needs them", i.e. stay eager

    try:
        for base in type(processing_class).__mro__:
            base_module = getattr(base, "__module__", "") or ""
            if "transformers.models." not in base_module:
                continue
            modelling = base_module.replace(".processing_", ".modeling_")
            module = sys.modules.get(modelling)
            if module is not None and hasattr(module, marker):
                return True
    except Exception:  # noqa: BLE001
        return True
    return False


def dataset_column_names(dataset: Any) -> tuple:
    """Backing column names, or () when the split cannot answer."""
    names = getattr(dataset, "column_names", None)
    if isinstance(names, dict):
        return tuple({c for value in names.values() for c in (value or [])})
    if names is None:
        return ()
    return tuple(names)


def text_column_defect(dataset: Any, text_field: str) -> Optional[str]:
    """Why ``text_field`` cannot be tokenized lazily, or None when it can.

    The eager map reads every row inside the trainer constructor, so a null or a
    non-string row fails there, in seconds, before anything else has happened.
    The lazy view reads a row only when the sampler draws it, so the same data
    fails at whatever step that turns out to be -- possibly hours in, with the
    run's checkpoints and its optimizer state behind it.  That is the one way
    this feature can make a failing run worse rather than slower, so the shapes
    that cause it are refused up front.

    Both checks are metadata, not rows.  The dtype comes off the schema, and
    Arrow tracks ``null_count`` per chunk, so neither one reads or tokenizes
    anything.  A ``select``ed split keeps the full backing table, so its null
    count covers rows the view does not contain: that over-reports, which vetoes
    a split that might have been fine, and never the other way round.
    """
    try:
        from datasets import Value
        features = getattr(dataset, "features", None) or {}
        feature = features.get(text_field)
    except Exception:  # noqa: BLE001 - unreadable schema stays eager
        return f"the type of '{text_field}' could not be read"

    if not isinstance(feature, Value) or feature.dtype not in ("string", "large_string"):
        described = getattr(feature, "dtype", None) or type(feature).__name__
        return f"'{text_field}' holds {described}, not strings"

    try:
        nulls = int(dataset.data.column(text_field).null_count)
    except Exception:  # noqa: BLE001
        return f"'{text_field}' could not be checked for null rows"
    if nulls > 0:
        return f"'{text_field}' has {nulls:,} null row{'' if nulls == 1 else 's'}"
    return None


def resolve_worker_count(desired: Optional[int] = None) -> int:
    """How many DataLoader workers this host can spare, 0 for "do not".

    Sized from the same shared policy that sizes ``dataset_num_proc`` (CPU
    affinity and cgroup quota, not raw ``os.cpu_count()``), then capped at
    :data:`MAX_ONLINE_WORKERS`.
    """
    if not platform_supports_dataloader_workers():
        return 0
    try:
        from utils.hardware import dataset_map_num_proc
        available = dataset_map_num_proc(desired, serial_as_none = True)
    except Exception:  # noqa: BLE001
        available = None
    if not available or available < MIN_ONLINE_WORKERS:
        return 0
    return int(min(available, MAX_ONLINE_WORKERS))


def prewarm_batch_count(grad_accum: int, workers: int, prefetch_factor: int) -> int:
    """Microbatches to pull before ``train()``.

    ``grad_accum`` because step 1 is not complete until that many have landed,
    and ``workers * prefetch_factor`` because that is the depth the DataLoader
    keeps in flight -- filling it is what makes the steady state steady.
    """
    return max(1, int(grad_accum or 1), int(workers or 0) * int(prefetch_factor or 0))


def _epoch_count(num_train_epochs: Optional[float], max_steps: Optional[int]) -> float:
    """Epochs this run will actually perform.

    ``max_steps > 0`` wins over ``num_train_epochs`` in both TRL and
    transformers, and a step-capped run cannot be assumed to be one epoch, so it
    is reported as unknown (``float("inf")``) unless the caller resolved it.
    """
    if max_steps and int(max_steps) > 0:
        return float("inf")
    try:
        return float(num_train_epochs if num_train_epochs is not None else 1.0)
    except (TypeError, ValueError):
        return float("inf")


def decide_online_tokenization(
    *,
    dataset: Any,
    eval_dataset: Any = None,
    processing_class: Any = None,
    model: Any = None,
    text_field: str = "text",
    packing: bool = False,
    is_vlm: bool = False,
    is_audio: bool = False,
    is_audio_vlm: bool = False,
    is_deepseek_ocr: bool = False,
    is_cpt: bool = False,
    raw_text_mode: bool = False,
    has_custom_collator: bool = False,
    train_on_completions: bool = False,
    dataset_streaming: bool = False,
    num_train_epochs: Optional[float] = 1.0,
    max_steps: Optional[int] = 0,
    grad_accum: int = 1,
    row_count: Optional[int] = None,
    workers: Optional[int] = None,
    prefetch_factor: int = DEFAULT_PREFETCH_FACTOR,
    resolved_max_steps_epochs: Optional[float] = None,
) -> OnlineTokenizationDecision:
    """Decide whether this run may tokenize online.  Pure, GPU-free, testable.

    Every gate is a veto.  The order is correctness first, then cost: a caller
    reading the log wants "off (VLM)" rather than "off (dataset too small)" when
    both are true.
    """
    checks: list = []

    def veto(reason: str) -> OnlineTokenizationDecision:
        checks.append((reason, False))
        return OnlineTokenizationDecision(enabled = False, reason = reason, checks = tuple(checks))

    override = env_override()
    if override is False:
        return veto(f"{ENV_FLAG}=0")

    # ---- correctness gates: never overridable ----
    if not platform_supports_dataloader_workers():
        if sys.platform in ("win32", "darwin"):
            return veto(f"{sys.platform} spawns DataLoader workers")
        return veto(
            f"DataLoader workers would start by "
            f"{dataloader_worker_start_method() or 'an unknown method'}, not fork"
        )
    if not trl_supports_skip_prepare_dataset():
        return veto("this TRL has no skip_prepare_dataset hook")
    if is_vlm or is_audio_vlm or is_deepseek_ocr:
        return veto("multimodal model")
    if is_audio:
        return veto("audio model")
    if is_cpt:
        return veto("continued pretraining")
    if raw_text_mode:
        return veto("raw-text mode")
    if has_custom_collator:
        return veto("custom data collator")
    if packing:
        return veto("packing enabled")
    if train_on_completions:
        return veto("train on completions")
    if dataset_streaming:
        return veto("streaming dataset")
    if not dataset_supports_with_transform(dataset):
        return veto("dataset is not a map-style datasets.Dataset")
    if processing_class is None or is_processor(processing_class):
        return veto("processor rather than a plain tokenizer")
    if not callable(processing_class):
        return veto("tokenizer is not callable")
    if model_needs_token_type_ids(model, processing_class):
        return veto("model needs token_type_ids")

    columns = dataset_column_names(dataset)
    if text_field not in columns:
        return veto(f"no '{text_field}' column to tokenize")
    already = [c for c in _PRETOKENIZED_COLUMNS if c in columns]
    if already:
        return veto(f"dataset already carries {already[0]}")
    defect = text_column_defect(dataset, text_field)
    if defect is not None:
        return veto(defect)

    if eval_dataset is not None:
        if not dataset_supports_with_transform(eval_dataset):
            return veto("eval split is not a map-style datasets.Dataset")
        eval_columns = dataset_column_names(eval_dataset)
        if text_field not in eval_columns:
            return veto(f"eval split has no '{text_field}' column")
        if any(c in eval_columns for c in _PRETOKENIZED_COLUMNS):
            return veto("eval split is already tokenized")
        eval_defect = text_column_defect(eval_dataset, text_field)
        if eval_defect is not None:
            return veto(f"eval split: {eval_defect}")

    resolved_workers = resolve_worker_count() if workers is None else int(workers)
    if resolved_workers < MIN_ONLINE_WORKERS:
        return veto("not enough CPU workers to stay ahead of the GPU")
    checks.append(("correctness gates", True))

    # ---- cost gates: the escape hatch may override these ----
    forced = override is True

    if row_count is None:
        try:
            row_count = len(dataset)
        except Exception:  # noqa: BLE001
            row_count = None
    if not forced and (row_count is None or row_count < MIN_ROWS_FOR_ONLINE):
        return veto(f"dataset smaller than {MIN_ROWS_FOR_ONLINE:,} rows")

    epochs = (
        float(resolved_max_steps_epochs)
        if resolved_max_steps_epochs is not None
        else _epoch_count(num_train_epochs, max_steps)
    )
    # The lazy view re-tokenizes on every pass, and the measured cost of the
    # online arm over 2.4 epochs was +2.9% of steady-state training time
    # (237.2s eager vs 244.1s online, identical loss).  One pass pays that once
    # against a 97s tokenize map; a multi-epoch run pays it again per epoch
    # while the saving stays fixed, so anything past a single pass keeps Arrow.
    if not forced and epochs > 1.0:
        detail = (
            "step-capped run of unknown length"
            if epochs == float("inf")
            else (f"{epochs:g} epochs")
        )
        return veto(f"more than one pass over the data ({detail})")

    checks.append(("cost gates", True))
    prewarm = prewarm_batch_count(grad_accum, resolved_workers, prefetch_factor)
    reason = "forced by " + ENV_FLAG if forced else "plain-text single-pass SFT run"
    return OnlineTokenizationDecision(
        enabled = True,
        reason = reason,
        workers = resolved_workers,
        prefetch_factor = int(prefetch_factor),
        prewarm_batches = prewarm,
        checks = tuple(checks),
    )


def resolve_add_special_tokens(processing_class: Any, sample_text: Optional[str]) -> bool:
    """The zoo's double-BOS rule, reproduced exactly.

    ``sft_prepare_dataset`` turns ``add_special_tokens`` off when the rendered
    text already begins with the BOS token, or when the chat template emits one.
    Getting this wrong shifts every row by one token, so it is copied rather
    than re-derived.
    """
    tokenizer = getattr(processing_class, "tokenizer", None)
    chat_template = getattr(processing_class, "chat_template", "") or ""
    if not chat_template and tokenizer is not None:
        chat_template = getattr(tokenizer, "chat_template", "") or ""

    bos_token = getattr(processing_class, "bos_token", None) or getattr(
        tokenizer, "bos_token", None
    )
    if bos_token is None:
        return True
    if isinstance(sample_text, (list, tuple)):
        sample_text = sample_text[0] if sample_text else None
    if sample_text is not None and str(sample_text).startswith(bos_token):
        return False
    if bos_token in chat_template:
        return False
    return True


def build_tokenizing_transform(
    tokenizer: Any, text_field: str, max_length: int, add_special_tokens: bool
):
    """A batched ``with_transform`` callable equivalent to the zoo's ``_tokenize``.

    ``with_transform`` hands the callable a dict of column *lists* and expects
    the same row count back, so the whole batch is encoded in one call -- the
    same batched encode the eager map does, only deferred to ``__getitem__``.

    Whatever the tokenizer returns is passed through, not just ``input_ids``.
    The eager map keeps the tokenizer's whole output (``remove_columns`` drops
    only the ORIGINAL columns), so a batch here carries ``attention_mask``
    exactly as it does there.  Returning a narrower schema is not free: the
    collator and the attention dispatcher both branch on which keys are
    present, and an A/B that differs in the keys is not measuring the same run.
    """

    def transform(batch: dict) -> dict:
        texts = batch[text_field]
        encoded = tokenizer(
            texts,
            truncation = True,
            max_length = max_length,
            add_special_tokens = add_special_tokens,
        )
        return dict(encoded)

    return transform


def attach_online_tokenization(
    dataset: Any, *, tokenizer: Any, text_field: str, max_length: int, add_special_tokens: bool
):
    """Return an immutable lazily-tokenizing view of ``dataset``.

    ``with_transform`` and not ``set_transform``: the caller's object is also
    held by the dataset preview and the row-count checks, and mutating it in
    place would silently change what those see.

    ``columns = [text_field]`` keeps the backing read down to the one column the
    transform needs, so a split carrying large unused columns does not pay to
    materialise them on every ``__getitem__``.

    The returned view is stamped with :data:`TRUNCATION_ATTESTATION_ATTR` so
    unsloth's ``max_length`` enforcement can trust the cap instead of reading
    every row to check it -- which on a lazy split is the eager tokenize pass
    all over again.
    """
    transform = build_tokenizing_transform(tokenizer, text_field, max_length, add_special_tokens)
    try:
        view = dataset.with_transform(transform, columns = [text_field])
    except TypeError:
        # Older/newer datasets without the `columns` kwarg: the transform reads
        # the field by name either way, so the only loss is the narrow read.
        view = dataset.with_transform(transform)
    try:
        setattr(view, TRUNCATION_ATTESTATION_ATTR, int(max_length))
    except Exception:  # noqa: BLE001 - a split that refuses attributes just gets scanned
        pass
    return view


def first_sample_text(dataset: Any, text_field: str) -> Optional[str]:
    """The first row's rendered text, for the double-BOS probe.  Never raises."""
    try:
        row = dataset[0]
    except Exception:  # noqa: BLE001
        try:
            row = next(iter(dataset))
        except Exception:  # noqa: BLE001
            return None
    if not isinstance(row, dict):
        return None
    value = row.get(text_field)
    if isinstance(value, (list, tuple)):
        value = value[0] if value else None
    return value if isinstance(value, str) else None


def online_config_args(decision: OnlineTokenizationDecision) -> dict:
    """The ``SFTConfig`` keys the online path needs, and nothing else.

    ``remove_unused_columns`` must be False: ``Trainer._remove_unused_columns``
    reads ``column_names``, which on a transformed split reports the BACKING
    table, so it would strip the very column the transform reads.
    """
    return {
        "dataset_kwargs": {"skip_prepare_dataset": True},
        "remove_unused_columns": False,
        "dataloader_num_workers": decision.workers,
        "dataloader_prefetch_factor": decision.prefetch_factor,
        "dataloader_persistent_workers": True,
    }


def memoize_train_dataloader(trainer: Any) -> bool:
    """Make the prewarmed train DataLoader the one ``train()`` actually uses.

    transformers memoizes only the EVAL loaders when persistent workers are on
    (``Trainer._get_dataloader`` stores into ``_eval_dataloaders``); the train
    loader is rebuilt on every call.  Without this, the barrier forks four
    workers, fills them, and then ``train()`` throws them away and forks four
    more -- so the prewarm warms the page cache and nothing else.

    ``_inner_training_loop`` calls ``get_train_dataloader()`` exactly once, so a
    one-shot memo changes no semantics; it also avoids handing the same dataset
    to ``accelerator.prepare`` twice.  Returns whether the memo was installed.

    The cache is parked on the trainer rather than closed over alone, so
    :func:`release_train_dataloader` can reach the loader it holds; a memo only
    reachable through the closure is a loader nothing can ever shut down.
    """
    getter = getattr(trainer, "get_train_dataloader", None)
    if getter is None or getattr(trainer, "_unsloth_online_memoized", False):
        return False

    cache: dict = {}

    def _memoized():
        if "loader" not in cache:
            cache["loader"] = getter()
        return cache["loader"]

    try:
        trainer.get_train_dataloader = _memoized
        trainer._unsloth_online_loader_cache = cache
        trainer._unsloth_online_memoized = True
    except Exception:  # noqa: BLE001 - a trainer that refuses attributes keeps today's behaviour
        return False
    return True


def _nested_loaders(loader: Any):
    """``loader`` and whatever it wraps, outermost first.

    ``accelerator.prepare`` hands back a ``DataLoaderShard`` in some accelerate
    versions and a wrapper holding ``base_dataloader`` in others; the worker
    processes belong to whichever object owns ``_iterator``.
    """
    seen: list = []
    current = loader
    for _ in range(4):  # a wrapper chain, not a graph: bounded on purpose
        if current is None or any(current is item for item in seen):
            break
        seen.append(current)
        current = getattr(current, "base_dataloader", None) or getattr(current, "dataloader", None)
    return seen


def _shutdown_loader_workers(loader: Any, shut: list) -> int:
    """Shut down every worker set ``loader`` (or a wrapper of it) still holds.

    ``shut`` carries the iterators already stopped across calls: an accelerate
    wrapper and the loader inside it hold the SAME iterator, so the walk visits
    one worker set twice; count it once, and still clear the reference on every
    level that holds it.
    """
    released = 0
    for candidate in _nested_loaders(loader):
        iterator = getattr(candidate, "_iterator", None)
        shutdown = getattr(iterator, "_shutdown_workers", None)
        if not callable(shutdown):
            continue
        try:
            if not any(iterator is seen for seen in shut):
                shut.append(iterator)
                released += len(getattr(iterator, "_workers", ()) or ())
                shutdown()
            candidate._iterator = None
        except Exception as exc:  # noqa: BLE001 - a wedged worker must not fail the run
            logger.warning(f"Online tokenization worker shutdown failed: {exc}")
    return released


def release_train_dataloader(trainer: Any) -> int:
    """Shut down the online run's persistent DataLoader workers.  Returns how many.

    The prewarmed train loader and, when the run evaluates, the eval loaders
    transformers memoized in ``_eval_dataloaders``: both were built from the
    same ``dataloader_num_workers`` / ``dataloader_persistent_workers``.

    ``dataloader_persistent_workers = True`` is what lets the barrier's workers
    survive into ``train()``, and it is equally what keeps them alive after
    ``train()`` returns: the memo holds the loader, the loader holds its
    iterator, and the iterator owns the worker processes, so nothing ever drops
    the last reference.  Studio then merges, quantizes and exports in that
    state -- the most memory-hungry part of a run -- with four forked children
    still resident, each one holding the parent's CUDA file descriptors because
    it was forked after the context was initialised.

    Idempotent, and never raises: it is called from a ``finally``, including on
    the paths where training never started.
    """
    released = 0
    cache = getattr(trainer, "_unsloth_online_loader_cache", None)
    loader = cache.pop("loader", None) if isinstance(cache, dict) else None

    # Put the real bound method back, so a trainer reused after this rebuilds a
    # loader instead of handing out the one whose workers just went away.
    try:
        trainer.__dict__.pop("get_train_dataloader", None)
        trainer._unsloth_online_memoized = False
        trainer._unsloth_online_loader_cache = None
    except Exception:  # noqa: BLE001
        pass

    shut: list = []
    released += _shutdown_loader_workers(loader, shut)

    # The worker count is a TrainingArguments setting, so an online run with
    # evaluation on gives the EVAL loader the same workers and the same
    # `persistent_workers = True`; transformers then keeps that prepared loader
    # in `_eval_dataloaders` (`Trainer._get_dataloader`, unchanged from 4.51.3
    # through 5.5.0), and torch keeps `_iterator` alive on a persistent-workers
    # loader once the eval loop has iterated it.  Nothing drops either, so the
    # eval workers outlive train() exactly as the train ones do.  Drop the memo
    # too: a trainer that evaluates after this rebuilds instead of iterating a
    # loader whose workers just went away.
    memo = getattr(trainer, "_eval_dataloaders", None)
    if isinstance(memo, dict):
        for key in list(memo.keys()):
            released += _shutdown_loader_workers(memo.pop(key, None), shut)
    return released


def quiet_tokenizer_fork_warning() -> None:
    """Silence the fast tokenizer's post-fork parallelism notice.

    Studio's chat-template map has already used the Rust tokenizer in parallel
    by the time DataLoader workers fork, so ``tokenizers`` prints its
    "process just got forked" notice and disables its own threads in the child.
    Disabling them explicitly is the same outcome -- one thread per worker
    process, which is what the worker count is for -- without the notice landing
    in a training log that has no terminal.
    """
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


__all__ = [
    "ENV_FLAG",
    "MAX_ONLINE_WORKERS",
    "MIN_ONLINE_WORKERS",
    "MIN_ROWS_FOR_ONLINE",
    "DEFAULT_PREFETCH_FACTOR",
    "TRUNCATION_ATTESTATION_ATTR",
    "OnlineTokenizationDecision",
    "attach_online_tokenization",
    "build_tokenizing_transform",
    "dataloader_worker_start_method",
    "dataset_column_names",
    "dataset_supports_with_transform",
    "decide_online_tokenization",
    "env_override",
    "first_sample_text",
    "is_processor",
    "memoize_train_dataloader",
    "model_needs_token_type_ids",
    "online_config_args",
    "platform_supports_dataloader_workers",
    "prewarm_batch_count",
    "quiet_tokenizer_fork_warning",
    "release_train_dataloader",
    "resolve_add_special_tokens",
    "resolve_worker_count",
    "text_column_defect",
    "trl_supports_skip_prepare_dataset",
]
