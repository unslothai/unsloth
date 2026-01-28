from __future__ import annotations

import queue
import threading
from dataclasses import dataclass
from typing import Any, Callable, Iterator, Optional

import psutil
from datasets import Dataset, IterableDataset


_STOP = object()
_ERROR = object()


def _resolve_worker_count(args) -> int:
    dataset_num_proc = getattr(args, "dataset_num_proc", None)
    if dataset_num_proc is not None:
        return max(int(dataset_num_proc), 1)
    cpu_count = psutil.cpu_count() or 1
    dataset_num_proc = max(cpu_count + 4, 2)
    memory_gb_left = psutil.virtual_memory().available / (1024**3)
    if memory_gb_left <= 4:
        dataset_num_proc = 1
    elif memory_gb_left <= 6:
        dataset_num_proc = min(2, dataset_num_proc)
    elif memory_gb_left <= 8:
        dataset_num_proc = min(4, dataset_num_proc)
    elif memory_gb_left <= 12:
        dataset_num_proc = min(6, dataset_num_proc)
    return max(int(dataset_num_proc), 1)


def _resolve_queue_size(args, worker_count: int) -> int:
    dataset_kwargs = getattr(args, "dataset_kwargs", None)
    if isinstance(dataset_kwargs, dict):
        queue_size = dataset_kwargs.get("queue_size")
        if queue_size is not None:
            return max(int(queue_size), 1)
    return max(worker_count * 2, 2)


def _resolve_batch_size(args, dataset: Any) -> int:
    dataset_kwargs = getattr(args, "dataset_kwargs", None)
    if isinstance(dataset_kwargs, dict):
        batch_size = dataset_kwargs.get("batch_size")
        if batch_size is not None:
            return max(int(batch_size), 1)
    if isinstance(dataset, IterableDataset):
        ex_iterable = getattr(dataset, "_ex_iterable", None)
        if ex_iterable is not None and getattr(ex_iterable, "batch_size", None):
            return int(ex_iterable.batch_size)
    return 1000


def _list_of_dicts_to_batch(rows: list[dict[str, Any]]) -> dict[str, list[Any]]:
    if not rows:
        return {}
    batch: dict[str, list[Any]] = {key: [] for key in rows[0].keys()}
    for row in rows:
        for key, value in row.items():
            batch[key].append(value)
    return batch


def _iter_batches(dataset: Any, batch_size: int) -> Iterator[dict[str, list[Any]]]:
    if isinstance(dataset, IterableDataset):
        buffer: list[dict[str, Any]] = []
        for row in dataset:
            buffer.append(row)
            if len(buffer) >= batch_size:
                yield _list_of_dicts_to_batch(buffer)
                buffer = []
        if buffer:
            yield _list_of_dicts_to_batch(buffer)
        return
    total = len(dataset)
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        yield dataset[start:end]


def _batch_length(batch: dict[str, list[Any]]) -> int:
    if not batch:
        return 0
    first = next(iter(batch.values()))
    return len(first)


def _build_seq_lengths(sequences: list[dict[str, list[int]]], field: str) -> list[int]:
    lengths: list[int] = []
    for seq in sequences:
        lengths.append(len(seq.get(field, [])))
    return lengths


@dataclass
class _WrappedPacker:
    max_length: int
    fields: tuple[str, ...]

    def __post_init__(self) -> None:
        self._buffers: dict[str, list[int]] = {field: [] for field in self.fields}

    def add_sequence(self, seq: dict[str, list[int]]) -> Iterable[dict[str, list[int]]]:
        for field in self.fields:
            self._buffers[field].extend(seq.get(field, []))
        return self._drain()

    def flush(self) -> Iterable[dict[str, list[int]]]:
        if not self._buffers:
            return []
        length = len(self._buffers[self.fields[0]])
        if length == 0:
            return []
        output = {field: list(self._buffers[field]) for field in self.fields}
        for field in self.fields:
            self._buffers[field].clear()
        return [output]

    def _drain(self) -> Iterable[dict[str, list[int]]]:
        outputs: list[dict[str, list[int]]] = []
        while len(self._buffers[self.fields[0]]) >= self.max_length:
            output = {}
            for field in self.fields:
                buffer = self._buffers[field]
                output[field] = buffer[: self.max_length]
                del buffer[: self.max_length]
            outputs.append(output)
        return outputs


@dataclass
class _WindowedBFDPacker:
    max_length: int
    fields: tuple[str, ...]
    window_size: int

    def __post_init__(self) -> None:
        self._buffer: list[dict[str, list[int]]] = []

    def add_sequence(self, seq: dict[str, list[int]]) -> Iterable[dict[str, list[int]]]:
        self._buffer.append(seq)
        if len(self._buffer) >= self.window_size:
            return self._pack_window(flush = False)
        return []

    def flush(self) -> Iterable[dict[str, list[int]]]:
        return self._pack_window(flush = True)

    def _pack_window(self, *, flush: bool) -> Iterable[dict[str, list[int]]]:
        if not self._buffer:
            return []
        sequences = self._buffer
        self._buffer = []
        lengths = _build_seq_lengths(sequences, self.fields[0])
        indices = sorted(range(len(sequences)), key = lambda i: lengths[i], reverse = True)

        bins: list[list[int]] = []
        remaining_spaces: list[int] = []
        remaining_bins: list[int] = []

        def _insert_space(space: int, bin_idx: int) -> None:
            if space <= 0:
                return
            pos = 0
            if remaining_spaces:
                import bisect

                pos = bisect.bisect_left(remaining_spaces, space)
            remaining_spaces.insert(pos, space)
            remaining_bins.insert(pos, bin_idx)

        for idx in indices:
            length = lengths[idx]
            pos = 0
            if remaining_spaces:
                import bisect

                pos = bisect.bisect_left(remaining_spaces, length)
            if pos < len(remaining_spaces):
                bin_idx = remaining_bins.pop(pos)
                space = remaining_spaces.pop(pos) - length
            else:
                bin_idx = len(bins)
                bins.append([])
                space = self.max_length - length
            bins[bin_idx].append(idx)
            _insert_space(space, bin_idx)

        outputs: list[dict[str, list[int]]] = []
        for bin_indices in bins:
            output: dict[str, list[int]] = {}
            for field in self.fields:
                combined: list[int] = []
                for idx in bin_indices:
                    combined.extend(sequences[idx].get(field, []))
                output[field] = combined
            output["seq_lengths"] = [lengths[idx] for idx in bin_indices]
            outputs.append(output)
        return outputs


def _make_packer(strategy: str, max_length: int, fields: tuple[str, ...], args) -> Any:
    if strategy == "wrapped":
        return _WrappedPacker(max_length = max_length, fields = fields)
    dataset_kwargs = getattr(args, "dataset_kwargs", None)
    window_size = None
    if isinstance(dataset_kwargs, dict):
        window_size = dataset_kwargs.get("pack_window_size")
    if window_size is None:
        window_size = max(256, min(4096, max_length * 4))
    return _WindowedBFDPacker(
        max_length = max_length, fields = fields, window_size = int(window_size)
    )


def _prepare_streaming_pipeline(
    dataset: Any,
    *,
    batch_size: int,
    worker_count: int,
    queue_size: int,
    tokenize_fn: Optional[
        Callable[[dict[str, list[Any]]], dict[str, list[Any]]]
    ] = None,
    keep_columns: Optional[set[str]] = None,
) -> Iterator[dict[str, list[Any]]]:
    input_queue: queue.Queue[Any] = queue.Queue(maxsize = queue_size)
    output_queue: queue.Queue[Any] = queue.Queue(maxsize = queue_size)

    def producer() -> None:
        if isinstance(dataset, IterableDataset):
            for batch in _iter_batches(dataset, batch_size):
                input_queue.put(("batch", batch))
        else:
            total = len(dataset)
            for start in range(0, total, batch_size):
                end = min(start + batch_size, total)
                input_queue.put(("slice", start, end))
        for _ in range(worker_count):
            input_queue.put(_STOP)

    def worker() -> None:
        try:
            while True:
                item = input_queue.get()
                if item is _STOP:
                    break
                batch = item
                if isinstance(batch, tuple):
                    mode = batch[0]
                    if mode == "slice":
                        _, start, end = batch
                        batch = dataset[start:end]
                    else:
                        _, batch = batch
                if tokenize_fn is not None:
                    batch = tokenize_fn(batch)
                if keep_columns is not None and batch:
                    batch = {
                        key: value
                        for key, value in batch.items()
                        if key in keep_columns
                    }
                output_queue.put(batch)
        except Exception as exc:
            output_queue.put((_ERROR, exc))
        finally:
            output_queue.put(_STOP)

    threads: list[threading.Thread] = []
    producer_thread = threading.Thread(target = producer, daemon = True)
    producer_thread.start()
    for _ in range(worker_count):
        thread = threading.Thread(target = worker, daemon = True)
        thread.start()
        threads.append(thread)

    finished = 0
    while finished < worker_count:
        batch = output_queue.get()
        if batch is _STOP:
            finished += 1
            continue
        if isinstance(batch, tuple) and len(batch) == 2 and batch[0] is _ERROR:
            raise RuntimeError(
                "Unsloth: streaming tokenization worker failed."
            ) from batch[1]
        if batch:
            yield batch

    producer_thread.join()
    for thread in threads:
        thread.join()


def _extract_chat_template(processing_class, tokenizer) -> str:
    chat_template = getattr(processing_class, "chat_template", "")
    if chat_template == "" and hasattr(processing_class, "tokenizer"):
        chat_template = getattr(tokenizer, "chat_template", "")
    if chat_template is None:
        chat_template = ""
    return chat_template


def sft_prepare_dataset(
    self,
    dataset: Dataset | IterableDataset,
    processing_class,
    args,
    packing: bool,
    formatting_func: Optional[Callable[[dict], str]],
    dataset_name: str,
):
    try:
        from trl.trainer.sft_trainer import ConstantLengthDataset

        if isinstance(dataset, ConstantLengthDataset):
            return dataset
    except Exception:
        pass

    # Skip dataset preparation if requested (preserve existing behavior).
    if (
        args is not None
        and getattr(args, "dataset_kwargs", None)
        and args.dataset_kwargs.get("skip_prepare_dataset", False)
    ):
        return dataset

    is_vlm = hasattr(processing_class, "tokenizer")
    tokenizer = processing_class.tokenizer if is_vlm else processing_class

    max_seq_length = getattr(args, "max_length", 0) if args is not None else 0
    if max_seq_length == 0:
        max_seq_length = getattr(args, "max_seq_length", 0) if args is not None else 0
    if max_seq_length == 0:
        max_seq_length = getattr(self, "max_seq_length", 0)
    if max_seq_length == 0:
        max_seq_length = getattr(self, "max_seq", 0)
    if max_seq_length == 0:
        raise RuntimeError("Unsloth: max_seq_length is 0! Please specify one!")

    dataset_text_field = (
        getattr(args, "dataset_text_field", "text") if args is not None else "text"
    )

    do_truncation = max_seq_length != 0
    do_formatting_func = False
    do_tokenize = True

    if getattr(dataset, "column_names", None) is not None:
        column_names = set(dataset.column_names)
    else:
        column_names = set(next(iter(dataset)).keys())

    used_column_names = ["input_ids"]
    if "attention_mask" in column_names:
        used_column_names.append("attention_mask")

    from transformers import DataCollatorForSeq2Seq, DataCollatorForLanguageModeling

    if "labels" in column_names:
        if is_vlm and not hasattr(tokenizer, "pad"):
            raise RuntimeError(
                f"Unsloth: {processing_class.__class__} does not have .pad!"
            )
        self.data_collator = DataCollatorForSeq2Seq(tokenizer)
        used_column_names.append("labels")
        do_tokenize = False
    elif "input_ids" in column_names:
        if is_vlm and not hasattr(tokenizer, "pad"):
            raise RuntimeError(
                f"Unsloth: {processing_class.__class__} does not have .pad!"
            )
        self.data_collator = DataCollatorForLanguageModeling(tokenizer, mlm = False)
        do_tokenize = False
    elif dataset_text_field not in column_names:
        do_formatting_func = True
        if formatting_func is None:
            raise RuntimeError("Unsloth: You must specify a `formatting_func`")

    if not do_tokenize and not packing:
        return dataset

    add_special_tokens = True
    if do_tokenize:
        if do_formatting_func:
            test_text = formatting_func(next(iter(dataset)))
            if not isinstance(test_text, list):
                raise ValueError(
                    "Unsloth: The `formatting_func` should return a list of processed strings."
                )
            test_text = test_text[0]
        else:
            sample_text = next(iter(dataset))[dataset_text_field]
            if isinstance(sample_text, list):
                test_text = sample_text[0] if sample_text else ""
            else:
                test_text = sample_text

        chat_template = _extract_chat_template(processing_class, tokenizer)
        bos_token = getattr(processing_class, "bos_token", None) or getattr(
            tokenizer, "bos_token", None
        )
        if bos_token is not None:
            if test_text.startswith(bos_token) or bos_token in chat_template:
                add_special_tokens = False
                print(
                    "Unsloth: We found double BOS tokens - we shall remove one automatically."
                )

    batch_size = _resolve_batch_size(args, dataset)
    worker_count = _resolve_worker_count(args)
    queue_size = _resolve_queue_size(args, worker_count)

    def tokenize_fn(batch: dict[str, list[Any]]) -> dict[str, list[Any]]:
        texts = (
            batch[dataset_text_field]
            if not do_formatting_func
            else formatting_func(batch)
        )
        tokenized = tokenizer(
            texts,
            truncation = do_truncation,
            max_length = max_seq_length,
            return_token_type_ids = False,
            add_special_tokens = add_special_tokens,
        )
        if not packing:
            for key, value in batch.items():
                if key not in tokenized:
                    tokenized[key] = value
        return tokenized

    keep_columns = None
    if packing:
        keep_columns = set(used_column_names)

    pack_fields = tuple(used_column_names)

    def example_generator() -> Iterator[dict[str, Any]]:
        stream = _prepare_streaming_pipeline(
            dataset,
            batch_size = batch_size,
            worker_count = worker_count,
            queue_size = queue_size,
            tokenize_fn = tokenize_fn if do_tokenize else None,
            keep_columns = keep_columns,
        )
        packer = (
            _make_packer(
                getattr(args, "packing_strategy", "bfd"),
                max_seq_length,
                pack_fields,
                args,
            )
            if packing
            else None
        )
        for batch in stream:
            count = _batch_length(batch)
            if count == 0:
                continue
            if not packing:
                for idx in range(count):
                    yield {key: value[idx] for key, value in batch.items()}
            else:
                for idx in range(count):
                    seq = {field: batch.get(field, [])[idx] for field in pack_fields}
                    for output in packer.add_sequence(seq):
                        yield output
        if packing:
            for output in packer.flush():
                yield output

    if do_tokenize and is_vlm and not hasattr(processing_class, "pad"):
        data_collator = DataCollatorForLanguageModeling(tokenizer, mlm = False)
        self.data_collator = data_collator

    dataset_kwargs = getattr(args, "dataset_kwargs", None) if args is not None else None
    keep_in_memory = None
    cache_dir = None
    if isinstance(dataset_kwargs, dict):
        keep_in_memory = dataset_kwargs.get("keep_in_memory")
        cache_dir = dataset_kwargs.get("cache_dir")
    return Dataset.from_generator(
        example_generator,
        keep_in_memory = keep_in_memory if keep_in_memory is not None else False,
        cache_dir = cache_dir,
    )
