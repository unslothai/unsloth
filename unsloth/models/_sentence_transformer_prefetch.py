# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Collate SentenceTransformerTrainer batches on a background thread; UNSLOTH_ST_PREFETCH=0 disables."""

import os
import queue
import threading

import torch

_DONE = object()


class _Raised:
    __slots__ = ("error",)

    def __init__(self, error):
        self.error = error


class PrefetchedDataLoader:
    """Mixed into the loader's own class (see prefetched), so isinstance checks such as accelerate's
    skip_first_batches on resume still see a DataLoaderShard / DataLoaderDispatcher."""

    _unsloth_depth = 2
    prefetched_batches = 0

    def __iter__(self):
        batches, stop = queue.Queue(self._unsloth_depth), threading.Event()
        source = super().__iter__
        # A new thread starts on device 0: under torchrun every rank would open a context there.
        device = torch.cuda.current_device() if torch.cuda.is_initialized() else None

        def produce():
            iterator = None
            try:
                if device is not None:
                    torch.cuda.set_device(device)
                iterator = source()
                for batch in iterator:
                    while not stop.is_set():
                        try:
                            batches.put(batch, timeout = 0.1)
                            break
                        except queue.Full:
                            continue
                    if stop.is_set():
                        break
            except BaseException as error:
                batches.put(_Raised(error))
                return
            finally:
                close = getattr(iterator, "close", None)
                if close is not None:
                    close()
            batches.put(_DONE)

        thread = threading.Thread(target = produce, name = "unsloth-st-prefetch", daemon = True)
        thread.start()
        try:
            while True:
                batch = batches.get()
                if batch is _DONE:
                    return
                if isinstance(batch, _Raised):
                    raise batch.error
                self.prefetched_batches += 1
                yield batch
        finally:
            # Stopped early (max_steps): let the producer finish so the loader's own cleanup runs.
            stop.set()
            while thread.is_alive():
                try:
                    batches.get(timeout = 0.1)
                except queue.Empty:
                    pass
            thread.join()


_SUBCLASSES = {}


def prefetched(loader, depth = 2):
    cls = type(loader)
    if not isinstance(loader, PrefetchedDataLoader):
        sub = _SUBCLASSES.get(cls)
        if sub is None:
            sub = _SUBCLASSES[cls] = type(
                f"Prefetched{cls.__name__}", (PrefetchedDataLoader, cls), {}
            )
        # A new instance sharing the loader's state: accelerate's adapters make __class__ read-only.
        clone = object.__new__(sub)
        object.__setattr__(clone, "__dict__", loader.__dict__)
        loader = clone
    loader._unsloth_depth = depth
    return loader


def prefetch_enabled(trainer) -> bool:
    if os.environ.get("UNSLOTH_ST_PREFETCH", "1") == "0":
        return False
    if not hasattr(getattr(trainer, "model", None), "_unsloth_trust_remote_code"):
        return False
    # Distributed loaders run collectives while iterating; off-thread they could interleave with all-reduces.
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        if torch.distributed.get_world_size() > 1:
            return False
    args = trainer.args
    return not getattr(args, "dataloader_num_workers", 0)


def patch_train_dataloader(trainer_class) -> None:
    if getattr(trainer_class, "_unsloth_prefetch_patched", False):
        return
    original = trainer_class.get_train_dataloader

    def get_train_dataloader(self, *args, **kwargs):
        loader = original(self, *args, **kwargs)
        if prefetch_enabled(self) and not isinstance(loader, PrefetchedDataLoader):
            loader = prefetched(loader)
        return loader

    get_train_dataloader.__wrapped__ = original
    trainer_class.get_train_dataloader = get_train_dataloader
    trainer_class._unsloth_prefetch_patched = True
