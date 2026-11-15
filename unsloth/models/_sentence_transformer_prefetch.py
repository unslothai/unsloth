# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""SentenceTransformerTrainer tokenizes every batch in its collator on the training thread, so the GPU
waits for it each step. A background thread fetches (and collates) the next batches while the current
one trains. Same batches, same order; UNSLOTH_ST_PREFETCH=0 turns it off."""

import os
import queue
import threading

_DONE = object()


class _Raised:
    __slots__ = ("error",)

    def __init__(self, error):
        self.error = error


class PrefetchedDataLoader:
    def __init__(
        self,
        loader,
        depth = 2,
    ):
        self._unsloth_loader = loader
        self._unsloth_depth = depth
        self.prefetched_batches = 0

    def __getattr__(self, name):
        return getattr(self._unsloth_loader, name)

    def __len__(self):
        return len(self._unsloth_loader)

    def __iter__(self):
        batches, stop = queue.Queue(self._unsloth_depth), threading.Event()
        loader = self._unsloth_loader

        def produce():
            iterator = iter(loader)
            try:
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


def prefetch_enabled(trainer) -> bool:
    if os.environ.get("UNSLOTH_ST_PREFETCH", "1") == "0":
        return False
    # Only models loaded by FastSentenceTransformer.
    if not hasattr(getattr(trainer, "model", None), "_unsloth_trust_remote_code"):
        return False
    args = trainer.args
    # Worker processes already collate ahead of the training loop.
    return not getattr(args, "dataloader_num_workers", 0)


def patch_train_dataloader(trainer_class) -> None:
    if getattr(trainer_class, "_unsloth_prefetch_patched", False):
        return
    original = trainer_class.get_train_dataloader

    def get_train_dataloader(self, *args, **kwargs):
        loader = original(self, *args, **kwargs)
        if prefetch_enabled(self) and not isinstance(loader, PrefetchedDataLoader):
            loader = PrefetchedDataLoader(loader)
        return loader

    get_train_dataloader.__wrapped__ = original
    trainer_class.get_train_dataloader = get_train_dataloader
    trainer_class._unsloth_prefetch_patched = True
