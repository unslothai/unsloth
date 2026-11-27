# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Loader progress for the spawned diffusion training worker: diffusers has no per-weight callback, so observe
tensor placement / quantizer calls and library tqdm bars, crediting only successful returns; hooks always restored."""

from contextlib import contextmanager
from functools import wraps
import inspect
import threading
import time
from weakref import WeakKeyDictionary


@contextmanager
def report_model_load_progress(on_event):
    if on_event is None:
        yield
        return

    try:
        from diffusers.models import model_loading_utils as loading
        from diffusers.models import modeling_utils
        from diffusers.utils import logging as diffusion_logging
    except ImportError:
        yield
        return
    from transformers.utils import logging as transformer_logging

    try:
        from transformers import core_model_loading
    except ImportError:
        core_model_loading = None
    from tqdm.auto import tqdm

    lock = threading.RLock()
    counters = WeakKeyDictionary()
    last = [None, 0.0]

    def model_label(model):
        return (
            "Image encoder weights"
            if "Autoencoder" in type(model).__name__
            else "Training model weights"
        )

    def emit(done, total, unit, label):
        if not total or total <= 0:
            return
        with lock:
            identity = (unit, label, total)
            now = time.monotonic()
            if identity == last[0] and done not in (0, total) and now - last[1] < 0.2:
                return
            last[:] = [identity, now]
            on_event(
                dict(type = "model_load_progress", done = done, total = total, unit = unit, label = label)
            )

    def completed(model, name):
        with lock:
            counter = counters.get(model)
            if counter is None or name not in counter[0]:
                return
            counter[1].add(name)
            emit(len(counter[1]), len(counter[0]), "tensors", model_label(model))

    original_set = getattr(loading, "set_module_tensor_to_device", None)
    original_meta = getattr(loading, "load_model_dict_into_meta", None)
    if not callable(original_set) or not callable(original_meta):
        yield
        return
    meta_signature = inspect.signature(original_meta)
    if "model" not in meta_signature.parameters:
        yield
        return

    @wraps(original_set)
    def place(module, tensor_name, *args, **kwargs):
        result = original_set(module, tensor_name, *args, **kwargs)
        completed(module, tensor_name)
        return result

    class QuantizerProgress:
        def __init__(self, quantizer):
            self.quantizer = quantizer

        def __getattr__(self, name):
            return getattr(self.quantizer, name)

        def create_quantized_param(self, model, value, name, *args, **kwargs):
            result = self.quantizer.create_quantized_param(model, value, name, *args, **kwargs)
            completed(model, name)
            return result

    @wraps(original_meta)
    def load_meta(*args, **kwargs):
        bound = meta_signature.bind(*args, **kwargs)
        model = bound.arguments["model"]
        with lock:
            if model not in counters:
                # Snapshot names before quantization adds checkpoint metadata to state_dict.
                counters[model] = (set(model.state_dict()), set())
            names, done = counters[model]
            emit(len(done), len(names), "tensors", model_label(model))
        quantizer = bound.arguments.get("hf_quantizer")
        if quantizer is not None:
            bound.arguments["hf_quantizer"] = QuantizerProgress(quantizer)
        return original_meta(*bound.args, **bound.kwargs)

    class CountedTqdm(tqdm):
        def __init__(self, *args, unit, label, **kwargs):
            self.progress_unit, self.progress_label = unit, label
            # Count even when quiet logging disabled the bar; display() is muted and leave=False stops close() writing "\n".
            kwargs["disable"], kwargs["leave"] = False, False
            super().__init__(*args, **kwargs)
            emit(self.n, self.total, unit, label)

        def display(self, *args, **kwargs):
            pass

        def update(self, n = 1):
            result = super().update(n)
            emit(self.n, self.total, self.progress_unit, self.progress_label)
            return result

        def __iter__(self):
            try:
                for item in self.iterable:
                    yield item
                    # Credited on resume: an item whose load raised is never counted.
                    self.update(1)
            finally:
                self.close()

    class ProgressFactory:
        def __init__(self, original):
            self.original = original

        def __getattr__(self, name):
            return getattr(self.original, name)

        def __call__(self, *args, **kwargs):
            desc = kwargs.get("desc", "")
            if desc.startswith("Loading pipeline components"):
                return CountedTqdm(*args, unit = "components", label = "Model components", **kwargs)
            if desc == "Loading weights":
                return CountedTqdm(*args, unit = "tensors", label = "Text encoder weights", **kwargs)
            if desc == "Loading checkpoint shards":
                return CountedTqdm(*args, unit = "shards", label = "Checkpoint shards", **kwargs)
            return self.original(*args, **kwargs)

    patched = []

    def patch(module, name, replacement):
        original = getattr(module, name)
        patched.append((module, name, original))
        setattr(module, name, replacement)

    try:
        patch(loading, "set_module_tensor_to_device", place)
        patch(loading, "load_model_dict_into_meta", load_meta)
        if getattr(modeling_utils, "load_model_dict_into_meta", None) is original_meta:
            patch(modeling_utils, "load_model_dict_into_meta", load_meta)
        patch(diffusion_logging, "tqdm", ProgressFactory(diffusion_logging.tqdm))
        patch(transformer_logging, "tqdm", ProgressFactory(transformer_logging.tqdm))
        # Transformers 5 imports tqdm by name.
        if core_model_loading is not None and hasattr(core_model_loading, "tqdm"):
            patch(core_model_loading, "tqdm", ProgressFactory(core_model_loading.tqdm))
        yield
    finally:
        for module, name, original in reversed(patched):
            setattr(module, name, original)
