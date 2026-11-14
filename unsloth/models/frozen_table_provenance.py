# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Provenance of frozen lookup tables (unsloth_zoo frozen_table_store: Qwen3.8-Flash-Next's PLE n-gram table,
DeepSeek-V4.1's engram tables) next to every adapter save that leaves the tables out.

An adapter (single GPU, FSDP2 / tensor / pipeline parallel, a Trainer or DCP checkpoint's adapter) never carries
the tables: they stay in the base checkpoint and are read from it again. The adapter is only meaningful on the
same base tables, so the save writes `frozen_tables.json` (repo id, revision, file sizes, header and sampled
content digests, from unsloth_zoo) next to it, and loading the adapter checks the live model's tables against
it. UNSLOTH_FROZEN_TABLE_PROVENANCE = "warn" (default: a warning on a mismatch), "strict" (an error), "off".
"""

import functools
import json
import os
import sys
import warnings

__all__ = [
    "FILE_NAME",
    "table_provenance",
    "merge_provenance",
    "write_provenance",
    "check_provenance",
    "install",
]

FILE_NAME = "frozen_tables.json"
_MARK = "_unsloth_table_provenance"


def _mode():
    mode = os.environ.get("UNSLOTH_FROZEN_TABLE_PROVENANCE", "warn").strip().lower()
    return mode if mode in ("warn", "strict", "off") else "warn"


def _zoo():
    return sys.modules.get("unsloth_zoo.frozen_table_store")


def table_provenance(model):
    """unsloth_zoo's description of the bound frozen tables of `model`, or None (no tables / an older zoo)."""
    fn = getattr(_zoo(), "frozen_table_provenance", None)
    return fn(model) if fn is not None else None


def merge_provenance(parts):
    """One provenance from several (pipeline stages each hold their own tables)."""
    merged = None
    for part in parts:
        if not part:
            continue
        if merged is None:
            merged = {k: v for k, v in part.items() if k != "tables"}
            merged["tables"] = {}
        merged["tables"].update(part.get("tables") or {})
    return merged


def write_provenance(save_directory, provenance):
    """Write `provenance` as save_directory/frozen_tables.json; None when there is nothing to write."""
    if not provenance or _mode() == "off":
        return None
    os.makedirs(save_directory, exist_ok = True)
    path = os.path.join(os.fspath(save_directory), FILE_NAME)
    with open(path, "w") as f:
        json.dump(provenance, f, indent = 1, sort_keys = True)
    return path


def check_provenance(model, directory):
    """Compare directory/frozen_tables.json with the tables bound in `model`: [] when they are the same base
    tables (or there is nothing to check), else what differs (a warning, or an error in strict mode)."""
    mode = _mode()
    if mode == "off" or directory is None:
        return []
    path = os.path.join(os.fspath(directory), FILE_NAME)
    if not os.path.isfile(path):
        return []
    with open(path) as f:
        saved = json.load(f)
    compare = getattr(_zoo(), "compare_frozen_table_provenance", None)
    if compare is None:
        problems = ["unsloth_zoo cannot read frozen table provenance (update unsloth_zoo)"]
    else:
        problems = compare(saved, table_provenance(model))
    if problems:
        message = (
            f"Unsloth: the adapter in {directory} was saved on other frozen lookup tables than this model's "
            f"({'; '.join(problems[:4])}{'; ...' if len(problems) > 4 else ''}). It needs the same base model "
            "checkpoint it was trained on."
        )
        if mode == "strict":
            raise ValueError(message)
        warnings.warn(message, stacklevel = 3)
    return problems


def _adapter_dir(model_id, kwargs):
    if not isinstance(model_id, (str, os.PathLike)):
        return None
    folder = os.path.join(os.fspath(model_id), kwargs.get("subfolder") or "")
    return folder if os.path.isdir(folder) else None


def install():
    """Write the provenance next to every PEFT adapter save of a model with frozen tables, and check it on
    PeftModel.from_pretrained / load_adapter. Idempotent."""
    try:
        from peft import PeftModel
    except Exception:
        return False
    if getattr(PeftModel.save_pretrained, _MARK, False):
        return True
    save = PeftModel.save_pretrained

    @functools.wraps(save)
    def save_pretrained(self, save_directory, *args, **kwargs):
        out = save(self, save_directory, *args, **kwargs)
        if kwargs.get("is_main_process", True):
            write_provenance(save_directory, table_provenance(self))
        return out

    load = PeftModel.load_adapter

    @functools.wraps(load)
    def load_adapter(self, model_id, *args, **kwargs):
        out = load(self, model_id, *args, **kwargs)
        check_provenance(self, _adapter_dir(model_id, kwargs))
        return out

    original = PeftModel.__dict__["from_pretrained"].__func__

    @functools.wraps(original)
    def from_pretrained(cls, model, model_id, *args, **kwargs):
        out = original(cls, model, model_id, *args, **kwargs)
        check_provenance(out, _adapter_dir(model_id, kwargs))
        return out

    for fn in (save_pretrained, load_adapter, from_pretrained):
        setattr(fn, _MARK, True)
    PeftModel.save_pretrained = save_pretrained
    PeftModel.load_adapter = load_adapter
    PeftModel.from_pretrained = classmethod(from_pretrained)
    return True
