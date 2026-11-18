# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Write `frozen_tables.json` next to adapter saves and check it on load.

Adapters never carry the frozen lookup tables, so they are only valid on the same base tables.
UNSLOTH_FROZEN_TABLE_PROVENANCE = "warn" (default), "strict" (error) or "off".
"""

import functools
import inspect
import json
import os
import sys
import warnings

__all__ = [
    "FILE_NAME",
    "table_provenance",
    "merge_provenance",
    "write_provenance",
    "write_adapter_provenance",
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
    with open(path, "w", encoding = "utf-8") as f:
        json.dump(provenance, f, indent = 1, sort_keys = True)
    return path


def adapter_directories(
    model,
    save_directory,
    selected_adapters = None,
):
    """Where PeftModel.save_pretrained writes each adapter: "default" at save_directory, every other one in
    save_directory/<name> (every adapter in peft_config unless selected_adapters names some)."""
    names = list(selected_adapters) if selected_adapters is not None else None
    if names is None:
        names = list(getattr(model, "peft_config", None) or ()) or ["default"]
    root = os.fspath(save_directory)
    return [root if name == "default" else os.path.join(root, name) for name in names]


def write_adapter_provenance(
    model,
    save_directory,
    provenance,
    selected_adapters = None,
):
    """write_provenance next to each adapter a PEFT save wrote (see adapter_directories)."""
    if not provenance or _mode() == "off":
        return []
    return [
        write_provenance(directory, provenance)
        for directory in adapter_directories(model, save_directory, selected_adapters)
    ]


def _without_resident_tables(saved, model, live):
    """`saved` without the tables `model` holds resident (UNSLOTH_FROZEN_TABLE=gpu, or auto on a large GPU): those
    are ordinary embeddings with no bound table to describe, so a residence change is not other base tables. A saved
    table with no module at its path at all is kept, and still reported."""
    tables = saved.get("tables") if isinstance(saved, dict) else None
    if not tables or not hasattr(model, "named_modules"):
        return saved
    key = getattr(_zoo(), "_table_key", None) or (lambda name: name)
    bound = {key(name) for name in ((live or {}).get("tables") or {})}
    resident = {
        key(name)
        for name, module in model.named_modules()
        if type(module).__name__ != "UnslothFrozenTableEmbedding"
    }
    kept = {k: v for k, v in tables.items() if key(k) in bound or key(k) not in resident}
    return saved if len(kept) == len(tables) else {**saved, "tables": kept}


def check_provenance(model, directory):
    """Compare directory/frozen_tables.json with the tables bound in `model`: [] when they are the same base
    tables (or there is nothing to check), else what differs (a warning, or an error in strict mode)."""
    mode = _mode()
    if mode == "off" or directory is None:
        return []
    path = os.path.join(os.fspath(directory), FILE_NAME)
    if not os.path.isfile(path):
        return []
    with open(path, encoding = "utf-8") as f:
        saved = json.load(f)
    compare = getattr(_zoo(), "compare_frozen_table_provenance", None)
    if compare is None:
        problems = ["unsloth_zoo cannot read frozen table provenance (update unsloth_zoo)"]
    else:
        live = table_provenance(model)
        problems = compare(_without_resident_tables(saved, model, live), live)
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


_HUB_OPTIONS = (
    "subfolder",
    "revision",
    "repo_type",
    "cache_dir",
    "token",
    "local_files_only",
    "force_download",
    "proxies",
    "endpoint",
)


def _hub_sidecar(model_id, kwargs):
    """The directory holding the Hub adapter's frozen_tables.json (downloaded into the Hub cache with the load's
    revision / subfolder / token / cache options), or None when the repo has none (an adapter of a model without
    frozen tables, or saved before provenance existed). In strict mode an unanswerable question (offline and not
    cached, a network error) is an error: the file may exist."""
    from huggingface_hub import hf_hub_download

    try:
        from huggingface_hub.errors import (
            EntryNotFoundError,
            LocalEntryNotFoundError,
            RepositoryNotFoundError,
            RevisionNotFoundError,
        )
    except ImportError:  # huggingface_hub < 0.23
        from huggingface_hub.utils import (
            EntryNotFoundError,
            LocalEntryNotFoundError,
            RepositoryNotFoundError,
            RevisionNotFoundError,
        )
    options = {k: kwargs[k] for k in _HUB_OPTIONS if kwargs.get(k) is not None}
    if "token" not in options and kwargs.get("use_auth_token") is not None:
        options["token"] = kwargs["use_auth_token"]
    try:
        return os.path.dirname(hf_hub_download(os.fspath(model_id), FILE_NAME, **options))
    except (
        LocalEntryNotFoundError
    ) as e:  # before EntryNotFoundError: a subclass of it on older hubs
        error = e
    except (EntryNotFoundError, RepositoryNotFoundError, RevisionNotFoundError):
        return None  # no sidecar, or no such repo (the adapter load itself reports that)
    except OSError as e:
        error = e
    if _mode() == "strict":
        raise ValueError(
            f"Unsloth: cannot read {FILE_NAME} of the adapter {model_id} ({type(error).__name__}: {error}), so its "
            "frozen lookup tables cannot be checked (UNSLOTH_FROZEN_TABLE_PROVENANCE=strict)."
        ) from error
    return None


def _adapter_dir(model, model_id, kwargs):
    """Where the adapter's frozen_tables.json is: its local directory, or (a Hub id, only for a model with frozen
    tables) the Hub cache directory it was downloaded to; None when there is nothing to check."""
    if _mode() == "off" or not isinstance(model_id, (str, os.PathLike)):
        return None
    root = os.fspath(model_id)
    if os.path.isdir(root):
        folder = os.path.join(root, kwargs.get("subfolder") or "")
        return folder if os.path.isdir(folder) else None
    if table_provenance(model) is None:
        return None  # nothing of this model to check: no Hub request
    return _hub_sidecar(model_id, kwargs)


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
    try:
        save_signature = inspect.signature(save)
    except (TypeError, ValueError):
        save_signature = None

    @functools.wraps(save)
    def save_pretrained(self, save_directory, *args, **kwargs):
        out = save(self, save_directory, *args, **kwargs)
        selected, main = kwargs.get("selected_adapters"), kwargs.get("is_main_process", True)
        if save_signature is not None:
            try:
                bound = save_signature.bind(self, save_directory, *args, **kwargs).arguments
                selected = bound.get("selected_adapters", selected)
                main = bound.get("is_main_process", main)
            except TypeError:
                pass
        if main:
            write_adapter_provenance(self, save_directory, table_provenance(self), selected)
        return out

    load = PeftModel.load_adapter

    @functools.wraps(load)
    def load_adapter(self, model_id, *args, **kwargs):
        # Before PEFT touches the model: a refused adapter must leave it as it was.
        check_provenance(self, _adapter_dir(self, model_id, kwargs))
        return load(self, model_id, *args, **kwargs)

    original = PeftModel.__dict__["from_pretrained"].__func__

    @functools.wraps(original)
    def from_pretrained(cls, model, model_id, *args, **kwargs):
        # Before PEFT injects the adapter's layers into `model` (in place).
        check_provenance(model, _adapter_dir(model, model_id, kwargs))
        return original(cls, model, model_id, *args, **kwargs)

    for fn in (save_pretrained, load_adapter, from_pretrained):
        setattr(fn, _MARK, True)
    PeftModel.save_pretrained = save_pretrained
    PeftModel.load_adapter = load_adapter
    PeftModel.from_pretrained = classmethod(from_pretrained)
    return True
