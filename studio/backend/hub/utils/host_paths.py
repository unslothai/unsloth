# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Keep the host's filesystem layout inside the host. For an ``sk-unsloth`` API key every host
path in an inventory payload is replaced by an opaque reference, stable for the life of the
server so a client can still group rows. Output encoding, not an authorization decision.
"""

from __future__ import annotations

import hmac
import os
import re
import secrets
import threading
import time
from collections import OrderedDict
from contextvars import ContextVar
from functools import lru_cache
from hashlib import sha256
from typing import Any, Iterable, Mapping, Optional

# One list, so a new route cannot introduce a path field the drift test misses.
HOST_PATH_SCALAR_FIELDS = frozenset(
    {
        "cache_path",
        "load_cache_path",
        "hf_cache_dir",
        "models_dir",
        "repo_path",
        "snapshot_path",
        "local_path",
        "model_local_path",
        "dataset_local_path",
        "dataset_path",
        "tensorboard_dir",
    }
)

# Referenced, not blanked: blanking left `can_resume` true beside nothing to resume with.
# The two snapshot pins are here for the same reason one layer down. Resume replays the run's
# config, and a blanked pin is FALSY, so the preflight stops pinning and re-resolves the newest
# cached revision instead: the checkpoint then continues against different base weights, with no
# warning, because a pin that was never asked for cannot go missing.
HOST_PATH_HANDLE_FIELDS = frozenset(
    {
        "output_dir",
        "checkpoint_path",
        "resume_from_checkpoint",
        "model_snapshot_path",
        "dataset_snapshot_path",
    }
)

# Scrubbed, not blanked: the only account of WHY a run failed.
HOST_PATH_TEXT_FIELDS = frozenset({"error_message", "error", "detail", "message"})

# Emptied, not referenced: scan ROOTS carry layout and nothing actionable.
HOST_PATH_LIST_FIELDS = frozenset(
    {
        "exact_paths",
        "lmstudio_dirs",
        "ollama_dirs",
        "hermes_dirs",
        "scanned_dirs",
        "output_dirs",
        "dataset_paths",
    }
)

# Referenced entry by entry: emptying leaves a resumable checkpoint with no dataset to resume on.
HOST_PATH_HANDLE_LIST_FIELDS = frozenset({"local_datasets", "local_eval_datasets"})

# A path on the inventory objects and a repo id elsewhere, so never redacted by field name alone.
HOST_PATH_AMBIGUOUS_FIELD = "path"

# Decided by a SIBLING: a repo id for most adapters, a path for a locally trained one.
HOST_PATH_CONDITIONAL_FIELD = "base_model"
HOST_PATH_CONDITIONAL_SOURCE_FIELD = "base_model_source"
HOST_PATH_CONDITIONAL_SOURCE_LOCAL = "local"


def _conditional_path_is_local(payload: Mapping) -> bool:
    return payload.get(HOST_PATH_CONDITIONAL_SOURCE_FIELD) == HOST_PATH_CONDITIONAL_SOURCE_LOCAL


# A row whose IDENTITY is a path: blanking `path` hides nothing while `id`, `load_id` and
# `inventory_id` spell it out. The rest are decided by VALUE, not field name.
HOST_PATH_IDENTITY_FIELDS = (
    "id",
    "load_id",
    "repo_id",
    "model_name",
    "active_model",
    "model_identifier",
    "dataset_name",
    "base_repo",
)
HOST_PATH_IDENTITY_LIST_FIELDS = ("loaded", "loading")
# The subset a LOCAL row is named by, referenced on source alone; the others only on value.
HOST_PATH_ROW_IDENTITY_FIELDS = ("id", "load_id")
HOST_PATH_ENCODED_IDENTITY_FIELD = "inventory_id"
HOST_PATH_ROW_SOURCE_FIELD = "source"
# `hf_cache` is absent: those rows are named by repo id, and otherwise value decides.
HOST_PATH_LOCAL_SOURCES = frozenset({"models_dir", "lmstudio", "ollama", "hermes", "custom"})


def _row_identity_is_a_path(payload: Mapping) -> bool:
    return payload.get(HOST_PATH_ROW_SOURCE_FIELD) in HOST_PATH_LOCAL_SOURCES


def _identity_value_is_a_path(value: Any) -> bool:
    """Whatever the row's source says: a copy outside the active cache pins a row to a path."""
    return isinstance(value, str) and bool(value) and _looks_absolute(value)


def _referenced_identity(value: Any) -> Any:
    if not isinstance(value, str) or not value:
        return value
    return cache_reference(value) or ""


def _referenced_inventory_id(value: Any) -> Any:
    """`<source>:<format>:<identity>[:<variant>]`; the shape is kept because clients split on it."""
    if not isinstance(value, str) or not value:
        return value
    parts = value.split(":")
    if len(parts) < 3:
        return _referenced_identity(value)
    parts[2] = (cache_reference(parts[2]) or "").removeprefix(_REFERENCE_PREFIX)
    return ":".join(parts)


# Sibling written beside a redacted scalar, so a client keeps the identity the path gave it.
CACHE_REFERENCE_FIELD = "cache_ref"

_REFERENCE_PREFIX = "ref:"

# Per-process, so a reference cannot be brute-forced from guessed home directories.
_REFERENCE_KEY = secrets.token_bytes(32)


def host_paths_visible(via_api_key: Any) -> bool:
    """Non-bool means no HTTP request: an in-process call gets a truthy unresolved ``Depends``."""
    if not isinstance(via_api_key, bool):
        return True
    return not via_api_key


_REFERENCE_LIMIT = 8192
# Evicted by AGE: with no row limit, count-based eviction drops entries of the response being
# built RIGHT NOW. The ceiling still bounds the table.
_REFERENCE_PIN_SECONDS = 120.0
_REFERENCE_CEILING = 65536
_reference_paths: "OrderedDict[str, tuple[str, float]]" = OrderedDict()
_reference_lock = threading.Lock()


def cache_reference(value: Any) -> Optional[str]:
    text = _as_text(value)
    if not text:
        return None
    digest = hmac.new(_REFERENCE_KEY, os.fsencode(text), sha256).hexdigest()
    reference = f"{_REFERENCE_PREFIX}{digest[:32]}"
    now = time.monotonic()
    with _reference_lock:
        _reference_paths[reference] = (text, now)
        _reference_paths.move_to_end(reference)
        while len(_reference_paths) > _REFERENCE_LIMIT:
            oldest, (_path, issued) = next(iter(_reference_paths.items()))
            if (
                now - issued < _REFERENCE_PIN_SECONDS
                and len(_reference_paths) <= _REFERENCE_CEILING
            ):
                break
            _reference_paths.pop(oldest, None)
    return reference


# Per REQUEST, so one caller's substitutions cannot reach another's response.
_request_handles: "ContextVar[Optional[dict[str, str]]]" = ContextVar(
    "unsloth_request_inventory_handles", default = None
)


def note_resolved_handle(handle: str, path: str) -> None:
    if not isinstance(handle, str) or not isinstance(path, str) or not handle or not path:
        return
    if handle == path:
        return
    known = _request_handles.get()
    if known is None:
        known = {}
        _request_handles.set(known)
    known[path] = handle


def restore_inventory_handles(payload: Any) -> Any:
    """A substring swap, not a field list: the path also appears in labels and error details."""
    known = _request_handles.get()
    if not known:
        return payload
    return _restore(payload, known)


def _restore(payload: Any, known: "dict[str, str]") -> Any:
    dumped = _dump_model(payload)
    if dumped is not None:
        return _restore(dumped, known)
    if isinstance(payload, Mapping):
        return {key: _restore(value, known) for key, value in payload.items()}
    if isinstance(payload, (list, tuple)):
        restored = [_restore(item, known) for item in payload]
        if not isinstance(payload, tuple):
            return restored
        # A NamedTuple takes its fields one by one and a plain tuple the iterable; deciding by
        # try/except turns the plain `("abc",)` into `("a", "b", "c")` without raising.
        if hasattr(payload, "_fields"):
            try:
                return type(payload)(*restored)
            except Exception:  # noqa: BLE001 -- a subclass with its own constructor
                return tuple(restored)
        try:
            return type(payload)(restored)
        except Exception:  # noqa: BLE001 -- same
            return tuple(restored)
    if isinstance(payload, str):
        text = payload
        # Longest first, so a path that is a prefix of another does not claim its text.
        for path in sorted(known, key = len, reverse = True):
            if path in text:
                text = _swap_at_boundaries(text, path, known[path])
        return text
    return payload


# A plain substring swap turns a sibling (`/srv/models/foo-private` beside a resolved
# `/srv/models/foo`) into `ref:<digest>-private`, which resolves to nothing and no longer reads
# as a path, so the layout rides out in it. The component must END at the boundary, so a letter,
# digit, `-`, `_`, `.` or space means a DIFFERENT path; declining is safe, since an unswapped
# path is still removed on the way out.
_HANDLE_LEFT_BOUNDARY = r"(?<![\w:/.\\])"
_HANDLE_RIGHT_BOUNDARY = r"(?![^\s\\/\n\":;,=])"


@lru_cache(maxsize = 4096)
def _boundary_pattern(path: str) -> "re.Pattern[str]":
    return re.compile(_HANDLE_LEFT_BOUNDARY + re.escape(path) + _HANDLE_RIGHT_BOUNDARY)


def _swap_at_boundaries(text: str, path: str, handle: str) -> str:
    try:
        pattern = _boundary_pattern(path)
    except Exception:  # noqa: BLE001 -- an uncompilable path must not fail the response
        return text
    return pattern.sub(lambda _match: handle, text)


def resolve_host_path_reference(value: Any) -> Optional[str]:
    """Not an authorization decision: callers stay subject to the checks a named path gets."""
    text = _as_text(value)
    if not text or not text.startswith(_REFERENCE_PREFIX):
        return None
    with _reference_lock:
        entry = _reference_paths.get(text)
    return entry[0] if entry else None


def redact_host_paths(
    payload: Any,
    *,
    via_api_key: bool,
    echo: Iterable[Any] = (),
) -> Any:
    """``echo`` holds values THIS CALLER supplied, returned as written; see ``_echoed``."""
    if host_paths_visible(via_api_key):
        return payload
    return _redact(payload, redact_ambiguous_path = False, echo = _echo_set(echo))


def redact_inventory_host_paths(
    payload: Any,
    *,
    via_api_key: bool,
    echo: Iterable[Any] = (),
) -> Any:
    if host_paths_visible(via_api_key):
        return payload
    return _redact(payload, redact_ambiguous_path = True, echo = _echo_set(echo))


def _echo_set(echo: Iterable[Any]) -> frozenset:
    return frozenset(text for text in (_as_text(value) for value in echo) if text)


def response_leaks_host_path(
    payload: Any,
    roots: Iterable[Any] = (),
    *,
    ignore: Iterable[str] = (),
) -> Optional[str]:
    """For tests and the drift gate. ``ignore`` makes keeping a path explicit at each call."""
    needles = [text for text in (_as_text(root) for root in roots) if text]
    return _find_leak(payload, needles, ambiguous_is_path = True, ignore = frozenset(ignore))


def _echoed(value: Any, echo: frozenset) -> bool:
    """A reference cannot stand in for a path the caller NAMED: the HMAC is process-stable, so
    referencing caller-chosen input is a confirmation oracle."""
    text = _as_text(value)
    return bool(text) and text in echo


def _redact(
    payload: Any,
    *,
    redact_ambiguous_path: bool,
    echo: frozenset = frozenset(),
) -> Any:
    dumped = _dump_model(payload)
    if dumped is not None:
        return _redact(dumped, redact_ambiguous_path = redact_ambiguous_path, echo = echo)
    if isinstance(payload, Mapping):
        out: dict[Any, Any] = {}
        reference: Optional[str] = None
        # Read before the walk: the sibling that decides it may come after it in the dump.
        base_model_is_local = _conditional_path_is_local(payload)
        identity_is_a_path = redact_ambiguous_path and _row_identity_is_a_path(payload)
        for key, value in payload.items():
            if key in HOST_PATH_IDENTITY_FIELDS and (
                (identity_is_a_path and key in HOST_PATH_ROW_IDENTITY_FIELDS)
                or _identity_value_is_a_path(value)
            ):
                out[key] = value if _echoed(value, echo) else _referenced_identity(value)
                continue
            if key in HOST_PATH_IDENTITY_LIST_FIELDS and isinstance(value, (list, tuple)):
                out[key] = [
                    _referenced_identity(item)
                    if _identity_value_is_a_path(item) and not _echoed(item, echo)
                    else item
                    for item in value
                ]
                continue
            if identity_is_a_path and key == HOST_PATH_ENCODED_IDENTITY_FIELD:
                out[key] = value if _echoed(value, echo) else _referenced_inventory_id(value)
                continue
            if (
                key == HOST_PATH_CONDITIONAL_FIELD
                and not base_model_is_local
                and _identity_value_is_a_path(value)
            ):
                out[key] = value if _echoed(value, echo) else _referenced_identity(value)
                continue
            if key in HOST_PATH_HANDLE_FIELDS:
                # Only where the value really is a path: a relative output dir names no layout.
                out[key] = (
                    _referenced_identity(value)
                    if _identity_value_is_a_path(value) and not _echoed(value, echo)
                    else value
                )
                continue
            if (
                key in HOST_PATH_SCALAR_FIELDS
                or (redact_ambiguous_path and key == HOST_PATH_AMBIGUOUS_FIELD)
                or (base_model_is_local and key == HOST_PATH_CONDITIONAL_FIELD)
            ):
                # Only the row-level cache dir: one per scan root is layout again.
                if (
                    key in {"cache_path", "repo_path"}
                    and reference is None
                    and not _echoed(value, echo)
                ):
                    reference = cache_reference(value)
                out[key] = None if value is None else ""
                continue
            if key in HOST_PATH_HANDLE_LIST_FIELDS and isinstance(value, (list, tuple)):
                out[key] = [
                    _referenced_identity(item)
                    if _identity_value_is_a_path(item) and not _echoed(item, echo)
                    else item
                    for item in value
                ]
                continue
            if key in HOST_PATH_LIST_FIELDS:
                out[key] = []
                continue
            if key in HOST_PATH_TEXT_FIELDS and isinstance(value, str):
                out[key] = redact_paths_in_text(value)
                continue
            out[key] = _redact(value, redact_ambiguous_path = redact_ambiguous_path, echo = echo)
        # After the walk: the field is declared on the models, so the dump's None would win.
        if reference is not None and not out.get(CACHE_REFERENCE_FIELD):
            out[CACHE_REFERENCE_FIELD] = reference
        return out
    if isinstance(payload, (list, tuple)):
        redacted = [
            _redact(item, redact_ambiguous_path = redact_ambiguous_path, echo = echo)
            for item in payload
        ]
        if not isinstance(payload, tuple):
            return redacted
        # A NamedTuple's constructor takes the fields one by one, so a list rebuild raises.
        if hasattr(payload, "_fields"):
            try:
                return type(payload)(*redacted)
            except Exception:
                return tuple(redacted)
        return type(payload)(redacted)
    return payload


def _find_leak(
    payload: Any,
    needles: list[str],
    *,
    ambiguous_is_path: bool,
    ignore: frozenset[str] = frozenset(),
) -> Optional[str]:
    dumped = _dump_model(payload)
    if dumped is not None:
        return _find_leak(dumped, needles, ambiguous_is_path = ambiguous_is_path, ignore = ignore)
    if isinstance(payload, Mapping):
        base_model_is_local = _conditional_path_is_local(payload)
        for key, value in payload.items():
            if key in ignore:
                continue
            text = _as_text(value)
            is_path_field = (
                key in HOST_PATH_SCALAR_FIELDS
                or key in HOST_PATH_HANDLE_FIELDS
                or (ambiguous_is_path and key == HOST_PATH_AMBIGUOUS_FIELD)
                or (base_model_is_local and key == HOST_PATH_CONDITIONAL_FIELD)
            )
            if is_path_field and text and _looks_absolute(text):
                return f"{key}={text}"
            if key in HOST_PATH_IDENTITY_FIELDS and _identity_value_is_a_path(text):
                return f"{key}={text}"
            if (key in HOST_PATH_LIST_FIELDS or key in HOST_PATH_HANDLE_LIST_FIELDS) and isinstance(
                value, (list, tuple)
            ):
                for item in value:
                    item_text = _as_text(item)
                    if item_text and _looks_absolute(item_text):
                        return f"{key}[]={item_text}"
            if text and any(needle in text for needle in needles):
                return f"{key}={text}"
            found = _find_leak(value, needles, ambiguous_is_path = ambiguous_is_path, ignore = ignore)
            if found is not None:
                return found
        return None
    if isinstance(payload, (list, tuple)):
        for item in payload:
            found = _find_leak(item, needles, ambiguous_is_path = ambiguous_is_path, ignore = ignore)
            if found is not None:
                return found
        return None
    text = _as_text(payload)
    if text and any(needle in text for needle in needles):
        return text
    return None


def short_path_for_log(value: Any) -> str:
    text = _as_text(value)
    if not text:
        return ""
    # A RELATIVE path is returned as written: `.../models/repo` invents a parent it lacks.
    if not _is_absolute_for_log(text):
        return text
    parts = [part for part in text.replace("\\", "/").split("/") if part]
    if len(parts) <= 2:
        # Rebuilding turned `/srv/cache` into `srv/cache`.
        return text
    return ".../" + "/".join(parts[-2:])


def _is_absolute_for_log(text: str) -> bool:
    """Judged as WRITTEN, not against this host's OS: a Windows path read on Linux is still one."""
    return bool(text.startswith(("/", "\\")) or re.match(r"^[A-Za-z]:[\\/]", text))


# Quotes, whitespace, punctuation and `=` terminate the run, so a two-path line is not swallowed
# whole. Spaces are allowed INSIDE a component.
_PATH_COMPONENT = r"[^\s\\/\n\":;,=](?:[^\\/\n\":;,=]*[^\s\\/\n\":;,=])?"

_ABSOLUTE_PATH_RE = re.compile(
    # Not preceded by a word character, colon or slash, so a URL and a ratio like "3/4" survive.
    # Components are a DENYLIST of terminators, or `client(acme)` stops the run at the `(`.
    r"(?<![\w:/.])(?:\\\\[^\\/\s]+[\\/]|[A-Za-z]:[\\/]|/)"
    r"(?:" + _PATH_COMPONENT + r"[\\/])*" + _PATH_COMPONENT
)


def scrub_paths(text: Any) -> str:
    """Shorten every absolute path inside a log message. Not a security control: the log is local."""
    # str(), not _as_text: the usual argument is an exception, whose message is the whole point.
    message = text if isinstance(text, str) else ("" if text is None else str(text))
    if not message:
        return ""
    return _ABSOLUTE_PATH_RE.sub(lambda match: short_path_for_log(match.group(0)), message)


_REDACTED_PATH = "<path>"

# The leftover when a directory name contains a terminator (`Acme, Inc` ends the run early): a
# terminator plus text carrying one more separator is replaced too, that last condition being
# what keeps the terminators terminating. The set must match `_PATH_COMPONENT`'s, `=` included,
# or `/srv/cache/foo=bar/config.json` publishes `<path>=bar/config.json`.
_REDACTED_TAIL_TEXT = r"[^\s\\/\":;,=][^\\/\":;,=]*"
_REDACTED_TAIL_RE = re.compile(
    re.escape(_REDACTED_PATH)
    + r"(?:(?:[:;,=][ ]?"
    + _REDACTED_TAIL_TEXT
    + r")+"
    + r"(?:[\\/]"
    + _REDACTED_TAIL_TEXT
    + r")+)+"
)


def redact_paths_in_text(text: Any) -> str:
    """Removed, not shortened: the component ``scrub_paths`` keeps is layout once text leaves."""
    message = text if isinstance(text, str) else ("" if text is None else str(text))
    if not message:
        return ""
    return _REDACTED_TAIL_RE.sub(_REDACTED_PATH, _ABSOLUTE_PATH_RE.sub(_REDACTED_PATH, message))


def redact_inventory_error_detail(detail: Any, *, via_api_key: bool) -> Any:
    """The response redactors only see a payload that was BUILT; a raising route comes here."""
    if host_paths_visible(via_api_key):
        return detail
    if isinstance(detail, str):
        return redact_paths_in_text(detail)
    if isinstance(detail, Mapping):
        return {
            key: redact_inventory_error_detail(value, via_api_key = via_api_key)
            for key, value in detail.items()
        }
    if isinstance(detail, (list, tuple)):
        redacted = [
            redact_inventory_error_detail(value, via_api_key = via_api_key) for value in detail
        ]
        if not isinstance(detail, tuple):
            return redacted
        # A NamedTuple takes its fields one by one; handing it the list raises TypeError, and
        # this runs while a route is already raising, so the caller gets a 500 not the refusal.
        if hasattr(detail, "_fields"):
            try:
                return type(detail)(*redacted)
            except Exception:  # noqa: BLE001 -- a subclass with its own constructor
                return tuple(redacted)
        try:
            return type(detail)(redacted)
        except Exception:  # noqa: BLE001 -- same
            return tuple(redacted)
    return detail


def redact_load_progress(progress: Any, *, via_api_key: bool) -> Any:
    """A worker-thread failure stores ``str(exc)``, naming the path a ``ref:`` load resolved to,
    answered on a LATER request whose handle map is empty."""
    if not isinstance(progress, Mapping):
        return progress
    if host_paths_visible(via_api_key) or not progress.get("error"):
        return dict(progress)
    return {**progress, "error": redact_paths_in_text(progress["error"])}


def raised_inventory_detail(detail: Any, *, via_api_key: bool) -> Any:
    """For a payload that was RAISED, since the route's wrappers never run. Restore first so a
    path the caller owns comes back as its handle, then redact what is left."""
    return redact_inventory_error_detail(restore_inventory_handles(detail), via_api_key = via_api_key)


def _dump_model(payload: Any) -> Optional[dict]:
    """A pydantic response object as a plain dict, or ``None``. Duck-typed: imported beneath."""
    dump = getattr(payload, "model_dump", None)
    if dump is None or isinstance(payload, type) or isinstance(payload, Mapping):
        return None
    try:
        dumped = dump()
    except Exception:
        return None
    return dumped if isinstance(dumped, dict) else None


def _looks_absolute(text: str) -> bool:
    if text.startswith(_REFERENCE_PREFIX):
        return False
    if text.startswith("/") or text.startswith("\\\\"):
        return True
    return len(text) > 2 and text[1] == ":" and text[2] in "\\/"


def _as_text(value: Any) -> str:
    if value is None or isinstance(value, bool):
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, os.PathLike):
        return os.fspath(value)
    return ""
