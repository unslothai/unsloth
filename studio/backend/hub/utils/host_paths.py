# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Keep the host's filesystem layout inside the host.

For an ``sk-unsloth`` API key every host path in an inventory payload is replaced by an opaque
reference, stable for the life of the server so a client can still group and de-duplicate rows.
Not an authorization decision: output encoding for one caller class.
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
from hashlib import sha256
from typing import Any, Iterable, Mapping, Optional

# One list, so a new route cannot quietly introduce a path field the drift test misses.
HOST_PATH_SCALAR_FIELDS = frozenset(
    {
        "cache_path",
        "load_cache_path",
        "hf_cache_dir",
        "models_dir",
        "repo_path",
        "snapshot_path",
        "local_path",
        # The training half: a run persists where it wrote, and the history routes answer with it.
        "model_local_path",
        "model_snapshot_path",
        "dataset_local_path",
        "dataset_snapshot_path",
        "dataset_path",
        # Read off TrainingStartRequest rather than guessed: the UI copies these into config_json.
        "tensorboard_dir",
    }
)

# Referenced, not blanked: blanking left `can_resume` true beside nothing to resume with.
HOST_PATH_HANDLE_FIELDS = frozenset({"output_dir", "checkpoint_path", "resume_from_checkpoint"})

# Error text: scrubbed, not blanked, being the only account of WHY a run failed.
HOST_PATH_TEXT_FIELDS = frozenset({"error_message", "error", "detail", "message"})

# Emptied, not referenced: scan ROOTS carry layout and nothing actionable.
HOST_PATH_LIST_FIELDS = frozenset(
    {
        "exact_paths",
        "lmstudio_dirs",
        "ollama_dirs",
        "hermes_dirs",
        "scanned_dirs",
        # The list-valued training equivalents, emptied for the same reason.
        "output_dirs",
        "dataset_paths",
    }
)

# Referenced entry by entry: emptying these leaves the replay with a resumable checkpoint and
# no dataset to resume it against.
HOST_PATH_HANDLE_LIST_FIELDS = frozenset({"local_datasets", "local_eval_datasets"})

# ``path`` is a path on the inventory objects and a repo id elsewhere, so it is only redacted
# inside the models the routes name here, never by field name alone.
HOST_PATH_AMBIGUOUS_FIELD = "path"

# Decided by a SIBLING: ``base_model`` carries ``base_model_name_or_path`` verbatim, a repo id
# for most adapters and a path for one trained locally, so field-name blanking costs every row.
HOST_PATH_CONDITIONAL_FIELD = "base_model"
HOST_PATH_CONDITIONAL_SOURCE_FIELD = "base_model_source"
HOST_PATH_CONDITIONAL_SOURCE_LOCAL = "local"


def _conditional_path_is_local(payload: Mapping) -> bool:
    return payload.get(HOST_PATH_CONDITIONAL_SOURCE_FIELD) == HOST_PATH_CONDITIONAL_SOURCE_LOCAL


# A row whose IDENTITY is a path: blanking `path` hides nothing while `id`, `load_id` and
# `inventory_id` still spell it out, and a row with no identity cannot be told from the next.
#
# `repo_id`, `model_name`, `active_model`, `model_identifier` and `dataset_name` are decided by
# VALUE, not field name: a path exactly when a load or run PERSISTED what it resolved, which the
# status routes then answer with long after the resolving request ended.
HOST_PATH_IDENTITY_FIELDS = (
    "id",
    "load_id",
    "repo_id",
    "model_name",
    "active_model",
    "model_identifier",
    "dataset_name",
)
# The same identities, published as lists. Each ENTRY is decided on its own value.
HOST_PATH_IDENTITY_LIST_FIELDS = ("loaded", "loading")
# The subset a LOCAL row is named by, referenced on the row's source alone; the others only
# when the value itself is a path.
HOST_PATH_ROW_IDENTITY_FIELDS = ("id", "load_id")
HOST_PATH_ENCODED_IDENTITY_FIELD = "inventory_id"
HOST_PATH_ROW_SOURCE_FIELD = "source"
# `hf_cache` is absent: those rows are named by repo id, and where they are not, value decides.
HOST_PATH_LOCAL_SOURCES = frozenset({"models_dir", "lmstudio", "ollama", "hermes", "custom"})


def _row_identity_is_a_path(payload: Mapping) -> bool:
    return payload.get(HOST_PATH_ROW_SOURCE_FIELD) in HOST_PATH_LOCAL_SOURCES


def _identity_value_is_a_path(value: Any) -> bool:
    """Whether THIS identity spells out a host path, whatever its row's source says: a copy
    outside the active cache pins a cache row to an absolute snapshot path."""
    return isinstance(value, str) and bool(value) and _looks_absolute(value)


def _referenced_identity(value: Any) -> Any:
    if not isinstance(value, str) or not value:
        return value
    return cache_reference(value) or ""


def _referenced_inventory_id(value: Any) -> Any:
    """`<source>:<format>:<url-encoded identity>[:<variant>]` with the identity referenced. The
    shape is kept because clients split on it."""
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
    """Whether this caller class may see host paths. Anything other than a bool means there was
    no HTTP request to classify: an in-process call receives its unresolved ``Depends`` marker,
    which is truthy and would read as "an API key asked"."""
    if not isinstance(via_api_key, bool):
        return True
    return not via_api_key


# References this process has handed out, so a caller can ACT on a row without being told where
# the file is. Per-process key, so one can never be forged or carried in from elsewhere.
_REFERENCE_LIMIT = 8192
# Evicted by AGE, not count: one row costs several entries and the inventory has no row limit,
# so count-based eviction drops entries of the response being built RIGHT NOW. The hard ceiling
# bounds the table: past it the oldest goes whatever its age.
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


# Resolved path -> the reference the caller sent, for THIS request only, so one caller's
# substitutions cannot reach another's response. What goes out must be the string that came in.
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
    """Put every handle back where its resolved path ended up in *payload*. A substring swap, not
    a field list, because the path also appears in labels and error details."""
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
        try:
            return type(payload)(*restored)
        except Exception:  # noqa: BLE001 -- a plain tuple, not a NamedTuple
            return tuple(restored)
    if isinstance(payload, str):
        text = payload
        # Longest first, so a path that is a prefix of another does not claim its text.
        for path in sorted(known, key = len, reverse = True):
            if path in text:
                text = text.replace(path, known[path])
        return text
    return payload


def resolve_host_path_reference(value: Any) -> Optional[str]:
    """The path a reference stands for, or None. Not an authorization decision: callers stay
    subject to the checks a directly named path gets."""
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
    """Return *payload* with every host path removed, if this caller may not see them. Returns the
    identical object otherwise, so a browser session pays nothing. ``echo`` holds the values THIS
    CALLER supplied, returned as written; see ``_echoed``."""
    if host_paths_visible(via_api_key):
        return payload
    return _redact(payload, redact_ambiguous_path = False, echo = _echo_set(echo))


def redact_inventory_host_paths(
    payload: Any,
    *,
    via_api_key: bool,
    echo: Iterable[Any] = (),
) -> Any:
    """``redact_host_paths`` plus the ambiguous ``path`` field, for the routes whose ``path`` is a
    directory on this host."""
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
    """The first host path found in *payload*, or ``None``. For tests and the drift gate.
    ``ignore`` exists so keeping a path is written down at each call rather than by loosening
    this function."""
    needles = [text for text in (_as_text(root) for root in roots) if text]
    return _find_leak(payload, needles, ambiguous_is_path = True, ignore = frozenset(ignore))


def _echoed(value: Any, echo: frozenset) -> bool:
    """Whether *value* is one this caller just sent, and so is theirs to be handed back. A
    reference cannot stand in for a path the caller NAMED: it is one stable HMAC for the life of
    the process, so referencing caller-chosen input is an online confirmation oracle."""
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
                # Referenced, not blanked: nothing else here names the base.
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
                # Only the row-level cache dir earns a reference: one per scan root would say
                # "these two rows share a root", which is layout again.
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
                # Scrubbed, not blanked: the only account of why a run ended.
                out[key] = redact_paths_in_text(value)
                continue
            out[key] = _redact(value, redact_ambiguous_path = redact_ambiguous_path, echo = echo)
        # After the walk: the field is declared on the response models, so a dump carries it as None
        # and writing it inside the loop let that None win.
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
            # An identity field is a repo id most of the time, so the VALUE decides.
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
        # Already the two components this keeps, and rebuilding turned `/srv/cache` into `srv/cache`.
        return text
    return ".../" + "/".join(parts[-2:])


def _is_absolute_for_log(text: str) -> bool:
    """The three spellings of absolute, judged as WRITTEN rather than against this host's OS: a
    Windows path logged on Windows is still one when the line is read on Linux."""
    return bool(text.startswith(("/", "\\")) or re.match(r"^[A-Za-z]:[\\/]", text))


# An absolute POSIX or Windows drive path. Quotes, whitespace, sentence punctuation and `=`
# terminate the run, so a ratio or a URL is left alone and a two-path line is not swallowed
# whole; spaces are allowed INSIDE a component (`Program Files`).
_PATH_COMPONENT = r"[^\s\\/\n\":;,=](?:[^\\/\n\":;,=]*[^\s\\/\n\":;,=])?"

_ABSOLUTE_PATH_RE = re.compile(
    # Not preceded by a word character, colon or slash, so a URL and a ratio like "3/4" survive. UNC
    # is the third spelling of absolute on Windows. The trailing group is `*`, not `+`: `/tmp` is as
    # absolute as a deep path. The component set is a DENYLIST of terminators, not an allowlist, or
    # `client(acme)` stops the run at the punctuation. A DOT terminates, so `./models/repo` keeps its
    # leading separator.
    r"(?<![\w:/.])(?:\\\\[^\\/\s]+[\\/]|[A-Za-z]:[\\/]|/)"
    r"(?:" + _PATH_COMPONENT + r"[\\/])*" + _PATH_COMPONENT
)


def scrub_paths(text: Any) -> str:
    """Shorten every absolute path inside a log message. Pairs with
    ``download_registry.scrub_secrets``. Not a security control, since the log is local."""
    # str(), not _as_text: the usual argument is an exception, whose message is the whole point.
    message = text if isinstance(text, str) else ("" if text is None else str(text))
    if not message:
        return ""
    return _ABSOLUTE_PATH_RE.sub(lambda match: short_path_for_log(match.group(0)), message)


_REDACTED_PATH = "<path>"

# What is left over when a directory name contains a terminator: `Acme, Inc` ends the run early.
# So a tail that continues the path -- a terminator then text carrying at least one more
# separator -- is replaced too, and that last condition is what keeps the terminators
# terminating. The cost is over-removing prose, the safe direction for text that leaves the host.
_REDACTED_TAIL_TEXT = r"[^\s\\/\":;,][^\\/\":;,]*"
_REDACTED_TAIL_RE = re.compile(
    re.escape(_REDACTED_PATH)
    + r"(?:(?:[:;,][ ]?"
    + _REDACTED_TAIL_TEXT
    + r")+"
    + r"(?:[\\/]"
    + _REDACTED_TAIL_TEXT
    + r")+)+"
)


def redact_paths_in_text(text: Any) -> str:
    """Remove every absolute path from a message, rather than shortening it. ``scrub_paths`` keeps
    the last component because a local log is read by the person who owns the filesystem; for text
    that LEAVES the host that component is still their layout."""
    message = text if isinstance(text, str) else ("" if text is None else str(text))
    if not message:
        return ""
    return _REDACTED_TAIL_RE.sub(_REDACTED_PATH, _ABSOLUTE_PATH_RE.sub(_REDACTED_PATH, message))


def redact_inventory_error_detail(detail: Any, *, via_api_key: bool) -> Any:
    """An error detail with host paths removed for a caller that may not see them: the response
    redactors only see a payload that was BUILT, so a route that raises hands over the exception
    instead. Structured details are walked too."""
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
        return type(detail)(redacted) if isinstance(detail, tuple) else redacted
    return detail


def _dump_model(payload: Any) -> Optional[dict]:
    """A pydantic response object as a plain dict, or ``None``. Duck-typed, since this module is
    imported beneath the routes."""
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
