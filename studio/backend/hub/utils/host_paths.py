# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Keep the host's filesystem layout inside the host.

The local model inventory is a filesystem listing, so its rows carry the directory each copy
was found in. That is the right answer for the browser session, which is the operator looking
at their own machine and which hands the path straight back to the delete and load endpoints.
It is not the right answer for an ``sk-unsloth`` API key: that caller is on the API boundary,
it is explicitly denied the operator's ambient Hugging Face credential, and the absolute cache
root, the home directory it sits under and the account name inside it are host detail it never
needs in order to enumerate what is downloaded.

So the inventory still answers every authenticated caller, with the same repo ids, sizes,
capabilities and partial state as before, and every host path in the payload is replaced for
an API key by an opaque reference. The reference is stable for the life of the server, so a
client can still tell two rows in the same repo apart, group a companion asset with its model
and de-duplicate a repo cached under two roots, which is all the path was carrying for it.

Nothing here is an authorization decision: the caller was already authenticated, and the
routes deliberately keep serving them. Read it as output encoding for one caller class.

One field is deliberately NOT redacted. ``load_id`` on a cached row is a snapshot directory
exactly when the row's bare repo id cannot reach that copy: it sits in a legacy or default cache
while another is active, or ``refs/main`` resolves to an older or torn revision. It is then the
handle the load and train endpoints take, so blanking it would leave an API key able to see a
model and unable to load it, and substituting the repo id would load a different revision. The
endpoints do not accept an opaque reference yet, so this module cannot give the caller something
better; it is a smaller disclosure than the inventory, since it names one model the caller can
already list rather than the shape of the cache, and ``response_leaks_host_path`` makes every
test that tolerates it say so at the call site.
"""

from __future__ import annotations

import hmac
import os
import re
import secrets
from hashlib import sha256
from typing import Any, Iterable, Mapping, Optional

# Scalar fields whose value is a path on this host. Kept as one list so a new route cannot
# quietly introduce a path field that the drift test does not know about.
HOST_PATH_SCALAR_FIELDS = frozenset(
    {
        "cache_path",
        "load_cache_path",
        "hf_cache_dir",
        "models_dir",
        "repo_path",
        "snapshot_path",
        "local_path",
    }
)

# List-of-path fields. Emptied rather than referenced: they name scan ROOTS, which carry the
# host's layout and nothing a caller can act on.
HOST_PATH_LIST_FIELDS = frozenset(
    {
        "exact_paths",
        "lmstudio_dirs",
        "ollama_dirs",
        "hermes_dirs",
        "scanned_dirs",
    }
)

# ``path`` is a path on the inventory objects and a repo id or a display name elsewhere, so it
# is only redacted inside the models the routes name here, never by field name alone.
HOST_PATH_AMBIGUOUS_FIELD = "path"

# Conditionally a host path, decided by a SIBLING rather than by the route. ``base_model`` is
# ``adapter_config.json``'s ``base_model_name_or_path`` verbatim: a Hub repo id for most
# adapters, and an absolute path on this machine for one trained against a local base. It
# cannot join HOST_PATH_SCALAR_FIELDS, which would blank a legitimate repo id on every LoRA
# row and take away the thing a caller actually needs. The scan already writes which one it
# is -- ``_base_model_source`` answers "local" only after resolving the value on this
# filesystem -- so the pair decides it, and a row marked local is redacted like any other path.
HOST_PATH_CONDITIONAL_FIELD = "base_model"
HOST_PATH_CONDITIONAL_SOURCE_FIELD = "base_model_source"
HOST_PATH_CONDITIONAL_SOURCE_LOCAL = "local"


def _conditional_path_is_local(payload: Mapping) -> bool:
    """Whether this mapping's ``base_model`` is a host path rather than a repo id."""
    return payload.get(HOST_PATH_CONDITIONAL_SOURCE_FIELD) == HOST_PATH_CONDITIONAL_SOURCE_LOCAL


# Sibling written beside a redacted scalar, so a client keeps the identity the path gave it.
CACHE_REFERENCE_FIELD = "cache_ref"

_REFERENCE_PREFIX = "ref:"

# Per-process, so a reference cannot be brute-forced back into a path by a caller who guesses
# at candidate home directories, and cannot be correlated across installs. Stable for the life
# of the server, which is the only window a client compares references in.
_REFERENCE_KEY = secrets.token_bytes(32)


def host_paths_visible(via_api_key: Any) -> bool:
    """Whether this caller class may see host paths.

    One place, so the rule reads the same in every route: the browser session sees the machine
    it is running on, an API key does not. ``via_api_key`` comes from
    ``auth.authentication.authenticated_via_api_key``, the same dependency the ambient Hugging
    Face token boundary is drawn with, rather than a second notion of caller class.

    Anything other than a bool means there was no HTTP request to classify: a route function
    called in process, by the CLI or by a test, receives its unresolved ``Depends`` marker,
    which is truthy and would otherwise read as "an API key asked". An in-process caller is the
    operator's own process, and there is no API key in the picture, so it sees paths.
    """
    if not isinstance(via_api_key, bool):
        return True
    return not via_api_key


def cache_reference(value: Any) -> Optional[str]:
    """An opaque, stable, non-reversible stand-in for one host path."""
    text = _as_text(value)
    if not text:
        return None
    digest = hmac.new(_REFERENCE_KEY, os.fsencode(text), sha256).hexdigest()
    return f"{_REFERENCE_PREFIX}{digest[:32]}"


def redact_host_paths(payload: Any, *, via_api_key: bool) -> Any:
    """Return *payload* with every host path removed, if this caller may not see them.

    Walks dicts and sequences so one call covers a route that answers a row, a list of rows or
    a response object with rows nested inside it. Returns the payload unchanged for a caller
    that may see paths, including the identical object, so the browser session pays nothing.
    """
    if host_paths_visible(via_api_key):
        return payload
    return _redact(payload, redact_ambiguous_path = False)


def redact_inventory_host_paths(payload: Any, *, via_api_key: bool) -> Any:
    """``redact_host_paths`` plus the ambiguous ``path`` field.

    For the routes whose ``path`` is documented as a directory on this host: the scan folder
    list, the models folder and the local model rows.
    """
    if host_paths_visible(via_api_key):
        return payload
    return _redact(payload, redact_ambiguous_path = True)


def response_leaks_host_path(
    payload: Any,
    roots: Iterable[Any] = (),
    *,
    ignore: Iterable[str] = (),
) -> Optional[str]:
    """The first host path found in *payload*, or ``None``. For tests and for the drift gate.

    Looks for any of *roots* as a substring and for an absolute path in any known path field,
    so a new field spelled differently is still caught by the root scan. ``ignore`` names the
    fields a caller has decided must keep a path, and is there so that decision has to be
    written down at each call rather than weakened by loosening this function.
    """
    needles = [text for text in (_as_text(root) for root in roots) if text]
    return _find_leak(payload, needles, ambiguous_is_path = True, ignore = frozenset(ignore))


def _redact(payload: Any, *, redact_ambiguous_path: bool) -> Any:
    dumped = _dump_model(payload)
    if dumped is not None:
        return _redact(dumped, redact_ambiguous_path = redact_ambiguous_path)
    if isinstance(payload, Mapping):
        out: dict[Any, Any] = {}
        reference: Optional[str] = None
        # Read before the walk: the sibling that decides it may come after it in the dump.
        base_model_is_local = _conditional_path_is_local(payload)
        for key, value in payload.items():
            if (
                key in HOST_PATH_SCALAR_FIELDS
                or (redact_ambiguous_path and key == HOST_PATH_AMBIGUOUS_FIELD)
                or (base_model_is_local and key == HOST_PATH_CONDITIONAL_FIELD)
            ):
                # Only the row-level cache directory earns a reference: one per scan root would
                # say "these two rows came from the same root", which is host layout again.
                if key in {"cache_path", "repo_path"} and reference is None:
                    reference = cache_reference(value)
                out[key] = None if value is None else ""
                continue
            if key in HOST_PATH_LIST_FIELDS:
                out[key] = []
                continue
            out[key] = _redact(value, redact_ambiguous_path = redact_ambiguous_path)
        # After the walk, not during it: the field is declared on the response models, so a
        # model dump carries it as None and writing it inside the loop let that None win.
        if reference is not None and not out.get(CACHE_REFERENCE_FIELD):
            out[CACHE_REFERENCE_FIELD] = reference
        return out
    if isinstance(payload, (list, tuple)):
        redacted = [_redact(item, redact_ambiguous_path = redact_ambiguous_path) for item in payload]
        if not isinstance(payload, tuple):
            return redacted
        # A NamedTuple is a tuple whose constructor takes the fields one by one, so rebuilding
        # it from the list raises and the route 500s on a payload it was only meant to edit.
        # Rebuild it the way its own type is built, and keep the plain-tuple case as it was.
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
                or (ambiguous_is_path and key == HOST_PATH_AMBIGUOUS_FIELD)
                or (base_model_is_local and key == HOST_PATH_CONDITIONAL_FIELD)
            )
            if is_path_field and text and _looks_absolute(text):
                return f"{key}={text}"
            if key in HOST_PATH_LIST_FIELDS and isinstance(value, (list, tuple)):
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
    """One path, shortened to the two components that identify it.

    The log is the operator's, so this is hygiene rather than a boundary: an inventory warning
    wants to say WHICH repo directory it skipped, and it can do that without writing out the
    home directory, the account name inside it and the cache root on every line.
    """
    text = _as_text(value)
    if not text:
        return ""
    parts = [part for part in text.replace("\\", "/").split("/") if part]
    if len(parts) <= 2:
        return "/".join(parts)
    return ".../" + "/".join(parts[-2:])


# An absolute POSIX path or a Windows drive path. Deliberately conservative about what may sit
# inside one: quotes, whitespace and the punctuation that ends a sentence all terminate the run,
# so a message that merely mentions a ratio or a URL is left alone.
_ABSOLUTE_PATH_RE = re.compile(
    # Not preceded by a word character, a colon or a slash, so a URL's "//host/path" and a
    # ratio like "3/4" are left as they were written.
    # A UNC share is the third spelling of absolute on Windows, and a scan folder on a network
    # share is written that way, so the whole server and share name used to stay in the line.
    r"(?<![\w:/])(?:\\\\[^\\/\s]+[\\/]|[A-Za-z]:[\\/]|/)(?:[\w.\-+@ ]+[\\/])+[\w.\-+@]*"
)


def scrub_paths(text: Any) -> str:
    """Shorten every absolute path inside a log message.

    Pairs with ``download_registry.scrub_secrets``, which does the same for credentials: between
    them a line built from an exception carries neither the operator's token nor their filesystem
    layout. Not a security control, since the log is local.
    """
    # str(), not _as_text: the usual argument is an exception, whose message is the whole point.
    message = text if isinstance(text, str) else ("" if text is None else str(text))
    if not message:
        return ""
    return _ABSOLUTE_PATH_RE.sub(lambda match: short_path_for_log(match.group(0)), message)


def _dump_model(payload: Any) -> Optional[dict]:
    """A pydantic response object as a plain dict, or ``None`` when it is not one.

    Duck-typed rather than importing pydantic: the routes return whichever of the two shapes is
    convenient, and this module is imported beneath them.
    """
    dump = getattr(payload, "model_dump", None)
    if dump is None or isinstance(payload, type) or isinstance(payload, Mapping):
        return None
    try:
        dumped = dump()
    except Exception:
        return None
    return dumped if isinstance(dumped, dict) else None


def _looks_absolute(text: str) -> bool:
    """POSIX and Windows both, since a response is read on the machine that asked, not the host."""
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
