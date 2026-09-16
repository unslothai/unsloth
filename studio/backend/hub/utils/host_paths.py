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
import threading
from collections import OrderedDict
from contextvars import ContextVar
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
        # The training half. A run persists where it wrote and what it was given, and those
        # fields answer from the history routes for as long as the run exists, so an API-key
        # caller could read the server's home, cache and output layout out of any completed
        # run even with the model identity referenced. `output_dir` is the one on every
        # summary; the rest are the request fields the detail route echoes back, and they are
        # the same four the remote-code scan takes plus the dataset pair.
        "output_dir",
        "model_local_path",
        "model_snapshot_path",
        "dataset_local_path",
        "dataset_snapshot_path",
        "dataset_path",
        "checkpoint_path",
        "resume_from_checkpoint",
        # The persisted request's own spellings, read off TrainingStartRequest rather than
        # guessed: a run created through the UI copies these into config_json verbatim.
        "tensorboard_dir",
    }
)

# Error text, not a path field: the whole string is scrubbed rather than blanked, because the
# message is the only thing telling the caller WHY a run failed and a filename is usually only
# part of it. A trainer records `str(e)` here, and filesystem and model-loading errors quote
# the file they failed on.
HOST_PATH_TEXT_FIELDS = frozenset({"error_message", "error", "detail", "message"})

# List-of-path fields. Emptied rather than referenced: they name scan ROOTS, which carry the
# host's layout and nothing a caller can act on.
HOST_PATH_LIST_FIELDS = frozenset(
    {
        "exact_paths",
        "lmstudio_dirs",
        "ollama_dirs",
        "hermes_dirs",
        "scanned_dirs",
        # The list-valued training equivalents, emptied for the same reason: a list of output
        # directories is the host's layout however many entries it has.
        "output_dirs",
        "dataset_paths",
        "local_datasets",
        "local_eval_datasets",
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


# A row whose IDENTITY is a path. `_local_model_info` names a filesystem-backed model by the
# file it found: `id` and `load_id` are the absolute load path, and `inventory_id` carries the
# same path URL-encoded as its third component. Blanking `path` on such a row therefore hides
# nothing, and the load-handle exception recorded above is about CACHED rows, whose handle is
# a snapshot directory the caller already named by repo id.
#
# These rows are not blanked, because a row with no identity cannot be told apart from the
# next one: each identity becomes its opaque reference, which is stable within the response
# and across the life of the server and reverses to nothing.
# `repo_id` and `model_name` are here for the same reason as `load_id` and on the same
# terms: they are a repo id almost always, and an absolute path when the thing they name is
# a local folder -- which is what a load or a training run started from an inventory
# reference records, and what `/images/status`, `/video/status` and the training run list
# hand back long after the request that resolved the reference has ended. The value decides,
# so a repo id is never touched.
# `active_model` and `model_identifier` are the chat half of the same thing: a chat or GGUF
# row loaded from an inventory reference keeps the RESOLVED path as its resident identity, and
# `GET /api/inference/status` publishes it for as long as that model stays loaded -- long
# after the request that resolved the reference ended, so there is no handle left to put back.
# Value-decided like the two above it: almost always a repo id, and never touched when it is.
HOST_PATH_IDENTITY_FIELDS = (
    "id",
    "load_id",
    "repo_id",
    "model_name",
    "active_model",
    "model_identifier",
)
# The same identities, published as lists by the chat status route. Each ENTRY is decided on
# its own value, so a list of repo ids with one local path in it keeps every id.
HOST_PATH_IDENTITY_LIST_FIELDS = ("loaded", "loading")
# The subset a LOCAL row is named by, which is referenced on the strength of the row's
# source alone. The other two are only ever referenced when the value itself is a path: a
# local row can carry a `repo_id` that is not one, and blanking that would take away the
# only thing the caller could act on.
HOST_PATH_ROW_IDENTITY_FIELDS = ("id", "load_id")
HOST_PATH_ENCODED_IDENTITY_FIELD = "inventory_id"
HOST_PATH_ROW_SOURCE_FIELD = "source"
# `hf_cache` is absent because those rows are named by repo id. Where they are NOT -- a cached
# copy outside the active cache, or a `refs/main` that points at an unusable revision, both of
# which put an absolute snapshot path in `load_id` -- the value itself is the test, below.
HOST_PATH_LOCAL_SOURCES = frozenset({"models_dir", "lmstudio", "ollama", "hermes", "custom"})


def _row_identity_is_a_path(payload: Mapping) -> bool:
    """Whether this row is named by a file on this host rather than by a repo id."""
    return payload.get(HOST_PATH_ROW_SOURCE_FIELD) in HOST_PATH_LOCAL_SOURCES


def _identity_value_is_a_path(value: Any) -> bool:
    """Whether THIS identity spells out a host path, whatever its row's source says.

    The source tells you what a row is usually named by, and a cache row is usually named by
    its repo id. It is not always: a copy outside the active cache, or a `refs/main` that
    lands on a torn revision, pins the row to an absolute snapshot path instead, and that
    value carries the cache root, the home directory and the account name. There is nothing
    ambiguous left to weigh once the value is an absolute path, so this does not wait for
    `redact_ambiguous_path`: the cache listing is redacted without it.
    """
    return isinstance(value, str) and bool(value) and _looks_absolute(value)


def _referenced_identity(value: Any) -> Any:
    """One identity field, as its opaque reference. Non-strings and blanks are left alone."""
    if not isinstance(value, str) or not value:
        return value
    return cache_reference(value) or ""


def _referenced_inventory_id(value: Any) -> Any:
    """`<source>:<format>:<url-encoded identity>[:<variant>]` with the identity referenced.

    The shape is kept because clients split on it; only the component that spells out the
    host path is replaced.
    """
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


# References this process has handed out, so a caller can ACT on a row it was shown without
# being told where the file is. Only references issued by this process are in here, so
# resolving one can never name a path that was not already listed to somebody; the key is
# per-process, so a reference cannot be forged or carried in from anywhere else. Bounded and
# oldest-first, because a long-lived server lists a great many rows; a reference that has
# aged out simply does not resolve, and the load fails the way an unknown model does.
_REFERENCE_LIMIT = 8192
_reference_paths: "OrderedDict[str, str]" = OrderedDict()
_reference_lock = threading.Lock()


def cache_reference(value: Any) -> Optional[str]:
    """An opaque, stable stand-in for one host path, resolvable only in this process."""
    text = _as_text(value)
    if not text:
        return None
    digest = hmac.new(_REFERENCE_KEY, os.fsencode(text), sha256).hexdigest()
    reference = f"{_REFERENCE_PREFIX}{digest[:32]}"
    with _reference_lock:
        _reference_paths[reference] = text
        _reference_paths.move_to_end(reference)
        while len(_reference_paths) > _REFERENCE_LIMIT:
            _reference_paths.popitem(last = False)
    return reference


# The handles resolved while serving THIS request, resolved path -> the reference the caller
# actually sent. Per-request rather than global: a context variable is copied into each
# request's task, so one caller's substitutions cannot reach another's response.
#
# Resolving a reference is what makes a redacted row loadable, and the resolved path then
# comes back out in the answer -- `ValidateModelResponse.identifier`, `LoadResponse.model`
# and the label beside it, and any error detail that quotes what was asked for. A caller who
# may not see host paths could therefore enumerate a redacted listing and read the path
# straight back out of the load it performed with the reference. What goes out is the string
# that came in.
_request_handles: "ContextVar[Optional[dict[str, str]]]" = ContextVar(
    "unsloth_request_inventory_handles", default = None
)


def note_resolved_handle(handle: str, path: str) -> None:
    """Record that *handle* stood for *path* while serving this request."""
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
    """Put every handle back where its resolved path ended up in *payload*.

    A substring swap rather than a field list, because the path does not only appear as the
    identity: it is the display label when there is nothing better to call the model, it is
    embedded in the inference identifier, and it is quoted in the detail of anything that
    goes wrong. A field list would cover the first of those and be wrong about the rest.
    """
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
    """The path a reference stands for, or None for anything that is not one of ours.

    The counterpart of `cache_reference`, and the reason a redacted row is still ACTIONABLE:
    an API-key caller is shown `ref:...` as a local model's identity and hands it straight
    back when it asks to load that model. Without this the load would fail on a row the
    caller was invited to pick.

    Not an authorization decision. It answers only "which path did this process already
    list under this name", and every caller of it stays subject to whatever it applies to a
    path a caller names directly.
    """
    text = _as_text(value)
    if not text or not text.startswith(_REFERENCE_PREFIX):
        return None
    with _reference_lock:
        return _reference_paths.get(text)


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
        identity_is_a_path = redact_ambiguous_path and _row_identity_is_a_path(payload)
        for key, value in payload.items():
            if key in HOST_PATH_IDENTITY_FIELDS and (
                (identity_is_a_path and key in HOST_PATH_ROW_IDENTITY_FIELDS)
                or _identity_value_is_a_path(value)
            ):
                out[key] = _referenced_identity(value)
                continue
            if key in HOST_PATH_IDENTITY_LIST_FIELDS and isinstance(value, (list, tuple)):
                out[key] = [
                    _referenced_identity(item) if _identity_value_is_a_path(item) else item
                    for item in value
                ]
                continue
            if identity_is_a_path and key == HOST_PATH_ENCODED_IDENTITY_FIELD:
                out[key] = _referenced_inventory_id(value)
                continue
            if (
                key == HOST_PATH_CONDITIONAL_FIELD
                and not base_model_is_local
                and _identity_value_is_a_path(value)
            ):
                # A row with no `base_model_source` beside it -- the model-details lookup,
                # which reports a LoRA's `base_model_name_or_path` verbatim -- still says
                # plainly what it is when the VALUE is an absolute path. Referenced rather
                # than blanked, because unlike the inventory row above there is nothing else
                # in this answer naming the base, and the reference is the handle the caller
                # can hand back.
                out[key] = _referenced_identity(value)
                continue
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
            if key in HOST_PATH_TEXT_FIELDS and isinstance(value, str):
                # Scrubbed, not blanked: a persisted failure message is the only account of why
                # a run ended, and the path inside it is usually one clause of it.
                out[key] = redact_paths_in_text(value)
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
            # An identity field is named by a repo id most of the time and by a host path the
            # rest of it, so the value decides rather than the field name. Without this the
            # gate reported a clean response for a cache row pinned to a snapshot directory.
            if key in HOST_PATH_IDENTITY_FIELDS and _identity_value_is_a_path(text):
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
# One path component: no separator, no quote, no colon, none of the marks that end a clause,
# and no leading or trailing whitespace. Spaces INSIDE are allowed, because directory names
# have them.
_PATH_COMPONENT = r"[^\s\\/\n\":;,](?:[^\\/\n\":;,]*[^\s\\/\n\":;,])?"

_ABSOLUTE_PATH_RE = re.compile(
    # Not preceded by a word character, a colon or a slash, so a URL's "//host/path" and a
    # ratio like "3/4" are left as they were written.
    # A UNC share is the third spelling of absolute on Windows, and a scan folder on a network
    # share is written that way, so the whole server and share name used to stay in the line.
    # The trailing group is `*` rather than `+`: a root with a SINGLE component under it
    # (`/tmp`, `/cache`, `C:\\cache`) is just as absolute as a deep one, and requiring a
    # second separator left exactly those unredacted. A configured cache root is very often
    # spelled that way, and it is the value the models-folder error puts in its detail.
    # A component is anything that is not a separator, a quote, a colon or one of the marks
    # that end a clause, and it may not begin or end with whitespace. An ALLOWLIST was the
    # bug: a directory called `client(acme)`, `o'connor` or `Models (private)` fell outside
    # it, so the run stopped at the punctuation and only the fragments around it were
    # replaced -- `/srv/client(acme)/models` came back as `<path>(acme)<path>`, with the
    # customer name still in the line this exists to clean. Turned around, the excluded set
    # is the short one and it is the terminators: a colon so `Skipping /x: denied` keeps its
    # reason, a quote so a quoted path ends at the quote, and a comma or semicolon so a list
    # of paths is still a list.
    r"(?<![\w:/])(?:\\\\[^\\/\s]+[\\/]|[A-Za-z]:[\\/]|/)"
    r"(?:" + _PATH_COMPONENT + r"[\\/])*" + _PATH_COMPONENT
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


_REDACTED_PATH = "<path>"


def redact_paths_in_text(text: Any) -> str:
    """Remove every absolute path from a message, rather than shortening it.

    The stronger sibling of ``scrub_paths``. That one keeps the last component because a
    local log is read by the person who owns the filesystem, and a name is what makes the
    line useful. This one is for text that LEAVES the host, where the last component is
    still the operator's layout: a home directory name, a drive letter, a share.
    """
    message = text if isinstance(text, str) else ("" if text is None else str(text))
    if not message:
        return ""
    return _ABSOLUTE_PATH_RE.sub(_REDACTED_PATH, message)


def redact_inventory_error_detail(detail: Any, *, via_api_key: bool) -> Any:
    """An error detail with host paths removed for a caller that may not see them.

    The response redactors only ever see a payload that was BUILT. A route that raises
    while building one hands the client the exception instead, and these details are
    formatted with the path in them, so the one caller the redaction exists for got the
    host path out of the 500 rather than out of the 200.

    Strings are scrubbed; a structured detail is walked so a dict or list detail is covered
    too, since nothing stops a route from raising one.
    """
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
