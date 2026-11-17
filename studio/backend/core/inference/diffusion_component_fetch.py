# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""``load_components`` that raises ``ComponentFetchError`` instead of registering a failed component as None.

diffusers masks a failed safetensors fetch with a 404 on the ``.bin`` twin, so such a model component is retried once
with ``use_safetensors=True`` to load it or surface the real cause (e.g. a full disk).
"""

from __future__ import annotations

import errno
import re
import shutil
from typing import Any, Iterable, Iterator, Optional

_DISK_FULL_ERRNOS = frozenset({errno.ENOSPC, errno.EDQUOT})
# hf_xet reports it as text only.
_DISK_FULL_TEXT = re.compile(
    r"no space left on device|os error 28|\[errno 28\]|disk quota exceeded", re.I
)
_NETWORK_NAMES = frozenset(
    {
        "ConnectionError",
        "ConnectTimeout",
        "ConnectError",
        "ReadTimeout",
        "ReadError",
        "Timeout",
        "TimeoutException",
        "RemoteProtocolError",
        "ProtocolError",
        "ChunkedEncodingError",
        "IncompleteRead",
        "RemoteDisconnected",
    }
)
_PICKLE_FALLBACK_MISS = re.compile(
    r"does not appear to have a file named \S+\.bin|\.bin\b.*(not found|404)", re.I
)


class ComponentFetchError(RuntimeError):
    """``kind`` is one of ``disk_full``, ``not_found``, ``access``, ``offline``, ``network`` or ``error``."""

    def __init__(self, message: str, *, component: str, source: str, kind: str) -> None:
        super().__init__(message)
        self.component = component
        self.source = source
        self.kind = kind


def _exception_chain(exc: BaseException) -> Iterator[BaseException]:
    seen: set[int] = set()
    pending: list[Optional[BaseException]] = [exc]
    while pending:
        link = pending.pop(0)
        if link is None or id(link) in seen:
            continue
        seen.add(id(link))
        yield link
        pending += [link.__cause__, link.__context__]


def fetch_failure_kind(exc: BaseException) -> str:
    links = list(_exception_chain(exc))
    for link in links:
        if isinstance(link, OSError) and link.errno in _DISK_FULL_ERRNOS:
            return "disk_full"
        if _DISK_FULL_TEXT.search(str(link)):
            return "disk_full"
    from hub.utils.hf_errors import hf_error_status

    for link in links:
        if not isinstance(link, Exception):
            continue
        status = hf_error_status(link)
        if status == 404:
            return "not_found"
        if status in (401, 403):
            return "access"
        if status == 503:
            return "offline"
    for link in links:
        if type(link).__name__ in _NETWORK_NAMES or isinstance(
            link, (ConnectionError, TimeoutError)
        ):
            return "network"
    # diffusers does not chain the hub error on every version.
    if any("does not appear to have a file named" in str(link) for link in links):
        return "not_found"
    return "error"


def _free_space_note(cache_dir: Optional[str]) -> str:
    try:
        root: Any = cache_dir
        if not root:
            from hub.utils.hf_cache_state import hf_cache_root
            root = hf_cache_root(create = False)
        if not root:
            return ""
        free = shutil.disk_usage(str(root)).free
    except Exception:  # noqa: BLE001 -- the reading only decorates the message
        return ""
    return f" Only {free / 1e9:.1f} GB is available in {root}."


def _first_line(exc: BaseException, limit: int = 300) -> str:
    text = str(exc).strip().splitlines()
    line = text[0] if text else ""
    return line if len(line) <= limit else line[: limit - 3] + "..."


def component_fetch_error(
    component: str,
    source: str,
    exc: BaseException,
    *,
    cache_dir: Optional[str] = None,
    also_failed: Iterable[str] = (),
) -> ComponentFetchError:
    kind = fetch_failure_kind(exc)
    what = f"the {component} component of {source}"
    if kind == "disk_full":
        # Same wording as the download preflight (hub/workers/hf_download.py).
        reason = (
            f"Not enough disk space to download {what}.{_free_space_note(cache_dir)} "
            "Free up space and try again."
        )
    elif kind == "not_found":
        reason = (
            f"Could not download {what}: its files were not found on Hugging Face "
            f"({_first_line(exc)})."
        )
    elif kind == "access":
        reason = (
            f"Could not download {what}: Hugging Face refused access. Set a valid HF token, or accept the "
            "model's terms on its Hugging Face page, then load again."
        )
    elif kind == "offline":
        reason = (
            f"Could not load {what}: it is not in the local cache and Hugging Face could not be reached. "
            "Reconnect and load again."
        )
    elif kind == "network":
        reason = (
            f"Could not download {what}: the connection to Hugging Face failed "
            f"({type(exc).__name__}: {_first_line(exc)}). Check the network and load again."
        )
    else:
        reason = f"Could not load {what}: {type(exc).__name__}: {_first_line(exc)}"
    others = [name for name in also_failed if name != component]
    if others:
        reason += f" (also failed: {', '.join(others)})"
    return ComponentFetchError(reason, component = component, source = source, kind = kind)


def _is_model_mixin(type_hint: Any) -> bool:
    try:
        from diffusers import ModelMixin
        return isinstance(type_hint, type) and issubclass(type_hint, ModelMixin)
    except Exception:  # noqa: BLE001 -- not a diffusers model: no safetensors-only retry
        return False


def _pickle_fallback_miss(exc: BaseException) -> bool:
    return any(_PICKLE_FALLBACK_MISS.search(str(link)) for link in _exception_chain(exc))


def _spec_source(spec: Any, name: str, kwargs: dict) -> str:
    override = kwargs.get("pretrained_model_name_or_path")
    if isinstance(override, dict):
        override = override.get(name, override.get("default"))
    source = str(override or getattr(spec, "pretrained_model_name_or_path", None) or "?")
    subfolder = getattr(spec, "subfolder", None)
    if isinstance(kwargs.get("subfolder"), dict):
        subfolder = kwargs["subfolder"].get(name, subfolder)
    return f"{source} ({subfolder}/)" if subfolder else source


def _recording_load(name: str, spec: Any, load: Any, failures: dict[str, BaseException]) -> Any:
    def _load(**kwargs: Any) -> Any:
        try:
            return load(**kwargs)
        except Exception as exc:
            if (
                "use_safetensors" not in kwargs
                and _is_model_mixin(getattr(spec, "type_hint", None))
                and _pickle_fallback_miss(exc)
            ):
                try:
                    return load(**kwargs, use_safetensors = True)
                except Exception as retry_exc:
                    failures[name] = retry_exc
                    raise
            failures[name] = exc
            raise

    return _load


def load_components_strict(
    pipe: Any,
    *,
    optional: Iterable[str] = (),
    **kwargs: Any,
) -> None:
    """``optional`` components may fail and stay None (logged by diffusers)."""
    specs = getattr(pipe, "_component_specs", None)
    failures: dict[str, BaseException] = {}
    patched: list[Any] = []
    if isinstance(specs, dict):
        for name, spec in specs.items():
            load = getattr(spec, "load", None)
            if not callable(load):
                continue
            try:
                spec.load = _recording_load(name, spec, load, failures)
            except Exception:  # noqa: BLE001 -- a spec that refuses the attribute is loaded unrecorded
                continue
            patched.append(spec)
    try:
        pipe.load_components(**kwargs)
    finally:
        for spec in patched:
            try:
                vars(spec).pop("load", None)
            except TypeError:
                pass
    skip = set(optional)
    missing = [n for n in failures if n not in skip and getattr(pipe, n, None) is None]
    if not missing:
        return
    name = missing[0]
    spec = specs.get(name) if isinstance(specs, dict) else None
    raise component_fetch_error(
        name,
        _spec_source(spec, name, kwargs),
        failures[name],
        cache_dir = kwargs.get("cache_dir") if isinstance(kwargs.get("cache_dir"), str) else None,
        also_failed = missing,
    ) from failures[name]
