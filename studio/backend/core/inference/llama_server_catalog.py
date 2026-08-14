"""Installed-version argument discovery for ``llama-server``.

The subprocess result is cached by binary revision and shared with the legacy
capability detector in :mod:`core.inference.llama_cpp`.  Catalog parsing stays
pure so fixture tests can cover upstream help-format changes without launching
a process.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import re
import subprocess
import threading
import time
from typing import Callable, Iterable, Mapping, Optional

from core.inference.llama_server_args import (
    BLOCKED_FLAG_POLICIES,
    flag_policy,
    is_managed_flag,
    managed_flag_groups,
    managed_flags,
    overlaps_studio_control,
    safe_flag_policy,
    scrub_llama_server_env,
)


_PROBE_RETRY_SECONDS = 30.0
_PROBE_CACHE_MAX_ENTRIES = 4
_DIGEST_CHUNK_BYTES = 1024 * 1024
_REMOVED_MARKER = "argument has been removed"
_GROUP_RE = re.compile(r"^-{2,}\s*(?P<name>.*?)\s*-{2,}$")
_FLAG_RE = re.compile(r"-{1,2}[A-Za-z][A-Za-z0-9_.-]*")
_DEFAULT_RE = re.compile(
    r"\bdefault\s*:\s*(?:'([^']*)'|\"([^\"]*)\"|([^,\s)\]]+))",
    re.IGNORECASE,
)
_ENV_RE = re.compile(
    r"[\[(]?\b(?:env|environment(?:\s+variable)?)\s*:\s*([A-Z][A-Z0-9_]*)",
    re.IGNORECASE,
)


@dataclass(frozen = True)
class LlamaServerArgument:
    name: str
    aliases: tuple[str, ...]
    value_hint: Optional[str]
    choices: tuple[str, ...]
    description: str
    default_value: Optional[str]
    env_var: Optional[str]
    group: Optional[str]
    deprecated: bool

    def as_public_dict(self) -> dict[str, object]:
        policy = flag_policy(self.name)
        safe_policy = safe_flag_policy(self.name)
        return {
            "name": self.name,
            "aliases": list(self.aliases),
            "value_hint": self.value_hint,
            "choices": list(self.choices),
            "description": self.description,
            "default_value": self.default_value,
            "env_var": self.env_var,
            "group": self.group,
            "deprecated": self.deprecated,
            "managed_by_studio": is_managed_flag(self.name),
            "overlaps_studio_control": overlaps_studio_control(self.name),
            "policy_category": policy.category if policy is not None else "Unclassified",
            "value_arity": (
                policy.value_arity
                if policy is not None
                else safe_policy.value_arity
                if safe_policy is not None
                else int(bool(self.value_hint))
            ),
        }


@dataclass(frozen = True)
class LlamaServerHelpProbe:
    binary: Optional[str]
    fingerprint: Optional[tuple[str, int, int, str]]
    available: bool
    installed_tag: Optional[str]
    error_code: Optional[str]
    help_text: str
    arguments: tuple[LlamaServerArgument, ...]
    returncode: Optional[int]

    @property
    def authoritative(self) -> bool:
        """Whether this help output can safely prove an option is absent."""
        return self.available and self.returncode == 0

    def as_public_catalog(self) -> dict[str, object]:
        return {
            "available": self.available,
            "authoritative": self.authoritative,
            "installed_tag": self.installed_tag,
            "error_code": self.error_code,
            # Admission remains useful when the installed help cannot be probed.
            # Completion metadata may disappear; managed protection may not.
            "managed_flags": list(managed_flags()),
            "managed_flag_groups": [list(group) for group in managed_flag_groups()],
            "blocked_categories": sorted(
                {policy.category for policy in BLOCKED_FLAG_POLICIES}
            ),
            "arguments": [argument.as_public_dict() for argument in self.arguments]
            if self.available
            else [],
        }


_CAPABILITY_EXACT_NAMES = frozenset(
    {
        "--rpc",
        "--model",
        "--mmproj",
        "--mmproj-auto",
        "--no-mmproj",
        "--no-mmproj-auto",
        "--path",
        "--host",
        "--port",
        "--reuse-port",
        "--api-prefix",
        "--alias",
        "--tags",
        "--parallel",
        "--embedding",
        "--embeddings",
        "--rerank",
        "--reranking",
        "--pooling",
        "--timeout",
        "--sse-ping-interval",
        "--threads-http",
        "--metrics",
        "--props",
        "--slots",
        "--no-slots",
        "--sleep-idle-seconds",
        "--log-disable",
        "--lora-init-without-apply",
    }
)
_CAPABILITY_NAME_FRAGMENTS = (
    "-url",
    "-repo",
    "--hf-",
    "--lora",
    "--control-vector",
    "--grammar-file",
    "--json-schema-file",
    "--chat-template-file",
    "--lookup-cache-",
    "--log-file",
    "--log-prompts-dir",
    "--slot-save-path",
    "--media-path",
    "--models-",
    "--api-key",
    "--ssl-",
    "--cors-",
    "--ui",
    "--webui",
    "--tools",
    "--agent",
    "--mcp-",
)


def capability_policy_gaps(
    arguments: Iterable[LlamaServerArgument],
) -> tuple[str, ...]:
    """Documented capability flags not covered by the checked-in policy.

    This is intentionally a focused compatibility audit, not a promise to
    classify every future llama.cpp feature.  It catches the naming and help
    text shapes used by all capability-bearing flags in the installed build.
    """
    gaps: set[str] = set()
    for argument in arguments:
        spellings = (argument.name, *argument.aliases)
        if any(flag_policy(spelling) is not None for spelling in spellings):
            continue
        names = " ".join(spellings).lower()
        description = argument.description.lower()
        suspicious = (
            any(spelling in _CAPABILITY_EXACT_NAMES for spelling in spellings)
            or any(fragment in names for fragment in _CAPABILITY_NAME_FRAGMENTS)
            or (
                argument.name.endswith("-default")
                and "download weights from the internet" in description
            )
            or (
                argument.name.endswith("-spec")
                and "download weights from the internet" in description
            )
        )
        if suspicious:
            gaps.add(argument.name)
    return tuple(sorted(gaps))


@dataclass
class _ProbeFlight:
    event: threading.Event
    generation: int
    result: Optional[LlamaServerHelpProbe] = None


_probe_cache: OrderedDict[tuple[str, int, int, str], LlamaServerHelpProbe] = OrderedDict()
_probe_retry_after: dict[tuple[str, int, int, str], float] = {}
_probe_inflight: dict[tuple[str, int, int, str], _ProbeFlight] = {}
_latest_fingerprint_by_path: OrderedDict[str, tuple[str, int, int, str]] = OrderedDict()
_probe_cache_lock = threading.Lock()
_probe_cache_generation = 0


def clear_llama_server_help_cache() -> None:
    """Clear the process-local probe cache (primarily for focused tests)."""
    global _probe_cache_generation
    with _probe_cache_lock:
        _probe_cache.clear()
        _probe_retry_after.clear()
        _latest_fingerprint_by_path.clear()
        _probe_cache_generation += 1


def llama_server_binary_fingerprint(binary: str) -> tuple[str, int, int, str]:
    """Content-aware identity for one executable revision.

    Metadata remains in the key for diagnostics and cheap revision visibility,
    while the streaming digest detects in-place replacement that deliberately
    preserves size and timestamps.
    """
    path = Path(binary)
    resolved = str(path.resolve())
    for _attempt in range(2):
        before = path.stat()
        digest = hashlib.sha256()
        with path.open("rb") as binary_file:
            while chunk := binary_file.read(_DIGEST_CHUNK_BYTES):
                digest.update(chunk)
        after = path.stat()
        if (before.st_mtime_ns, before.st_size) == (after.st_mtime_ns, after.st_size):
            return (resolved, after.st_mtime_ns, after.st_size, digest.hexdigest())
    raise OSError("llama-server changed while its content identity was read")


def _remember_latest_fingerprint(fingerprint: tuple[str, int, int, str]) -> None:
    resolved = fingerprint[0]
    _latest_fingerprint_by_path[resolved] = fingerprint
    _latest_fingerprint_by_path.move_to_end(resolved)
    while len(_latest_fingerprint_by_path) > _PROBE_CACHE_MAX_ENTRIES:
        _latest_fingerprint_by_path.popitem(last = False)


def _publish_cached_probe(
    fingerprint: tuple[str, int, int, str], probe: LlamaServerHelpProbe
) -> None:
    _probe_cache[fingerprint] = probe
    _probe_cache.move_to_end(fingerprint)
    if probe.authoritative:
        _probe_retry_after.pop(fingerprint, None)
    else:
        _probe_retry_after[fingerprint] = time.monotonic() + _PROBE_RETRY_SECONDS
    while len(_probe_cache) > _PROBE_CACHE_MAX_ENTRIES:
        evicted, _ = _probe_cache.popitem(last = False)
        _probe_retry_after.pop(evicted, None)


def _installed_tag(binary: str) -> Optional[str]:
    try:
        from utils.llama_cpp_freshness import read_install_marker

        marker = read_install_marker(binary) or {}
        value = marker.get("release_tag") or marker.get("tag")
        return str(value) if value else None
    except Exception:
        return None


def _missing_probe(binary: Optional[str], error_code: str) -> LlamaServerHelpProbe:
    return LlamaServerHelpProbe(
        binary = binary,
        fingerprint = None,
        available = False,
        installed_tag = None,
        error_code = error_code,
        help_text = "",
        arguments = (),
        returncode = None,
    )


def probe_llama_server_help(
    binary: Optional[str],
    *,
    env: Optional[Mapping[str, str]] = None,
    run: Optional[Callable[..., object]] = None,
) -> LlamaServerHelpProbe:
    """Run and cache ``llama-server --help`` for one binary revision.

    Failures use stable error categories; exception text and stderr never cross
    the public catalog boundary.  Transient failures are cached briefly so a
    broken binary cannot make every settings-page open wait for the timeout.
    """
    if not binary:
        return _missing_probe(None, "not_installed")
    path = Path(binary)
    if not path.is_file():
        return _missing_probe(str(path), "not_installed")
    try:
        fingerprint = llama_server_binary_fingerprint(str(path))
    except OSError:
        return _missing_probe(str(path), "binary_unreadable")

    leader = False
    with _probe_cache_lock:
        _remember_latest_fingerprint(fingerprint)
        cached = _probe_cache.get(fingerprint)
        if cached is not None:
            retry_after = _probe_retry_after.get(fingerprint)
            if (
                cached.authoritative
                or retry_after is None
                or time.monotonic() < retry_after
            ):
                _probe_cache.move_to_end(fingerprint)
                return cached
            _probe_cache.pop(fingerprint, None)
            _probe_retry_after.pop(fingerprint, None)
        flight = _probe_inflight.get(fingerprint)
        if flight is None:
            flight = _ProbeFlight(
                event = threading.Event(),
                generation = _probe_cache_generation,
            )
            _probe_inflight[fingerprint] = flight
            leader = True

    if not leader:
        flight.event.wait()
        if flight.result is not None:
            return flight.result
        return _missing_probe(str(path), "probe_failed")

    runner = run or subprocess.run
    help_text = ""
    returncode: Optional[int] = None
    error_code: Optional[str] = None
    try:
        probe_env = dict(os.environ if env is None else env)
        scrub_llama_server_env(probe_env)
        result = runner(
            [str(path), "--help"],
            capture_output = True,
            text = True,
            encoding = "utf-8",
            errors = "replace",
            timeout = 10,
            check = False,
            env = probe_env,
        )
        returncode = int(getattr(result, "returncode", 1))
        help_text = (getattr(result, "stdout", "") or "") + "\n" + (
            getattr(result, "stderr", "") or ""
        )
        if not help_text.strip():
            error_code = "probe_failed" if returncode != 0 else "empty_help"
    except subprocess.TimeoutExpired:
        error_code = "probe_timeout"
    except (OSError, subprocess.SubprocessError):
        error_code = "probe_failed"
    except Exception:
        error_code = "probe_failed"

    arguments = tuple(parse_llama_server_help(help_text)) if help_text.strip() else ()
    if arguments:
        # Some builds print structurally valid help and then fail optional device
        # initialization. Keep the declarations for completion, but a partial
        # listing cannot authoritatively label omitted options as unknown.
        available = True
        error_code = "probe_nonzero" if returncode not in {None, 0} else None
    else:
        available = False
        if error_code is None:
            error_code = (
                "probe_failed" if returncode not in {None, 0} else "unrecognized_help"
            )
    probe = LlamaServerHelpProbe(
        binary = str(path),
        fingerprint = fingerprint,
        available = available,
        installed_tag = _installed_tag(str(path)),
        error_code = error_code,
        help_text = help_text,
        arguments = arguments,
        returncode = returncode,
    )
    with _probe_cache_lock:
        active_flight = _probe_inflight.get(fingerprint)
        if active_flight is flight:
            # A replacement may finish before an older in-flight probe. Return
            # the older result to its waiters but never make it current again.
            if (
                flight.generation == _probe_cache_generation
                and _latest_fingerprint_by_path.get(fingerprint[0]) == fingerprint
            ):
                _publish_cached_probe(fingerprint, probe)
            flight.result = probe
            _probe_inflight.pop(fingerprint, None)
            flight.event.set()
    return probe


def _looks_like_value_hint(token: str) -> bool:
    """Whether one declaration-column token describes an option value."""
    if not token or token.startswith("-"):
        return False
    if re.fullmatch(r"(?:[A-Z][A-Z0-9_-]*|<[^>]+>|\[[^]]+]|\{[^}]+})", token):
        return True
    # Installed llama.cpp uses mixed/lowercase shapes without brackets:
    # MiB0,MiB1,...; FNAME:SCALE,...; and the finite --spec-type list.
    if "," in token and re.fullmatch(r"[A-Za-z0-9_,.+/:<>{}\[\]-]+", token):
        return True
    return False


def _declaration_parts(line: str) -> Optional[tuple[list[str], Optional[str], str]]:
    body = line.strip()
    if not body.startswith("-") or body in {"-", "--"}:
        return None

    flags: list[str] = []
    cursor = 0
    attached_hint: Optional[str] = None
    while True:
        match = _FLAG_RE.match(body, cursor)
        if match is None:
            break
        flags.append(match.group(0))
        cursor = match.end()
        if cursor < len(body) and body[cursor] == "=":
            hint_match = re.match(r"=([^,\s]+)", body[cursor:])
            if hint_match:
                attached_hint = hint_match.group(1)
                cursor += hint_match.end()
        separator = re.match(r"(?:\s*,\s*|\s+)(?=-)", body[cursor:])
        if separator is None:
            break
        cursor += separator.end()

    if not flags:
        return None
    remainder = body[cursor:]
    leading = len(remainder) - len(remainder.lstrip())
    remainder = remainder.lstrip()
    if not remainder:
        return flags, attached_hint, ""
    if leading >= 2:
        return flags, attached_hint, remainder
    # Current llama.cpp also uses structured multi-word metavariables, such as
    # ``<tensor name pattern>=<buffer type>,...``. Splitting on the first space
    # makes these options look boolean to the editor.
    structured_hint = re.match(
        r"(?P<hint><[^>]+>(?:\s*[=:]\s*<[^>]+>)*(?:,\.\.\.)?)(?:\s+(?P<rest>.*))?$",
        remainder,
    )
    if structured_hint:
        hint = structured_hint.group("hint")
        rest = structured_hint.group("rest") or ""
        return flags, attached_hint or hint, rest.strip()
    first, space, rest = remainder.partition(" ")
    if _looks_like_value_hint(first):
        return flags, attached_hint or first, rest.strip() if space else ""
    return flags, attached_hint, remainder


def _metadata(description: str) -> tuple[Optional[str], Optional[str]]:
    default = _DEFAULT_RE.search(description)
    env = _ENV_RE.search(description)
    return (
        next((value for value in default.groups() if value is not None), None) if default else None,
        env.group(1).upper() if env else None,
    )


def _choices(value_hint: Optional[str], description: str) -> tuple[str, ...]:
    sources: list[str] = []
    if value_hint:
        sources.append(value_hint.strip())
    sources.extend(
        match.group(0)
        for match in re.finditer(r"[\[({<][^\])}>]+[\])}>]", description)
    )
    for source in sources:
        body = source.strip().strip("[](){}<>")
        # Ellipses/repetition, indexed device metavariables, and structured
        # FNAME:SCALE syntax are open shapes, never completion enums.
        if ".." in body or "..." in body or ":" in body:
            continue
        if not ("|" in body or "," in body):
            continue
        values = [part.strip().strip("'\"") for part in re.split(r"[|,]", body)]
        if len(values) < 2 or not all(
            value and re.fullmatch(r"[A-Za-z0-9_.+/-]+", value) for value in values
        ):
            continue
        return tuple(dict.fromkeys(values))
    return ()


def parse_llama_server_help(help_text: str) -> list[LlamaServerArgument]:
    """Parse upstream help categories and multiline option declarations."""
    records: list[tuple[Optional[str], list[str], Optional[str], list[str]]] = []
    group: Optional[str] = None
    current: Optional[tuple[Optional[str], list[str], Optional[str], list[str]]] = None

    for raw_line in (help_text or "").splitlines():
        stripped = raw_line.strip()
        group_match = _GROUP_RE.match(stripped)
        if group_match:
            if current is not None:
                records.append(current)
                current = None
            group = group_match.group("name").strip() or None
            continue
        declaration = _declaration_parts(raw_line)
        if declaration is not None:
            if current is not None:
                records.append(current)
            flags, value_hint, inline_description = declaration
            current = (group, flags, value_hint, [inline_description] if inline_description else [])
            continue
        if current is not None and stripped:
            current[3].append(stripped)
    if current is not None:
        records.append(current)

    parsed: list[LlamaServerArgument] = []
    seen: set[str] = set()
    for record_group, flags, value_hint, description_lines in records:
        description = " ".join(description_lines).strip()
        if _REMOVED_MARKER in description.lower():
            continue
        name = next((flag for flag in flags if flag.startswith("--")), flags[0])
        if name in seen:
            continue
        seen.add(name)
        default_value, env_var = _metadata(description)
        parsed.append(
            LlamaServerArgument(
                name = name,
                aliases = tuple(flag for flag in flags if flag != name),
                value_hint = value_hint,
                choices = _choices(value_hint, description),
                description = description,
                default_value = default_value,
                env_var = env_var,
                group = record_group,
                deprecated = bool(re.search(r"\bdeprecated\b", description, re.IGNORECASE)),
            )
        )
    return parsed


def get_llama_server_argument_catalog(
    binary: Optional[str],
    *,
    env: Optional[Mapping[str, str]] = None,
    run: Optional[Callable[..., object]] = None,
) -> dict[str, object]:
    """Return the stable Studio-facing catalog payload for ``binary``."""
    return probe_llama_server_help(binary, env = env, run = run).as_public_catalog()
