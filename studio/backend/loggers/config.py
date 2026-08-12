# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Structured logging configuration via structlog.

Environment-specific formats (JSON for prod, console for dev), ISO timestamps,
context-var integration, log-level filtering, and logger caching.
"""

import logging
import os
import sys
from typing import Optional

import structlog

from loggers.handlers import filter_sensitive_data


class _DropTorchDtypeDeprecation(logging.Filter):
    """Drop transformers' once-per-run "`torch_dtype` is deprecated" warning_once.
    It is emitted via logging (not warnings), so a warnings filter cannot catch it."""

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        return not ("torch_dtype" in msg and "deprecated" in msg)


def _env_int(name: str, default: int) -> int:
    try:
        raw = (os.environ.get(name) or "").strip()
        return int(raw) if raw else default
    except ValueError:
        return default


# Cap on the rendered traceback in a single log record. A traceback is normally a few
# KB, but an exception whose message embeds a request body is not: a binary upload
# rejected by request validation produced one 2.2 MB line. Keep the head (where the
# raising frame is) and the tail (where the actual exception type and message are),
# and say how much was dropped. 0 disables the cap.
_MAX_EXC_CHARS = _env_int("UNSLOTH_STUDIO_MAX_EXCEPTION_CHARS", 16384)
_EXC_TAIL_CHARS = 2048
# The middleware logs the same exception twice: once rendered as a traceback under
# "exception" and once as str(exc) under "error". Capping only the first still lets an
# exception whose message embeds the request body through, so bound both.
_MAX_ERROR_CHARS = 2048


def _truncate_middle(text: str, limit: int, tail: int) -> str:
    """Keep the head and the tail of `text`, saying how much was dropped.

    The head holds the raising frame and the tail the exception type and message, so
    both ends are worth keeping. `tail` is clamped so a cap smaller than the tail
    cannot make the head negative and hand back nearly the whole string.
    """
    if limit <= 0 or len(text) <= limit:
        return text
    tail = max(1, min(tail, limit // 4))
    head = limit - tail
    dropped = len(text) - limit
    return (
        text[:head] + f"\n... [{dropped} chars omitted; "
        "raise UNSLOTH_STUDIO_MAX_EXCEPTION_CHARS to see it all] ...\n" + text[-tail:]
    )


def truncate_exception(event_dict: dict) -> dict:
    """Structlog processor: bound the rendered exception, its message and the event."""
    if _MAX_EXC_CHARS <= 0:
        return event_dict
    text = event_dict.get("exception")
    if isinstance(text, str):
        event_dict["exception"] = _truncate_middle(text, _MAX_EXC_CHARS, _EXC_TAIL_CHARS)
    message_cap = min(_MAX_ERROR_CHARS, _MAX_EXC_CHARS)
    error = event_dict.get("error")
    if isinstance(error, str):
        event_dict["error"] = _truncate_middle(error, message_cap, _EXC_TAIL_CHARS)
    # f-string call sites interpolate the exception straight into the message
    # (routes/inference.py: logger.error(f"...: {e}", exc_info = True)), so the event
    # itself is a third copy that can carry the whole payload.
    event = event_dict.get("event")
    if isinstance(event, str):
        event_dict["event"] = _truncate_middle(event, message_cap, _EXC_TAIL_CHARS)
    # logger.error("stream error: %s", exc) keeps the exception under positional_args,
    # and the chain has no PositionalArgumentsFormatter, so the renderer stringifies it
    # untouched. Render and cap it here instead.
    args = event_dict.get("positional_args")
    if isinstance(args, (list, tuple)) and args:
        event_dict["positional_args"] = [
            _truncate_middle(a, message_cap, _EXC_TAIL_CHARS)
            if isinstance(a, str)
            else _truncate_middle(str(a), message_cap, _EXC_TAIL_CHARS)
            for a in args
        ]
    return event_dict


def _truncate_exception_processor(logger, method_name, event_dict):
    return truncate_exception(event_dict)


def _plain_tracebacks_enabled() -> bool:
    """Whether to echo a readable traceback under each JSON error record.
    ``UNSLOTH_STUDIO_PLAIN_TRACEBACKS=0`` turns it off."""
    return (os.environ.get("UNSLOTH_STUDIO_PLAIN_TRACEBACKS") or "").strip().lower() not in (
        "0",
        "off",
        "no",
        "false",
    )


# Prefix on every echoed traceback line. Deliberately NOT whitespace: RFC 8259 lets a
# parser skip leading space/tab, so json.loads('  {"event": ...}') SUCCEEDS, and an
# exception message carrying a newline plus a JSON object would produce a line that
# reads as a genuine record (CWE-117 log injection). Exception messages carry
# request-derived text -- the cap above exists because a rejected upload embedded a
# whole request body in one -- so that payload is reachable, not theoretical. "| "
# cannot begin a JSON value, so no echoed line can be mistaken for a record, and a
# reader that wants records only can drop these lines on the prefix alone.
_TRACEBACK_ECHO_PREFIX = "| "


def _echoable(exception: str) -> str:
    """The traceback as lines that can never be read as a log record.

    ``splitlines`` also splits on \\r, \\x0b, \\x0c, \\x85 and U+2028/9, so rejoining on
    \\n normalises every separator a message could smuggle in, including the \\r the
    export worker's log reader treats as a line break."""
    return "\n".join(f"{_TRACEBACK_ECHO_PREFIX}{part}" for part in exception.rstrip().splitlines())


def with_readable_traceback(renderer):
    """Wrap the JSON renderer so an exception is ALSO written as a real multi-line
    traceback, on the lines after the record.

    ~/.unsloth/studio/logs is a tee of stdout, and stdout is JSON, so every traceback
    reached the person reading that file as one enormous line with its newlines escaped
    to ``\\n``. That is correct JSON and unreadable prose: the reported Image Transform
    failure ("RuntimeError: Input type (float) and bias type (c10::BFloat16)...") was
    found in the log with, in the reporter's words, all its newlines mangled -- which is
    the state the log is in for every crash anyone is ever asked to send in.

    The JSON record is emitted UNCHANGED, escaped exception field and all, so anything
    parsing the file record-by-record still reads every record it read before, and every
    line the echo adds is prefixed so it can neither parse as a record nor be confused
    for one. That the file already carries non-JSON lines is not something this relies on
    by chance -- faulthandler dumps native stacks straight to the same handle.

    Emitted as part of the SAME return value rather than written to another stream, so
    one ``print`` under ``PrintLogger``'s lock keeps the record and its traceback both
    adjacent and correctly ordered. A processor runs BEFORE that print, so a side-channel
    write here would put the traceback above the record it belongs to, and the export
    worker reads stdout and stderr on two separate pipes, which would let them reorder
    again in the dialog.

    Only for the JSON renderer. ConsoleRenderer (development) already prints tracebacks
    as tracebacks, and wrapping it would print each one twice."""

    def _render(logger, method_name, event_dict):
        exception = event_dict.get("exception")
        line = renderer(logger, method_name, event_dict)
        if (
            isinstance(exception, str)
            and exception.strip()
            and isinstance(line, str)
            and _plain_tracebacks_enabled()
        ):
            return f"{line}\n{_echoable(exception)}"
        return line

    return _render


# Set alongside HF_HUB_DISABLE_PROGRESS_BARS when the value is Studio's default rather
# than the operator's, so allow_progress_bars() can tell them apart.
_PROGRESS_BARS_DEFAULTED = "UNSLOTH_STUDIO_PROGRESS_BARS_DEFAULTED"

# huggingface_hub's own spelling of truth (utils/_runtime.py ENV_VARS_TRUE_VALUES),
# so "off" and "no" mean "keep the bars" here exactly as they do there.
_ENV_TRUE = frozenset({"1", "on", "yes", "true"})

# Set once this process has deliberately taken its bars back (the export worker draws
# them, the training worker reads them). quiet_third_party_progress_bars() then stops
# being a switch a later call can flip the other way.
_BARS_RESTORED = False


def _env_is_true(value: str) -> bool:
    return (value or "").strip().lower() in _ENV_TRUE


def _verbose_logging_requested() -> bool:
    """True when `unsloth studio --verbose` asked for every line back. The CLI signals
    it by zeroing both access-log dedup windows, which is what the workers inherit."""

    def _zero(name: str) -> bool:
        raw = (os.environ.get(name) or "").strip()
        try:
            return raw != "" and int(raw) <= 0
        except ValueError:
            return False

    return _zero("UNSLOTH_STUDIO_ACCESS_LOG_DEDUP_MS") and _zero(
        "UNSLOTH_STUDIO_ACCESS_LOG_POLL_DEDUP_MS"
    )


class _NullStream:
    """Somewhere for a progress bar to write that is not the log."""

    def write(self, _data):
        return 0

    def flush(self):
        pass

    def isatty(self):
        return False


def _silence_datasets_bar_output() -> None:
    """Keep the datasets bar object, drop only what it writes.

    datasets exposes no env var, and its disable_progress_bar() works by forcing
    tqdm(disable = True), which never registers the bar in tqdm._instances.
    utils/datasets/chat_templates.py polls that set to publish
    "Applying chat template ... 42%" to the UI, so disabling the bar outright would
    freeze that status for the whole of a long format job. Pointing the bar at a null
    stream keeps the counter (and the status) alive while the log stays clean.
    """
    if "datasets" not in sys.modules:
        return
    try:
        from datasets.utils.tqdm import tqdm as bar_cls

        if getattr(bar_cls, "_unsloth_output_silenced", False):
            return
        original_init = bar_cls.__init__

        def _quiet_init(self, *args, **kwargs):
            kwargs.setdefault("file", _NullStream())
            original_init(self, *args, **kwargs)

        bar_cls.__init__ = _quiet_init
        bar_cls._unsloth_output_silenced = True
    except Exception:  # noqa: BLE001 - a datasets build without it just stays noisy
        pass


def _redirect_every_bar_output() -> None:
    """Point every tqdm bar at a null stream, disabling none of them.

    tqdm.std.tqdm.__init__ is the one funnel: huggingface_hub's, datasets' and
    transformers' bar classes all subclass it and reach it through super().
    """
    try:
        from tqdm.std import tqdm as bar_cls

        if getattr(bar_cls, "_unsloth_every_output_silenced", False):
            return
        original_init = bar_cls.__init__

        def _quiet_init(self, *args, **kwargs):
            kwargs.setdefault("file", _NullStream())
            original_init(self, *args, **kwargs)

        bar_cls.__init__ = _quiet_init
        bar_cls._unsloth_every_output_silenced = True
    except Exception:  # noqa: BLE001 - an unfamiliar tqdm just stays noisy
        pass


def keep_progress_bars_countable() -> None:
    """Keep the bar objects alive in a process that READS them, output dropped.

    core/training/worker.py runs a poller over tqdm._instances to turn the Hub
    download bar and "Loading checkpoint shards" into the UI's status line, which is
    the only progress a user sees between "Loading model..." and the first step. A
    disabled bar is never registered in _instances (tqdm/std.py drops it), so the
    inherited HF_HUB_DISABLE_PROGRESS_BARS default would leave that status frozen for
    the whole of a multi-GB download. Same trade as datasets: keep the counter, drop
    the writes, so nothing reaches the log either way.

    Only Studio's own default is undone; an operator who set the variable themselves
    asked for no bars and keeps getting none. Afterwards
    quiet_third_party_progress_bars() is a no-op in this process, so a later call
    cannot re-disable what the poller reads.

    Call it BEFORE huggingface_hub is imported: hub reads the variable once, into a
    module constant, and enable_progress_bars() then refuses to override it. The
    training worker does, ahead of its setup_logging call.
    """
    value = os.environ.get("HF_HUB_DISABLE_PROGRESS_BARS")
    if value is None or not _env_is_true(value):
        # Nothing quieted them here: --verbose, or an operator who asked to keep them.
        return
    if not os.environ.get(_PROGRESS_BARS_DEFAULTED):
        # The operator turned them off; that is not ours to undo.
        return
    _redirect_every_bar_output()
    allow_progress_bars()


def quiet_bar_kwargs() -> dict:
    """tqdm kwargs that keep a bar counting but stop it writing to the log.

    For Studio's own explicit bars (the dataset conversion loops), which no library
    switch reaches. Empty when the operator asked to keep bars, so nothing changes.
    """
    value = os.environ.get("HF_HUB_DISABLE_PROGRESS_BARS")
    if value is None or not _env_is_true(value):
        return {}
    return {"file": _NullStream()}


def allow_progress_bars() -> None:
    """Undo an inherited Studio default so this process can draw progress bars.

    Called by the export worker, whose stdout is forwarded to the export dialog and
    whose Hub upload bar is the only live byte progress a long push_to_hub has. An
    operator-set HF_HUB_DISABLE_PROGRESS_BARS is left alone.
    """
    global _BARS_RESTORED
    _BARS_RESTORED = True
    if os.environ.pop(_PROGRESS_BARS_DEFAULTED, None):
        os.environ.pop("HF_HUB_DISABLE_PROGRESS_BARS", None)


def quiet_third_party_progress_bars() -> None:
    """Turn off the tqdm bars transformers / diffusers / huggingface_hub draw
    during an in-process model load.

    A bar is written with carriage returns to a terminal, so in Studio's log it
    lands as a burst of lines like

        Loading weights:   8%|>         | 30/398 [00:00<00:01, 277.16it/s][A

    and, because tqdm writes to a different stream than the structlog JSON
    writer with no line discipline between them, a bar can land mid-record:
    ``Loading pipeline components...:  20%|...|{"timestamp": ...}``. That line
    is no longer parseable JSON, so anything reading the log record-by-record
    loses the record.

    Nothing is lost by dropping them: download and load progress already reach
    the UI as real events (``hub_download_progress``, ``inference_load_progress``)
    and via /api/inference/{images,video}/load-progress. Only the bars go; the
    libraries' warnings and errors are untouched.

    The subprocess workers already do this by exporting
    HF_HUB_DISABLE_PROGRESS_BARS (hub/services/download_lifecycle.py,
    core/inference/stt_download_worker.py); the server process, which loads the
    RAG embedder at boot and every diffusers pipeline in-process, did not.

    Respects an explicit operator override: if HF_HUB_DISABLE_PROGRESS_BARS is
    already set, its value wins, parsed the way huggingface_hub parses it. Only
    modules that are ALREADY imported get the API call, so this never forces a heavy
    import at logging-setup time, and never caches a Hub copy that a subprocess is
    about to replace with its transformers sidecar. `--verbose` skips it entirely.
    """
    if _BARS_RESTORED:
        # This process took its bars back on purpose: the export worker shows them, the
        # training worker reads them out of tqdm._instances (where a disabled bar is
        # never registered) and has already redirected their output.
        return
    if _verbose_logging_requested() and os.environ.get("HF_HUB_DISABLE_PROGRESS_BARS") is None:
        # --verbose promises everything back, so it must not install this default
        # either; the flag is inherited by the workers, which would stay quiet.
        return
    if os.environ.get("HF_HUB_DISABLE_PROGRESS_BARS") is None:
        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
        # Marks the value as ours rather than the operator's, so a process that needs
        # bars back (the export worker streams Hub upload progress into the export
        # dialog) can tell the difference. Inherited by every child process.
        os.environ[_PROGRESS_BARS_DEFAULTED] = "1"
    elif not _env_is_true(os.environ["HF_HUB_DISABLE_PROGRESS_BARS"]):
        # Operator asked to keep them; leave every library alone.
        return

    # Only touch Hub if something already imported it. Importing it here would cache
    # the base environment's copy before a subprocess prepends its transformers
    # sidecar to sys.path, leaving that process on an incompatible Hub.
    if "huggingface_hub" in sys.modules:
        try:
            from huggingface_hub.utils import disable_progress_bars
            disable_progress_bars()
        except Exception:  # noqa: BLE001 — quieting logs must never break startup
            pass

    # transformers derives its own _tqdm_active from the hub flag at import time,
    # so a module imported BEFORE this ran still needs the explicit call.
    #
    # datasets is handled separately (see _silence_datasets_bar_output): its `Map:` and
    # `Standardizing chat format (num_proc=8):` bars from dataset preparation were the
    # ones actually landing inside JSON records, but the UI reads their counter, so
    # only the output goes. datasets is imported long after logging setup, which is why
    # this function is safe to call again once a library is in.
    for _mod in ("transformers", "diffusers"):
        module = sys.modules.get(_mod)
        if module is None:
            continue
        try:
            module.utils.logging.disable_progress_bar()
        except Exception:  # noqa: BLE001
            pass
    _silence_datasets_bar_output()


class LogConfig:
    """Structured logging configuration for the application."""

    @staticmethod
    def setup_logging(
        service_name: str = "unsloth-studio-backend",
        env: Optional[str] = None,
        quiet_progress_bars: bool = True,
    ) -> structlog.BoundLogger:
        """Configure structured logging for the application.
        Args:
            service_name: Name of the service for logging identification
            env: Environment (development/production), affects logging format
            quiet_progress_bars: Turn third-party tqdm bars off. False for a process
                whose stdout is a user-facing progress stream (the export worker).
        """
        # Log level from environment; fall back to INFO if invalid.
        log_level_name = os.getenv("LOG_LEVEL", "INFO").upper()
        log_level = getattr(logging, log_level_name, logging.INFO)

        # Non-ASCII on a non-UTF-8 stream raises UnicodeEncodeError (Windows,
        # LANG=C), so key off the stream, not the platform.
        for stream in (sys.stdout, sys.stderr):
            if getattr(stream, "encoding", "") and not str(stream.encoding).lower().replace(
                "-", ""
            ).startswith("utf8"):
                if hasattr(stream, "reconfigure"):
                    try:
                        stream.reconfigure(encoding = "utf-8", errors = "replace")
                    except Exception:
                        pass

        structlog.configure(
            processors = [
                # Ordered to control output field order.
                structlog.processors.TimeStamper(fmt = "iso"),  # timestamp first
                structlog.processors.add_log_level,  # level second
                structlog.contextvars.merge_contextvars,
                structlog.processors.format_exc_info,
                filter_sensitive_data,
                # After redaction, not before: redact_native_paths replaces exact
                # strings, so cutting the middle out of a traceback first could leave
                # half a path behind for it to miss.
                _truncate_exception_processor,
                # Flatten the extra field into the main dict.
                lambda logger, method_name, event_dict: {
                    "timestamp": event_dict.get("timestamp"),
                    "level": event_dict.get("level"),
                    "event": event_dict.get("event"),
                    **(event_dict.get("extra", {})),  # Flatten extra into main dict
                    **{
                        k: v
                        for k, v in event_dict.items()
                        if k not in ["timestamp", "level", "event", "extra"]
                    },
                },
                (
                    # Preserve order; the wrapper adds the human-readable traceback copy.
                    with_readable_traceback(structlog.processors.JSONRenderer(sort_keys = False))
                    if env == "production"
                    else structlog.dev.ConsoleRenderer()
                ),
            ],
            wrapper_class = structlog.make_filtering_bound_logger(log_level),
            logger_factory = structlog.PrintLoggerFactory(file = sys.stdout),
            cache_logger_on_first_use = True,
        )

        # Silence third-party tqdm bars; they carry no signal and corrupt JSON records.
        if quiet_progress_bars:
            quiet_third_party_progress_bars()

        # Drop transformers' cosmetic "`torch_dtype` is deprecated" warning_once (see filter).
        _dtype_filter = _DropTorchDtypeDeprecation()
        for _name in (
            "transformers.configuration_utils",
            "transformers.modeling_utils",
            "transformers.pipelines.base",
        ):
            logging.getLogger(_name).addFilter(_dtype_filter)

        return structlog.get_logger(service_name)
