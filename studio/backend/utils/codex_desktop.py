# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The Codex desktop app (Codex.app, or ChatGPT.app on builds where Codex lives inside ChatGPT)
switched to Unsloth, and back.

Shared by `unsloth start codex --app` and Studio's Settings → Agents → Codex Desktop card, so both
write the same lines into ~/.codex/config.toml, keep the same restore record (one at a time, and
either one cleans up after the other crashed) and put the file back the same way.

The switch makes Unsloth the active provider: ``model_provider`` points at an
``[model_providers.unsloth_api]`` table that carries a per-session Studio key, ``model`` at the
loaded model, and ``model_catalog_json`` at a one-entry catalog, because the app lists only the
active provider's catalog. Every line it replaces is recorded, and switching back restores exactly
those lines (or the byte-for-byte backup when nothing else changed), leaving anything the user
changed in the meantime alone.

App discovery, quitting and launching follow `ollama launch codex-app` (bundle id
com.openai.codex, ChatGPT.app or Codex.app under /Applications or ~/Applications, the
codex://threads/new?mode=codex launch URL, a graceful quit that never escalates to a kill).
"""

from __future__ import annotations

import contextlib
import importlib.util
import json
import os
import plistlib
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, Optional

from . import claude_desktop as _cd

PROFILE = "unsloth_api"
PROVIDER_HEADER = f"[model_providers.{PROFILE}]"
PROVIDER_TABLES = (PROVIDER_HEADER, PROVIDER_HEADER[:-1] + ".")
ENV_KEY = "UNSLOTH_STUDIO_AUTH_TOKEN"
# See the long note on this value in unsloth_cli/commands/start.py: a local first turn can be
# silent for longer than Codex's 5-minute default, and each reconnect restarts prompt processing.
STREAM_IDLE_TIMEOUT_MS = 1_200_000
CATALOG_NAME = "unsloth-model-catalog.json"
BACKUP_SUFFIX = ".unsloth-backup"
STATE_NAME = "codex.json"
KEY_NAME = "Codex app"

BUNDLE_ID = "com.openai.codex"
LAUNCH_URL = "codex://threads/new?mode=codex"
DOWNLOAD_URL = "https://chatgpt.com/download"
APP_NAMES = ("ChatGPT.app", "Codex.app")
FALLBACK_PROMPT_NAME = "codex_fallback_prompt.md"


# --------------------------------------------------------------------------------------------
# config.toml
# --------------------------------------------------------------------------------------------


def provider_table(base: str, key: Optional[str] = None) -> str:
    # The desktop app gets no shell env, so it needs the key itself rather than env_key.
    auth = f"experimental_bearer_token = {json.dumps(key)}" if key else f'env_key = "{ENV_KEY}"'
    return (
        f"{PROVIDER_HEADER}\n"
        'name = "Unsloth Studio"\n'
        f"base_url = {json.dumps(base + '/v1')}\n"
        f"{auth}\n"
        'wire_api = "responses"\n'
        "requires_openai_auth = false\n"
        f"stream_idle_timeout_ms = {STREAM_IDLE_TIMEOUT_MS}\n"
    )


def merge_config(
    existing: str,
    base: str,
    key: Optional[str] = None,
) -> str:
    chunks = re.split(r"(?m)^(?=\[)", existing)  # preamble, then one chunk per table
    if key is None and not re.search(r"(?m)^\s*oss_provider\s*=", chunks[0]):
        if chunks[0] and not chunks[0].endswith("\n"):
            chunks[0] += "\n"
        chunks[0] += f'oss_provider = "{PROFILE}"\n'
    text = "".join(c for c in chunks if not c.startswith(PROVIDER_TABLES))
    if not text.endswith("\n"):
        text += "\n"
    if not text.endswith("\n\n"):
        text += "\n"
    return text + provider_table(base, key)


def set_toml_key(text: str, name: str, line: str) -> tuple:
    """Set a root key, returning (text, the line it replaced or None)."""
    chunks = re.split(r"(?m)^(?=\[)", text)
    match = re.search(rf"(?m)^[ \t]*{name}[ \t]*=[^\r\n]*", chunks[0])
    if match is None:
        return f"{line}\n{text}", None
    chunks[0] = chunks[0][: match.start()] + line + chunks[0][match.end() :]
    return "".join(chunks), match.group(0)


def unset_toml_key(text: str, name: str, line: str, previous: Optional[str]) -> Optional[str]:
    """Put back the line ``line`` replaced, or None when the key no longer holds ``line``."""
    chunks = re.split(r"(?m)^(?=\[)", text)
    match = re.search(rf"(?m)^[ \t]*{name}[ \t]*=([^\r\n]*)(\r?\n)?", chunks[0])
    if match is None or match.group(0).strip() != line:
        return None
    restored = "" if previous is None else previous + (match.group(2) or "")
    chunks[0] = chunks[0][: match.start()] + restored + chunks[0][match.end() :]
    return "".join(chunks)


def app_lines(model_id: str) -> dict:
    return {
        "model_provider": f'model_provider = "{PROFILE}"',
        "model": f"model = {json.dumps(model_id)}",
        "model_catalog_json": f"model_catalog_json = {json.dumps(CATALOG_NAME)}",
    }


def build(text: Optional[str], base: str, key: str, model_id: str, previous) -> tuple:
    """(new config text, the lines and provider table it replaced)."""
    previous = dict(previous or {})
    tables = [c for c in re.split(r"(?m)^(?=\[)", text or "") if c.startswith(PROVIDER_TABLES)]
    previous.setdefault("table", "".join(tables) or None)
    merged = merge_config(text or "", base, key)
    for name, line in app_lines(model_id).items():
        merged, old = set_toml_key(merged, name, line)
        previous.setdefault(name, old)
    return merged, previous


def restore(text: str, state: dict) -> tuple:
    """(text with the switch undone, the keys left alone because they changed meanwhile)."""
    skipped = []
    chunks = re.split(r"(?m)^(?=\[)", text)
    ours = provider_table(state["base"], state["key"]).strip()

    def is_ours(chunk: str) -> bool:
        # An editor on Windows may have rewritten the file with CRLF line endings.
        return chunk.replace("\r\n", "\n").strip() == ours

    if any(is_ours(chunk) for chunk in chunks):
        table = state["previous"].get("table") or ""
        text = "".join(table if is_ours(chunk) else chunk for chunk in chunks)
    else:
        skipped.append(PROVIDER_HEADER)
    for name, line in app_lines(state["model"]).items():
        restored = unset_toml_key(text, name, line, state["previous"].get(name))
        if restored is None:
            skipped.append(name)
        else:
            text = restored
    return text, skipped


def retarget(text: str, state: dict, model_id: str) -> Optional[str]:
    """Point a live switch at another model, keeping what it recorded to restore. None when the
    model line is no longer the one the switch wrote (the user picked another in the app)."""
    ours = app_lines(state["model"])["model"]
    chunks = re.split(r"(?m)^(?=\[)", text)
    match = re.search(r"(?m)^[ \t]*model[ \t]*=[^\r\n]*", chunks[0])
    if match is None or match.group(0).strip() != ours:
        return None
    line = app_lines(model_id)["model"]
    chunks[0] = chunks[0][: match.start()] + line + chunks[0][match.end() :]
    return "".join(chunks)


def fallback_prompt_path() -> Optional[Path]:
    """Codex's own unknown-model prompt, shipped with the CLI (see unsloth_cli/start.py)."""
    candidates = []
    with contextlib.suppress(Exception):
        spec = importlib.util.find_spec("unsloth_cli")
        if spec is not None and spec.origin:
            candidates.append(Path(spec.origin).parent / FALLBACK_PROMPT_NAME)
    candidates.append(Path(__file__).resolve().parents[3] / "unsloth_cli" / FALLBACK_PROMPT_NAME)
    return next((path for path in candidates if path.is_file()), None)


def model_catalog(
    model: dict,
    visibility: str = "none",
    prompt: Optional[str] = None,
) -> dict:
    """Conservative metadata for an Unsloth model unknown to Codex's built-in catalog."""
    if prompt is None:
        path = fallback_prompt_path()
        if path is None:
            raise FileNotFoundError(f"{FALLBACK_PROMPT_NAME} is missing from this install.")
        prompt = path.read_text(encoding = "utf-8")
    model_id = model["id"]
    window = model.get("context_length") or model.get("max_context_length")
    entry = {
        "slug": model_id,
        "display_name": model_id,
        "description": "Model served by Unsloth Studio",
        "supported_reasoning_levels": [],
        "shell_type": "default",
        "visibility": visibility,
        "supported_in_api": True,
        "priority": 99,
        "availability_nux": None,
        "upgrade": None,
        "base_instructions": prompt,
        "supports_reasoning_summaries": False,
        "supports_reasoning_summary_parameter": False,
        "support_verbosity": False,
        "default_verbosity": None,
        "apply_patch_tool_type": "freeform",
        "truncation_policy": {"mode": "bytes", "limit": 10_000},
        "supports_parallel_tool_calls": False,
        "experimental_supported_tools": [],
    }
    if window:
        entry["context_window"] = int(window)
        entry["max_context_window"] = int(window)
    return {"models": [entry]}


def load_toml(text: Optional[str]) -> Optional[dict]:
    """The parsed config ({} for an empty file), or None when no TOML parser is available.
    Raises ValueError when the text isn't valid TOML."""
    if text is None or not text.strip():
        return {}
    try:
        import tomllib  # novermin
    except ImportError:
        try:
            import tomli as tomllib
        except ImportError:
            return None
    try:
        data = tomllib.loads(text)
    except Exception as exc:
        raise ValueError(str(exc)) from exc
    return data


def same_config(path: Path, text: str, other: str) -> bool:
    if text.strip() == other.strip():
        return True
    try:
        parsed = load_toml(text)
        return parsed is not None and parsed == load_toml(other)
    except Exception:
        return False


def read_file(path: Path) -> Optional[str]:
    """The file's text, None when missing. Raises ValueError for a symlink, which is never
    rewritten (the write would replace the link, or follow it somewhere unexpected)."""
    if path.is_symlink():
        raise ValueError(
            f"{path} is a symlink, so Unsloth won't rewrite it. Replace the link with a plain "
            "file and try again."
        )
    try:
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except FileNotFoundError:
        return None
    with open(fd, encoding = "utf-8", newline = "") as handle:
        return handle.read()


write_file = _cd.write_file
file_mode = _cd.file_mode


def restore_files(
    state: dict,
    *,
    read: Callable = read_file,
    write: Callable = write_file,
    same: Callable = same_config,
) -> list:
    """Undo the switch in config.toml and remove the catalog. Returns notes for the user; the
    caller revokes state["key_id"] and removes the state file."""
    path = Path(state["path"])
    text = read(path)
    skipped = []
    if text is not None:
        new_text, skipped = restore(text, state)
        backup = Path(state["backup"]) if state.get("backup") else None
        original = read(backup) if backup else None
        if original is not None and same(path, new_text, original):
            new_text = original
        if state.get("created") and same(path, new_text, ""):
            path.unlink()
        elif new_text != text:
            write(path, new_text)
        if original is not None and new_text == original:
            if state.get("mode") is not None:
                os.chmod(path, state["mode"])
            backup.unlink(missing_ok = True)
    if "model_catalog_json" not in skipped:
        path.with_name(CATALOG_NAME).unlink(missing_ok = True)
    return [
        f"Left {name} in {path} as is: it changed while Codex was on Unsloth." for name in skipped
    ]


# --------------------------------------------------------------------------------------------
# The restore record, shared with `unsloth start codex --app`
# --------------------------------------------------------------------------------------------


def state_path(agents_root: Path) -> Path:
    # The CLI's `_app_state_path("codex")`: one record, so only one switch holds the app.
    return Path(agents_root) / "app" / STATE_NAME


def read_state(path: Path) -> Optional[dict]:
    try:
        state = json.loads(Path(path).read_text(encoding = "utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(state, dict) or not state.get("path"):
        return None
    return state


write_state = _cd.write_state
owner_record = _cd.owner_record
owner_alive = _cd.owner_alive


def config_path(home: Optional[Path] = None) -> Path:
    # The app is opened from the Dock, so it never sees CODEX_HOME, nor a redirected $HOME.
    return Path(home or _cd.user_home()) / ".codex" / "config.toml"


# --------------------------------------------------------------------------------------------
# The app itself
# --------------------------------------------------------------------------------------------


def can_launch(platform: Optional[str] = None) -> bool:
    # `ollama launch codex-app` also drives Windows; Studio's card is macOS-only for now.
    return (platform or sys.platform) == "darwin"


def app_candidates(home: Optional[Path] = None) -> list:
    home = Path(home or _cd.user_home())
    roots = (Path("/Applications"), home / "Applications")
    return [root / name for root in roots for name in APP_NAMES]


def bundle_id(app: Path) -> Optional[str]:
    try:
        with open(Path(app) / "Contents" / "Info.plist", "rb") as handle:
            value = plistlib.load(handle).get("CFBundleIdentifier")
    except Exception:
        return None
    return value if isinstance(value, str) else None


def app_path(home: Optional[Path] = None) -> Optional[Path]:
    """The installed Codex app. ChatGPT.app counts only when it is the Codex build: the older
    ChatGPT chat app (com.openai.chat) lives at the same path and has no Codex in it."""
    for candidate in app_candidates(home):
        if candidate.is_dir() and bundle_id(candidate) == BUNDLE_ID:
            return candidate
    return None


def _process_patterns(app: Optional[Path]) -> list:
    if app is not None:
        return [f"{app}/Contents/MacOS/"]
    return [f"{name}/Contents/MacOS/{name[:-4]}" for name in APP_NAMES]


def is_running(app: Optional[Path] = None) -> bool:
    if sys.platform != "darwin":
        return False
    for pattern in _process_patterns(app or app_path()):
        try:
            result = subprocess.run(["pgrep", "-f", pattern], capture_output = True, timeout = 10)
        except (OSError, subprocess.SubprocessError):
            continue
        if result.returncode == 0 and result.stdout.strip():
            return True
    return False


def quit_app(timeout: float = 30.0) -> None:
    # By bundle id, so ChatGPT.app and Codex.app are both covered; a quit the app answers with a
    # dialog would otherwise block the caller.
    with contextlib.suppress(subprocess.TimeoutExpired, OSError):
        subprocess.run(
            ["osascript", "-e", f'tell application id "{BUNDLE_ID}" to quit'],
            capture_output = True,
            timeout = timeout,
            env = _cd.launch_env(),
        )


def open_app(app: Optional[Path] = None) -> None:
    """Open the app on a new Codex thread, with the environment a Dock launch would give it."""
    env = _cd.launch_env()
    commands = []
    if app is not None:
        commands.append(["/usr/bin/open", "-a", str(app), LAUNCH_URL])
        commands.append(["/usr/bin/open", str(app)])
    commands.append(["/usr/bin/open", "-b", BUNDLE_ID])
    for command in commands:
        with contextlib.suppress(OSError, subprocess.SubprocessError):
            if subprocess.run(command, capture_output = True, timeout = 30, env = env).returncode == 0:
                return
    raise OSError("Couldn't open the Codex app.")


def wait_for_exit(
    timeout: float = 30.0,
    poll: float = 0.2,
    app: Optional[Path] = None,
) -> bool:
    deadline = time.monotonic() + timeout
    while is_running(app):
        if time.monotonic() >= deadline:
            return False
        time.sleep(poll)
    return True
