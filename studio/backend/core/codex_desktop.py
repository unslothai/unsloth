# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The Codex desktop app switched to Studio from the app's Settings → Agents, owned by this backend.

The switch itself (config.toml lines, the catalog, the restore record) is utils.codex_desktop, the
same code `unsloth start codex --app` runs. Here the backend holds the switch instead of a
terminal: it lasts exactly as long as this process, is switched back on Disconnect and on a normal
Studio quit, and a crash is switched back the next time Studio starts. The record is shared with
the CLI, so only one of them holds Codex at a time and either one cleans up after the other.

Codex talks to Studio's own /v1 directly with a per-session key (no gateway, unlike Claude
Desktop), so it names one model. The card picks it; the pick is loaded in Studio before Codex is
pointed at it, and changing it while connected rewrites the model line and restarts Codex, which
reads its model list only at startup.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Callable, Optional

from loggers import get_logger
from utils import claude_desktop as cd
from utils import codex_desktop as cx

logger = get_logger(__name__)

SHUTDOWN_QUIT_TIMEOUT = 2.5
QUIT_TIMEOUT = 30.0
PREFS_NAME = "codex-desktop-model.json"


class CodexDesktopError(Exception):
    def __init__(self, status: int, code: str, message: str):
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message


_lock = threading.RLock()
_live: Optional[dict] = None


def agents_root() -> Path:
    from utils.paths.storage_roots import studio_root
    return studio_root() / "auth" / "agents"


def _state_path() -> Path:
    return cx.state_path(agents_root())


def _prefs_path() -> Path:
    return agents_root() / "app" / PREFS_NAME


def saved_model() -> Optional[str]:
    """The model picked on the card, kept across sessions; None follows the loaded model."""
    try:
        data = json.loads(_prefs_path().read_text(encoding = "utf-8"))
    except (OSError, ValueError):
        return None
    model = data.get("model") if isinstance(data, dict) else None
    return model.strip() if isinstance(model, str) and model.strip() else None


def save_model(model: Optional[str]) -> None:
    path = _prefs_path()
    path.parent.mkdir(parents = True, exist_ok = True, mode = 0o700)
    cd.write_file(path, json.dumps({"model": model}, indent = 2) + "\n")


def loaded_models() -> list:
    from routes.inference import _openai_model_objects
    return [entry for entry in _openai_model_objects() or () if isinstance(entry, dict)]


def _loaded_entry(model_id: Optional[str], loaded: list) -> Optional[dict]:
    if model_id is None:
        return next((entry for entry in loaded if entry.get("id")), None)
    for entry in loaded:
        if cd.loaded_matches(model_id, [entry]):
            return {**entry, "id": model_id}
    return None


def revoke_key(state: dict) -> bool:
    from auth import storage

    key, key_id = state.get("key"), state.get("key_id")
    if key_id is None:
        return True
    try:
        username = storage.validate_api_key(key) if isinstance(key, str) else None
        if username is None:
            return True
        return bool(storage.revoke_api_key(username, int(key_id)))
    except Exception as exc:
        logger.warning("Couldn't revoke the Codex app key %s: %s", key_id, exc)
        return False


def mint_key(username: str) -> tuple:
    from auth import storage
    key, row = storage.create_api_key(username = username, name = cx.KEY_NAME)
    return key, int(row["id"])


def status(loaded: Optional[Callable] = None) -> dict:
    loaded = loaded or loaded_models
    with _lock:
        state = cx.read_state(_state_path())
        live = _live
    supported = cx.can_launch()
    app = cx.app_path() if supported else None
    owner = (state or {}).get("owner") or {}
    held_elsewhere = state is not None and live is None and cx.owner_alive(owner)
    try:
        current = _loaded_entry(None, loaded())
    except Exception:
        current = None
    return {
        "supported": supported,
        "installed": app is not None,
        "app_path": str(app) if app is not None else None,
        "download_url": cx.DOWNLOAD_URL,
        "connected": live is not None,
        "held_by": ((owner.get("kind") or "cli") if held_elsewhere else None),
        "held_by_pid": owner.get("pid") if held_elsewhere else None,
        "running": cx.is_running(app) if supported else False,
        "model": live["model"] if live is not None else None,
        "loaded_model": current.get("id") if current else None,
        "selected_model": saved_model(),
    }


def _switch_back(state: dict, revoke: Callable) -> list:
    notes = cx.restore_files(state)
    if not revoke(state):
        notes.append(f"Couldn't revoke the '{cx.KEY_NAME}' key; delete it in Settings → API.")
    _state_path().unlink(missing_ok = True)
    return notes


def _quit(timeout: float) -> bool:
    cx.quit_app(timeout = timeout)
    return cx.wait_for_exit(timeout = timeout)


def recover(revoke: Optional[Callable] = None) -> list:
    """Switch back a switch whose owner (a Studio or CLI that crashed) is gone. Run at startup.
    An open Codex is left alone: it is pointed at a dead key either way, and quitting someone's
    app unasked at Studio startup is worse than one failed request."""
    revoke = revoke or revoke_key
    with _lock:
        if _live is not None:
            return []
        state = cx.read_state(_state_path())
        if state is None or cx.owner_alive(state.get("owner")):
            return []
        notes = _switch_back(state, revoke)
    logger.info("Switched the Codex app back from an Unsloth session that did not exit cleanly.")
    for note in notes:
        logger.info("Codex app: %s", note)
    return notes


def _resolve_model(
    upstream: str, key: str, requested: Optional[str], loaded: Callable, load: Callable
) -> dict:
    """The catalog entry Codex will use: the pick (loaded first if it isn't), else the loaded
    model."""
    entry = _loaded_entry(requested, loaded())
    if entry is not None:
        return entry
    if requested is None:
        raise CodexDesktopError(
            409, "no_model", "Load a model in Studio, or pick one here, then connect Codex."
        )
    try:
        load(upstream, key, requested)
    except Exception as exc:
        raise CodexDesktopError(
            409, "load_failed", f"Studio couldn't load {requested}: {exc}"
        ) from exc
    return _loaded_entry(requested, loaded()) or {"id": requested}


def connect(
    upstream: str,
    username: str,
    restart: Optional[bool],
    model: Optional[str] = None,
    *,
    loaded: Optional[Callable] = None,
    load: Optional[Callable] = None,
    mint: Optional[Callable] = None,
    revoke: Optional[Callable] = None,
) -> dict:
    """Point the Codex app at this Studio. ``restart`` answers "Codex is open — restart it?" as
    for Claude Desktop. ``model`` is the pick (saved for next time); None uses the saved pick, or
    whatever Studio has loaded."""
    global _live
    loaded = loaded or loaded_models
    load = load or cd.load_model
    mint = mint or mint_key
    revoke = revoke or revoke_key
    with _lock:
        if _live is not None:
            return {"status": "connected"}
        if not cx.can_launch():
            raise CodexDesktopError(
                409, "not_supported", "Unsloth can only switch the Codex app on macOS so far."
            )
        app = cx.app_path()
        if app is None:
            raise CodexDesktopError(
                409,
                "not_installed",
                "The Codex app isn't installed (looked for ChatGPT.app and Codex.app in "
                f"/Applications and ~/Applications). Install it from {cx.DOWNLOAD_URL}.",
            )
        recover(revoke)
        path = _state_path()
        held = cx.read_state(path)
        if held is not None:
            raise CodexDesktopError(
                409,
                "in_use",
                "The Codex app is already on Unsloth for `unsloth start codex --app` "
                f"(PID {(held.get('owner') or {}).get('pid')}). Stop that one first.",
            )
        config = cx.config_path()
        try:
            text = cx.read_file(config)
            cx.load_toml(text)
        except ValueError as exc:
            raise CodexDesktopError(
                409, "bad_config", f"Couldn't use {config}: {exc} Nothing was changed."
            ) from exc
        if model is not None:
            save_model(model or None)
        requested = saved_model() if model is None else (model or None)
        running = cx.is_running(app)
        if running and restart is None:
            return {"status": "needs_restart"}

        state = None
        key_id = key = None
        try:
            key, key_id = mint(username)
            entry = _resolve_model(upstream, key, requested, loaded, load)
            if running and restart:
                if not _quit(QUIT_TIMEOUT):
                    raise CodexDesktopError(
                        409,
                        "quit_failed",
                        "Codex did not quit. Quit it yourself, then connect again.",
                    )
                # Codex may write its config while it shuts down, so read it again.
                text = cx.read_file(config)
            new_text, previous = cx.build(text, upstream, key, entry["id"], None)
            cx.load_toml(new_text)
            backup = None
            if text is not None:
                backup = config.with_name(config.name + cx.BACKUP_SUFFIX)
                cx.write_file(backup, text)
            state = {
                "path": str(config),
                "base": upstream,
                "model": entry["id"],
                "key": key,
                "key_id": key_id,
                "owner": cx.owner_record("studio"),
                "created": text is None,
                "mode": cx.file_mode(config),
                "previous": previous,
                "backup": str(backup) if backup else None,
            }
            cx.write_state(path, state)
            cx.write_file(
                config.with_name(cx.CATALOG_NAME),
                json.dumps(cx.model_catalog(entry, "list"), indent = 2) + "\n",
            )
            cx.write_file(config, new_text)
            if not running or restart:
                cx.open_app(app)
        except BaseException as exc:
            if state is not None:
                try:
                    cx.restore_files(state)
                except Exception:
                    logger.warning("Couldn't undo a failed Codex switch", exc_info = True)
                path.unlink(missing_ok = True)
            if key_id is not None:
                revoke({"key": key, "key_id": key_id})
            if isinstance(exc, ValueError):
                raise CodexDesktopError(409, "bad_config", str(exc)) from exc
            if isinstance(exc, FileNotFoundError):
                raise CodexDesktopError(500, "missing_prompt", str(exc)) from exc
            raise
        _live = state
        logger.info("Codex app switched to Unsloth (%s).", entry["id"])
        return {"status": "pending_restart" if running and not restart else "connected"}


def set_model(
    model: Optional[str],
    restart: Optional[bool],
    *,
    loaded: Optional[Callable] = None,
    load: Optional[Callable] = None,
) -> dict:
    """Pick the model (None: follow what Studio has loaded). Not connected: saved for Connect.
    Connected: loaded in Studio, written into config.toml and the catalog, and an open Codex is
    restarted (``restart`` True), left for its next launch (False) or asked about (None)."""
    global _live
    loaded = loaded or loaded_models
    load = load or cd.load_model
    model = model.strip() if isinstance(model, str) and model.strip() else None
    with _lock:
        live = _live
        if live is None:
            save_model(model)
            return {"status": "saved"}
        app = cx.app_path()
        running = cx.is_running(app)
        if running and restart is None:
            return {"status": "needs_restart"}
        entry = _resolve_model(live["base"], live["key"], model, loaded, load)
        save_model(model)
        if entry["id"] == live["model"]:
            if not running and restart and app is not None:
                cx.open_app(app)  # "Start Codex" with nothing to change
            return {"status": "connected"}
        if running and restart and not _quit(QUIT_TIMEOUT):
            raise CodexDesktopError(
                409, "quit_failed", "Codex did not quit. Quit it yourself, then try again."
            )
        config = Path(live["path"])
        try:
            text = cx.read_file(config) or ""
        except ValueError as exc:
            raise CodexDesktopError(409, "bad_config", str(exc)) from exc
        new_text = cx.retarget(text, live, entry["id"])
        if new_text is None:
            raise CodexDesktopError(
                409,
                "changed",
                f"The model in {config} was changed in Codex; disconnect and connect again.",
            )
        state = {**live, "model": entry["id"]}
        cx.write_state(_state_path(), state)
        cx.write_file(
            config.with_name(cx.CATALOG_NAME),
            json.dumps(cx.model_catalog(entry, "list"), indent = 2) + "\n",
        )
        cx.write_file(config, new_text)
        _live = state
        if restart and app is not None:
            cx.open_app(app)
        return {"status": "pending_restart" if running and not restart else "connected"}


def disconnect(
    restart: Optional[bool] = None,
    revoke: Optional[Callable] = None,
    quit_timeout: float = QUIT_TIMEOUT,
) -> dict:
    """Put Codex's own config back and revoke the session key. Like `ollama launch codex-app`'s
    restore, an open Codex is restarted onto its own settings (``restart`` True; None asks first,
    False leaves it running until the user restarts it), and a closed one stays closed."""
    global _live
    revoke = revoke or revoke_key
    with _lock:
        live = _live
        if live is None:
            state = cx.read_state(_state_path())
            if state is not None and cx.owner_alive(state.get("owner")):
                raise CodexDesktopError(
                    409,
                    "in_use",
                    "The Codex app is on Unsloth for `unsloth start codex --app`; stop it in that "
                    "terminal (Ctrl+C).",
                )
            return {"status": "disconnected", "notes": recover(revoke)}
        app = cx.app_path()
        running = cx.is_running(app)
        if running and restart is None:
            return {"status": "needs_restart", "notes": []}
        notes = []
        quit_ok = False
        if running and restart:
            quit_ok = _quit(quit_timeout)
            if not quit_ok:
                notes.append("Codex did not quit; quit and reopen it to use your usual setup.")
        try:
            notes = _switch_back(live, revoke) + notes
        finally:
            _live = None
        if running and not restart:
            notes.append("Restart Codex to go back to your usual setup.")
        if quit_ok and app is not None:
            try:
                cx.open_app(app)
            except OSError:
                notes.append("Couldn't reopen Codex; open it yourself.")
    logger.info("Codex app switched back to its own config.")
    return {"status": "disconnected", "notes": notes}


def shutdown(revoke: Optional[Callable] = None) -> None:
    """Studio is quitting: quit an open Codex (its key is about to be revoked) and put its config
    back now; anything left is recovered on the next start."""
    global _live
    revoke = revoke or revoke_key
    with _lock:
        live = _live
        if live is None:
            return
        if cx.is_running():
            _quit(SHUTDOWN_QUIT_TIMEOUT)
        try:
            notes = _switch_back(live, revoke)
        finally:
            _live = None
    for note in notes:
        logger.info("Codex app: %s", note)
