# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The Codex desktop app switched to Studio from the app's Settings → Agents, owned by this backend.

The switch itself (config.toml lines, the catalog, the restore record) is utils.codex_desktop, the
same code `unsloth start codex --app` runs. Here the backend holds the switch instead of a
terminal: it lasts exactly as long as this process, is switched back on Disconnect and on a normal
Studio quit, and a crash is switched back the next time Studio starts. The record is shared with
the CLI, so only one of them holds Codex at a time and either one cleans up after the other.

The card lists any number of Studio's models (like `ollama launch codex-app`); they become
Codex's catalog, so each one is in Codex's own model picker, and the first is the default. Codex
talks to a loopback gateway with the per-session key: Studio's /v1 answers only for loaded models,
so the gateway loads a listed model the first time Codex asks for it. An empty list follows
whatever Studio has loaded. Changing the list while connected rewrites the catalog and restarts
Codex, which reads its model list only at startup.
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
# How many models the card lists, as `ollama launch codex-app` (codexDesktopMaxModels).
MAX_MODELS = 5


class CodexDesktopError(Exception):
    def __init__(self, status: int, code: str, message: str):
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message


_lock = threading.RLock()
_live: Optional[dict] = None
_gateway: Optional[cx.Gateway] = None


def agents_root() -> Path:
    from utils.paths.storage_roots import studio_root
    return studio_root() / "auth" / "agents"


def _state_path() -> Path:
    return cx.state_path(agents_root())


def _prefs_path() -> Path:
    return agents_root() / "app" / PREFS_NAME


def normalize_models(models) -> list:
    """Model ids in order, stripped, without blanks or repeats, at most MAX_MODELS."""
    out = []
    for model in models or ():
        if isinstance(model, str) and model.strip() and model.strip() not in out:
            out.append(model.strip())
    return out[:MAX_MODELS]


def saved_models() -> list:
    """The models listed on the card, kept across sessions; empty follows the loaded model.
    Reads the one-model file earlier builds wrote, too."""
    try:
        data = json.loads(_prefs_path().read_text(encoding = "utf-8"))
    except (OSError, ValueError):
        return []
    if not isinstance(data, dict):
        return []
    if isinstance(data.get("models"), list):
        return normalize_models(data["models"])
    return normalize_models([data.get("model")])


def save_models(models) -> None:
    path = _prefs_path()
    path.parent.mkdir(parents = True, exist_ok = True, mode = 0o700)
    cd.write_file(path, json.dumps({"models": normalize_models(models)}, indent = 2) + "\n")


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
    live_models = listed_models(live) if live is not None else []
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
        "models": live_models,
        "loaded_model": current.get("id") if current else None,
        "selected_models": saved_models(),
        "max_models": MAX_MODELS,
    }


def listed_models(state: dict) -> list:
    """The models a live switch lists in Codex's catalog, in order."""
    models = [m for m in state.get("models") or () if isinstance(m, str) and m]
    return models or ([state["model"]] if state.get("model") else [])


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


def _resolve_models(
    upstream: str, key: str, requested: list, loaded: Callable, load: Callable
) -> list:
    """The catalog entries Codex will list: the picks in order (the first, Codex's default,
    loaded now if it isn't, so a bad pick fails here rather than in Codex; the rest load when
    Codex first asks for them), else the loaded model."""
    current = loaded()
    if not requested:
        entry = _loaded_entry(None, current)
        if entry is None:
            raise CodexDesktopError(
                409, "no_model", "Load a model in Studio, or pick one here, then connect Codex."
            )
        return [entry]
    first = requested[0]
    if _loaded_entry(first, current) is None:
        try:
            load(upstream, key, first)
        except Exception as exc:
            raise CodexDesktopError(
                409, "load_failed", f"Studio couldn't load {first}: {exc}"
            ) from exc
        current = loaded()
    return [_loaded_entry(model, current) or {"id": model} for model in requested]


def _start_gateway(upstream: str, key: str, load: Callable, loaded: Callable) -> cx.Gateway:
    def listed() -> list:
        live = _live
        # Following the loaded model never loads one back: Codex just uses what Studio has.
        return [] if not live or live.get("follow") else list(live.get("models") or ())

    return cx.Gateway(
        upstream,
        key,
        listed,
        lambda model_id: cd.loaded_matches(model_id, loaded()),
        lambda model_id: load(upstream, key, model_id),
    ).start()


def _stop_gateway() -> None:
    global _gateway
    gateway, _gateway = _gateway, None
    if gateway is not None:
        gateway.stop()


def _write_catalog(config: Path, entries: list) -> None:
    cx.write_file(
        config.with_name(cx.CATALOG_NAME),
        json.dumps(cx.model_catalog(entries, "list"), indent = 2) + "\n",
    )


def connect(
    upstream: str,
    username: str,
    restart: Optional[bool],
    models: Optional[list] = None,
    *,
    loaded: Optional[Callable] = None,
    load: Optional[Callable] = None,
    mint: Optional[Callable] = None,
    revoke: Optional[Callable] = None,
) -> dict:
    """Point the Codex app at this Studio. ``restart`` answers "Codex is open — restart it?" as
    for Claude Desktop. ``models`` is the list (saved for next time); None uses the saved list,
    and an empty list follows whatever Studio has loaded."""
    global _live, _gateway
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
        if models is not None:
            save_models(models)
        requested = saved_models() if models is None else normalize_models(models)
        running = cx.is_running(app)
        if running and restart is None:
            return {"status": "needs_restart"}

        state = None
        gateway = None
        key_id = key = None
        try:
            key, key_id = mint(username)
            entries = _resolve_models(upstream, key, requested, loaded, load)
            gateway = _start_gateway(upstream, key, load, loaded)
            if running and restart:
                if not _quit(QUIT_TIMEOUT):
                    raise CodexDesktopError(
                        409,
                        "quit_failed",
                        "Codex did not quit. Quit it yourself, then connect again.",
                    )
                # Codex may write its config while it shuts down, so read it again.
                text = cx.read_file(config)
            new_text, previous = cx.build(text, gateway.base_url, key, entries[0]["id"], None)
            cx.load_toml(new_text)
            backup = None
            if text is not None:
                backup = config.with_name(config.name + cx.BACKUP_SUFFIX)
                cx.write_file(backup, text)
            state = {
                "path": str(config),
                "base": gateway.base_url,
                "model": entries[0]["id"],
                "models": [entry["id"] for entry in entries],
                "follow": not requested,
                "key": key,
                "key_id": key_id,
                "owner": cx.owner_record("studio"),
                "created": text is None,
                "mode": cx.file_mode(config),
                "previous": previous,
                "backup": str(backup) if backup else None,
            }
            cx.write_state(path, state)
            _write_catalog(config, entries)
            cx.write_file(config, new_text)
            _live, _gateway = state, gateway
            if not running or restart:
                cx.open_app(app)
        except BaseException as exc:
            _live = None
            if _gateway is gateway:
                _gateway = None
            if gateway is not None:
                gateway.stop()
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
        logger.info("Codex app switched to Unsloth (%s).", ", ".join(state["models"]))
        return {"status": "pending_restart" if running and not restart else "connected"}


def set_models(
    models: Optional[list],
    restart: Optional[bool],
    *,
    loaded: Optional[Callable] = None,
    load: Optional[Callable] = None,
) -> dict:
    """Set the list (empty: follow what Studio has loaded). Not connected: saved for Connect.
    Connected: the first is loaded in Studio, the catalog and the model line are rewritten, and an
    open Codex is restarted (``restart`` True), left for its next launch (False) or asked about
    (None), since Codex reads its model list only at startup."""
    global _live
    loaded = loaded or loaded_models
    load = load or cd.load_model
    models = normalize_models(models)
    with _lock:
        live = _live
        if live is None:
            save_models(models)
            return {"status": "saved"}
        app = cx.app_path()
        running = cx.is_running(app)
        if running and restart is None:
            return {"status": "needs_restart"}
        entries = _resolve_models(live["base"], live["key"], models, loaded, load)
        save_models(models)
        ids = [entry["id"] for entry in entries]
        if ids == listed_models(live) and bool(live.get("follow")) == (not models):
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
        new_text = cx.retarget(text, live, ids[0])
        if new_text is None:
            raise CodexDesktopError(
                409,
                "changed",
                f"The model in {config} was changed in Codex; disconnect and connect again.",
            )
        # Keep every id this session ever listed: Codex may still hold one in its model line.
        owned = ids + [model for model in cx.owned_models(live) if model not in ids]
        state = {**live, "model": ids[0], "models": ids, "follow": not models}
        state["previous_models"] = owned[len(ids) :]
        cx.write_state(_state_path(), state)
        _write_catalog(config, entries)
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
            _stop_gateway()
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
            _stop_gateway()
    for note in notes:
        logger.info("Codex app: %s", note)
