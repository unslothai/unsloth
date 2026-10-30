# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Claude Desktop switched to Studio from the app's Settings → Agents, owned by this backend.

The switch itself (profile files, backups, the loopback gateway) is utils.claude_desktop, the same
code `unsloth start claude --app` runs. Here the backend holds the gateway instead of a terminal:
it lives exactly as long as this process, is switched back on Disconnect and on a normal Studio
quit, and a crash is switched back the next time Studio starts. The record is shared with the CLI,
so only one of them holds Claude at a time and either one cleans up after the other.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Callable, NamedTuple, Optional

from loggers import get_logger
from utils import claude_desktop as cd

logger = get_logger(__name__)

# Studio's desktop shell gives the backend ~5 s after SIGTERM before SIGKILL; leave room for the rest
# of shutdown. Whatever doesn't finish is switched back on the next start.
SHUTDOWN_QUIT_TIMEOUT = 2.5
QUIT_TIMEOUT = 30.0


class ClaudeDesktopError(Exception):
    def __init__(self, status: int, code: str, message: str):
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message


class _Live(NamedTuple):
    gateway: cd.Gateway
    state: dict


_lock = threading.RLock()
_live: Optional[_Live] = None


def agents_root() -> Path:
    # Same directory as the CLI's `_agents_config_root()` (STUDIO_HOME/auth/agents).
    from utils.paths.storage_roots import studio_root
    return studio_root() / "auth" / "agents"


def _state_path() -> Path:
    return cd.state_path(agents_root())


MAPPINGS_NAME = "claude-desktop-models.json"


def _mappings_path() -> Path:
    return agents_root() / "app" / MAPPINGS_NAME


def saved_mappings() -> dict:
    """The slot → model choice from the card, kept across sessions (not secret). Defaults to the
    first slot following whatever Studio has loaded."""
    try:
        data = json.loads(_mappings_path().read_text(encoding = "utf-8"))
    except (OSError, ValueError):
        data = None
    mappings = cd.normalize_mappings(
        (data or {}).get("mappings") if isinstance(data, dict) else None
    )
    return mappings or dict(cd.DEFAULT_MAPPINGS)


def save_mappings(mappings: dict) -> dict:
    normalized = cd.normalize_mappings(mappings)
    if not normalized:
        raise ClaudeDesktopError(400, "no_mapping", "Choose a model for at least one Claude model.")
    path = _mappings_path()
    path.parent.mkdir(parents = True, exist_ok = True, mode = 0o700)
    cd.write_file(path, json.dumps({"mappings": normalized}, indent = 2) + "\n")
    return normalized


def loaded_models() -> list:
    from routes.inference import _openai_model_objects
    return [entry for entry in _openai_model_objects() or () if isinstance(entry, dict)]


def is_loaded(model_id: str) -> bool:
    try:
        return cd.loaded_matches(model_id, loaded_models())
    except Exception:
        return False


def loaded_model() -> Optional[cd.GatewayModel]:
    """The model Claude Desktop is routed to: whatever Studio has loaded right now."""
    from routes.inference import _openai_model_objects

    for entry in _openai_model_objects() or ():
        if isinstance(entry, dict) and entry.get("id"):
            return cd.gateway_model(entry)
    return None


def revoke_key(state: dict) -> bool:
    """Revoke the key a switch minted, whether Studio or the CLI made it."""
    from auth import storage

    key, key_id = state.get("key"), state.get("key_id")
    if key_id is None:
        return True
    try:
        username = storage.validate_api_key(key) if isinstance(key, str) else None
        if username is None:
            return True  # already revoked or expired
        return bool(storage.revoke_api_key(username, int(key_id)))
    except Exception as exc:
        logger.warning("Couldn't revoke the Claude Desktop key %s: %s", key_id, exc)
        return False


def mint_key(username: str) -> tuple:
    from auth import storage
    key, row = storage.create_api_key(username = username, name = cd.KEY_NAME)
    return key, int(row["id"])


def _owner_kind(state: Optional[dict]) -> Optional[str]:
    if state is None:
        return None
    return (state.get("owner") or {}).get("kind") or "cli"


def status(model_provider: Optional[Callable] = None) -> dict:
    model_provider = model_provider or loaded_model
    with _lock:
        state = cd.read_state(_state_path())
        live = _live
    supported = cd.can_launch()
    app = cd.app_path() if supported else None
    owner = (state or {}).get("owner") or {}
    held_elsewhere = state is not None and live is None and cd.owner_alive(owner)
    try:
        current = model_provider()
        model = current.model_id if current is not None else None
    except Exception:
        model = None
    return {
        "supported": supported,
        "installed": app is not None,
        "app_path": str(app) if app is not None else None,
        "download_url": cd.DOWNLOAD_URL,
        "connected": live is not None,
        "held_by": _owner_kind(state) if held_elsewhere else None,
        "held_by_pid": owner.get("pid") if held_elsewhere else None,
        "running": cd.is_running() if supported else False,
        "model": model,
        "mappings": saved_mappings(),
        "routes": cd.route_names(),
    }


def _switch_back(state: dict, quit_timeout: float, revoke: Callable) -> list:
    notes = cd.switch_back(state, quit_timeout = quit_timeout)
    if not revoke(state):
        notes.append(f"Couldn't revoke the '{cd.KEY_NAME}' key; delete it in Settings → API.")
    _state_path().unlink(missing_ok = True)
    return notes


def recover(revoke: Optional[Callable] = None, quit_timeout: float = 10.0) -> list:
    """Switch back a switch whose owner (a Studio or CLI that crashed) is gone. Run at startup."""
    revoke = revoke or revoke_key
    with _lock:
        if _live is not None:
            return []
        state = cd.read_state(_state_path())
        if state is None or cd.owner_alive(state.get("owner")):
            return []
        notes = _switch_back(state, quit_timeout, revoke)
    logger.info("Switched Claude Desktop back from an Unsloth session that did not exit cleanly.")
    for note in notes:
        logger.info("Claude Desktop: %s", note)
    return notes


def connect(
    upstream: str,
    username: str,
    restart: Optional[bool],
    *,
    model_provider: Optional[Callable] = None,
    mint: Optional[Callable] = None,
    revoke: Optional[Callable] = None,
) -> dict:
    """Point Claude Desktop at a gateway this backend runs.

    ``restart`` is the answer to "Claude is open — restart it?": None asks (``needs_restart``,
    nothing changed), True quits and reopens it, False switches the files and leaves the open
    app alone until the user restarts it (``pending_restart``)."""
    global _live
    model_provider = model_provider or loaded_model
    mint = mint or mint_key
    revoke = revoke or revoke_key
    with _lock:
        if _live is not None:
            return {"status": "connected"}
        if not cd.can_launch():
            raise ClaudeDesktopError(
                409, "not_supported", "Unsloth can only switch Claude Desktop on macOS so far."
            )
        app = cd.app_path()
        if app is None:
            raise ClaudeDesktopError(
                409,
                "not_installed",
                "Claude Desktop isn't installed (looked for /Applications/Claude.app and "
                f"~/Applications/Claude.app). Install it from {cd.DOWNLOAD_URL}.",
            )
        recover(revoke)
        path = _state_path()
        held = cd.read_state(path)
        if held is not None:
            raise ClaudeDesktopError(
                409,
                "in_use",
                "Claude Desktop is already on Unsloth for `unsloth start claude --app` "
                f"(PID {(held.get('owner') or {}).get('pid')}). Stop that one first.",
            )
        targets = cd.target_paths()
        try:
            cd.snapshot(targets)
        except ValueError as exc:
            raise ClaudeDesktopError(
                409, "bad_config", f"{exc} Fix it and try again; nothing was changed."
            ) from exc
        running = cd.is_running()
        if running and restart is None:
            return {"status": "needs_restart"}

        gateway = state = None
        key_id = None
        try:
            key, key_id = mint(username)
            token = cd.new_token()
            gateway = cd.Gateway(
                upstream,
                key,
                model_provider,
                token,
                mappings = saved_mappings,
                is_loaded = is_loaded,
                ensure_loaded = lambda model_id: cd.load_model(upstream, key, model_id),
            ).start()
            state = cd.begin(
                path,
                targets,
                {"base": upstream, "key": key, "key_id": key_id, "gateway": gateway.base_url},
                cd.owner_record("studio"),
            )
            if running and restart:
                cd.quit_app()
                if not cd.wait_for_exit(timeout = QUIT_TIMEOUT):
                    raise ClaudeDesktopError(
                        409,
                        "quit_failed",
                        "Claude Desktop did not quit. Quit it yourself, then connect again.",
                    )
            try:
                # Claude writes its settings while it shuts down, so switch only once it is gone.
                cd.switch(path, state, targets, token)
            except ValueError as exc:
                raise ClaudeDesktopError(409, "bad_config", str(exc)) from exc
            if not running or restart:
                cd.open_app(app)
        except BaseException:
            if state is not None:
                cd.abort(path, state)
            if gateway is not None:
                gateway.stop()
            if key_id is not None:
                revoke({"key": key, "key_id": key_id})
            raise
        _live = _Live(gateway, state)
        logger.info("Claude Desktop switched to Unsloth (gateway %s).", gateway.base_url)
        return {"status": "pending_restart" if running and not restart else "connected"}


def set_mappings(mappings: dict, restart: Optional[bool]) -> dict:
    """Save the slot → model choice. Requests follow it at once; Claude reads the model names it
    shows only at startup, so an open, connected Claude is restarted (``restart`` True), left for
    its next launch (False) or asked about first (None → ``needs_restart``, nothing saved)."""
    normalized = cd.normalize_mappings(mappings)
    if not normalized:
        raise ClaudeDesktopError(400, "no_mapping", "Choose a model for at least one Claude model.")
    with _lock:
        live = _live
        running = live is not None and cd.is_running()
        if running and restart is None and normalized != saved_mappings():
            return {"status": "needs_restart"}
        save_mappings(normalized)
        if live is None:
            return {"status": "saved"}
        if not running:
            # "Start Claude" on the card, as in Ollama's settings: open it on the new mapping.
            if restart:
                app = cd.app_path()
                if app is not None:
                    cd.open_app(app)
            return {"status": "connected"}
        if not restart:
            return {"status": "pending_restart"}
        cd.quit_app()
        if not cd.wait_for_exit(timeout = QUIT_TIMEOUT):
            raise ClaudeDesktopError(
                409, "quit_failed", "Claude Desktop did not quit. Quit it yourself to apply."
            )
        app = cd.app_path()
        if app is not None:
            cd.open_app(app)
        return {"status": "connected"}


def disconnect(revoke: Optional[Callable] = None, quit_timeout: float = QUIT_TIMEOUT) -> list:
    """Quit Claude Desktop, put its own profile back and stop the gateway."""
    global _live
    revoke = revoke or revoke_key
    with _lock:
        live = _live
        if live is None:
            state = cd.read_state(_state_path())
            if state is not None and cd.owner_alive(state.get("owner")):
                raise ClaudeDesktopError(
                    409,
                    "in_use",
                    "Claude Desktop is on Unsloth for `unsloth start claude --app`; stop it in "
                    "that terminal (Ctrl+C).",
                )
            return recover(revoke, quit_timeout)
        try:
            notes = _switch_back(live.state, quit_timeout, revoke)
        finally:
            live.gateway.stop()
            _live = None
    logger.info("Claude Desktop switched back to its own profile.")
    return notes


def shutdown(revoke: Optional[Callable] = None) -> None:
    """Studio is quitting: switch Claude back now; anything left is recovered on the next start."""
    if _live is None:
        return
    for note in disconnect(revoke, quit_timeout = SHUTDOWN_QUIT_TIMEOUT):
        logger.info("Claude Desktop: %s", note)
