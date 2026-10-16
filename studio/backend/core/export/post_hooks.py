# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Post-export hooks: admin-configured commands Studio can run on a finished export directory.

A hook turns an export into something Studio does not produce itself -- another runtime's format, a smoke test, a copy to
a model store. Hooks are defined only in ``post_export_hooks.json`` under the Studio root (override the path with
``UNSLOTH_POST_EXPORT_HOOKS``). A request names a hook id and picks option values from the choices the file declares, so
the API never accepts a command. Placeholders are substituted one argument at a time (no shell):
``{export_dir}``, ``{export_type}``, ``{base_model}`` and every declared option name. A command may print
``UNSLOTH_HOOK_OUTPUT=<path>`` to name what it produced; Studio lists that path with the export's outputs.
The command's environment carries ``UNSLOTH_POST_EXPORT=1``, ``UNSLOTH_EXPORT_DIR`` and ``UNSLOTH_EXPORT_TYPE``.

A hook is shown as a switch under the Merged and LoRA methods ("Also convert to ..."), or -- with a ``method``
block -- as an export method of its own: Studio exports ``source`` (``auto``: the LoRA adapter for an adapter run,
else the merged model) to a temporary folder, runs the hook on it, and removes that folder once the hook succeeds.

Example::

    {"hooks": [{
        "id": "my-runtime",
        "label": "MyRuntime",
        "description": "Converts the export with my-runtime-convert.",
        "applies_to": ["merged", "lora"],
        "command": ["my-runtime-convert", "{export_dir}", "--preset", "{preset}"],
        "options": [{"name": "preset", "label": "Preset", "default": "auto",
                     "choices_command": ["my-runtime-convert", "--list-presets", "--json"], "choices_key": "presets"}],
        "method": {"title": "MyRuntime", "description": "A MyRuntime model file.", "source": "auto"},
        "timeout_s": 3600
    }]}
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from loggers import get_logger

logger = get_logger(__name__)

_CHOICES_TIMEOUT_S = 20
_PLACEHOLDERS = ("export_dir", "export_type", "base_model")
# A hook with "method" is offered as its own export method; "source" is the export Studio feeds it
# ("auto": the LoRA adapter for an adapter run, else the merged model).
_METHOD_SOURCES = ("auto", "lora", "merged")


def hooks_file() -> Path:
    env = os.environ.get("UNSLOTH_POST_EXPORT_HOOKS", "").strip()
    if env:
        return Path(env).expanduser()
    from utils.paths.storage_roots import studio_root

    return Path(studio_root()) / "post_export_hooks.json"


def _str_list(v: Any) -> bool:
    return isinstance(v, list) and bool(v) and all(isinstance(x, str) and x for x in v)


def _normalise(h: Any) -> Optional[Dict[str, Any]]:
    """One hook entry checked field by field; None when any field has the wrong shape."""
    if not isinstance(h, dict) or not isinstance(h.get("id"), str) or not h["id"]:
        return None
    if not _str_list(h.get("command")):
        return None
    applies = h.get("applies_to", ["merged", "lora"])
    if not _str_list(applies):
        return None
    options = h.get("options", [])
    if not isinstance(options, list) or not all(
        isinstance(o, dict) and isinstance(o.get("name"), str) and o["name"] for o in options
    ):
        return None
    timeout = h.get("timeout_s", 3600)
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or timeout <= 0:
        return None
    for key in ("label", "description"):
        if key in h and not isinstance(h[key], str):
            return None
    method = h.get("method")
    if method is not None:
        if (
            not isinstance(method, dict)
            or not isinstance(method.get("title"), str)
            or not method["title"]
            or method.get("source", "auto") not in _METHOD_SOURCES
            or any(not isinstance(method.get(k, ""), str) for k in ("description", "tooltip"))
        ):
            return None
        method = {
            "title": method["title"],
            "description": method.get("description", ""),
            "tooltip": method.get("tooltip", ""),
            "source": method.get("source", "auto"),
        }
    return {
        **h,
        "applies_to": applies,
        "options": options,
        "timeout_s": float(timeout),
        "method": method,
    }


def _load() -> List[Dict[str, Any]]:
    path = hooks_file()
    if not path.is_file():
        return []
    try:
        raw = json.loads(path.read_text(encoding = "utf-8"))
    except Exception as e:
        logger.warning(f"post-export hooks: cannot read {path}: {e}")
        return []
    if not isinstance(raw, dict) or not isinstance(raw.get("hooks", []), list):
        logger.warning(
            f'post-export hooks: {path} must be an object with a "hooks" list; ignoring it'
        )
        return []
    hooks = []
    for h in raw.get("hooks", []):
        hook = _normalise(h)
        if hook is None:
            logger.warning(f"post-export hooks: skipping malformed entry {h!r}")
            continue
        hooks.append(hook)
    return hooks


def _as_choices(values: Any) -> List[str]:
    if not isinstance(values, list):
        return []
    return [str(v) for v in values if isinstance(v, (str, int, float)) and not isinstance(v, bool)]


def _choices(opt: Dict[str, Any]) -> List[str]:
    cmd = opt.get("choices_command")
    if not _str_list(cmd):
        return _as_choices(opt.get("choices", []))
    assert isinstance(cmd, list)
    try:
        out = subprocess.run(
            cmd, capture_output = True, text = True, timeout = _CHOICES_TIMEOUT_S, check = True
        ).stdout
        data = json.loads(out)
        values = data.get(opt.get("choices_key", "choices"), []) if isinstance(data, dict) else data
        if not isinstance(values, list):
            raise ValueError(f"expected a JSON list of choices, got {type(values).__name__}")
        return _as_choices(values)
    except Exception as e:
        logger.warning(f"post-export hooks: choices_command {cmd!r} failed: {e}")
        return _as_choices(opt.get("choices", []))


def list_hooks(export_type: Optional[str] = None) -> List[Dict[str, Any]]:
    """Hooks for the UI: id, label, description, applies_to, options with resolved choices. Commands stay server-side."""
    result = []
    for h in _load():
        applies = h.get("applies_to") or ["merged", "lora"]
        if export_type and export_type not in applies:
            continue
        exe = h["command"][0]
        result.append(
            {
                "id": h["id"],
                "label": h.get("label", h["id"]),
                "description": h.get("description", ""),
                "applies_to": applies,
                "available": bool(shutil.which(exe) or os.path.isfile(exe)),
                "method": h["method"],
                "options": [
                    {
                        "name": o["name"],
                        "label": o.get("label", o["name"]),
                        "default": str(o.get("default", "")),
                        "choices": _choices(o),
                    }
                    for o in h.get("options", [])
                    if isinstance(o, dict) and o.get("name")
                ],
            }
        )
    return result


def _base_model_of(export_dir: Path) -> str:
    for name, key in (
        ("export_metadata.json", "base_model"),
        ("adapter_config.json", "base_model_name_or_path"),
        ("config.json", "_name_or_path"),
    ):
        p = export_dir / name
        if p.is_file():
            try:
                v = json.loads(p.read_text(encoding = "utf-8-sig")).get(key)
                if v:
                    return str(v)
            except Exception:
                pass
    return ""


def _export_type_of(export_dir: Path) -> str:
    if (export_dir / "adapter_config.json").is_file():
        return "lora"
    if (export_dir / "config.json").is_file():
        return "merged"
    return "other"


def run_hook(
    hook_id: str,
    export_dir: str,
    options: Dict[str, str],
    allowed_root: Path,
    log: Callable[[str, str], None],
    cancel: Optional[threading.Event] = None,
    discard_export: bool = False,
) -> Tuple[bool, str, Dict[str, Any]]:
    """Run hook ``hook_id`` on ``export_dir`` (which must lie inside ``allowed_root``), streaming output to ``log``.

    ``discard_export`` removes ``export_dir`` once the hook succeeds -- only for a hook configured as an export
    method, only strictly inside ``allowed_root``, and never when the hook's output lies inside it.
    """
    hook = next((h for h in _load() if h["id"] == hook_id), None)
    if hook is None:
        return False, f"Unknown post-export hook '{hook_id}'", {}
    target = Path(export_dir).resolve()
    root = Path(allowed_root).resolve()
    if not target.is_dir() or not (target == root or root in target.parents):
        return False, "Export directory is not inside this account's exports folder", {}
    if discard_export and (not hook.get("method") or target == root):
        return False, f"Hook '{hook_id}' is not an export method; its export is never removed", {}
    requested = target
    if _export_type_of(target) == "other":
        # A run folder holding one checkpoint folder ({run}/{checkpoint}/): use the checkpoint.
        inner = [d for d in target.iterdir() if d.is_dir() and _export_type_of(d) != "other"]
        if len(inner) == 1:
            target = inner[0]
    export_type = _export_type_of(target)
    applies = hook.get("applies_to") or ["merged", "lora"]
    if export_type not in applies:
        return False, f"Hook '{hook_id}' does not apply to a {export_type} export", {}

    values: Dict[str, str] = {
        "export_dir": str(target),
        "export_type": export_type,
        "base_model": _base_model_of(target),
    }
    for opt in hook.get("options", []):
        name = opt["name"]
        value = str(options.get(name, opt.get("default", "")))
        allowed = _choices(opt)
        if allowed and value not in allowed:
            return False, f"Option '{name}' must be one of the hook's choices", {}
        values[name] = value

    def sub(arg: str) -> str:
        for k, v in values.items():
            arg = arg.replace("{" + k + "}", v)
        return arg

    argv = [sub(str(a)) for a in hook["command"]]
    timeout = float(hook.get("timeout_s", 3600))
    log(f"[post-export:{hook_id}] {' '.join(argv)}", "stdout")
    t0 = time.time()
    tail: List[str] = []
    produced = ""
    try:
        env = {
            **os.environ,
            "UNSLOTH_POST_EXPORT": "1",
            "UNSLOTH_EXPORT_DIR": str(target),
            "UNSLOTH_EXPORT_TYPE": export_type,
        }
        # Own process group on POSIX, so a timeout or cancel also stops what the command started.
        proc = subprocess.Popen(
            argv,
            stdout = subprocess.PIPE,
            stderr = subprocess.STDOUT,
            text = True,
            bufsize = 1,
            env = env,
            start_new_session = os.name == "posix",
        )
    except OSError as e:
        return False, f"Hook '{hook_id}' could not start: {e}", {}
    assert proc.stdout is not None
    # A watcher enforces the timeout and cancel even while the command prints nothing.
    stopped: List[str] = []
    finished = threading.Event()

    def _watch() -> None:
        while not finished.wait(0.5):
            if cancel is not None and cancel.is_set():
                stopped.append("cancelled")
            elif time.time() - t0 > timeout:
                stopped.append("timeout")
            else:
                continue
            if os.name == "posix":
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                    return
                except OSError:
                    pass
            proc.kill()
            return

    watcher = threading.Thread(target = _watch, daemon = True)
    watcher.start()
    try:
        for line in proc.stdout:
            line = line.rstrip()
            if line.startswith("UNSLOTH_HOOK_OUTPUT="):
                produced = line.split("=", 1)[1].strip()
                continue
            if line:
                log(line, "stdout")
                tail = (tail + [line])[-20:]
        rc = proc.wait()
    finally:
        finished.set()
        watcher.join()
    if stopped:
        log(f"[post-export:{hook_id}] stopped ({stopped[0]})", "stderr")
        return False, f"Hook '{hook_id}' stopped ({stopped[0]})", {"tail": tail}
    secs = round(time.time() - t0, 1)
    details = {
        "hook": hook_id,
        "returncode": rc,
        "seconds": secs,
        "tail": tail,
        "output_path": produced or None,
    }
    if rc != 0:
        log(f"[post-export:{hook_id}] failed (exit {rc}) after {secs} s", "stderr")
        return False, f"Hook '{hook_id}' failed (exit {rc})", details
    log(f"[post-export:{hook_id}] done in {secs} s", "stdout")
    if discard_export:
        out = Path(produced).resolve() if produced else None
        if out is not None and (out == requested or requested in out.parents):
            log(
                f"[post-export:{hook_id}] kept {requested}: the hook's output is inside it",
                "stdout",
            )
        else:
            shutil.rmtree(requested, ignore_errors = True)
            details["discarded"] = str(requested)
            log(f"[post-export:{hook_id}] removed the intermediate export {requested}", "stdout")
    return True, f"{hook.get('label', hook_id)} finished in {secs} s", details
