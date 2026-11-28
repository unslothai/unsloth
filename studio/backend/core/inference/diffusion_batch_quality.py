# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The quality gate on a measured batching win: a race (diffusion_batch_race) only times a shared call and checks it
bit for bit against a rerun; it cannot tell whether a batched row drifts from the image that request renders alone.
That is an offline G2c verdict: held-out prompts x seeds rendered batched and alone on one execution path, scored with
LPIPS against an eager reference, paired upper bound of (batched - solo) at most ``EPS`` over at least ``MIN_PAIRS``.

Verdicts are keyed by (family, device class, execution path): the device class is the card's name (one verdict covers
every card of that model), the path is ``eager`` or ``graph`` (what the race's winning arm really ran, persisted as
``path``), or ``any``. A measured win is served only where a PASS verdict exists for its family, card and path; a missing or failing
verdict keeps today's serial rules. ``SHIPPED`` holds the verdicts measured for releases; a local file
(``<cache>/batch_quality.json``, written by ``record``) adds this machine's own."""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Optional

EPS = 0.01
MIN_PAIRS = 24
PATHS = ("eager", "graph", "any")
FILE_NAME = "batch_quality.json"

# (family name, device class, path) -> verdict. Only PASS verdicts measured on a release build belong here.
SHIPPED: dict[tuple[str, str, str], dict] = {}

# FLUX.3 klein, RTX PRO 6000 (G4), Low VRAM int8, 16 steps, batch 2, 8 prompts x 3 seeds vs eager bf16.
_KLEIN_G4 = {
    "verdict": "PASS",
    "n": 24,
    "upper": 0.0061,
    "mean": 0.0024,
    "eps": EPS,
    "size": 2,
    "steps": 16,
    "source": "lfl_klein_g4 (studio-batch-flux3-anima, #19837 e5b8e3838 base)",
}
for _path in ("eager", "graph"):
    SHIPPED[("flux.3-klein", "NVIDIA RTX PRO 6000 Blackwell Server Edition", _path)] = dict(
        _KLEIN_G4, path = _path
    )

_LOCK = threading.Lock()
_DEVICE_CLASS: dict[str, Optional[str]] = {}


def _file() -> Optional[Path]:
    try:
        from utils.paths.storage_roots import cache_root
        return cache_root() / FILE_NAME
    except Exception:  # noqa: BLE001 - no Studio home: shipped verdicts only
        return None


def _key(family: str, device: str, path: str) -> str:
    return f"{family}|{device}|{path}"


def _read() -> dict:
    path = _file()
    if path is None or not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding = "utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:  # noqa: BLE001 - an unreadable file is no verdict
        return {}


def path_of(verdict: Any) -> str:
    """The execution path a race verdict served on: its recorded ``path``, else (a verdict from before it was recorded)
    the arm name, which is a guess for a load without graph handles."""
    if isinstance(verdict, dict):
        if verdict.get("path") in ("eager", "graph"):
            return verdict["path"]
        return "eager" if verdict.get("arm") == "batched_eager" else "graph"
    return "eager" if verdict == "batched_eager" else "graph"


def device_class(state: Any) -> Optional[str]:
    """The card's model name (``NVIDIA A100-SXM4-40GB``), None off CUDA / ROCm or when it cannot be read."""
    named = getattr(state, "device_name", None)
    if isinstance(named, str) and named:
        return named
    from .diffusion_batch_race import card_of

    card = card_of(state)
    if card is None:
        return None
    if card not in _DEVICE_CLASS:
        try:
            import torch
            _DEVICE_CLASS[card] = str(torch.cuda.get_device_name(int(card.split(":")[1])))
        except Exception:  # noqa: BLE001
            _DEVICE_CLASS[card] = None
    return _DEVICE_CLASS[card]


def passes(verdict: Any) -> bool:
    """A stored verdict that lets a measured win serve: PASS, enough pairs, upper bound within ``EPS``."""
    if not isinstance(verdict, dict) or verdict.get("verdict") != "PASS":
        return False
    try:
        return int(verdict.get("n", 0)) >= MIN_PAIRS and float(verdict.get("upper")) <= EPS
    except (TypeError, ValueError):
        return False


def lookup(family: Optional[str], device: Optional[str], path: str) -> Optional[dict]:
    """The verdict that applies (this path's own, else ``any``); local entries over shipped ones."""
    if not family or not device:
        return None
    local = _read()
    for p in (path, "any"):
        found = local.get(_key(family, device, p))
        if isinstance(found, dict):
            return found
        found = SHIPPED.get((family, device, p))
        if isinstance(found, dict):
            return found
    return None


def _family(state: Any) -> Optional[str]:
    return getattr(getattr(state, "family", None), "name", None)


def quality_ok(state: Any, path: str) -> bool:
    return passes(lookup(_family(state), device_class(state), path))


def any_quality_ok(state: Any) -> bool:
    """Whether some path of this family on this card has a PASS verdict (status may report the measured mode)."""
    family, device = _family(state), device_class(state)
    return any(passes(lookup(family, device, p)) for p in ("eager", "graph"))


def gate(verdict: Any, state: Any) -> Any:
    """``verdict`` as it may serve: a promoted one without a PASS for its family, card and path is not promoted."""
    if not isinstance(verdict, dict) or not verdict.get("promoted"):
        return verdict
    path = path_of(verdict)
    if quality_ok(state, path):
        return verdict
    family, device = _family(state), device_class(state)
    found = lookup(family, device, path)
    reason = (
        f"no G2c verdict for {family} on {device} ({path})"
        if found is None
        else f"G2c verdict for {family} on {device} ({path}) does not pass"
    )
    return dict(verdict, promoted = False, quality_gate = reason)


def record(family: str, device: str, path: str, verdict: dict) -> bool:
    """Persist an offline G2c verdict (``verdict``, ``n``, ``upper`` and whatever else the harness reports)."""
    if path not in PATHS:
        raise ValueError(f"path must be one of {PATHS}")
    target = _file()
    if target is None:
        return False
    with _LOCK:
        data = _read()
        data[_key(family, device, path)] = dict(verdict, t = verdict.get("t", time.time()))
        target.parent.mkdir(parents = True, exist_ok = True)
        tmp = target.with_name(f".{target.name}.{os.getpid()}.{threading.get_ident()}.tmp")
        tmp.write_text(json.dumps(data, indent = 1, sort_keys = True), encoding = "utf-8")
        os.replace(tmp, target)
    return True
