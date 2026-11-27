# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Leave an unread vision tower out of a text encoder.

FLUX.3 and Krea-2 encode their prompt through Qwen3-VL's language model only: no FLUX.3 mode (text-to-image, a
single reference or several) and no Krea-2 mode ever hands the encoder ``pixel_values``, since FLUX.3 routes its
reference images through its own VAE. Qwen3-VL-4B's vision tower is still ~0.42 B bf16 parameters (0.83 GB, 315
tensors, in the dense and the int8 ConvRot files alike), which every load reads, holds in host RAM and places or
streams with the encoder. ``vision_tower_skipped`` builds the encoder with an empty stand-in for the tower so its
checkpoint tensors are never read; ``drop_vision_tower`` frees a tower that was already built.

Families whose edit modes run the tower (Qwen-Image edit, Qwen-Image-2.1, MiniMax-H3 reference) are not listed and
keep it. ``UNSLOTH_DIFFUSION_TE_DROP_VISION=0`` keeps it everywhere.
"""

from __future__ import annotations

import logging
import os
import re
import threading
from contextlib import contextmanager
from typing import Any, Iterator, Optional

DROP_VISION_ENV = "UNSLOTH_DIFFUSION_TE_DROP_VISION"
# Families whose text encoder never runs its vision tower.
TEXT_ONLY_VISION_FAMILIES = frozenset({"flux.3-klein", "flux.3-klein-base", "krea-2"})
# Checkpoint prefixes of the tower: ``visual.`` on a bare Qwen3VLModel, ``model.visual.`` under the conditional
# generation class (and its 4.x checkpoint name, ``visual.`` at the root).
VISION_TOWER_PREFIXES = ("visual.", "model.visual.")
_UNEXPECTED_VISUAL = r"(^|\.)visual\."

_local = threading.local()


def drop_enabled() -> bool:
    return (os.environ.get(DROP_VISION_ENV) or "").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )


def family_drops_vision_tower(family: Optional[str]) -> bool:
    return drop_enabled() and (family or "").strip().lower() in TEXT_ONLY_VISION_FAMILIES


def is_vision_tower_key(name: str) -> bool:
    return str(name).startswith(VISION_TOWER_PREFIXES)


def _holds_visual(module: Any) -> bool:
    """``module`` registers a ``visual`` child. Read off ``_modules`` rather than an ``isinstance`` on a fresh
    ``import torch.nn``, which a stubbed ``torch`` in ``sys.modules`` would answer wrongly."""
    children = getattr(module, "_modules", None)
    return isinstance(children, dict) and children.get("visual") is not None


def _tower_owner(text_encoder: Any) -> Any:
    """The module that holds ``visual``: the encoder itself (Qwen3VLModel, Qwen3_5Model) or its ``model``
    (the conditional-generation classes)."""
    if _holds_visual(text_encoder):
        return text_encoder
    inner = getattr(text_encoder, "model", None)
    if _holds_visual(inner):
        return inner
    return None


def drop_vision_tower(text_encoder: Any) -> bool:
    """Free the encoder's vision tower; the text forward never touches it (images enter only through
    ``pixel_values``). True when a tower was dropped."""
    owner = _tower_owner(text_encoder)
    if owner is None:
        return False
    try:
        owner.visual = None
    except Exception:  # noqa: BLE001 - a frozen module keeps its tower
        return False
    return True


def has_vision_tower(text_encoder: Any) -> bool:
    return _tower_owner(text_encoder) is not None


def _skipping() -> bool:
    return bool(getattr(_local, "depth", 0))


def _stand_in_class(tower_class: type) -> Any:
    """An empty module built in place of ``tower_class``. Its base is the ``nn.Module`` in the tower class's own MRO,
    the one transformers was imported with, so the stand-in registers as a child even when the first scope opens while
    a test has a stub ``torch`` in ``sys.modules``."""
    module_base = next(
        c
        for c in tower_class.__mro__
        if c.__name__ == "Module" and c.__module__ == "torch.nn.modules.module"
    )

    class SkippedVisionTower(module_base):
        """Empty stand-in built in place of the vision tower; transformers 5.x collects its ignore pattern, so the
        tower's checkpoint tensors are skipped without an unexpected-key report."""

        _keys_to_ignore_on_load_unexpected = [_UNEXPECTED_VISUAL]

    return SkippedVisionTower


_PATCH_LOCK = threading.Lock()
_PATCHED: set = set()


def _vision_classes() -> list:
    classes = []
    try:
        from transformers.models.qwen3_vl import modeling_qwen3_vl
    except Exception:  # noqa: BLE001 - no Qwen3-VL in this transformers: nothing to skip
        return classes
    cls = getattr(modeling_qwen3_vl, "Qwen3VLVisionModel", None)
    if isinstance(cls, type):
        classes.append(cls)
    return classes


def _install() -> bool:
    """Route ``Qwen3VLVisionModel._from_config`` (4.57 builds the tower through it; 5.x through
    ``AutoModel.from_config``, which calls it) to the stand-in while this thread is inside the scope. Installed once;
    outside a scope it is the inherited classmethod."""
    classes = _vision_classes()
    if not classes:
        return False
    with _PATCH_LOCK:
        for cls in classes:
            if cls in _PATCHED:
                continue
            original = cls._from_config
            stand_in = _stand_in_class(cls)

            def _from_config(
                klass,
                config,
                *args,
                _original = original,
                _stand_in = stand_in,
                **kwargs,
            ):
                if _skipping():
                    _local.built = getattr(_local, "built", 0) + 1
                    return _stand_in()
                return _original.__func__(klass, config, *args, **kwargs)

            cls._from_config = classmethod(_from_config)
            _PATCHED.add(cls)
    return True


class _UnusedVisualKeysFilter(logging.Filter):
    """transformers 4.x names the skipped tower tensors as unused checkpoint weights; drop that record only when
    every key it lists is a tower key."""

    _KEYS = re.compile(r"'([^']+)'")

    def filter(self, record: logging.LogRecord) -> bool:
        if not _skipping():
            return True
        try:
            msg = record.getMessage()
        except Exception:  # noqa: BLE001
            return True
        if "were not used when" not in msg:
            return True
        head = msg.split("- This IS expected", 1)[0]
        keys = self._KEYS.findall(head.split(":", 1)[-1])
        return not (keys and all(re.search(_UNEXPECTED_VISUAL, k) for k in keys))


_FILTER = _UnusedVisualKeysFilter()


@contextmanager
def vision_tower_skipped(enabled: bool = True) -> Iterator[None]:
    """Inside, a Qwen3-VL ``from_pretrained`` builds no vision tower and reads none of its tensors; call
    ``drop_vision_tower`` on the result to clear the stand-in. Per thread, so a concurrent load of a family that runs
    the tower is unaffected."""
    if not enabled or not _install():
        yield
        return
    log = logging.getLogger("transformers.modeling_utils")
    _local.depth = getattr(_local, "depth", 0) + 1
    log.addFilter(_FILTER)
    try:
        yield
    finally:
        _local.depth -= 1
        if not _local.depth:
            log.removeFilter(_FILTER)
