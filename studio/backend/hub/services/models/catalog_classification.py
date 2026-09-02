# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Dependency-light task classification for model inventory rows."""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Optional

from hub.services.models.common import (
    _is_diffusers_pipeline_dir,
    _is_mmproj_filename,
    _iter_gguf_paths,
    _local_path_can_chat,
)
from utils.gguf_archs import SPEECH_GGUF_ARCHS, is_speech_gguf_architecture
from utils.paths.path_utils import file_contents_available_locally


_DIFFUSION_GGUF_ARCHS = frozenset({"flux", "flux2", "qwen_image", "qwenimage", "z_image", "zimage"})
_UNSUPPORTED_DIFFUSION_GGUF_ARCHS = frozenset(
    {"sd1", "sd3", "sdxl", "aura", "hidream", "cosmos", "hyvid"}
)
_AMBIGUOUS_DIFFUSION_GGUF_ARCHS = frozenset({"lumina2"})
_PLACEHOLDER_DIFFUSION_GGUF_ARCHS = frozenset({"pig", "cow"})
_VIDEO_GGUF_ARCHS = frozenset({"ltxv", "wan"})
_VIDEO_GEN_TASK = "text-to-video"

# TTS-only GGUF archs llama.cpp cannot load, tagged speech so the chat picker keeps them out of
# llama-server. One shared definition rather than a copy per layer: the chat gate, this classifier
# and the media preflight all have to agree.
_SPEECH_GGUF_ARCHS = SPEECH_GGUF_ARCHS
_SPEECH_TASK = "text-to-speech"
_UNSUPPORTED_DIFFUSION_TASK = "image-diffusion-unsupported"
_H3_DENOISER_GGUF_PREFIXES = ("minimax_h3_fl2va", "minimax_h3_ref2va")
_LOADABLE_MEDIA_GGUF_TASKS = frozenset({"text-to-image", _VIDEO_GEN_TASK})
_MAX_TASK_CLASSIFY_GGUFS = 64
_TASK_CLASSIFY_WALK_SECONDS = 0.75
_TASK_CLASSIFY_READ_SECONDS = 1.5
_QWEN3_ASR_HINT = re.compile(
    r"(?<![a-z0-9])qwen3[-_. ]*asr[-_. ]*(?:0[._]6|1[._]7)b(?![a-z0-9])",
    re.IGNORECASE,
)
_ORPHEUS_GGUF_HINT = re.compile(
    r"(?<![a-z0-9])orpheus[-_. ]*3b(?![a-z0-9])",
    re.IGNORECASE,
)


class _LocalProbeModel:
    def __init__(self, model, path: str):
        self._model = model
        self.path = path

    def __getattr__(self, name):
        return getattr(self._model, name)


def _local_probe_model(model):
    if getattr(model, "source", None) != "hf_cache" or _hf_cache_snapshot_repo_id(model.path):
        return model
    try:
        from hub.utils.inventory_scan import resolve_hf_cache_realpath
        path = resolve_hf_cache_realpath(Path(model.path))
    except Exception:
        path = None
    return _LocalProbeModel(model, path) if path and path != model.path else model


def _is_h3_bundle_gguf_hint(hint: Optional[str]) -> bool:
    if not hint:
        return False
    name = str(hint).strip().lower().rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
    if name.endswith(".gguf") and name.startswith(_H3_DENOISER_GGUF_PREFIXES):
        return True
    try:
        from hub.utils.gguf import is_h3_bundle_repo
        return is_h3_bundle_repo(hint)
    except Exception:
        return False


def _gguf_architecture(path: str) -> Optional[str]:
    # Every read inventory classification makes goes through here, so this is where "never open a
    # cloud placeholder" holds: opening one recalls its whole payload. Load-time inspection calls
    # read_gguf_architecture directly and still hydrates, on purpose.
    if not file_contents_available_locally(path):
        return None
    from utils.models.gguf_metadata import read_gguf_architecture
    return read_gguf_architecture(path)


def _gguf_family_buildable(name_hints: tuple[Optional[str], ...]) -> bool:
    try:
        from core.inference.diffusion_engine_router import family_buildable_here
        from core.inference.diffusion_families import detect_family_for_pick
        for hint in name_hints:
            if not hint:
                continue
            family = detect_family_for_pick(hint)
            if family is not None:
                return family_buildable_here(family, model_kind = "gguf")
    except Exception:
        return True
    return True


def _video_family_buildable(family) -> bool:
    try:
        from core.inference.diffusion_families import family_pipeline_available
        return family_pipeline_available(family)
    except Exception:
        return True


def _hint_leaf(hint: str) -> str:
    """The last path segment of *hint*, leaving a bare name or repo id untouched."""
    return str(hint).strip().rstrip("/\\").rsplit("/", 1)[-1].rsplit("\\", 1)[-1]


def _name_hint_media_task(
    name_hints: tuple[Optional[str], ...], unmatched: Optional[str]
) -> Optional[str]:
    """Media task from name and path hints alone, for a GGUF whose header settles nothing.

    Both callers are files whose architecture cannot answer the question: a denoiser that
    declares a placeholder arch, and a cloud file we must not open. They differ only in what
    an unrecognised name means, hence *unmatched*.
    """
    from core.inference.video_families import detect_video_family

    for hint in name_hints:
        family = detect_video_family(hint) if hint else None
        if family is not None:
            if not getattr(family, "is_moe", False) and _video_family_buildable(family):
                return _VIDEO_GEN_TASK
            return _UNSUPPORTED_DIFFUSION_TASK
    from core.inference.diffusion_families import detect_family_for_pick

    if any(detect_family_for_pick(hint) is not None for hint in name_hints if hint):
        return (
            "text-to-image" if _gguf_family_buildable(name_hints) else _UNSUPPORTED_DIFFUSION_TASK
        )
    return unmatched


def _unhydrated_gguf_task(name_hints: tuple[Optional[str], ...]) -> Optional[str]:
    """Task for a GGUF still held as a cloud placeholder, read from its name alone.

    The pickers filter On Device rows on an exact task, so an unclassified denoiser drops out
    of Images and Video -- the pages whose pick is what would hydrate it -- and lists in Chat
    instead, where a background auto-load recalls it into llama.cpp. The name is the only
    evidence available, and it is the same evidence _arch_to_task already routes a
    placeholder-arch denoiser on.

    Speech stays out on purpose. A row tagged automatic-speech-recognition is dropped from
    every filesystem list, and a text-to-speech GGUF row whose codec is unknown fails Audio's
    routing check closed, so guessing either from a name hides the model outright. Unknown
    keeps it in Chat, which is where a GGUF with nothing but a name belongs.
    """
    if any(_is_h3_bundle_gguf_hint(hint) for hint in name_hints):
        return _VIDEO_GEN_TASK
    # Leaves only: family detection matches a keyword in ANY path segment, so a chat GGUF under
    # .../FLUX.1-dev-GGUF/extra/ read as text-to-image.
    return _name_hint_media_task(tuple(_hint_leaf(hint) for hint in name_hints if hint), None)


def _arch_to_task(arch: Optional[str], name_hints: tuple[Optional[str], ...] = ()) -> Optional[str]:
    if any(_is_h3_bundle_gguf_hint(hint) for hint in name_hints):
        return _VIDEO_GEN_TASK
    if arch is None:
        return None
    normalized = arch.lower()
    if normalized == "qwen3" and any(
        _QWEN3_ASR_HINT.search(str(hint)) for hint in name_hints if hint
    ):
        return "automatic-speech-recognition"
    if normalized == "llama" and any(
        _ORPHEUS_GGUF_HINT.search(str(hint)) for hint in name_hints if hint
    ):
        return _SPEECH_TASK
    if normalized in _PLACEHOLDER_DIFFUSION_GGUF_ARCHS:
        return _name_hint_media_task(name_hints, _UNSUPPORTED_DIFFUSION_TASK)
    if is_speech_gguf_architecture(normalized):
        return _SPEECH_TASK
    if normalized in _DIFFUSION_GGUF_ARCHS:
        return (
            "text-to-image" if _gguf_family_buildable(name_hints) else _UNSUPPORTED_DIFFUSION_TASK
        )
    if normalized in _VIDEO_GGUF_ARCHS:
        from core.inference.video_families import detect_video_family

        family = detect_video_family("", override = normalized)
        if family is None:
            for hint in name_hints:
                if hint:
                    family = detect_video_family(hint)
                    if family is not None:
                        break
        if (
            family is not None
            and not getattr(family, "is_moe", False)
            and _video_family_buildable(family)
        ):
            return _VIDEO_GEN_TASK
        return _UNSUPPORTED_DIFFUSION_TASK
    if normalized in _AMBIGUOUS_DIFFUSION_GGUF_ARCHS:
        from core.inference.diffusion_engine_router import family_buildable_here
        from core.inference.diffusion_families import detect_family_for_pick, family_gguf_loadable

        for hint in name_hints:
            if not hint:
                continue
            family = detect_family_for_pick(hint)
            if family is not None:
                loadable = family_gguf_loadable(family) and family_buildable_here(
                    family, model_kind = "gguf"
                )
                return "text-to-image" if loadable else _UNSUPPORTED_DIFFUSION_TASK
        return _UNSUPPORTED_DIFFUSION_TASK
    if normalized in _UNSUPPORTED_DIFFUSION_GGUF_ARCHS:
        return _UNSUPPORTED_DIFFUSION_TASK
    return "text-generation"


def _arch_to_audio_type(
    arch: Optional[str], name_hints: tuple[Optional[str], ...] = ()
) -> Optional[str]:
    """Decoder provenance for a GGUF classified as speech."""
    if arch is None:
        return None
    normalized = arch.strip().lower()
    if normalized == "llama" and any(
        _ORPHEUS_GGUF_HINT.search(str(hint)) for hint in name_hints if hint
    ):
        return "snac"
    if is_speech_gguf_architecture(normalized):
        return "csm"
    return None


def _is_trailing_split_shard(name: str) -> bool:
    try:
        from utils.models.model_config import _GGUF_SPLIT_FILE_RE
    except ImportError:
        return False
    match = _GGUF_SPLIT_FILE_RE.match(name)
    return match is not None and match.group("index") != "00001"


def _task_classify_sort_key(root: Path, path: Path) -> tuple[str, str]:
    try:
        relative = path.relative_to(root).as_posix()
    except ValueError:
        relative = path.name
    return (relative.lower(), relative)


def _gguf_folder_task(
    root: Path,
    id_hints: tuple[Optional[str], ...],
    deadline: Optional[float] = None,
) -> Optional[str]:
    if deadline is None:
        deadline = time.monotonic() + _TASK_CLASSIFY_WALK_SECONDS
    fallback: Optional[str] = None
    try:
        scored: list[tuple[tuple[str, str], Path]] = []
        # Recorded as the tail is dropped, never inferred from scored afterwards: trimming cuts back to the
        # cap, so an overflowing folder would be indistinguishable from one that fit.
        overflowed = False
        for path in _iter_gguf_paths(root, deadline):
            name = path.name
            if _is_mmproj_filename(name) or _is_trailing_split_shard(name):
                continue
            scored.append((_task_classify_sort_key(root, path), path))
            if len(scored) > _MAX_TASK_CLASSIFY_GGUFS * 2:
                scored.sort(key = lambda item: item[0])
                del scored[_MAX_TASK_CLASSIFY_GGUFS:]
                overflowed = True
        scored.sort(key = lambda item: item[0])
        paths = [path for _, path in scored[:_MAX_TASK_CLASSIFY_GGUFS]]
        # The walk gives up at its own deadline and the cap drops the tail, so either can leave a sibling unseen.
        complete = (
            not overflowed
            and len(scored) <= _MAX_TASK_CLASSIFY_GGUFS
            and time.monotonic() < deadline
        )
    except Exception:
        return None
    unsupported: Optional[str] = None
    speech: Optional[str] = None
    read_deadline = time.monotonic() + _TASK_CLASSIFY_READ_SECONDS
    for index, path in enumerate(paths):
        if index and time.monotonic() >= read_deadline:
            complete = False
            break
        hints = id_hints + (path.name,)
        try:
            if file_contents_available_locally(path):
                task = _arch_to_task(_gguf_architecture(str(path)), name_hints = hints)
            else:
                # Its header stays unread, so a name that says nothing leaves the candidate unclassified rather than
                # voting text-generation for the whole folder.
                task = _unhydrated_gguf_task(hints)
        except Exception:
            # Unread, so unranked: this file might have been the runnable sibling.
            complete = False
            continue
        if task is None:
            # A truncated header gives no architecture, and _arch_to_task answers None.
            complete = False
            continue
        if task in _LOADABLE_MEDIA_GGUF_TASKS:
            return task
        # Speech is last resort: nothing here runs a llama-csm GGUF, so answering speech while a sibling is
        # loadable hides that sibling.
        if task == _SPEECH_TASK:
            if speech is None:
                speech = task
        elif task == _UNSUPPORTED_DIFFUSION_TASK:
            unsupported = unsupported or task
        elif task is not None and fallback is None:
            fallback = task
    # Speech only on a whole folder: it is the one answer that HIDES a row rather than filing it.
    return unsupported or fallback or (speech if complete else None)


def _repo_gguf_task(repo_info, selected: Optional[Path] = None) -> Optional[str]:
    repo_id = getattr(repo_info, "repo_id", None)
    try:
        return _gguf_folder_task(selected or Path(repo_info.repo_path), (repo_id,))
    except Exception:
        return None


def _gguf_path_audio_type(
    path: str | Path, id_hints: tuple[Optional[str], ...] = ()
) -> Optional[str]:
    model_path = Path(path)
    try:
        paths = (
            [model_path]
            if model_path.suffix.lower() == ".gguf" and model_path.is_file()
            else _iter_gguf_paths(model_path)
        )
        for gguf_path in paths:
            audio_type = _arch_to_audio_type(
                _gguf_architecture(str(gguf_path)),
                name_hints = id_hints + (gguf_path.name,),
            )
            if audio_type is not None:
                return audio_type
    except Exception:
        return None
    return None


def _repo_gguf_audio_type(repo_info, selected: Optional[Path] = None) -> Optional[str]:
    repo_id = getattr(repo_info, "repo_id", None)
    try:
        return _gguf_path_audio_type(selected or Path(repo_info.repo_path), (repo_id,))
    except Exception:
        return None


def _gguf_path_task(path: str | Path, id_hints: tuple[Optional[str], ...] = ()) -> Optional[str]:
    model_path = Path(path)
    try:
        if model_path.suffix.lower() == ".gguf" and model_path.is_file():
            hints = id_hints + (model_path.name,)
            if not file_contents_available_locally(model_path):
                return _unhydrated_gguf_task(hints)
            return _arch_to_task(
                _gguf_architecture(str(model_path)),
                name_hints = hints,
            )
        return _gguf_folder_task(model_path, id_hints)
    except Exception:
        return None


def _hf_cache_snapshot_repo_id(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    parts = str(path).replace("\\", "/").rstrip("/").split("/")
    if len(parts) >= 3 and parts[-2] == "snapshots" and parts[-3].startswith("models--"):
        from core.inference.model_ids import hf_cache_repo_id
        return hf_cache_repo_id(path)
    return None


def _local_family_needles(model) -> tuple[str, ...]:
    needles = [model.model_id, model.display_name, Path(model.id).name]
    try:
        needles.append(
            _hf_cache_snapshot_repo_id(model.path) or _hf_cache_snapshot_repo_id(model.id)
        )
    except Exception:
        pass
    try:
        from core.inference.diffusion import resolve_local_single_file
        single = resolve_local_single_file(model.path)
        if single:
            needles.append(single)
    except Exception:
        pass
    return tuple(needle for needle in needles if needle)


def _local_model_can_chat(model) -> Optional[bool]:
    model = _local_probe_model(model)
    return _local_path_can_chat(model.path, getattr(model, "base_model", None))


def _local_model_task(model) -> Optional[str]:
    model = _local_probe_model(model)
    path = model.path
    id_hints = (model.model_id, model.display_name, model.id)
    if model.model_format == "gguf" or Path(path).suffix.lower() == ".gguf":
        return _gguf_path_task(path, id_hints)
    try:
        from core.inference.native_audio import native_audio_type_from_local_path
        if native_audio_type_from_local_path(path):
            return "text-to-speech"
    except Exception:
        pass
    if not _local_is_diffusers(model):
        return None
    try:
        from core.inference.video import _is_trusted_video_repo
        from core.inference.video_families import detect_video_family
        for needle in _local_family_needles(model):
            family = detect_video_family(needle)
            if family is not None and _is_trusted_video_repo(path):
                return _VIDEO_GEN_TASK if _video_family_buildable(family) else None
    except Exception:
        pass
    try:
        from core.inference.diffusion_engine_router import family_buildable_here
        from core.inference.diffusion_families import detect_family, detect_family_by_pipeline_index

        families = (
            detect_family_by_pipeline_index(path),
            *(detect_family(needle) for needle in _local_family_needles(model)),
        )
        for family in families:
            if family is not None:
                return (
                    "text-to-image"
                    if family_buildable_here(family, model_kind = "pipeline")
                    else None
                )
        return None
    except Exception:
        return "text-to-image"


def _local_model_audio_type(model) -> Optional[str]:
    model = _local_probe_model(model)
    path = model.path
    if model.model_format == "gguf" or Path(path).suffix.lower() == ".gguf":
        return _gguf_path_audio_type(path, (model.model_id, model.display_name, model.id))
    try:
        from core.inference.native_audio import native_audio_type_from_local_path
        native_audio_type = native_audio_type_from_local_path(path)
        if native_audio_type is not None:
            return native_audio_type
    except Exception:
        pass
    try:
        from utils.audio_tokens import detect_local_tts_audio_type
        return detect_local_tts_audio_type(path)
    except Exception:
        return None


def _local_model_classification_for_task(
    model, task: Optional[str]
) -> tuple[Optional[str], Optional[str]]:
    """Add decoder provenance to an already classified local-row task."""
    audio_type = _local_model_audio_type(model) if task is None or task == _SPEECH_TASK else None
    if task is None and audio_type is not None:
        from utils.audio_tokens import is_output_audio_type
        if is_output_audio_type(audio_type):
            task = _SPEECH_TASK
    return task, audio_type


def _local_model_classification(model) -> tuple[Optional[str], Optional[str]]:
    """Return picker task and decoder provenance from one local-row probe."""
    return _local_model_classification_for_task(model, _local_model_task(model))


def _local_is_diffusers(model) -> bool:
    model = _local_probe_model(model)
    try:
        path = Path(model.path)
        if path.is_dir() and _is_diffusers_pipeline_dir(path):
            return True
    except Exception:
        pass
    try:
        from core.inference.diffusion_families import detect_family
        if any(detect_family(needle) is not None for needle in _local_family_needles(model)):
            return True
    except Exception:
        pass
    try:
        from core.inference.video_families import detect_video_family
        return any(
            detect_video_family(needle) is not None for needle in _local_family_needles(model)
        )
    except Exception:
        return False


def _repo_has_pipeline_index(repo_info, selected: Optional[Path] = None) -> bool:
    if selected is not None:
        return _is_diffusers_pipeline_dir(selected)
    from hub.utils import inventory_scan
    return inventory_scan.repo_has_pipeline_index(repo_info)


def _repo_is_diffusers(repo_info, selected: Optional[Path] = None) -> bool:
    if _repo_has_pipeline_index(repo_info, selected):
        return True
    repo_id = getattr(repo_info, "repo_id", "") or ""
    try:
        from core.inference.diffusion_families import detect_family
        if detect_family(repo_id) is not None:
            return True
    except Exception:
        pass
    # _cached_repo_task returns None for an unbuildable video repo, so a single-file video checkpoint
    # with no pipeline index carried no task and no diffusers flag, and an inconclusive config leaves
    # can_chat set: the video weights reach the text loader.
    try:
        from core.inference.video_families import detect_video_family
        return detect_video_family(repo_id) is not None
    except Exception:
        return False


def _is_sd_cpp_companion_repo(repo_id: str) -> bool:
    try:
        from core.inference.diffusion_families import sd_cpp_companion_only_repo_ids
        return (repo_id or "").strip().lower() in sd_cpp_companion_only_repo_ids()
    except Exception:
        return False


def _cached_repo_task(repo_info, selected: Optional[Path] = None) -> Optional[str]:
    repo_id = getattr(repo_info, "repo_id", "") or ""
    try:
        from core.inference.video import _is_trusted_video_repo
        from core.inference.video_families import detect_video_family

        family = detect_video_family(repo_id)
        if family is not None:
            if not _is_trusted_video_repo(repo_id) or not _video_family_buildable(family):
                return None
            return _VIDEO_GEN_TASK
    except Exception:
        pass
    if not _repo_is_diffusers(repo_info, selected):
        return None
    try:
        from core.inference.diffusion import _is_trusted_diffusion_repo
        from core.inference.diffusion_families import detect_family, family_pipeline_available

        if _is_sd_cpp_companion_repo(repo_id):
            return None
        family = detect_family(repo_id)
        if not _is_trusted_diffusion_repo(repo_id) or family is None:
            return None
        if not family_pipeline_available(family):
            return None
        return "text-to-image"
    except Exception:
        return "text-to-image"
