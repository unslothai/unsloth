# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Draft model arguments and device selection."""

import os
import re

from typing import (
    Collection,
    Iterable,
    List,
    Mapping,
    Optional,
)
from core.inference.llama_server_args import (
    _flag_name,
    _GPU_LAYER_FLAGS,
    parse_split_mode_override,
    split_policy_starves_devices,
)

from .launch_flags import (
    _accumulated_spec_types,
    _args_place_tensors_on_cpu,
    _env_fixes_gpu_layers,
    _env_places_tensors_on_cpu,
)
from .memory_policy import _extra_args_set_any_flag


def _extra_args_requests_separate_draft(
    extra_args: Optional[Iterable[str]], env: Optional[Mapping[str, str]] = None
) -> bool:
    """True if a non-MTP model draft mode (draft-simple/draft-eagle3) is selected, which
    loads a separate draft model the budget must reserve (draft-mtp ->
    _extra_args_requests_mtp; ngram-* load no model). Accumulating for the same reason as
    _extra_args_requests_mtp: a later --spec-default or --spec-type cannot clear an
    inherited draft-simple, so reading only the last under-reserves the model it loads."""
    return bool(
        _accumulated_spec_types(extra_args, env)
        & {"draft-simple", "draft-eagle3", "draft-dspark", "draft-dflash"}
    )


def _extra_args_spec_draft_n_max(extra_args: Optional[Iterable[str]]) -> Optional[int]:
    """Draft depth from extras (``--spec-draft-n-max`` or legacy ``--draft-max``), else None."""
    if not extra_args:
        return None
    args = [str(a) for a in extra_args]
    found: Optional[int] = None
    for i, raw in enumerate(args):
        flag = _flag_name(raw)
        _, eq, inline = raw.partition("=")
        if flag not in ("--spec-draft-n-max", "--draft-max"):
            continue
        value = inline if eq else (args[i + 1] if i + 1 < len(args) else "")
        try:
            found = int(value)
        except (TypeError, ValueError):
            continue
    return found


def _extra_args_mtp_draft_path(
    extra_args: Optional[Iterable[str]], env: Optional[Mapping[str, str]] = None
) -> Optional[str]:
    """Separate drafter path from extras (local --model-draft/-md or HF
    --spec-draft-hf/-hfd/...), else the LLAMA_ARG_SPEC_DRAFT_MODEL/_HF_REPO env,
    else None. An HF repo isn't a local file, so the budget can't size it (falls
    back to the flat reserve), but recognizing it avoids sizing the wrong one."""
    return _extra_args_mtp_draft_source(extra_args, env)[0]


_HF_DRAFT_FLAGS = frozenset({"--spec-draft-hf", "-hfd", "-hfrd", "--hf-repo-draft"})
_LOCAL_DRAFT_FLAGS = frozenset({"--model-draft", "--spec-draft-model", "-md"})


def _extra_args_mtp_draft_source(
    extra_args: Optional[Iterable[str]], env: Optional[Mapping[str, str]] = None
) -> tuple[Optional[str], bool]:
    """The drafter the launch resolves to, and whether it is a repo id.

    The two travel together on purpose. Draft flags are last-wins, so a remote flag
    followed by a local one leaves a local path as the winner; a caller that asks
    only "does any remote flag appear" would then price a filesystem path as a
    Hugging Face repository and charge a reserve for something llama-server will
    not load. Only the winning flag decides which of the two the value is.
    """
    args = [str(a) for a in extra_args] if extra_args else []
    found: Optional[str] = None
    found_remote = False
    for i, raw in enumerate(args):
        flag = _flag_name(raw)
        _, eq, inline = raw.partition("=")
        if flag not in _HF_DRAFT_FLAGS and flag not in _LOCAL_DRAFT_FLAGS:
            continue
        value = inline if eq else (args[i + 1] if i + 1 < len(args) else "")
        if value and not value.startswith("-"):
            found = value
            found_remote = flag in _HF_DRAFT_FLAGS
    if found is not None:
        return found, found_remote
    e = os.environ if env is None else env
    # The value lookup's own precedence; with no flag to read, the env spelling says
    # local or remote. The last two are pre-b8955 (2025-08-30 to 2026-04-28).
    for var, remote in (
        ("LLAMA_ARG_SPEC_DRAFT_MODEL", False),
        ("LLAMA_ARG_SPEC_DRAFT_HF_REPO", True),
        ("LLAMA_ARG_MODEL_DRAFT", False),
        ("LLAMA_ARG_HFD_REPO", True),
    ):
        value = e.get(var)
        if value:
            return value, remote
    return None, False


def _normalized_load_mode(value: Optional[str]) -> Optional[str]:
    """Canonical --load-mode, with llama.cpp's own default reading as unset.

    "auto" IS the default, so a load asking for it and a load asking for nothing
    launch the same command; the dedupe has to see them as equal or picking Auto
    would reload a server already running it.
    """
    if value is None:
        return None
    mode = str(value).strip().lower()
    return None if mode in {"", "auto"} else mode


_BREAKDOWN_DEVICE_ROW_RE = re.compile(
    r"\|\s*-\s*(?P<name>[^|(]+?)\s*\([^|]*\|\s*\d+\s*=\s*\d+\s*\+\s*\(\s*\d+\s*=\s*(?P<model>\d+)\s*\+"
)
_BREAKDOWN_HOST_ROW_RE = re.compile(
    r"\|\s*-\s*(?P<name>[^|]+?)\s*\|\s*\d+\s*=\s*(?P<model>\d+)\s*\+"
)


def _parse_metal_memory_breakdown(output: str) -> Optional[tuple[int, int, int]]:
    """Parse the first llama.cpp table as ``(metal_model_mib, host_model_mib, rows)``.

    Require one Metal device row and sum the model column from every host row.
    """
    rows: list[str] = []
    for line in output.splitlines():
        if "common_memory_breakdown_print" not in line:
            if rows:
                break
            continue
        if "memory breakdown" in line:
            if rows:
                break
            continue
        rows.append(line)
    metal: list[int] = []
    host = 0
    host_rows = 0
    for row in rows:
        device = _BREAKDOWN_DEVICE_ROW_RE.search(row)
        if device:
            if not device.group("name").strip().startswith("MTL"):
                return None
            metal.append(int(device.group("model")))
            continue
        hosted = _BREAKDOWN_HOST_ROW_RE.search(row)
        if hosted:
            host += int(hosted.group("model"))
            host_rows += 1
            continue
        return None
    if len(metal) != 1 or host_rows == 0:
        return None
    return metal[0], host, len(rows)


# The KV cache dtypes llama.cpp can map, shared by the emission guard and the
# budget. Module level because both the draft-cache block and the MTP reserve
# need it, and the main cache path's copy is a local inside load_model.
_VALID_KV_CACHE_TYPES: frozenset[str] = frozenset(
    {"f16", "bf16", "q8_0", "q4_0", "q4_1", "q5_0", "q5_1", "iq4_nl", "f32"}
)


def _extra_args_draft_cache_types(
    extra_args: Optional[Iterable[str]], env: Optional[Mapping[str, str]] = None
) -> tuple[Optional[str], Optional[str]]:
    """Draft KV cache types (k_type, v_type), each from extras else the
    LLAMA_ARG_SPEC_DRAFT_CACHE_TYPE_K/_V env, else None (f16). K and V are
    independent: a one-sided override must not apply to both."""
    args = [str(a) for a in extra_args] if extra_args else []
    k_flags = {"--cache-type-k-draft", "--spec-draft-type-k", "-ctkd"}
    v_flags = {"--cache-type-v-draft", "--spec-draft-type-v", "-ctvd"}
    k_type: Optional[str] = None
    v_type: Optional[str] = None
    for i, raw in enumerate(args):
        flag = _flag_name(raw)
        _, eq, inline = raw.partition("=")
        if flag not in k_flags and flag not in v_flags:
            continue
        value = inline if eq else (args[i + 1] if i + 1 < len(args) else "")
        if not value or value.startswith("-"):
            continue
        if flag in k_flags:
            k_type = value
        else:
            v_type = value
    e = os.environ if env is None else env
    if k_type is None:
        k_type = e.get("LLAMA_ARG_SPEC_DRAFT_CACHE_TYPE_K") or None
    if v_type is None:
        v_type = e.get("LLAMA_ARG_SPEC_DRAFT_CACHE_TYPE_V") or None
    return k_type, v_type


_DRAFT_GPU_LAYER_FLAGS = frozenset(
    {"--spec-draft-ngl", "-ngld", "--gpu-layers-draft", "--n-gpu-layers-draft"}
)


def _extra_args_draft_gpu_layers(
    extra_args: Optional[Iterable[str]], env: Optional[Mapping[str, str]] = None
) -> Optional[str]:
    """The effective draft layer count for the SEPARATE drafter, as written.

    Last-wins over argv, falling back to the ``LLAMA_ARG_N_GPU_LAYERS_DRAFT`` twin
    the child reads before argv. None when neither is set."""
    args = [str(a) for a in extra_args] if extra_args else []
    last_ngl: Optional[str] = None
    for i, raw in enumerate(args):
        flag = _flag_name(raw)
        _, eq, inline = raw.partition("=")
        value = inline if eq else (args[i + 1] if i + 1 < len(args) else "")
        if flag in _DRAFT_GPU_LAYER_FLAGS:
            last_ngl = value
    if last_ngl is None:
        last_ngl = (os.environ if env is None else env).get("LLAMA_ARG_N_GPU_LAYERS_DRAFT")
    return last_ngl


def _draft_is_split_across_host(
    extra_args: Optional[Iterable[str]],
    env: Optional[Mapping[str, str]] = None,
    *,
    n_draft_layers: Optional[int] = None,
) -> bool:
    """True when a fixed POSITIVE draft layer count splits the drafter between the
    GPU and host RAM.

    ``_extra_args_draft_offloaded_to_cpu`` is the whole-drafter question and only
    answers ``-ngld 0``. Anything between that and the drafter's own block count
    leaves the layers past the count on the host -- llama.cpp derives
    ``i_gpu_start`` from the count and hands every layer below it the CPU buffer
    list (llama-model.cpp), and each layer's KV follows its device
    (llama-kv-cache.cpp). llama.cpp's default is -1, "all", so only a non-negative
    count is a placement at all, and only a count ABOVE ``n_draft_layers`` places
    the drafter whole: ``i_gpu_start = max(n_layer_all + 1 - n_gpu_layers, 0)``
    (llama-model.cpp), so the count has to clear the block count by one to reach
    zero, and ``-ngld`` equal to the block count still leaves one block -- and its
    KV -- on the host. That is the same threshold the main model already uses
    (``requested <= n_layers`` in ``_partially_offloads_layers``, and
    ``0 <= gpu_layers <= n_layers`` in the pipelining check). An unreadable block
    count (None) says nothing either way, so a positive count stands as a split
    rather than being waved through.
    """
    raw = _extra_args_draft_gpu_layers(extra_args, env)
    if raw is None:
        return False
    try:
        count = int(str(raw).strip())
    except (TypeError, ValueError):
        # std::stoi would throw and the child never starts: nothing to misprice.
        return False
    if count <= 0:
        return False
    return not (n_draft_layers and count > n_draft_layers)


_SIDECAR_ADAPTER_FLAGS = (
    "--lora",
    "--lora-scaled",
    "--control-vector",
    "--control-vector-scaled",
)


def _sidecar_adapter_paths(extra_args: Optional[Iterable[str]]) -> list[str]:
    """Every adapter file the child loads and holds RESIDENT, in argv order.

    None of these have an env twin -- common/arg.cpp declares all four without
    ``set_env`` -- so the tokens are the whole surface. Each operand is a
    comma-separated row (``parse_csv_row``), and the ``-scaled`` spellings split
    every entry on ':' into exactly FNAME and SCALE and throw otherwise, so a
    malformed entry is a child that never starts and has no placement to misprice;
    it is skipped rather than priced. ACCUMULATED, not last-wins: every handler
    ``push_back``s onto ``lora_adapters`` / ``control_vectors``, so a second
    ``--lora`` adds to the first rather than replacing it.

    The ``-scaled`` pair also has a LEGACY spelling: builds up to the turn of the
    year declared it ``{"--lora-scaled"}, "FNAME", "SCALE"`` and took the scale as a
    SECOND token, and Unsloth runs whatever llama-server it is pointed at
    (``LLAMA_SERVER_PATH``, or one found on PATH), which is why the pass-through
    validator still admits that shape (``_OPTIONAL_SECOND_VALUE_FLAGS``). Read
    colon-first, since that is today's syntax; only when the colon parse priced
    NOTHING and the token after the operand is a bare number is this the two-token
    form, and then the operand is the whole filename -- which is also the one
    reading that survives a Windows drive letter. On a build that wants the colon
    the same tokens are upstream's own throw, so pricing the file there costs at
    most a fit that never launched.
    """
    args = [str(a) for a in extra_args] if extra_args else []
    paths: list[str] = []
    for i, raw in enumerate(args):
        flag = _flag_name(raw)
        if flag not in _SIDECAR_ADAPTER_FLAGS:
            continue
        _, eq, inline = raw.partition("=")
        operand = inline if eq else (args[i + 1] if i + 1 < len(args) else "")
        priced = 0
        for entry in (piece for piece in operand.split(",") if piece):
            if not flag.endswith("-scaled"):
                paths.append(entry)
                priced += 1
                continue
            # Upstream demands exactly 2 parts, so a bare or drive-lettered path throws.
            head, colon, tail = entry.partition(":")
            if not (colon and head) or ":" in tail:
                continue
            try:
                float(tail)
            except ValueError:
                continue
            paths.append(head)
            priced += 1
        if priced or not operand or eq or not flag.endswith("-scaled"):
            continue
        # Legacy "FNAME SCALE": the scale is the token AFTER the operand.
        scale = args[i + 2] if i + 2 < len(args) else ""
        try:
            float(scale.strip())
        except ValueError:
            continue
        paths.append(operand)
    return paths


def _sidecar_adapter_bytes(extra_args: Optional[Iterable[str]]) -> Optional[int]:
    """Resident bytes the pass-through adapters add, or None when one cannot be sized.

    LoRA tensors are allocated on the BASE tensor's buffer type
    (``ggml_backend_buffer_get_type(model_tensor->buffer)``, llama-adapter.cpp) and
    a control vector on ``model.select_buft(il)``, so they follow the layer
    placement this fit just priced rather than living anywhere it can ignore.
    Neither path mmaps -- the file is read into a staging buffer and pushed with
    ``ggml_backend_tensor_set`` -- so these bytes are resident whatever
    ``--load-mode`` says, and they sit ON TOP of a placement llama.cpp's own fitter
    never accounted for (common/fit.cpp knows nothing about adapters).

    File size is the ceiling on the allocation (the loader skips ``_norm.weight``
    pairs and keeps every other tensor at its stored type), and overstating a term
    only ever costs a fit, so the stat is charged whole. An unreadable file is the
    "engaged but unsized" case the fit abstains on.
    """
    total = 0
    for path in _sidecar_adapter_paths(extra_args):
        try:
            total += max(0, os.stat(path).st_size)
        except OSError:
            return None
    return total


def _extra_args_draft_offloaded_to_cpu(
    extra_args: Optional[Iterable[str]], env: Optional[Mapping[str, str]] = None
) -> bool:
    """True if the SEPARATE draft model is on CPU (so the budget must not charge
    its weights+KV): --spec-draft-ngl 0, or --spec-draft-device naming only
    cpu/none, else the LLAMA_ARG_N_GPU_LAYERS_DRAFT env the child honors (the
    device flag has no env). An embedded MTP head follows the main -ngl, so these
    draft-only flags don't move it. Last-wins, so only each flag's final value counts.

    A count BETWEEN 0 and the drafter's block count is a split, not an offload; see
    ``_draft_is_split_across_host``."""
    last_ngl = _extra_args_draft_gpu_layers(extra_args, env)
    if last_ngl is not None:
        try:
            if int(last_ngl) == 0:
                return True
        except (TypeError, ValueError):
            pass
    last_dev = _extra_args_draft_device(extra_args)
    if last_dev is not None:
        devs = [d.strip().lower() for d in last_dev.split(",") if d.strip()]
        if devs and all(d in ("cpu", "none") for d in devs):
            return True
    return False


def _extra_args_device(
    extra_args: Optional[Iterable[str]], flags: Collection[str]
) -> Optional[str]:
    """Return the last value for any device flag in ``flags``."""
    args = [str(a) for a in extra_args] if extra_args else []
    value: Optional[str] = None
    for i, raw in enumerate(args):
        flag = _flag_name(raw)
        _, eq, inline = raw.partition("=")
        if flag in flags:
            value = inline if eq else (args[i + 1] if i + 1 < len(args) else "")
    return value


def _extra_args_main_device(extra_args: Optional[Iterable[str]]) -> Optional[str]:
    """Return the last explicit main-device value, if any."""
    return _extra_args_device(extra_args, {"--device", "-dev"})


def _without_subsequence(tokens: List[str], run: List[str]) -> List[str]:
    """``tokens`` with the first contiguous occurrence of ``run`` removed.

    Only the exact flags Unsloth appended go, so a user's own identical flag
    elsewhere in the command survives. Unchanged when ``run`` is not present.
    """
    if not run:
        return list(tokens)
    for i in range(len(tokens) - len(run) + 1):
        if tokens[i : i + len(run)] == run:
            return [*tokens[:i], *tokens[i + len(run) :]]
    return list(tokens)


def _count_subsequence(tokens: List[str], run: List[str]) -> int:
    """How many non-overlapping contiguous occurrences of ``run`` are in ``tokens``."""
    if not run:
        return 0
    total = 0
    i = 0
    while i <= len(tokens) - len(run):
        if tokens[i : i + len(run)] == run:
            total += 1
            i += len(run)
        else:
            i += 1
    return total


def _subsequence_index(tokens: List[str], run: List[str], hint: int) -> int:
    """Where ``run`` sits in ``tokens``, given it was appended at ``hint``.

    A stored index goes stale the moment anything narrows the argv ahead of it,
    and the arch gate does exactly that (dropping ``--split-mode`` /
    ``--tensor-split``), so the block can only ever move LEFT. Search back from
    the hint, then forward, so a future narrowing site cannot silently shift the
    slice and take neighbouring tokens with it.
    """
    span = len(tokens) - len(run)
    if not run or span < 0:
        return max(0, min(hint, len(tokens)))
    for i in range(min(hint, span), -1, -1):
        if tokens[i : i + len(run)] == run:
            return i
    for i in range(min(hint, span) + 1, span + 1):
        if tokens[i : i + len(run)] == run:
            return i
    return max(0, min(hint, len(tokens)))


_CPU_DEVICE_VALUES = frozenset({"cpu", "none"})


def _pass_through_overrides_placement(
    extra_args: Optional[Iterable[str]],
    source_env: Mapping[str, str],
    credited: int,
    *,
    is_vulkan_backend: bool = False,
    gpu_indices: Optional[Iterable[int]] = None,
) -> bool:
    """Whether pass-through args or env run a placement other than the one a fit priced.

    ``extra_args`` / ``source_env`` are what the child really gets (a ``gpu_ids`` pin
    strips its device flags and env twins first); ``credited`` is how many devices the
    fit charged. See ``LlamaCppBackend._fit_derived_load_mode`` for each term.
    """
    # A --device naming FEWER devices than this fit charges leaves the credit
    # paying for cards the child never opens. llama.cpp REPLACES the list on every
    # occurrence (parse_device_list) and reads LLAMA_ARG_DEVICE before argv, so the
    # last of the two wins. Counted, not name-matched: CUDA/ROCm ordinals are
    # assigned after a visibility mask this launch has not written yet.
    _dev_value = _extra_args_main_device(extra_args)
    if _dev_value is None:
        _dev_value = source_env.get("LLAMA_ARG_DEVICE")
    _dev_names = [d.strip() for d in str(_dev_value or "").split(",") if d.strip()]
    _credited = credited
    gpu_indices = list(gpu_indices) if gpu_indices is not None else None
    # DISTINCT names: llama.cpp appends every entry without deduplicating and
    # keeps the list verbatim, so "CUDA0,CUDA0" is ONE physical pool listed twice,
    # and counting it as two leaves the credit paying for a card that is not there.
    _device_narrows = bool(_dev_names) and len(set(_dev_names)) < _credited
    # Vulkan is the one backend where names ARE comparable (_vulkan_device_pin ->
    # "Vulkan<i>"), so hold a surviving --device to an exact match. That catches
    # what a count cannot: a same-sized list naming DIFFERENT cards, and a longer
    # one adding devices this footprint never charged for.
    _vulkan_pin_replaced = bool(
        is_vulkan_backend
        and gpu_indices is not None
        and _dev_names
        and {name.strip().lower() for name in _dev_names} != {f"vulkan{idx}" for idx in gpu_indices}
    )
    return bool(
        _extra_args_set_any_flag(extra_args, _GPU_LAYER_FLAGS)
        # And its env twin, the ONLY layer count the fitting path leaves: it
        # emits no -ngl, only Manual mode clears LLAMA_ARG_N_GPU_LAYERS, and the
        # fitter will not lower a count it did not set, so an inherited one stands
        # and loads a prefix out of host RAM.
        or _env_fixes_gpu_layers(source_env)
        # With the env twin: only an explicit gpu_ids pin clears LLAMA_ARG_DEVICE,
        # so an inherited "none" puts the whole load in host RAM.
        or _device_selection_is_cpu(extra_args, source_env)
        or _device_narrows
        or _vulkan_pin_replaced
        # Pass-through and last-wins: --split-mode none, or a --tensor-split that
        # zeroes a device, leaves the pooled credit paying for cards held empty.
        or split_policy_starves_devices(extra_args, _credited, source_env)
        or _args_place_tensors_on_cpu(extra_args)
        or _env_places_tensors_on_cpu(source_env)
    )


def _device_selection_is_cpu(
    extra_args: Optional[Iterable[str]], env: Optional[Mapping[str, str]] = None
) -> bool:
    """True when the effective ``--device`` selection names no GPU.

    llama.cpp then runs the model on the CPU whatever the layer count says, so
    the weights are in pageable host RAM. argv wins over the env twin, and an
    unreadable value answers True like every other unknown in the gate.
    """
    value = _extra_args_main_device(extra_args)
    if value is None and env:
        value = env.get("LLAMA_ARG_DEVICE")
    if value is None:
        return False
    devices = [d.strip().lower() for d in str(value).split(",") if d.strip()]
    return not devices or all(d in _CPU_DEVICE_VALUES for d in devices)


def _split_mode_confines_to_one_device(
    args: Optional[Iterable[str]], env: Optional[Mapping[str, str]] = None
) -> bool:
    """Whether the effective split mode confines all weights to one device."""
    try:
        mode = parse_split_mode_override(args)
    except ValueError:
        return False
    if mode is None and env:
        mode = env.get("LLAMA_ARG_SPLIT_MODE")
    return str(mode or "").strip().lower() == "none"


def _extra_args_draft_device(extra_args: Optional[Iterable[str]]) -> Optional[str]:
    """Return the last explicit draft-device value, if any."""
    return _extra_args_device(extra_args, {"--spec-draft-device", "-devd", "--device-draft"})


_GPU_DEVICE_TOKEN_RE = re.compile(r"(CUDA|ROCm)(\d+)$", re.IGNORECASE)
# One group per companion: llama.cpp is last-wins within a group, so only its last flag counts.
_COMPANION_DEVICE_FLAG_GROUPS = (
    frozenset({"--mmproj-device", "-mmdev"}),
    frozenset({"--spec-draft-device", "-devd", "--device-draft"}),
)


def _widen_pin_ids_for_companion_devices(
    cmd: List[str],
    pin_ids: list[int],
    inherited_ids: Optional[list[int]],
    *,
    allowed_ids: Optional[Iterable[int]] = None,
) -> tuple[list[int], str]:
    """Fit a pinned GPU mask to the companion devices the argv names (#11810).

    Companion ``CUDA<n>`` / ``ROCm<n>`` tokens use the unpinned numbering (position in
    ``inherited_ids``, else the physical index). Cards they name are appended after the
    pinned ones, the winning flag is renumbered to the child's mask, and a main
    ``--device`` is added so ``-ngl -1`` does not spread over the extra cards. Tokens
    the parent cannot see are left alone. ``allowed_ids`` (explicit gpu_ids) limits
    widening to those cards. Returns the mask and a log note, empty when unchanged.
    """
    if not pin_ids:
        return list(pin_ids), ""
    main_ids = [int(i) for i in pin_ids]
    allowed = None if allowed_ids is None else {int(i) for i in allowed_ids}
    # A main --device past the pinned positions would resolve to a widened companion card.
    for token in str(_extra_args_main_device(cmd) or "").split(","):
        match = _GPU_DEVICE_TOKEN_RE.match(token.strip())
        if match and int(match.group(2)) >= len(main_ids):
            return list(main_ids), ""
    # (value index, prefix before "=" or None, [(token, physical id or None)])
    sites: list[tuple[int, Optional[str], list[tuple[str, Optional[int]]]]] = []
    prefix_word = None
    last_of_group: dict[int, int] = {}
    for i, raw in enumerate(cmd):
        for group, flags in enumerate(_COMPANION_DEVICE_FLAG_GROUPS):
            if _flag_name(str(raw)) in flags:
                last_of_group[group] = i
    for i in sorted(last_of_group.values()):
        raw = str(cmd[i])
        head, eq, inline = raw.partition("=")
        if eq:
            at, value, lead = i, inline, head + "="
        elif i + 1 < len(cmd):
            at, value, lead = i + 1, str(cmd[i + 1]), None
        else:
            continue
        tokens: list[tuple[str, Optional[int]]] = []
        for token in value.split(","):
            match = _GPU_DEVICE_TOKEN_RE.match(token.strip())
            physical = None
            if match:
                prefix_word = prefix_word or match.group(1)
                n = int(match.group(2))
                if inherited_ids is None:
                    physical = n
                elif n < len(inherited_ids):
                    physical = int(inherited_ids[n])
                if allowed is not None and physical not in main_ids and physical not in allowed:
                    physical = None
            tokens.append((token, physical))
        if any(physical is not None for _, physical in tokens):
            sites.append((at, lead, tokens))
    if not sites:
        return list(main_ids), ""
    widened = list(main_ids)
    for _, _, tokens in sites:
        for _, physical in tokens:
            if physical is not None and physical not in widened:
                widened.append(physical)
    changed = False
    for at, lead, tokens in sites:
        rewritten = []
        for token, physical in tokens:
            if physical is None:
                rewritten.append(token)
                continue
            match = _GPU_DEVICE_TOKEN_RE.match(token.strip())
            rewritten.append(f"{match.group(1)}{widened.index(physical)}")
        value = ",".join(rewritten)
        new = f"{lead}{value}" if lead is not None else value
        if new != cmd[at]:
            cmd[at] = new
            changed = True
    if len(widened) > len(main_ids) and _extra_args_main_device(cmd) is None:
        # Pinned cards lead the mask, so the main model's devices are the first positions.
        cmd.extend(["--device", ",".join(f"{prefix_word}{k}" for k in range(len(main_ids)))])
        changed = True
    if not changed and widened == main_ids:
        return list(main_ids), ""
    return widened, f"{main_ids} -> {widened}"


def _extra_args_draft_device_pin(extra_args: Optional[Iterable[str]]) -> Optional[str]:
    """Return a GPU draft-device override; cpu/none do not conflict with a pin."""
    last_dev = _extra_args_draft_device(extra_args)
    if last_dev is None:
        return None
    devs = [d.strip() for d in last_dev.split(",") if d.strip()]
    if not devs or all(d.lower() in ("cpu", "none") for d in devs):
        return None
    return last_dev
