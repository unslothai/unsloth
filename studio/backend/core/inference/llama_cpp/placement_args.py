# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Paravirtual Metal placement, context checkpoints and projector offload."""

import functools
import os
import subprocess
import sys

from typing import (
    Iterable,
    List,
    Literal,
    Mapping,
    NamedTuple,
    Optional,
    Sequence,
)
from core.inference.llama_server_args import (
    _flag_name,
    effective_ctx_checkpoints,
    LLAMA_CTX_CHECKPOINTS_DEFAULT,
    parse_split_mode_override,
    strip_shadowing_flags,
)

from .launch_flags import (
    _LLAMA_ARG_FALSE_VALUES,
    _LLAMA_ARG_TRUE_VALUES,
    _env_ctx_checkpoints_override,
)
from .log import logger
from .memory_policy import _extra_args_set_any_flag


@functools.lru_cache(maxsize = 1)
def _metal_device_is_paravirtual() -> bool:
    """True when Metal is a virtualised Apple GPU, whose offload can corrupt output.

    Seen on our own macos-14/15 runners: offload returned an ordered walk through the
    character set while gpu_layers=0 on the same model, quant and binary was coherent,
    and MLX stayed correct. It does not reproduce on every build or quant (gemma-3-270m
    Q4_K_M on b9000 and b10090 was byte identical at gpu_layers 0 and 99), so
    UNSLOTH_ALLOW_PARAVIRTUAL_METAL=1 keeps the GPU on a VM known good. Physical Apple
    Silicon never matches.

    Three probes, cheapest first. MLX names the device outright (~40 ms) but is not on
    every Mac. hw.model then answers the headless case that matters most: measured on
    macos-14/15, SPDisplaysDataType returns zero bytes on a VM with no display so it
    cannot see a cloud or CI Mac at all, while hw.model still reports VirtualMac2,1 or
    VMM-x86_64 against Mac<n>,<n> on real hardware. SPDisplaysDataType stays last for
    desktop VMs, which do have a display and do name the paravirtual chipset.
    """
    if sys.platform != "darwin":
        return False
    if os.environ.get("UNSLOTH_ALLOW_PARAVIRTUAL_METAL", "").strip().lower() in (
        "1",
        "true",
        "yes",
    ):
        logger.info("UNSLOTH_ALLOW_PARAVIRTUAL_METAL set: keeping Metal offload.")
        return False
    name = ""
    try:
        import mlx.core as mx
        name = str(mx.device_info().get("device_name") or "")
    except Exception:
        name = ""
    if "paravirtual" not in name.lower():
        for _probe_cmd, _match in (
            # VMM-x86_64 is the Intel-guest identifier and contains neither
            # "virtual" nor "paravirtual"; upstream ships Metal on macOS x64 too.
            (["sysctl", "-n", "hw.model"], ("virtual", "vmm-")),
            (["system_profiler", "SPDisplaysDataType"], ("paravirtual",)),
        ):
            try:
                probe = subprocess.run(
                    _probe_cmd,
                    capture_output = True,
                    text = True,
                    timeout = 30,
                    encoding = "utf-8",
                    errors = "replace",
                )
            except Exception:
                continue
            _out = (probe.stdout or "").lower()
            if any(m in _out for m in _match):
                name = f"{name} {probe.stdout}".strip()
                break
    _low = name.lower()
    if "paravirtual" in _low or "virtual" in _low or "vmm-" in _low:
        logger.warning(
            "Metal device looks virtualised (%s). Offload can return corrupt output on "
            "paravirtual Apple GPUs, so GGUF inference will run on CPU. MLX is "
            "unaffected. Set UNSLOTH_ALLOW_PARAVIRTUAL_METAL=1 to keep the GPU.",
            name.strip()[:120] or "unknown",
        )
        return True
    return False


def _paravirtual_probe_answered(server_caps: Mapping[str, object]) -> bool:
    """False when the --help probe failed, so every capability read as absent.

    Both companion flags predate the oldest build Unsloth can launch (--no-mmproj-offload
    is b5178, --gpu-layers-draft is from 2023, and the base argv already requires b6325),
    so "capability missing" means the probe did not answer, not that the build lacks it:
    one malformed inherited LLAMA_ARG_* makes llama-server --help exit non-zero and the
    all-false result is cached for the process. Dropping the projector or the drafter on
    that would be a self-inflicted outage, so drops require a conclusive probe and the
    pins cover the unanswered case.
    """
    return not server_caps.get("mtp_probe_inconclusive")


def _paravirtual_draft_ngl_flag(server_caps: Mapping[str, object]) -> Optional[str]:
    """The draft-layer flag to pin with, or None only when the build truly lacks one."""
    flag = server_caps.get("spec_draft_ngl_flag")
    if flag:
        return str(flag)
    # Unanswered probe: use the spelling every launchable build has had since 2023
    # rather than treating the drafter as unpinnable.
    if not _paravirtual_probe_answered(server_caps):
        return "--gpu-layers-draft"
    return None


def _argv_keeps_projector_on_gpu(
    args: Sequence[str], env: Optional[Mapping[str, str]] = None
) -> bool:
    """Whether *args* still load a projector onto the device. arg.cpp applies
    ``LLAMA_ARG_MMPROJ(_URL)`` before argv, so the env alone still loads one."""
    tokens = [str(a) for a in args]
    env_map = env or {}

    def _flag(token: str) -> str:
        return token.split("=", 1)[0].replace("_", "-").lower()

    from_argv = any(_flag(a) in ("--mmproj", "-mm") for a in tokens)
    from_env = any(
        str(env_map.get(name) or "").strip()
        for name in ("LLAMA_ARG_MMPROJ", "LLAMA_ARG_MMPROJ_URL")
    )
    if not from_argv and not from_env:
        return False
    return _resolved_mmproj_offload(tokens, env_map) is not False


def _paravirtual_mmproj_pinnable(server_caps: Mapping[str, object]) -> bool:
    """True when --no-mmproj-offload can be emitted, including on an unanswered probe."""
    return bool(
        server_caps.get("supports_no_mmproj_offload")
        or not _paravirtual_probe_answered(server_caps)
    )


def ctx_checkpoints_allocated(server_caps: Mapping[str, object]) -> bool:
    """Return false only when a PARSED help listing confirms no checkpoint flag.

    A wrapper that exits 0 and prints nothing leaves help_probe_ok True over an empty
    catalogue, which is silence, not absence -- the same rule the parser applies to its
    own fail-open flags. Reading it as absence priced a recurrent model's whole snapshot
    pool at zero while the child kept llama.cpp's default.
    """
    if server_caps.get("ctx_checkpoints_flag"):
        return True
    if not server_caps.get("help_probe_ok"):
        return True
    return not bool(server_caps.get("flags"))


def ctx_checkpoints_default_for_caps(server_caps: Mapping[str, object]) -> int:
    """The count this build keeps unflagged: what it advertised, else upstream's today."""
    advertised = server_caps.get("ctx_checkpoints_default")
    if isinstance(advertised, int) and advertised >= 0:
        return advertised
    return LLAMA_CTX_CHECKPOINTS_DEFAULT


def effective_ctx_checkpoints_for_caps(
    server_caps: Mapping[str, object],
    extra_args: Optional[Iterable[str]],
    requested: Optional[int],
    *,
    per_checkpoint_bytes: int = 0,
    n_parallel: int = 1,
    total_host_bytes: Optional[int] = None,
    env: Optional[Mapping[str, str]] = None,
) -> int:
    """Resolve the count from capabilities, arguments, and host budget.

    A confirmed unsupported build uses zero. If a detected binary has no confirmed alias,
    Studio cannot emit its field or cap, so only pass-through extras override the default.
    """
    if not ctx_checkpoints_allocated(server_caps):
        return 0
    emittable = bool(server_caps.get("ctx_checkpoints_flag"))
    describes_a_child = bool(server_caps.get("found"))
    return effective_ctx_checkpoints(
        extra_args,
        requested if (emittable or not describes_a_child) else None,
        supports_flag = True,
        per_checkpoint_bytes = per_checkpoint_bytes,
        n_parallel = n_parallel,
        total_host_bytes = total_host_bytes if emittable else None,
        upstream_default = ctx_checkpoints_default_for_caps(server_caps),
        inherited = _env_ctx_checkpoints_override(env),
    )


def _idle_slot_clearing_active(cmd: Sequence[str], *, supports_cache_ram: bool) -> bool:
    """True when *cmd* leaves llama-server clearing idle slots' KV.

    ``--cache-idle-slots`` defaults ON but requires cache-ram, and ``--cache-ram 0``
    force-disables it (``server-context.cpp``). So the starting point is the BINARY, not
    the command line: a server old enough to lack ``--cache-ram`` never had idle-slot
    clearing either, while a server that has it clears by default at 8192 MiB. An absent
    flag means that default is in force, not that clearing is off -- the common case,
    since Studio only emits ``--cache-ram`` when something asked it to.

    Last-wins, matching llama.cpp's own argument handling.
    """
    enabled = bool(supports_cache_ram)
    for i, arg in enumerate(cmd):
        if arg == "--cache-ram":
            # A missing or unreadable value reads as no clearing, not as the default: the
            # conservative answer costs a re-cost, the other hands out resident cells.
            try:
                enabled = int(cmd[i + 1]) > 0
            except (IndexError, TypeError, ValueError):
                enabled = False
        elif arg == "--cache-idle-slots":
            enabled = True
        elif arg == "--no-cache-idle-slots":
            enabled = False
    return enabled


def _mmproj_env_is_audio_only(path: Optional[str]) -> bool:
    """True only when *path* names a readable projector with no vision tower.

    Anything else -- unset, absent from disk, unreadable, or declaring images -- is
    False, so the caller clears it. Deliberately asymmetric: keeping a projector the
    user switched off is the worse mistake of the two, and an unknown declaration
    reads image-capable upstream.
    """
    if not path:
        return False
    try:
        from utils.models.gguf_metadata import mmproj_capabilities
        has_audio, accepts_image = mmproj_capabilities(str(path))
    except Exception as e:
        logger.debug(f"inherited mmproj capability read failed: {e}")
        return False
    return has_audio and not accepts_image


# Both spellings of the projector-placement boolean. llama.cpp pairs them as one
# option and takes the last occurrence, so naming either one in the pass-through
# extras is the user taking ownership of the placement.
_MMPROJ_OFFLOAD_FLAGS: frozenset = frozenset({"--mmproj-offload", "--no-mmproj-offload"})


def _extra_args_set_mmproj_offload(extra_args: Optional[Sequence[str]]) -> bool:
    """True when the pass-through extras name either projector-placement spelling.

    Matched through _flag_name rather than by string equality: arg.cpp folds
    underscores to dashes before matching, so --mmproj_offload is the same flag.
    """
    return _extra_args_set_any_flag(extra_args, _MMPROJ_OFFLOAD_FLAGS)


def _extra_args_mmproj_offload_value(extra_args: Optional[Sequence[str]]) -> Optional[bool]:
    """The offload the extras actually resolve to, or None when they name neither.

    Detecting ownership is not enough to budget for: --no-mmproj-offload means the
    child keeps the projector in host RAM, so charging its bytes against VRAM
    shrinks the context for memory nothing on the GPU takes. Last occurrence wins,
    as llama.cpp pairs the two spellings into one option.
    """
    if not extra_args:
        return None
    value: Optional[bool] = None
    for raw in extra_args:
        flag = _flag_name(str(raw))
        if flag == "--mmproj-offload":
            value = True
        elif flag == "--no-mmproj-offload":
            value = False
    return value


# arg.cpp applies set_env options BEFORE argv, so an inherited placement is a choice
# Unsloth can reverse with nothing but a llama.cpp warning to show for it, and the child
# inherits this process's environment. Both spellings: get_value_from_env checks the
# LLAMA_ARG_NO_ form first and forces falsey on PRESENCE, empty value included.
_MMPROJ_OFFLOAD_ENV_VAR = "LLAMA_ARG_MMPROJ_OFFLOAD"
_MMPROJ_OFFLOAD_NEG_ENV_VAR = "LLAMA_ARG_NO_MMPROJ_OFFLOAD"

# Gemma 4 expects video at up to 1 fps; llama-server defaults to 4.
_LLAMA_VIDEO_FPS = "1"
_VIDEO_FPS_ENV_VAR = "LLAMA_ARG_VIDEO_FPS"


def _video_fps_flags(
    server_caps: Mapping[str, object], env: Optional[Mapping[str, str]] = None
) -> list[str]:
    """Return the default FPS flag unless unsupported or set in the environment.

    User extras are appended later and retain their last-wins override.
    """
    if not server_caps.get("supports_video_fps"):
        return []
    source = os.environ if env is None else env
    if source.get(_VIDEO_FPS_ENV_VAR) is not None:
        return []
    return ["--video-fps", _LLAMA_VIDEO_FPS]


def requested_video_fps(
    extra_args: Optional[Sequence[str]] = None, env: Optional[Mapping[str, str]] = None
) -> Optional[float]:
    """The frame rate the USER asked llama-server for, if they asked at all.

    Preprocessing has to know this: it transcodes before llama-server samples,
    and the server can only duplicate frames the transcode already dropped. So
    an explicit 8 fps has to reach the encoder, or the override silently buys
    nothing but duplicates.

    ARGV FIRST, then the environment. llama.cpp reads the env var while
    registering the option and argv overrides it afterwards, so an inherited
    LLAMA_ARG_VIDEO_FPS=1 alongside an advanced `--video-fps 8` samples at 8.
    Verified against b10976 on a 5s clip: env=1 alone costs 2796 prompt tokens,
    env=8 alone 19392, and env=1 with argv=8 also 19392.

    Within argv the LAST occurrence wins, which is how llama.cpp resolves a
    repeated flag ("only last value will be used"). Matching goes through
    `_flag_name`, so the underscore spelling llama.cpp also accepts
    (`--video_fps`, `--video_fps=8`) is caught rather than silently missed.

    None when the user said nothing, or said something that is not a rate.
    """
    args = [str(a) for a in (extra_args or ())]
    raw = None
    for i in range(len(args) - 1, -1, -1):
        if _flag_name(args[i]) != "--video-fps":
            continue
        raw = (
            args[i].split("=", 1)[1]
            if "=" in args[i]
            else (args[i + 1] if i + 1 < len(args) else None)
        )
        break
    if raw is None:
        source = os.environ if env is None else env
        raw = source.get(_VIDEO_FPS_ENV_VAR)
    try:
        rate = float(raw)
    except (TypeError, ValueError):
        return None
    return rate if rate > 0 else None


def _env_sets_mmproj_offload(env: Optional[Mapping[str, str]] = None) -> bool:
    """True when the child's environment names the projector placement at all.

    Ownership, not value: a value arg.cpp parses as neither truthy nor falsey throws
    out of common_params_parse, and a caller whose load is about to fail on their own
    variable is still the one who set it. Presence is the test for the same reason
    llama.cpp uses getenv, so ``LLAMA_ARG_NO_MMPROJ_OFFLOAD=`` counts.
    """
    source = os.environ if env is None else env
    return any(
        source.get(name) is not None
        for name in (_MMPROJ_OFFLOAD_ENV_VAR, _MMPROJ_OFFLOAD_NEG_ENV_VAR)
    )


def _env_mmproj_offload_value(env: Optional[Mapping[str, str]] = None) -> Optional[bool]:
    """The offload the child's environment resolves to, or None for silence.

    get_value_from_env's own order: the negative spelling short-circuits to falsey on
    presence, then the positive one goes through is_truthy / is_falsey. Anything else
    is left as None, since arg.cpp raises rather than picking a side and there is no
    placement to budget for.
    """
    source = os.environ if env is None else env
    if source.get(_MMPROJ_OFFLOAD_NEG_ENV_VAR) is not None:
        return False
    raw = source.get(_MMPROJ_OFFLOAD_ENV_VAR)
    if raw is None:
        return None
    value = str(raw).strip().lower()
    if value in _LLAMA_ARG_TRUE_VALUES:
        return True
    if value in _LLAMA_ARG_FALSE_VALUES:
        return False
    return None


def _resolved_mmproj_offload(
    extra_args: Optional[Sequence[str]], env: Optional[Mapping[str, str]] = None
) -> Optional[bool]:
    """Where the projector actually ends up, over extras and environment together.

    Same precedence the child applies: environment first, then argv on top, so a
    pass-through flag wins over an inherited variable and only silence in both falls
    through to None.
    """
    from_args = _extra_args_mmproj_offload_value(extra_args)
    return from_args if from_args is not None else _env_mmproj_offload_value(env)


_OVERRIDE_TENSOR_FLAGS: frozenset = frozenset(
    {
        "-ot",
        "--override-tensor",
        "-otd",
        "--override-tensor-draft",
        "--spec-draft-override-tensor",
    }
)


def _paravirtual_strip_gpu_overrides(
    extra_args: Optional[Iterable[str]], *, log_dropped: bool = True
) -> Optional[List[str]]:
    """Drop --override-tensor entries that place weights back on the corrupt device.

    ``-ot`` is the one placement flag ``--gpu-layers 0`` cannot answer: llama.cpp applies
    the override while selecting each weight's buffer type, before any layer is assigned
    to a device, so ``-ot ".*=Metal"`` puts weights on the virtualised GPU whatever the
    layer count says. It also accumulates rather than replaces, so a later flag cannot
    neutralise it; the entry has to go. ``-otd`` does the same for the drafter and would
    defeat the draft-layer pin.

    Only non-CPU targets are dropped: ``-ot exps=CPU`` moves weights the same way this
    fallback does, so stripping it would slow the very load it rescues.

    ``log_dropped=False`` for the duplicate-load comparators, which only inspect a
    request and must not narrate a drop no launch is performing, on every /load.
    """
    if not extra_args:
        return extra_args if extra_args is None else list(extra_args)
    args = [str(a) for a in extra_args]
    out: List[str] = []
    i = 0
    while i < len(args):
        tok = args[i]
        # _flag_name mirrors llama.cpp's underscore folding (arg.cpp does
        # std::replace(_ -> -) before matching), so --override_tensor is the same flag.
        base, _, inline = tok.partition("=")
        if _flag_name(base) in _OVERRIDE_TENSOR_FLAGS:
            value = inline if inline else (args[i + 1] if i + 1 < len(args) else "")
            targets = [
                part.rsplit("=", 1)[-1].strip().lower() for part in value.split(",") if "=" in part
            ]
            if targets and all(t == "cpu" for t in targets):
                out.append(tok)
                if not inline and i + 1 < len(args):
                    out.append(args[i + 1])
            elif log_dropped:
                logger.warning(
                    "Dropping %s %s: this Mac's Metal device is virtualised, so an "
                    "override that keeps weights on it would return corrupt output.",
                    base,
                    value,
                )
            i += 1 if inline else 2
            continue
        out.append(tok)
        i += 1
    return out


class ParavirtualPlacement(NamedTuple):
    """A load request's GPU placement after the virtualised-Metal CPU pin."""

    gpu_memory_mode: Literal["auto", "manual"]
    gpu_layers: int
    tensor_parallel: bool
    tensor_split: Optional[List[float]]
    n_cpu_moe: int
    extra_args: Optional[List[str]]


def paravirtual_normalized_request(
    *,
    gpu_memory_mode: Literal["auto", "manual"] = "auto",
    gpu_layers: int = -1,
    tensor_parallel: bool = False,
    tensor_split: Optional[List[float]] = None,
    n_cpu_moe: int = 0,
    extra_args: Optional[Iterable[str]] = None,
    log_dropped: bool = True,
) -> ParavirtualPlacement:
    """Map a requested placement to the one a virtualised Metal device launches.

    Single source of truth for the CPU pin, shared by the launch (load_model) and by
    BOTH duplicate-load comparators (``_already_in_target_state`` and the route's
    ``_request_matches_loaded_settings``). The launch records what it normalized, so a
    comparator judging the RAW request would mismatch its own state and tear down a
    healthy CPU server on every repeat Apply -- and, with an unbounded reload loop, 409
    or cancel a live generation.

    Pure, unconditional (callers gate on ``_metal_device_is_paravirtual()``) and
    idempotent, so re-applying it to a normalized request is a no-op.

    Deliberately NOT keyed on the request already being manual/0: that places the main
    model on CPU, but the extras are the caller's, and ``-ot ".*=Metal"`` is applied
    while each weight's buffer type is chosen, before any layer is assigned, so it puts
    weights back on the corrupt device -- and the route strips only the ``-ngl`` family.

    ``log_dropped=False`` for the comparators, which only inspect a request and must not
    narrate a rewrite no launch is performing.
    """
    normalized = extra_args if extra_args is None else list(extra_args)
    if normalized:
        # Extras are appended AFTER the managed --gpu-layers 0 and llama.cpp is
        # last-wins, so an `-ngl 99` would re-enable the corrupt offload. The route
        # strips these only in manual mode, so an Auto request still carries them.
        normalized = strip_shadowing_flags(
            normalized,
            strip_context = False,
            strip_cache = False,
            strip_spec = False,
            strip_template = False,
            strip_split_mode = False,
            strip_offload = True,
        )
        # strip_offload covers only the layer-count family; an --override-tensor aimed
        # at Metal survives it and places weights before any layer assignment.
        normalized = _paravirtual_strip_gpu_overrides(normalized, log_dropped = log_dropped)
    return ParavirtualPlacement(
        gpu_memory_mode = "manual",
        gpu_layers = 0,
        # Nothing is left on the GPU to split, and the launch drops these anyway;
        # normalize here too so the recorded state and the next request agree.
        tensor_parallel = False,
        tensor_split = None,
        n_cpu_moe = 0,
        extra_args = normalized,
    )


def _paravirtual_split_mode_pin(extra_args: Optional[Iterable[str]]) -> List[str]:
    """Flags that neutralise a pass-through ``--split-mode`` on a virtualised Mac.

    A user ``--split-mode`` is deliberately allowed (the Tensor Parallelism toggle
    cannot express row/none), but the paravirtual fallback has already pinned this load
    to CPU, and the flag is NOT inert at ``--gpu-layers 0``: llama.cpp builds a
    buffer-type list for every device in ``model->devices`` before any layer is
    assigned, and n_gpu_layers never reaches that step. So ``-sm row`` throws "device
    <X> does not support split buffers" on every backend except SYCL
    (``make_gpu_buft_list``, llama-model.cpp, since ggml-org/llama.cpp#24216 removed
    CUDA's split buffers), and ``-sm tensor`` throws "LLAMA_SPLIT_MODE_TENSOR not
    implemented for architecture" for every arch ``llm_arch_supports_sm_tensor``
    excludes. Either turns the CPU rescue into a server that refuses to start, and the
    zero-offload mask cannot save it: it writes only CUDA/HIP visibility vars, so the
    Metal device stays in the list on the one platform this fallback fires on.

    Overridden rather than stripped: llama.cpp is last-wins, so ``layer`` (the default,
    and a provable no-op at ``--gpu-layers 0`` -- llama-model.cpp computes the split
    points but never indexes them once ``act_gpu_layers`` is 0) neutralises the mode
    without rewriting ``extra_args``. The route's duplicate-load comparator compares
    stored extras verbatim and the UI does not round-trip the extras box, so stripping
    would leave the two permanently unequal and turn every later Apply into a real model
    swap. ``--tensor-split`` needs neither treatment for the same never-indexed reason.

    Returns [] when extras set no mode or already set ``layer``.
    """
    mode = (parse_split_mode_override(extra_args) or "").strip().lower()
    if not mode or mode == "layer":
        return []
    logger.warning(
        "Overriding --split-mode %s with the default layer split: this Mac's "
        "Metal device is virtualised, so the model runs on CPU and a GPU split "
        "mode would only fail the load.",
        mode,
    )
    return ["--split-mode", "layer"]
