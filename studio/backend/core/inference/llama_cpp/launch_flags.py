# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Launch argument and environment inspection."""

import ast
import functools
import os
import re
import sys
import threading

from pathlib import Path
from typing import (
    Iterable,
    List,
    Mapping,
    Optional,
)
from core.inference.llama_server_args import (
    _effective_tensor_parallel,
    _flag_name,
    _GPU_LAYER_FLAGS,
    parse_gpu_layers_override,
    parse_split_mode_override,
)

from .memory_policy import (
    _CPU_FFN_COUNT_FLAGS,
    _CPU_MOE_COUNT_FLAGS,
    _CPU_PLACEMENT_PRESENCE_FLAGS,
    _GENERATION_CPU_AFFINITY_FLAGS,
    _THREAD_OVERRIDE_FLAGS,
    _active_vram_fraction,
    _extra_args_set_any_flag,
    _extra_args_set_spec_type,
    _vram_usable_mib,
)


def _linux_math_core_count(
    cpu_root: Path = Path("/sys/devices/system/cpu"),
    logical_cpus: Optional[int] = None,
    vendor_id: Optional[str] = None,
    event_source_root: Path = Path("/sys/bus/event_source/devices"),
    pinnable_cpus: Optional[set[int]] = None,
    probe_affinity: Optional[bool] = None,
) -> Optional[int]:
    """Linux x86 cores useful to llama.cpp's default math pool.

    ``thread_siblings`` mirrors llama.cpp's physical-core count. Intel's hybrid
    PMU masks identify performance cores on kernels without ``cpu_capacity``;
    heterogeneous capacity remains the fallback. AMD capacity classes do not
    trigger llama.cpp's hybrid CPUID branch, so they retain every physical core.
    """
    if logical_cpus is not None and logical_cpus < 1:
        return None

    def cpu_list(path: Path) -> Optional[set[int]]:
        try:
            raw = path.read_text(encoding = "utf-8").strip()
            if not raw:
                return set()
            result: set[int] = set()
            for part in raw.split(","):
                bounds = [int(value) for value in part.split("-")]
                if len(bounds) == 1 and bounds[0] >= 0:
                    result.add(bounds[0])
                elif len(bounds) == 2 and 0 <= bounds[0] <= bounds[1]:
                    result.update(range(bounds[0], bounds[1] + 1))
                else:
                    return None
            return result
        except (OSError, ValueError):
            return None

    if vendor_id is None:
        try:
            vendor_match = re.search(
                r"^vendor_id\s*:\s*(\S+)",
                Path("/proc/cpuinfo").read_text(encoding = "utf-8"),
                flags = re.MULTILINE,
            )
            vendor_id = vendor_match.group(1) if vendor_match else ""
        except OSError:
            vendor_id = ""
    siblings: list[str] = []
    capacities: list[Optional[int]] = []
    cpu = 0
    while logical_cpus is None or cpu < logical_cpus:
        cpu_path = cpu_root / f"cpu{cpu}"
        try:
            sibling_set = (
                (cpu_path / "topology" / "thread_siblings").read_text(encoding = "utf-8").strip()
            )
        except OSError:
            break
        if not sibling_set:
            break
        siblings.append(sibling_set)
        try:
            capacities.append(int((cpu_path / "cpu_capacity").read_text(encoding = "utf-8").strip()))
        except (OSError, ValueError):
            capacities.append(None)
        cpu += 1
    if not siblings:
        return None
    online_cpus = cpu_list(cpu_root / "online")
    active_siblings = {
        sibling_set
        for cpu, sibling_set in enumerate(siblings)
        if not online_cpus or cpu in online_cpus
    }
    if not active_siblings:
        return None
    physical_count = len(active_siblings)
    if vendor_id == "GenuineIntel":
        native_count = len(online_cpus) if online_cpus else len(siblings)

        def hybrid_math_count(performance_cpus: set[int]) -> int:
            should_probe = pinnable_cpus is None and (
                cpu_root == Path("/sys/devices/system/cpu")
                if probe_affinity is None
                else probe_affinity
            )

            def count(pin_cpus: bool) -> int:
                result = 0
                cpu = 0
                while cpu < native_count:
                    if online_cpus and cpu not in online_cpus:
                        return physical_count
                    if pinnable_cpus is not None and cpu not in pinnable_cpus:
                        return physical_count
                    if pin_cpus:
                        try:
                            os.sched_setaffinity(0, {cpu})
                        except OSError:
                            return physical_count
                    if cpu not in performance_cpus:
                        cpu += 1
                        continue
                    result += 1
                    cpu += 2
                return result or physical_count

            if not should_probe:
                return count(False)
            if not hasattr(os, "sched_getaffinity") or not hasattr(os, "sched_setaffinity"):
                return physical_count

            answer = [physical_count]

            def probe() -> None:
                try:
                    saved_affinity = os.sched_getaffinity(0)
                except OSError:
                    return
                try:
                    answer[0] = count(True)
                finally:
                    try:
                        os.sched_setaffinity(0, saved_affinity)
                    except OSError:
                        pass

            worker = threading.Thread(target = probe)
            try:
                worker.start()
                worker.join()
            except RuntimeError:
                return physical_count
            return answer[0]

        core_mask = cpu_list(event_source_root / "cpu_core" / "cpus")
        atom_path = event_source_root / "cpu_atom"
        lowpower_path = event_source_root / "cpu_lowpower"
        hybrid_pmu = atom_path.exists() or lowpower_path.exists()
        if core_mask:
            return hybrid_math_count(core_mask)
        if hybrid_pmu:
            atom_mask = cpu_list(atom_path / "cpus") or cpu_list(lowpower_path / "cpus")
            if core_mask == set() and atom_mask:
                return physical_count
            return 1
        active_capacities = [
            capacity
            for cpu, capacity in enumerate(capacities)
            if not online_cpus or cpu in online_cpus
        ]
        if (
            all(capacity is not None for capacity in active_capacities)
            and len(set(active_capacities)) > 1
        ):
            fastest = max(capacity for capacity in active_capacities if capacity is not None)
            return hybrid_math_count(
                {
                    cpu
                    for cpu, capacity in enumerate(capacities)
                    if (not online_cpus or cpu in online_cpus) and capacity == fastest
                }
            )
    return physical_count


def _spilled_decode_threads(
    n_threads: Optional[int] = None, extra_args: Optional[Iterable[str]] = None
) -> Optional[int]:
    """How many threads the spilled-decode cost model should price.

    A positive override follows the launched command's managed-flag, then
    pass-through precedence. An inherited ``LLAMA_ARG_THREADS`` is ignored here
    because Studio removes it before spawn whenever argv has no thread flag; if
    argv does have one, that later flag wins. Explicit SMT oversubscription and
    explicit or inherited CPU affinity cannot be represented by this
    physical-core model, so those settings return ``None`` and the caller declines.
    Otherwise llama.cpp sizes its pool from
    ``common_cpu_get_num_math``, which counts physical cores
    and skips SMT siblings and efficiency cores. ``os.cpu_count()`` counts every
    hyperthread, so on a 6-core / 12-thread desktop it told the cost model the
    host had twice the cores the child would ever use, and the generation
    penalty came out about half of what a spill really costs there.

    Linux x86 reads the same sibling topology as llama.cpp and uses the kernel's
    capacity classes for hybrid CPUs. psutil is the cross-platform fallback. If
    neither can answer, match llama.cpp's last resort: keep up to four logical
    CPUs, otherwise halve them, and use four when no logical count is available.
    """
    requested: Optional[int] = None
    if n_threads is not None and n_threads > 0:
        requested = int(n_threads)
    args = [str(arg) for arg in extra_args] if extra_args else []
    if any(arg.partition("=")[0] in _GENERATION_CPU_AFFINITY_FLAGS for arg in args):
        return None
    for i, arg in enumerate(args):
        name, _, inline = arg.partition("=")
        if name not in _THREAD_OVERRIDE_FLAGS:
            continue
        value = inline if inline else (args[i + 1] if i + 1 < len(args) else "")
        try:
            requested = int(value.strip())
        except (TypeError, ValueError):
            continue
    physical: Optional[int] = None
    logical: Optional[int] = None
    affinity_count: Optional[int] = None
    if sys.platform.startswith("linux"):
        if hasattr(os, "sched_getaffinity"):
            try:
                affinity_count = len(os.sched_getaffinity(0))
            except OSError:
                pass
        if os.uname().machine.lower() in ("x86_64", "amd64"):
            physical = _linux_math_core_count()
    try:
        import psutil

        answer = psutil.cpu_count(logical = True)
        if answer and answer > 0:
            logical = int(answer)
        if physical is None:
            answer = psutil.cpu_count(logical = False)
            if answer and answer > 0:
                physical = int(answer)
    except Exception:
        pass
    if logical is None:
        logical = os.cpu_count()
    if affinity_count is not None and logical is not None and affinity_count < logical:
        return None
    if physical is None:
        physical = logical if logical and logical <= 4 else (logical // 2 if logical else 4)
    if requested is None:
        return physical
    actual = requested if requested > 0 else (logical or os.cpu_count() or physical)
    if actual > physical:
        return None
    return max(1, actual)


def _strip_flag_pairs(args: Iterable[str], flags: frozenset[str]) -> list[str]:
    """Drop each ``flag value`` pair naming one of ``flags``."""
    out: list[str] = []
    skip = False
    for arg in args:
        if skip:
            skip = False
            continue
        if str(arg) in flags:
            skip = True
            continue
        out.append(str(arg))
    return out


# common_params defaults in the bundled llama.cpp runtime.
_DEFAULT_LLAMA_N_BATCH = 2048
_DEFAULT_LLAMA_N_UBATCH = 512
# Non-causal image decoding requires one image chunk to fit in the micro-batch.
# These are the per-image limits from clip.cpp. Exact values avoid excess VRAM use.
_MMPROJ_NON_CAUSAL_IMAGE_TOKENS = {
    "gemma4v": 1120,
    "gemma4uv": 1120,
    "gemma3": 256,
    "deepseek4v": 384,
}
# A projector whose family we cannot read gets headroom instead of a ceiling.
_MMPROJ_UNKNOWN_UBATCH = 2048
_GEMMA4V_CAUSAL_TEXT_N_EMBD = frozenset({1536, 2560})  # E2B and E4B
_LLAMA_ARG_TRUE_VALUES = frozenset({"on", "enabled", "true", "1"})
_LLAMA_ARG_FALSE_VALUES = frozenset({"off", "disabled", "false", "0"})
_LLAMA_ARG_AUTO_VALUES = frozenset({"auto", "-1"})
_LLAMA_ARG_TRUE_OR_AUTO_VALUES = _LLAMA_ARG_TRUE_VALUES | _LLAMA_ARG_AUTO_VALUES
_LLAMA_ARG_TRUE_FALSE_AUTO_VALUES = _LLAMA_ARG_TRUE_OR_AUTO_VALUES | _LLAMA_ARG_FALSE_VALUES


def _diffusion_manual_ngl(gpu_memory_mode: str, gpu_layers: int) -> Optional[int]:
    """The --ngl the diffusion runner would launch with, or None for its default.

    Only an explicit manual count reaches the child; Auto (-1) and Unsloth mode defer.
    Shared with both dedup guards, so an inert manual preference cannot loop.
    """
    return int(gpu_layers) if (gpu_memory_mode == "manual" and int(gpu_layers) >= 0) else None


def _scale_diffusion_required_gb(required_gb: float, ngl: int, n_layers: Optional[int]) -> float:
    """GPU share of a partially offloaded diffusion GGUF, for the training guard.

    Only the ``ngl`` resident layers compete for VRAM; sizing the whole file 409s a
    split that fits. An unknown layer count keeps the full estimate."""
    if not n_layers or n_layers <= 0 or ngl >= n_layers:
        return required_gb
    return required_gb * (ngl / n_layers)


# The runner's own CUDA context, which the free reading taken before spawn cannot see.
_DIFFUSION_RUNNER_OVERHEAD_MIB = 512.0
# Context the fitted split leaves room for when the runner auto-sizes (maxtok 0).
_DIFFUSION_FIT_CTX = 4096


def _diffusion_auto_ngl(
    file_bytes: int,
    n_layers: Optional[int],
    n_head: Optional[int],
    maxtok: int,
    free_mib: float,
    total_mib: float,
    *,
    sysmem_fallback: bool = False,
) -> Optional[int]:
    """GPU layers for an Auto-mode diffusion load, or None to keep every layer on the GPU.

    Only a GGUF that cannot fit moves: all layers die in cudaMalloc, or on Windows spill into
    shared system memory until RAM and the page file run out (#6456). The split leaves room
    for the compute buffer (measured 1.75x the [n_head, N, N] fp32 scores) and the 8% of VRAM
    the runner's context probe keeps free (diffusion-gemma-visual-server.cpp)."""
    if not file_bytes or not n_layers or n_layers <= 0 or (free_mib <= 0 and total_mib <= 0):
        return None
    usable = _vram_usable_mib(
        free_mib, total_mib, _active_vram_fraction(), sysmem_fallback = sysmem_fallback
    )
    usable -= _DIFFUSION_RUNNER_OVERHEAD_MIB
    file_mib = file_bytes / (1024 * 1024)
    if file_mib <= usable:
        return None
    # Mirrors unsloth_zoo's _canvas_maxtok: larger requests are sent as 0 (auto-size).
    n_ctx = maxtok if 0 < maxtok <= 8192 else _DIFFUSION_FIT_CTX
    scores_mib = max(1, n_head or 1) * n_ctx * n_ctx * 4 / (1024 * 1024)
    ctx_reserve = 1.75 * scores_mib + 0.08 * (total_mib if total_mib > 0 else free_mib)
    # The whole file over the block count also charges token_embd, which stays on the CPU.
    per_layer_mib = file_mib / n_layers
    return max(0, min(n_layers - 1, int((usable - ctx_reserve) // per_layer_mib)))


def _source_declares_option(source: str, option: str) -> bool:
    """True iff ``source`` really registers ``option`` on an argparse parser.

    Parsed, not grepped: a substring test matches the flag in a comment and misses it
    when quoted differently. Falls back to a substring scan when the file will not parse.
    """
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return option in source
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (isinstance(func, ast.Attribute) and func.attr == "add_argument"):
            continue
        for arg in node.args:
            if isinstance(arg, ast.Constant) and arg.value == option:
                return True
    return False


def _shim_supports_ngl(shim_cmd: List[str]) -> bool:
    """Whether the resolved diffusion shim accepts --ngl.

    The package floor still allows an unsloth_zoo without the flag and argparse exits
    on an unknown option, so emitting it blindly breaks the launch. Read the source
    rather than spawning a probe: the launch path is already slow.
    """
    # _find_diffusion_assets emits a file override ([python, <path>]) or the installed
    # package ([python, -m, <module>]). Key on the shape, not the suffix (an override may
    # be extensionless, .pyw or SHIM.PY), and probe that exact file: a sibling shim.py
    # answers for a different shim than argparse will run.
    if len(shim_cmd) >= 2 and "-m" not in shim_cmd:
        candidates = [Path(shim_cmd[-1])]
    else:
        candidates = []
        try:
            import importlib.util
            spec = importlib.util.find_spec("unsloth_zoo")
            if spec is not None and spec.submodule_search_locations:
                pkg_dir = Path(list(spec.submodule_search_locations)[0])
                candidates.append(pkg_dir / "diffusion_studio" / "shim.py")
        except Exception:
            return False
    for path in candidates:
        try:
            source = path.read_text(encoding = "utf-8", errors = "ignore")
        except OSError:
            continue
        if _source_declares_option(source, "--ngl"):
            return True
    return False


def _visual_server_splits_gpus(visual_bin: Optional[str]) -> bool:
    """Whether the diffusion visual server survives a model split over several GPUs.

    Builds before the fix abort on the first request once split: with the prompt-KV store
    off (it is single-device) every [prompt|canvas] row asked for logits past the server's
    canvas-sized output cap. Fixed builds answer ``--capabilities`` with ``multi_gpu_split``;
    read that literal from the binary rather than spawning it, since an old build takes
    10-30 s to fail on the unknown argument.
    """
    if not visual_bin:
        return False
    try:
        st = os.stat(visual_bin)
    except OSError:
        return False
    return _binary_has_marker(visual_bin, st.st_mtime_ns, st.st_size, b"multi_gpu_split")


@functools.lru_cache(maxsize = 16)
def _binary_has_marker(path: str, _mtime_ns: int, _size: int, marker: bytes) -> bool:
    try:
        with open(path, "rb") as f:
            return marker in f.read()
    except OSError:
        return False


def _swa_full_from_args_or_env(
    extra_args: Optional[Iterable[str]], env: Optional[Mapping[str, str]] = None
) -> bool:
    """Whether llama.cpp receives the enable-only full-size SWA option."""
    if _extra_args_set_any_flag(extra_args, {"--swa-full"}):
        return True
    value = (os.environ if env is None else env).get("LLAMA_ARG_SWA_FULL")
    return value in _LLAMA_ARG_TRUE_VALUES


def _reasoning_from_env(env: Mapping[str, str]) -> Optional[bool]:
    value = env.get("LLAMA_ARG_REASONING")
    if value in _LLAMA_ARG_TRUE_VALUES:
        return True
    if value in _LLAMA_ARG_FALSE_VALUES:
        return False
    return None


def _reasoning_effort_from_env(env: Mapping[str, str]) -> Optional[str]:
    value = env.get("LLAMA_ARG_REASONING_EFFORT")
    return None if value == "default" else value


def _env_asks_for_the_native_context(env: Optional[Mapping[str, str]] = None) -> bool:
    """Whether an inherited LLAMA_ARG_CTX_SIZE is 0.

    llama.cpp resolves the variable through -c's own handler, so 0 pins the
    native length here exactly as on the command line. Anything unparseable is
    left to llama.cpp to reject."""
    value = (os.environ if env is None else env).get("LLAMA_ARG_CTX_SIZE")
    try:
        return int(str(value).strip()) == 0
    except (AttributeError, TypeError, ValueError):
        return False


def _env_ctx_checkpoints_override(env: Optional[Mapping[str, str]] = None) -> Optional[int]:
    """An inherited LLAMA_ARG_CTX_CHECKPOINTS count, else None.

    llama.cpp applies the variable before parsing argv (arg.cpp's env loop runs ahead of
    the flag loop), so an emitted --ctx-checkpoints would OVERRULE it. It is as explicit
    an instruction as a typed flag, so the automatic cap stands down for it and the
    estimate prices it. Anything unparseable is left to llama.cpp to reject.
    """
    value = (os.environ if env is None else env).get("LLAMA_ARG_CTX_CHECKPOINTS")
    try:
        count = int(str(value).strip())
    except (AttributeError, TypeError, ValueError):
        return None
    return count if count >= 0 else None


def _kv_offload_from_args(
    extra_args: Optional[Iterable[str]], env: Optional[Mapping[str, str]] = None
) -> bool:
    """Resolve llama.cpp's environment and last-wins KV-offload flags (default on):
    LLAMA_ARG_KV_OFFLOAD parses before argv, and ``-kvo`` / ``--kv-offload`` is the
    positive form, so a CLI re-enable beats a false env.

    Both env spellings, as with the projector pair above. ``-kvo`` is registered with
    a negative form (``{"-nkvo", "--no-kv-offload"}``, common/arg.cpp), so
    ``common_arg::get_value_from_env`` rewrites LLAMA_ARG_ to LLAMA_ARG_NO_, and if
    that variable EXISTS it forces the value to "0" and never reads the affirmative
    one (common/arg.cpp:127-141). Presence, empty value included, because llama.cpp
    tests it with getenv. Missing it is the optimistic direction here: the whole KV
    cache is then on the host (llama-kv-cache.cpp only upgrades a layer's buffer
    type inside ``if (offload)``) while the fit still lets free VRAM pay for it.
    """
    source_env = os.environ if env is None else env
    enabled = True
    if source_env.get("LLAMA_ARG_NO_KV_OFFLOAD") is not None:
        enabled = False
    else:
        value = source_env.get("LLAMA_ARG_KV_OFFLOAD")
        if value in _LLAMA_ARG_FALSE_VALUES:
            enabled = False
        elif value in _LLAMA_ARG_TRUE_VALUES:
            enabled = True
    for raw in extra_args or ():
        flag = _flag_name(str(raw))
        if flag in {"-kvo", "--kv-offload"}:
            enabled = True
        elif flag in {"-nkvo", "--no-kv-offload"}:
            enabled = False
    return enabled


def _is_positive_int(value: Optional[str]) -> bool:
    """Whether ``value`` parses as an int > 0 (llama.cpp's std::stoi)."""
    try:
        return int(str(value).strip()) > 0
    except (TypeError, ValueError):
        return False


def _args_place_tensors_on_cpu(extra_args: Optional[Iterable[str]] = None) -> bool:
    """True when extras keep weights on the CPU, whatever the layer count says.

    The arg-side twin of ``_env_places_tensors_on_cpu``, including its rule that
    a CPU-MoE or CPU-FFN count only counts when positive: ``--n-cpu-moe 0`` places nothing.
    """
    values = [str(raw) for raw in extra_args or ()]
    for i, raw in enumerate(values):
        flag = _flag_name(raw)
        # Any -ot counts even if it matches nothing: the list need only be
        # non-empty, and guessing at the pattern is not worth it.
        if flag in _CPU_PLACEMENT_PRESENCE_FLAGS:
            return True
        if flag in _CPU_MOE_COUNT_FLAGS | _CPU_FFN_COUNT_FLAGS:
            _, eq, inline = raw.partition("=")
            if _is_positive_int(inline if eq else (values[i + 1] if i + 1 < len(values) else "")):
                return True
    return False


def _env_places_tensors_on_cpu(env: Optional[Mapping[str, str]] = None) -> bool:
    """True when inherited env keeps weights on the CPU, whatever the argv says.

    The child inherits these, so they survive stripping the equivalent flags.
    Shared with the Model Memory host-residency gate, which must not call a load
    fully offloaded while one of them is holding tensors in RAM.
    """
    source_env = os.environ if env is None else env
    return bool(
        source_env.get("LLAMA_ARG_OVERRIDE_TENSOR")
        or source_env.get("LLAMA_ARG_CPU_MOE") in _LLAMA_ARG_TRUE_VALUES
        or _is_positive_int(source_env.get("LLAMA_ARG_N_CPU_MOE"))
        or _is_positive_int(source_env.get("LLAMA_ARG_N_CPU_FFN"))
    )


def _env_fixes_gpu_layers(env: Optional[Mapping[str, str]] = None) -> bool:
    """True when an inherited ``LLAMA_ARG_N_GPU_LAYERS`` owns the layer placement.

    The env twin of ``_GPU_LAYER_FLAGS``, and it bites harder than the tokens do:
    the fitting path emits ``--fit on`` and no ``-ngl`` at all, so an inherited
    count is the ONLY layer policy the child sees. llama.cpp's fitter then refuses
    to touch it -- ``common_params_fit_impl`` throws "n_gpu_layers already set by
    user" (common/fit.cpp:377) on any value but the default, and the caller
    downgrades that to a warning (common/fit.cpp:805) -- so the launch proceeds
    with the count as given, placing a prefix of the weights and running the rest
    out of host RAM the fit just credited to a card.

    ``auto`` / ``-1`` is that default (common/common.h:465,
    llama-model.cpp:2453), so it leaves the fitter free to choose and is not an
    override. Everything else is, ``all`` included: a count that stands is a count
    this fit did not price. Empty counts as unset -- llama.cpp's ``std::stoi``
    would throw on it and the child would never start.
    """
    raw = (os.environ if env is None else env).get("LLAMA_ARG_N_GPU_LAYERS")
    value = str(raw or "").strip().lower()
    return bool(value) and value not in _LLAMA_ARG_AUTO_VALUES


def _pipeline_parallel_disabled_by_args(
    extra_args: Optional[Iterable[str]],
    env: Optional[Mapping[str, str]] = None,
    n_layers: Optional[int] = None,
) -> bool:
    """Whether the launch turns OFF llama.cpp's pipeline parallelism, i.e. the 4x in
    ``_CTX_COMPUTE_SPLIT_MULT``. llama-context.cpp requires ``-sm layer``, KV offload
    on and an empty tensor-override list (among other things); anything here is back to
    1x, and charging the multiplier anyway would waste context. Unsloth passes none of
    it, but extra_args and the environment can.

    ``-cmoe`` / ``-ncmoe`` count: both push into ``tensor_buft_overrides`` exactly like
    ``-ot`` (common/arg.cpp). ``-otd`` does not - it targets the draft model.
    LLAMA_ARG_SPLIT_MODE is not read here: the launch pops a non-layer inherited value
    on the layer path, so it never reaches the child. Ambiguous input resolves to
    "still pipelined" - over-reserving only costs context, under-reserving OOMs the
    split. ``n_layers`` is the GGUF block count for the ``-ngl`` check below; None/0
    means unknown."""
    source_env = os.environ if env is None else env
    if _env_places_tensors_on_cpu(source_env):
        return True
    if not _kv_offload_from_args(extra_args, env = env):
        return True
    # -sm none/row keep the layer path and pass through, so they really do disable
    # it. -sm tensor does not: the layer branch is only reached after a downgrade,
    # which strips the flag (strip_split_mode_only) and leaves the child on layer.
    split_mode = parse_split_mode_override(extra_args)
    if split_mode is not None and split_mode.strip().lower() not in {"layer", "tensor"}:
        return True
    # Pipelining needs n_gpu_layers > n_layer_all, so a user -ngl (appended after
    # Unsloth's -ngl -1, last-wins) that cannot exceed the count loads a prefix and
    # turns it off. Negative (all layers) or above the count keeps the step, even
    # in between: n_layers is block_count, which IS n_layer_all (llama-model.cpp
    # reads the key straight into it); hparams.n_layer() is the smaller MTP-less one.
    if n_layers:
        try:
            gpu_layers = parse_gpu_layers_override(extra_args)
        except ValueError:
            gpu_layers = None  # malformed: unknown, so keep the step
        if gpu_layers is not None and 0 <= gpu_layers <= n_layers:
            return True
    # Same predicate the Model Memory host-residency gate uses; -ncmoe 0 pushes
    # no override, so it leaves pipelining on.
    return _args_place_tensors_on_cpu(extra_args)


def _kv_unified_from_args(
    extra_args: Optional[Iterable[str]],
    default: bool = False,
    env: Optional[Mapping[str, str]] = None,
) -> bool:
    """Resolve llama.cpp's environment and last-wins unified KV flags."""
    enabled = False
    value = (os.environ if env is None else env).get("LLAMA_ARG_KV_UNIFIED")
    if value in _LLAMA_ARG_TRUE_VALUES:
        enabled = True
    elif value in _LLAMA_ARG_FALSE_VALUES:
        enabled = False
    if default:
        # Unsloth's managed --kv-unified flag is appended after environment
        # parsing and before user extras.
        enabled = True
    for raw in extra_args or ():
        flag = _flag_name(str(raw))
        if flag in {"-kvu", "--kv-unified"}:
            enabled = True
        elif flag in {"-no-kvu", "--no-kv-unified"}:
            enabled = False
    return enabled


# llama_init_from_model forces flash attention off for LLM_ARCH_GROK whatever the launch
# asks for: Grok softcaps the KQ logits, which the fused kernel does not carry.
_FLASH_ATTN_INCOMPATIBLE_ARCHITECTURES = frozenset({"grok"})


def _architecture_forces_flash_attn_off(architecture: Optional[str]) -> bool:
    """Whether llama.cpp will disable flash attention for this architecture on its own."""
    return str(architecture or "").strip().lower() in _FLASH_ATTN_INCOMPATIBLE_ARCHITECTURES


def _planned_flash_attn_state(
    extra_args: Optional[Iterable[str]] = None,
    *,
    planned_cache_types: Optional[tuple[str, str]] = None,
    supports_flash_attn: bool = True,
    tensor_parallel: bool = False,
    architecture: Optional[str] = None,
    env: Optional[Mapping[str, str]] = None,
) -> bool:
    """One answer for the estimate and for the argv: the estimator floors V at f16 and pads
    it model-wide when flash attention is off, so resolving the two separately prices a
    different load, not a conservative one (#9697, #10489)."""
    if not supports_flash_attn:
        return False
    if _architecture_forces_flash_attn_off(architecture):
        # Ahead of every rule below, as llama.cpp puts it ahead of both its own upgrades.
        return False
    v_type = (planned_cache_types[1] if planned_cache_types else "f16") or "f16"
    quantized_v = str(v_type).strip().lower() not in {"f16", "bf16", "f32"}
    if quantized_v:
        # Not a choice: llama.cpp turns it on itself rather than refusing the load.
        return True
    # Prepended because ``load_model`` appends a managed ``--flash-attn on``: the env loses
    # to it and only the user's own extras beat it (#9697, #10489).
    effective_args = ["--flash-attn", "on", *(str(arg) for arg in extra_args or ())]
    if _asked_for_auto_flash_attn(effective_args, env = env):
        if _effective_tensor_parallel(extra_args, tensor_parallel, env):
            # Not undecided: llama.cpp upgrades AUTO to ENABLED under SPLIT_MODE_TENSOR.
            return True
        # ``auto`` is decided at load time and can come back no; size for the costlier answer.
        return False
    return _flash_attn_enabled_from_args(effective_args, default = True, env = env)


def _user_fit_disabled(
    extra_args: Optional[Iterable[str]] = None, *, env: Optional[Mapping[str, str]] = None
) -> bool:
    """Whether the USER has turned llama.cpp's fitter off, last-wins: Unsloth's own token
    comes first and a retry rewrites it, so only a user's later value survives a re-place."""
    values = [str(arg) for arg in extra_args] if extra_args else []
    asked: Optional[str] = None
    inherited = (os.environ if env is None else env).get("LLAMA_ARG_FIT")
    if inherited is not None and str(inherited).strip():
        asked = str(inherited).strip()
    for i, raw in enumerate(values):
        if _flag_name(raw) not in {"-fit", "--fit"}:
            continue
        _, eq, inline = raw.partition("=")
        value = inline if eq else "on"
        if not eq and i + 1 < len(values):
            value = values[i + 1]
        asked = value
    return str(asked).strip().lower() in {"off", "0", "false", "no", "disabled"}


def _placement_is_fitter_proof(
    extra_args: Optional[Iterable[str]] = None,
    *,
    gpu_layers: Optional[int] = None,
    env: Optional[Mapping[str, str]] = None,
) -> bool:
    """Whether a layer count llama.cpp's fitter refuses to move owns this placement:
    ``common_params_fit_impl`` throws "n_gpu_layers already set by user" (common/fit.cpp:377)
    for any value but -1, so re-enabling the fitter for a respawn buys nothing."""
    if gpu_layers is not None and gpu_layers >= 0:
        return True
    values = [str(arg) for arg in extra_args] if extra_args else []
    for i, raw in enumerate(values):
        if _flag_name(raw) not in _GPU_LAYER_FLAGS:
            continue
        _, eq, inline = raw.partition("=")
        value = inline if eq else ""
        if not eq and i + 1 < len(values):
            value = values[i + 1]
        value = value.strip().lower()
        if value and value not in _LLAMA_ARG_AUTO_VALUES:
            return True
    return bool(_env_fixes_gpu_layers(env))


def _reserved_flash_attn_state(
    planned: bool,
    extra_args: Optional[Iterable[str]] = None,
    *,
    tensor_parallel: bool = False,
    gpu_layers: Optional[int] = None,
    env: Optional[Mapping[str, str]] = None,
) -> bool:
    """``planned``, held to the conservative reading where a respawn cannot be re-placed.

    A flash-attention crash respawns with it off, which pads V model-wide, so where the
    fitter cannot re-place that respawn the larger cache lands on a placement chosen for the
    smaller one. Tensor mode has no such respawn, and pricing it would refuse loads that fit.
    """
    if not planned:
        return planned
    if _effective_tensor_parallel(extra_args, tensor_parallel, env):
        return planned
    if _user_fit_disabled(extra_args, env = env):
        return False
    if _placement_is_fitter_proof(extra_args, gpu_layers = gpu_layers, env = env):
        return False
    return planned


def _asked_for_auto_flash_attn(
    extra_args: Optional[Iterable[str]] = None, *, env: Optional[Mapping[str, str]] = None
) -> bool:
    """``_flash_attn_enabled_from_args`` folds auto into True because it answers what the
    argv will SAY; sizing needs to know nobody has decided yet."""
    asked: Optional[str] = None
    value = (os.environ if env is None else env).get("LLAMA_ARG_FLASH_ATTN")
    if value in _LLAMA_ARG_TRUE_FALSE_AUTO_VALUES:
        asked = value
    values = [str(arg) for arg in extra_args] if extra_args else []
    for i, raw in enumerate(values):
        if _flag_name(raw) not in {"-fa", "--flash-attn"}:
            continue
        _, eq, inline = raw.partition("=")
        value = inline if eq else "on"
        if not eq and i + 1 < len(values) and values[i + 1] in _LLAMA_ARG_TRUE_FALSE_AUTO_VALUES:
            value = values[i + 1]
        asked = value
    return asked in _LLAMA_ARG_AUTO_VALUES


def _flash_attn_enabled_from_args(
    args: Optional[Iterable[str]],
    default: bool = True,
    env: Optional[Mapping[str, str]] = None,
) -> bool:
    """Resolve llama.cpp's environment and last-wins flash-attention settings."""
    enabled = default
    # llama.cpp applies LLAMA_ARG_FLASH_ATTN before parsing argv (arg.cpp set_env),
    # so the CLI still wins. --flash-attn has no args_neg, so no LLAMA_ARG_NO_ twin.
    value = (os.environ if env is None else env).get("LLAMA_ARG_FLASH_ATTN")
    if value in _LLAMA_ARG_FALSE_VALUES:
        enabled = False
    elif value in _LLAMA_ARG_TRUE_OR_AUTO_VALUES:
        enabled = True
    values = [str(arg) for arg in args] if args else []
    for i, raw in enumerate(values):
        if _flag_name(raw) not in {"-fa", "--flash-attn"}:
            continue
        _, eq, inline = raw.partition("=")
        value = inline if eq else "on"
        if not eq and i + 1 < len(values) and values[i + 1] in _LLAMA_ARG_TRUE_FALSE_AUTO_VALUES:
            value = values[i + 1]
        if value in _LLAMA_ARG_FALSE_VALUES:
            enabled = False
        elif value in _LLAMA_ARG_TRUE_OR_AUTO_VALUES:
            enabled = True
    return enabled


def _effective_spec_type(
    extra_args: Optional[Iterable[str]], env: Optional[Mapping[str, str]] = None
) -> Optional[str]:
    """The last --spec-type label a launch carries: final CLI --spec-type (or
    --spec-default, which resolves non-MTP), else LLAMA_ARG_SPEC_TYPE. A display and
    comparison label only, NOT what llama.cpp enables: types accumulate there, so ask
    _extra_args_requests_mtp whether MTP is on. None if neither sets it."""
    args = [str(a) for a in extra_args] if extra_args else []
    cli_present = False
    cli_value: Optional[str] = None
    for i, raw in enumerate(args):
        flag = _flag_name(raw)
        _, eq, inline = raw.partition("=")
        if flag == "--spec-default":
            cli_present = True
            cli_value = "default"
            continue
        if flag != "--spec-type":
            continue
        cli_present = True
        cli_value = inline if eq else (args[i + 1] if i + 1 < len(args) else "")
    if cli_present:
        return cli_value
    return (os.environ if env is None else env).get("LLAMA_ARG_SPEC_TYPE")


def _child_spec_env(
    extra_args: Optional[Iterable[str]], env: Optional[Mapping[str, str]] = None
) -> Mapping[str, str]:
    """The spec env the child will actually see: empty unless extras name a
    --spec-type, because the launch scrubs LLAMA_ARG_SPEC_* whenever Unsloth owns the
    spec block. Predict from this, not os.environ, or you describe a server that is
    not the one being started."""
    if not _extra_args_set_spec_type(extra_args):
        return {}
    return os.environ if env is None else env


# The speculative types llama.cpp gives the TARGET context recurrent-state rollback
# snapshots for -- n_rs_seq = --spec-draft-n-max, per
# common_params_speculative::need_n_rs_seq. Every draft-model type but draft-simple,
# and both MTP spellings, since which one a build advertises varies.
_TARGET_ROLLBACK_SPEC_TYPES = frozenset(
    {"mtp", "draft-mtp", "draft-eagle3", "draft-dflash", "draft-dspark"}
)


# Mirrors llama.cpp is_recr_impl for SSM headers with no full_attention_interval.
_SSM_EVERY_LAYER_ARCHS = frozenset({"mamba", "mamba2", "falcon-h1"})
# Nemotron-H also zeroes head_count_kv on its MLP/MoE layers; its SSM layers have no FFN.
_SSM_FFN_FREE_LAYER_ARCHS = frozenset({"nemotron_h", "nemotron_h_moe"})


# Architectures whose TARGET context leaves the embedded MTP/NextN blocks out of
# its KV cache, so block_count must lose them before the target is priced.
#
# This is NOT every architecture that carries {arch}.nextn_predict_layers.
# llama_hparams::n_layer() subtracts nextn everywhere (llama-hparams.cpp:297),
# but llama_kv_cache walks hparams.n_layer_all (llama-kv-cache.cpp:100) and drops
# blocks only through an optional per-arch filter (:169). That filter is installed
# for the hybrids at llama-model.cpp:2289 (Qwen3-Next/Qwen3.5/MiniMax-01 and, via
# is_recr, Nemotron-H), for GLM-DSA and DeepSeek3.2 at :2129, and for the
# STEP35/HY_V3/MIMO2 group at :2356. Everywhere else -- deepseek2, glm4, glm4moe,
# bailingmoe2, cohere2moe, exaone4 -- filter is nullptr and the trailing MTP block
# DOES get target KV, so subtracting it under-reserves.
#
# Fail closed: an arch not named here keeps every block, because over-reserving
# costs context while under-reserving OOMs the load. Recurrent state is the one
# thing that always excludes nextn, llama_memory_recurrent sizing on n_layer()
# directly (llama-memory-recurrent.cpp:29), which is why
# _mamba_recurrent_state_bytes subtracts unconditionally.
_TARGET_KV_EXCLUDES_NEXTN_ARCHS = frozenset(
    {
        # Hybrid attention + recurrent, llama-model.cpp:2289
        "qwen3next",
        "qwen35",
        "qwen35moe",
        "minimax-01",
        "nemotron_h",
        "nemotron_h_moe",
        # MLA/DSA trunk with a dense MTP head, llama-model.cpp:2129
        "glm-dsa",
        "deepseek32",
        # Hybrid KDA + DSA trunk; both GLM-5.3-Flash ports filter blk.45 out
        "glm5next",
        "glm5-next",
        # Plain attention trunk with an explicit nextn filter, llama-model.cpp:2356
        "step35",
        "hy_v3",
        "mimo2",
    }
)


def _accumulated_spec_types(
    extra_args: Optional[Iterable[str]], env: Optional[Mapping[str, str]] = None
) -> set:
    """Every speculative type a launch carries, lowercased. Types accumulate rather
    than replace: llama.cpp applies the env first through the same handler and every
    --spec-type inserts at the end, so a later flag cannot clear an earlier one (even
    --spec-type none only appends NONE). Look at all of them, not just the last."""
    values = []
    args = [str(a) for a in extra_args] if extra_args else []
    for i, raw in enumerate(args):
        _, eq, inline = raw.partition("=")
        if _flag_name(raw) == "--spec-type":
            values.append(inline if eq else (args[i + 1] if i + 1 < len(args) else ""))
    env_value = (os.environ if env is None else env).get("LLAMA_ARG_SPEC_TYPE")
    if env_value:
        values.append(env_value)
    return {p.strip().lower() for v in values for p in v.split(",") if p.strip()}


def _extra_args_requests_mtp(
    extra_args: Optional[Iterable[str]], env: Optional[Mapping[str, str]] = None
) -> bool:
    """True if MTP lands in llama.cpp's spec-type vector, so the budget must reserve
    for it and the slots must clamp. Per _accumulated_spec_types, ANY source counts."""
    return bool(_accumulated_spec_types(extra_args, env) & {"mtp", "draft-mtp"})


def _extra_args_requests_dspark(
    extra_args: Optional[Iterable[str]], env: Optional[Mapping[str, str]] = None
) -> bool:
    """True if DSpark lands in llama.cpp's accumulated spec-type vector."""
    return "draft-dspark" in _accumulated_spec_types(extra_args, env)


def _dflash_drafter_breaks_on_images(
    model_path: Optional[str], dflash_path: Optional[str], projector: Optional[str]
) -> bool:
    """#13511: llama.cpp skips an M-RoPE image in the DFlash drafter (its rows share one
    position) and a NeoX drafter then rejects the jump, so the first image sent after any
    text fails the slot. Unreadable headers fail open."""
    if not (model_path and dflash_path and projector):
        return False
    try:
        from utils.models.gguf_metadata import gguf_has_mrope_sections, mmproj_accepts_image
        return bool(
            mmproj_accepts_image(projector)
            and gguf_has_mrope_sections(model_path) is True
            and gguf_has_mrope_sections(dflash_path) is False
        )
    except Exception:
        return False


def _extra_args_requests_dflash(
    extra_args: Optional[Iterable[str]], env: Optional[Mapping[str, str]] = None
) -> bool:
    """True if DFlash lands in llama.cpp's accumulated spec-type vector."""
    return "draft-dflash" in _accumulated_spec_types(extra_args, env)
