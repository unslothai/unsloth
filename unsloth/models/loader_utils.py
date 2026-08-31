# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from ..device_type import DEVICE_TYPE_TORCH
import hashlib
import importlib
import os
import torch
import re
import tempfile
import contextlib
import threading as _threading
import functools
from typing import Union
from .mapper import (
    INT_TO_FLOAT_MAPPER,
    FLOAT_TO_INT_MAPPER,
    MAP_TO_UNSLOTH_16bit,
    FLOAT_TO_FP8_BLOCK_MAPPER,
    FLOAT_TO_FP8_ROW_MAPPER,
    build_mappers,
    _add_with_lower,
    _add_lower_only,
)

# The alias helpers a fetched mapper.py may call, resolved to the INSTALLED
# implementations. `_get_new_mapper` reads such calls as data: it takes the table and
# the two literal strings out of the AST and applies them with these, so the fetched
# text never supplies behaviour.
_MAPPER_HELPERS = {
    "_add_with_lower": _add_with_lower,
    "_add_lower_only": _add_lower_only,
}

# https://github.com/huggingface/transformers/pull/26037 allows 4 bit loading!
from transformers import __version__ as transformers_version
from unsloth.models._utils import TorchAOConfig
from unsloth_zoo.utils import Version, get_quant_type
import gc
import traceback as _traceback

transformers_version = Version(transformers_version)
SUPPORTS_FOURBIT = transformers_version >= Version("4.37")

LOCAL_RANK_KEYS = ("LOCAL_RANK", "RANK")
WORLD_SIZE_KEYS = ("WORLD_SIZE",)

BAD_MAPPINGS = {
    "unsloth/Qwen3-32B-unsloth-bnb-4bit".lower(): "unsloth/Qwen3-32B-bnb-4bit".lower(),  # 32B dynamic quant is way too big
    "unsloth/Qwen3-30B-A3B-unsloth-bnb-4bit".lower(): "unsloth/Qwen3-30B-A3B".lower(),  # HF loads MoEs too slowly
    "unsloth/Qwen3-30B-A3B-bnb-4bit".lower(): "unsloth/Qwen3-30B-A3B".lower(),  # We rather do it on the fly
    "unsloth/Qwen3-30B-A3B-Base-unsloth-bnb-4bit".lower(): "unsloth/Qwen3-30B-A3B-Base".lower(),  # HF loads MoEs too slowly
    "unsloth/Qwen3-30B-A3B-Base-bnb-4bit".lower(): "unsloth/Qwen3-30B-A3B-Base".lower(),  # We rather do it on the fly
}


def _get_torchao_fp8_config(fp8_mode):
    # Lazy import so a broken optional vLLM install doesn't break `import unsloth`.
    from unsloth_zoo.vllm_utils import _get_torchao_fp8_config as _impl
    return _impl(fp8_mode)


def _get_env_int(keys):
    for key in keys:
        value = os.environ.get(key)
        if value is None:
            continue
        try:
            return int(value)
        except ValueError:
            continue
    return None


def _infer_distributed_ranks():
    if (
        torch.distributed.is_available()
        and getattr(torch.distributed, "is_initialized", lambda: False)()
    ):
        try:
            return torch.distributed.get_rank(), torch.distributed.get_world_size()
        except Exception:
            pass
    return _get_env_int(LOCAL_RANK_KEYS), _get_env_int(WORLD_SIZE_KEYS)


def is_distributed():
    rank, world_size = _infer_distributed_ranks()
    return (world_size or 1) > 1 or (rank is not None and rank > 0)


def prepare_device_map():
    rank, world_size = _infer_distributed_ranks()
    distributed = (world_size or 1) > 1 or (rank is not None and rank > 0)
    if not distributed:
        return None, False

    local_rank = 0 if rank is None else rank
    device_map = {"": f"{DEVICE_TYPE_TORCH}:{local_rank}"}
    try:
        if DEVICE_TYPE_TORCH == "cuda":
            torch.cuda.set_device(local_rank)
        elif DEVICE_TYPE_TORCH == "xpu" and hasattr(torch, "xpu"):
            torch.xpu.set_device(local_rank)
    except Exception:
        pass
    return device_map, True


UNSLOTH_DEVICE_MAP = "unsloth"

# Same planner, different answer when it declines. Every veto path in
# `resolve_unsloth_device_map` ends in "sequential", which fills the first device to its
# whole free budget before touching the next one -- right for a loader default, wrong for
# a caller that picked several cards on their combined capacity and would rather have an
# even split than a full first card. Naming the fallback is what makes that reachable
# without every such caller having to enumerate the veto reasons, which are unsloth's and
# can grow.
UNSLOTH_BALANCED_DEVICE_MAP = "unsloth_balanced"
_PLANNED_DEVICE_MAPS = {UNSLOTH_DEVICE_MAP: "sequential", UNSLOTH_BALANCED_DEVICE_MAP: "balanced"}

# A string, not None, so an explicit False stays distinguishable from "unset".
OFFLOAD_EMBEDDING_AUTO = "auto"


class _DefaultDeviceMap(str):
    """`"sequential"`, marked as the value nobody asked for.

    The env opt-in must upgrade the default and leave an explicit "sequential" alone, and a
    plain comparison cannot tell them apart. A `str` subclass keeps the distinction without
    changing the value: it equals, hashes and formats as `"sequential"` everywhere, and in
    `inspect.signature` too, which a sentinel object would not.
    """

    __slots__ = ()


DEFAULT_DEVICE_MAP = _DefaultDeviceMap("sequential")


def planner_hub_kwargs(loader_kwargs):
    """The Hub options the planner's own `AutoConfig` lookup needs.

    It resolves the config a second time, from `model_name`. Without these the lookup can
    go to the network behind a `local_files_only` request, or miss a cache-only model and
    turn a plan the load needed into a "sequential" fallback.
    """
    loader_kwargs = loader_kwargs or {}
    hub = {}
    if loader_kwargs.get("cache_dir") is not None:
        hub["cache_dir"] = loader_kwargs["cache_dir"]
    if _get_effective_local_files_only(loader_kwargs):
        hub["local_files_only"] = True
    # Resolving a remote class is a third lookup, and the planner honours `code_revision`
    # for it (`_HUB_KWARGS` in unsloth_zoo's `device_map_planner`). Left out, the plan is
    # built from the default revision's code and can name a tree the model does not have.
    if loader_kwargs.get("code_revision") is not None:
        hub["code_revision"] = loader_kwargs["code_revision"]
    return hub


def planner_config_overrides(loader_kwargs):
    """Config fields the caller overrides on the load, which the planner has to size for.

    The planner rebuilds the repo's config from a name, so an override living only in
    kwargs is invisible to it. `max_position_embeddings` is the one that changes weight
    sizes: raise it on learned position embeddings and the planned tensors come out
    smaller than the materialized ones, so a map that fitted on paper OOMs.
    `plan_device_map_for_pretrained` hands leftover kwargs to `AutoConfig.from_pretrained`,
    which applies them the way the load does.
    """
    value = (loader_kwargs or {}).get("max_position_embeddings")
    return {} if value is None else {"max_position_embeddings": value}


def planner_kwargs_with_max_memory(planner_kwargs, loader_kwargs):
    """The caller's transformers `max_memory` has to reach the planner as well.

    Once a plan is returned the load gets an explicit dict, and transformers consults
    `max_memory` only for a string `device_map` (`_get_device_map` gates the whole
    `infer_auto_device_map` branch on `isinstance(device_map, str)`). A budget that used to
    bound placement would be dropped in silence, and the plan could exceed their caps or
    use a card they withheld. An explicit `device_map_planner_kwargs["max_memory"]` wins.
    """
    budget = (loader_kwargs or {}).get("max_memory")
    if budget is None:
        return planner_kwargs
    merged = dict(planner_kwargs or {})
    merged.setdefault("max_memory", budget)
    return merged


def unmarked_device_map(device_map):
    """The default with its marker removed; anything else exactly as it came in.

    For a nested load that must not re-read the value as "nobody chose this". A bare `str()`
    would also flatten a caller's `{"": 0}` into text transformers reads as a device name.
    """
    return str(device_map) if isinstance(device_map, _DefaultDeviceMap) else device_map


def requested_device_map(device_map):
    """Head-aware planning is what a caller who chose nothing gets.

    Only the untouched default is upgraded: a dict, "auto", or a "sequential" the caller
    typed is a placement someone chose, and greedy fill is a different execution model.
    `UNSLOTH_AUTO_DEVICE_MAP=0` turns it off process-wide, for the multi-GPU operator who
    wants that fill back.
    """
    if device_map is DEFAULT_DEVICE_MAP and os.environ.get("UNSLOTH_AUTO_DEVICE_MAP", "1") == "1":
        return UNSLOTH_DEVICE_MAP
    return device_map


def planner_quantization_kwargs(
    load_in_4bit = False,
    load_in_8bit = False,
    quantization_config = None,
    extra_skip_modules = None,
):
    """The quantization the planner must size for, as the load will really apply it.

    The config or the flags, never both, since transformers refuses both and loader.py
    clears the flags whenever it forwards a config. Bare flags would describe a
    full-precision load and raise `DeviceMapInfeasible` on one that would have fit.

    The skip list travels with the flags: SKIP_QUANTIZATION_MODULES stays in compute dtype
    as `modules_to_not_convert`, and sizing it at 4bit understates the head device by GiBs
    on a large-vocab VLM. A pre-quantized checkpoint carries its own list in config.json.
    """
    if quantization_config is not None:
        return {"quantization_config": quantization_config}
    kwargs = {"load_in_4bit": load_in_4bit, "load_in_8bit": load_in_8bit}
    if load_in_4bit or load_in_8bit:
        try:
            from unsloth_zoo.peft_utils import SKIP_QUANTIZATION_MODULES
        except Exception:
            # Built on every quantized load, planning or not, so an older unsloth_zoo must
            # not turn a 4bit load into an ImportError. One without the shared list predates
            # the planner that consumes it, so this plan was going to decline anyway.
            return kwargs
        kwargs["llm_int8_skip_modules"] = SKIP_QUANTIZATION_MODULES + list(extra_skip_modules or [])
    return kwargs


def planner_model_class(config, trust_remote_code = False):
    """The model class the planner's own rules pick for `config`, or None if unknown.

    The planner never sees the auto class the load chose. `config` is whatever the caller
    passed, while the planner rebuilds the repo's from `model_name`; the two can disagree.
    """
    try:
        from unsloth_zoo.device_map_planner import _auto_class_for
        from ._utils import resolve_model_class

        auto_class = _auto_class_for(config, trust_remote_code = trust_remote_code)
        return resolve_model_class(auto_class, config)
    except Exception:
        # Unknown, not mismatched: an unsloth_zoo without this has no planner to feed.
        return None


def planner_class_mismatch_reason(loaded_class, planned_class):
    """Why the planner's model differs from the one being loaded, else None.

    Overriding the config's own choice gets a map for a module tree the model does not
    have: `num_labels` swaps in a `score` head where `lm_head` was planned, and dispatch
    refuses with "does not give any device for ... score.weight". Compared as model
    classes, since two distinct auto classes can resolve to the same VLM.
    """
    if loaded_class is None or planned_class is None or loaded_class is planned_class:
        return None
    return f"the load builds {loaded_class.__name__}, not the planned {planned_class.__name__}"


# accelerate's `max_memory` spellings, in the order it tries them: GiB/MiB/KiB are binary,
# GB/MB/KB are decimal, and a lowercase trailing `b` on a decimal unit means bits, not bytes.
_SIZE_UNITS = (
    ("GIB", 2**30, False),
    ("MIB", 2**20, False),
    ("KIB", 2**10, False),
    ("GB", 10**9, True),
    ("MB", 10**6, True),
    ("KB", 10**3, True),
)


def _as_bytes(size):
    """A `max_memory` budget in bytes, or None if it cannot be read as one.

    accelerate takes `"10GiB"` as readily as an int, so a caller writes what the load would
    have taken. The rules are reproduced rather than imported because this runs before
    anything has established accelerate is installed: an ImportError would read every budget
    as unparseable and drop the cap in silence. `test_the_local_size_parser_agrees_with_
    accelerate` holds the copy in step wherever accelerate is present.

    None leaves the measured free memory in place rather than dropping the device. A bool is
    unreadable here where accelerate would take its int value: `{0: True}` is a typo, and a
    one byte budget reads as "unusable" to the planner.
    """
    if isinstance(size, bool):
        return None
    if isinstance(size, int):
        return size if size >= 0 else None
    if not isinstance(size, str):
        return None
    upper = size.upper()
    for unit, scale, has_bit_form in _SIZE_UNITS:
        if not upper.endswith(unit):
            continue
        try:
            amount = int(float(size[: -len(unit)]) * scale)
        except ValueError:
            return None
        # Bits, if they spelled the unit "Gb" rather than "GB". Binary units have no such form.
        if has_bit_form and size.endswith("b"):
            amount //= 8
        return amount if amount >= 0 else None
    return None


def resolve_unsloth_device_map(
    device_map,
    model_name,
    *,
    fast_inference = False,
    full_finetuning = False,
    planner_kwargs = None,
    skip_reason = None,
    **config_kwargs,
):
    """Plan a head-aware multi-GPU map for `device_map = "unsloth"`, else return as-is.

    Opt-in only, so nothing an existing caller passes changes meaning. The plan is built
    on the meta device: no GPU memory, no weight download.

    Falls back to "sequential" wherever a plan cannot apply, since a model that loads the
    old way beats one that refuses to load at all. `DeviceMapInfeasible` is the exception:
    the planner raises it rather than spilling a bitsandbytes model to CPU, and swallowing
    it would hand the user an OOM instead of a diagnosis.

    `skip_reason` is the caller's veto, for when only the caller can tell the planner
    would describe a different model than the load builds.
    """
    # `isinstance` first: a caller's explicit dict is unhashable, so `in` alone raises.
    if not isinstance(device_map, str) or device_map not in _PLANNED_DEVICE_MAPS:
        return device_map
    _declined = _PLANNED_DEVICE_MAPS[device_map]

    def _fallback(reason):
        print(f"Unsloth: Not planning a device map; {reason}. Using `{_declined}`.")
        return _declined

    if skip_reason is not None:
        return _fallback(skip_reason)
    if fast_inference:
        return _fallback("vLLM places its own weights")
    if full_finetuning:
        return _fallback("full finetuning does not use the quantized planner")
    if is_distributed():
        # Every rank already owns the whole model on its own card; splitting on top of
        # that puts every rank on every card, a different execution model, not a bigger one.
        return _fallback("each rank of a distributed launch owns its own device")
    if DEVICE_TYPE_TORCH != "cuda":
        return _fallback(f"the planner has no memory budgets for {DEVICE_TYPE_TORCH}")
    try:
        device_count = torch.cuda.device_count()
    except Exception as error:
        return _fallback(f"the devices could not be counted ({error})")

    # Popped, not forwarded: `max_memory` is a named parameter of the planner, so a copy
    # left in `planner_kwargs` raises `TypeError: got multiple values for keyword argument`,
    # which the handler below turns into a silent "sequential", losing the cap and the plan.
    #
    # A caller's mapping replaces the measured one rather than editing it: its keys are the
    # devices they will let the load use. That is accelerate's reading too --
    # `_init_infer_auto_device_map` takes `devices = list(max_memory.keys())` and
    # `get_max_memory` never widens a supplied mapping -- so `{0: ..., 1: ...}` on a
    # four-GPU host means GPUs 2 and 3 are somebody else's.
    planner_kwargs = dict(planner_kwargs or {})
    requested_memory = planner_kwargs.pop("max_memory", None)
    requested_memory = dict(requested_memory) if requested_memory else None

    # Read before probing: `mem_get_info` initialises a CUDA context on each device it
    # touches, and a withheld card is likely busy with the workload it was withheld for.
    # One that refuses (ECC error, MIG parent, Exclusive_Process) would also drop the plan
    # to "sequential" over a device this load was never going to use.
    if requested_memory is None:
        probe = list(range(device_count))
    else:
        probe = [
            device
            for device in requested_memory
            if isinstance(device, int)
            and not isinstance(device, bool)
            and 0 <= device < device_count
        ]
    if len(probe) < 2:
        return _declined

    try:
        from unsloth_zoo.device_map_planner import plan_device_map_for_pretrained
    except Exception as error:
        return _fallback(f"the planner is unavailable ({error})")

    # Free, not total: this process's context and anything else resident on the card make
    # total an overcommit. Guarded because a card can still refuse mid-probe.
    try:
        max_memory = {index: torch.cuda.mem_get_info(index)[0] for index in probe}
    except Exception as error:
        return _fallback(f"free memory could not be read ({error})")

    if requested_memory is not None:
        budgets = {}
        for device, written in requested_memory.items():
            measured = max_memory.get(device)
            budget = _as_bytes(written)
            if budget is None:
                # Nothing to compare against: what we measured, or for a device we never
                # measured (cpu, disk) their value untouched for the planner to read.
                budgets[device] = measured if measured is not None else written
            else:
                # Under what is actually free: they may know of reservations we cannot
                # measure, but planning above free is how a plan OOMs on dispatch.
                budgets[device] = budget if measured is None else min(measured, budget)
        max_memory = budgets

    try:
        plan = plan_device_map_for_pretrained(
            model_name, max_memory = max_memory, **planner_kwargs, **config_kwargs
        )
    except Exception as error:
        if type(error).__name__ == "DeviceMapInfeasible":
            raise
        return _fallback(f"planning failed ({error})")

    if plan is None:
        return _declined
    print(plan.describe())
    return plan.device_map


def __get_model_name(
    model_name,
    load_in_4bit = True,
    INT_TO_FLOAT_MAPPER = None,
    FLOAT_TO_INT_MAPPER = None,
    MAP_TO_UNSLOTH_16bit = None,
    load_in_fp8 = False,
    FLOAT_TO_FP8_BLOCK_MAPPER = None,
    FLOAT_TO_FP8_ROW_MAPPER = None,
):
    model_name = str(model_name)
    lower_model_name = model_name.lower()

    assert load_in_fp8 in (True, False, "block")
    if load_in_fp8 != False:
        if load_in_fp8 == True and (os.environ.get("UNSLOTH_HAS_FBGEMM", "0") == "1"):
            if lower_model_name in FLOAT_TO_FP8_ROW_MAPPER:
                # Faster row scaling only works if FBGEMM works!
                return FLOAT_TO_FP8_ROW_MAPPER[lower_model_name]
            elif lower_model_name in FLOAT_TO_FP8_BLOCK_MAPPER:
                # Otherwise we use the slower blockwise type
                return FLOAT_TO_FP8_BLOCK_MAPPER[lower_model_name]
        else:
            if lower_model_name in FLOAT_TO_FP8_BLOCK_MAPPER:
                return FLOAT_TO_FP8_BLOCK_MAPPER[lower_model_name]
        # No pre-quantized model found. vllm >= 0.12.0 quantizes to FP8 on the
        # fly (return original name); older vllm falls through to offline quant.
        if importlib.util.find_spec("vllm") is not None:
            import vllm
            if Version(vllm.__version__) >= Version("0.12.0"):
                return model_name
        return None

    elif not SUPPORTS_FOURBIT and lower_model_name in INT_TO_FLOAT_MAPPER:
        model_name = INT_TO_FLOAT_MAPPER[lower_model_name]
        print(
            f"Unsloth: Your transformers version of {transformers_version} does not support native "
            f"4bit loading.\nThe minimum required version is 4.37.\n"
            f'Try `pip install --upgrade "transformers>=4.37"`\n'
            f"to obtain the latest transformers build, then restart this session.\n"
            f"For now, we shall load `{model_name}` instead (still 4bit, just slower downloading)."
        )
        return model_name

    elif not load_in_4bit and lower_model_name in INT_TO_FLOAT_MAPPER:
        new_model_name = INT_TO_FLOAT_MAPPER[lower_model_name]
        # logger.warning_once(
        #     f"Unsloth: You passed in `{model_name}` which is a 4bit model, yet you set\n"\
        #     f"`load_in_4bit = False`. We shall load `{new_model_name}` instead."
        # )
        return new_model_name

    elif not load_in_4bit and lower_model_name in MAP_TO_UNSLOTH_16bit:
        new_model_name = MAP_TO_UNSLOTH_16bit[lower_model_name]
        return new_model_name

    elif load_in_4bit and SUPPORTS_FOURBIT and lower_model_name in FLOAT_TO_INT_MAPPER:
        # Keep an explicit -bnb-4bit name; otherwise map to the dynamic version.
        if lower_model_name.endswith("-bnb-4bit"):
            return model_name

        new_model_name = FLOAT_TO_INT_MAPPER[lower_model_name]
        # logger.warning_once(
        #     f"Unsloth: You passed in `{model_name}` and `load_in_4bit = True`.\n"\
        #     f"We shall load `{new_model_name}` for 4x faster loading."
        # )
        return new_model_name

    return None


def _get_new_mapper():
    try:
        import requests
        import time

        new_mapper = (
            "https://raw.githubusercontent.com/unslothai/unsloth/main/unsloth/models/mapper.py"
        )
        # Capped WHILE reading, since `requests.get` buffers the whole body first, and
        # the deadline is total because `timeout` is per-read.
        byte_cap = 1_000_000
        deadline = time.monotonic() + 10
        chunks, total = [], 0
        # Redirects by hand: `requests` drains each intermediate body inside `get`.
        url = new_mapper
        for _ in range(5):
            response = requests.get(url, timeout = 3, stream = True, allow_redirects = False)
            with response:
                location = (
                    response.headers.get("location")
                    if (300 <= response.status_code < 400)
                    else None
                )
                if location is not None:
                    if time.monotonic() > deadline:
                        return {}, {}, {}, {}, {}
                else:
                    # `read1`: the timeout is per SOCKET READ, so a trickling peer
                    # resets it forever and the deadline is never reached.
                    raw = response.raw
                    # `requests` only decompresses inside `iter_content`.
                    try:
                        raw.decode_content = True
                    except AttributeError:
                        pass
                    read_once = getattr(raw, "read1", None)
                    while True:
                        if time.monotonic() > deadline:
                            return {}, {}, {}, {}, {}
                        if read_once is not None:
                            chunk = read_once(65_536)
                        else:
                            # Older urllib3 has no `read1`; one byte returns as promptly.
                            chunk = raw.read(1)
                        if not chunk:
                            break
                        chunks.append(chunk)
                        total += len(chunk)
                        if total > byte_cap:
                            return {}, {}, {}, {}, {}
                    encoding = response.encoding or "utf-8"
            if location is None:
                break
            url = requests.compat.urljoin(url, location)
        else:
            return {}, {}, {}, {}, {}
        new_mapper = b"".join(chunks).decode(encoding, errors = "replace")
        # Never exec the response: that is arbitrary code execution inside every
        # `from_pretrained` that hits an unmapped name. Only `__INT_TO_FLOAT_MAPPER` is
        # data, and the tables are returned rather than written into this module.
        import ast

        # `ast.parse` builds the whole tree before any literal-only check runs, so this
        # is no defence against exhaustion (python/cpython#95588); the byte cap above is.
        tree = ast.parse(new_mapper)
        # Every module-level literal dict, in source order; chosen below.
        literal_bindings = []
        for index, node in enumerate(tree.body):
            # `AnnAssign` too, or an annotation upstream turns the probe off silently.
            if isinstance(node, ast.AnnAssign):
                targets = [node.target.id] if isinstance(node.target, ast.Name) else []
                value = node.value
            elif isinstance(node, ast.Assign):
                targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
                value = node.value
            else:
                continue
            if value is None or not targets:
                continue
            try:
                literal = ast.literal_eval(value)
            except Exception:
                continue
            if not isinstance(literal, dict):
                continue
            for name in targets:
                literal_bindings.append((index, name, literal))

        def _binding_at(name, before):
            """What `name` holds when the statement at index `before` runs.

            The last assignment BEFORE that point: a rebind after the builder ran must
            not be read back over what the builder actually saw.
            """
            found = None
            for index, bound, literal in literal_bindings:
                if bound == name and (before is None or index <= before):
                    found = literal
            return found

        # Statements that really RUN at import: `ast.walk` reaches into function bodies
        # and dead branches, which would fabricate a mapping. Local, because the tests
        # run this body in a bare namespace.
        def _constant_truth(test):
            """True/False for a statically decidable condition, else None.

            `not True` is a UnaryOp, so it fell through and its body was read as run.
            """
            if isinstance(test, ast.Constant):
                return bool(test.value)
            if isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not):
                inner = _constant_truth(test.operand)
                return None if inner is None else not inner
            return None

        def _iterates_nothing(iterable):
            if isinstance(iterable, (ast.Tuple, ast.List, ast.Set)):
                return not iterable.elts
            if isinstance(iterable, ast.Dict):
                return not iterable.keys
            if isinstance(iterable, ast.Constant):
                # An empty literal iterates nothing; a nonempty one keeps its body.
                return isinstance(iterable.value, (str, bytes)) and not iterable.value
            return False

        def _class_bound(statement):
            # A class binding makes every LATER mutation a class attribute.
            targets = []
            if isinstance(statement, ast.Assign):
                targets = statement.targets
            elif isinstance(statement, ast.AnnAssign):
                targets = [statement.target]
            return {target.id for target in targets if isinstance(target, ast.Name)}

        def _executed_nodes(
            body,
            shadowed = frozenset(),
            class_body = False,
        ):
            # What ENDED the suite: `"return"` leaves the function, `True` the suite.
            for statement in body:
                if class_body:
                    # Only the statements ABOVE it still see the module global.
                    shadowed = shadowed | _class_bound(statement)
                if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                if isinstance(statement, (ast.Break, ast.Continue, ast.Return)):
                    return "return" if isinstance(statement, ast.Return) else True
                if isinstance(statement, ast.ClassDef):
                    # A class body, unlike a function body, RUNS at import.
                    yield from _executed_nodes(statement.body, shadowed, class_body = True)
                    continue
                if isinstance(statement, ast.If) and _constant_truth(statement.test) is not None:
                    # Undecidable tests keep their body.
                    branch = statement.body if _constant_truth(statement.test) else statement.orelse
                    ended = yield from _executed_nodes(branch, shadowed)
                    if ended:
                        return ended
                    continue
                if isinstance(statement, ast.While) and _constant_truth(statement.test) is not None:
                    if _constant_truth(statement.test):
                        ended = yield from _executed_nodes(statement.body, shadowed)
                        if ended == "return":
                            return ended
                    else:
                        ended = yield from _executed_nodes(statement.orelse, shadowed)
                        if ended:
                            return ended
                    continue
                if isinstance(statement, ast.For) and _iterates_nothing(statement.iter):
                    ended = yield from _executed_nodes(statement.orelse, shadowed)
                    if ended:
                        return ended
                    continue
                for field in ("body", "orelse", "finalbody"):
                    children = getattr(statement, field, None)
                    if isinstance(children, list):
                        yield from _executed_nodes(children, shadowed)
                for handler in getattr(statement, "handlers", []) or []:
                    yield from _executed_nodes(handler.body, shadowed)
                # Only the non-suite parts; a `Lambda` is excluded at the seed too.
                pending = [
                    child
                    for child in ast.iter_child_nodes(statement)
                    if not isinstance(
                        child,
                        (
                            ast.stmt,
                            ast.excepthandler,
                            ast.FunctionDef,
                            ast.AsyncFunctionDef,
                            ast.ClassDef,
                            ast.Lambda,
                        ),
                    )
                ]
                yield statement, shadowed
                while pending:
                    current = pending.pop()
                    yield current, shadowed
                    for child in ast.iter_child_nodes(current):
                        if isinstance(
                            child,
                            (
                                ast.FunctionDef,
                                ast.AsyncFunctionDef,
                                ast.ClassDef,
                                ast.Lambda,
                            ),
                        ):
                            continue
                        if isinstance(child, (ast.stmt, ast.excepthandler)):
                            continue
                        pending.append(child)

        # The aliases a newer mapper.py adds live inside `build_mappers`, which the
        # installed builder cannot know.
        def _calls_the_builder(node):
            """Whether this executed node IS the `build_mappers(...)` call.

            The node itself, not `ast.walk` over it: walking the parent statement
            descended back into the deferred children the yield had excluded, so
            `unused = lambda: build_mappers(...)` counted as a call.
            """
            return (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "build_mappers"
            )

        builder_body = []
        # From the EXECUTED nodes, with the statement each runs in: a source table
        # rebound after the call is not what the module exports.
        exported_names = (
            "INT_TO_FLOAT_MAPPER",
            "FLOAT_TO_INT_MAPPER",
            "MAP_TO_UNSLOTH_16bit",
            "FLOAT_TO_FP8_BLOCK_MAPPER",
            "FLOAT_TO_FP8_ROW_MAPPER",
        )

        def _binds_the_exports(statement):
            """Whether this statement assigns one of the exported tables."""
            if isinstance(statement, ast.AnnAssign):
                targets = [statement.target]
            elif isinstance(statement, ast.Assign):
                targets = statement.targets
            else:
                return False
            bound = set()
            for target in targets:
                if isinstance(target, (ast.Tuple, ast.List)):
                    bound |= {
                        element.id for element in target.elts if isinstance(element, ast.Name)
                    }
                elif isinstance(target, ast.Name):
                    bound.add(target.id)
            return bool(bound.intersection(exported_names))

        builder_call = None
        builder_index = None
        first_call = None
        # A call that ASSIGNS the exports replaces all five; the fallback binds nothing.
        builder_rebinds = False
        for index, statement in enumerate(tree.body):
            calls = [
                node for node, _shadowed in _executed_nodes([statement]) if _calls_the_builder(node)
            ]
            if not calls:
                continue
            if first_call is None:
                first_call = (calls[0], index)
            # The LAST call that BINDS the exports; a validation call may run first.
            if _binds_the_exports(statement):
                builder_call, builder_index = calls[0], index
                builder_rebinds = True
        if builder_call is None and first_call is not None:
            builder_call, builder_index = first_call
        builder_called = builder_call is not None
        if builder_called:
            for statement in tree.body:
                if isinstance(statement, ast.FunctionDef) and statement.name == "build_mappers":
                    builder_body = statement.body
                    break

        # The table the builder was HANDED, read where the call runs.
        source_name = "__INT_TO_FLOAT_MAPPER"
        source_table = None
        if builder_call is not None:
            argument = None
            if builder_call.args:
                argument = builder_call.args[0]
            else:
                for keyword in builder_call.keywords:
                    if keyword.arg is not None:
                        argument = keyword.value
                        break
            if isinstance(argument, ast.Name):
                source_name = argument.id
            elif argument is not None:
                try:
                    literal = ast.literal_eval(argument)
                except Exception:
                    literal = None
                if isinstance(literal, dict):
                    source_table = literal

        def _source_before_the_builder(name):
            """The source table as the builder receives it, mutations included.

            `.update({...})` and subscript assignment are ordinary ways to extend it
            before it is handed over; the mutation pass below reads only the EXPORTED
            tables. In execution order, and only up to the call.
            """
            current = None
            for node, _shadowed in _executed_nodes(tree.body):
                # The SELECTED call, by identity, so two alike calls are told apart.
                if node is builder_call or (builder_call is None and _calls_the_builder(node)):
                    break
                if isinstance(node, (ast.Assign, ast.AnnAssign)):
                    if node.value is None:
                        continue
                    targets = [node.target] if isinstance(node, ast.AnnAssign) else node.targets
                    for target in targets:
                        if isinstance(target, ast.Name) and target.id == name:
                            try:
                                literal = ast.literal_eval(node.value)
                            except Exception:
                                continue
                            if isinstance(literal, dict):
                                current = dict(literal)
                        elif (
                            isinstance(target, ast.Subscript)
                            and isinstance(target.value, ast.Name)
                            and target.value.id == name
                            and current is not None
                        ):
                            try:
                                current[ast.literal_eval(target.slice)] = ast.literal_eval(
                                    node.value
                                )
                            except Exception:
                                continue
                elif isinstance(node, ast.Delete):
                    for target in node.targets:
                        if isinstance(target, ast.Name) and target.id == name:
                            # Empty, not None: unbound must not fall back.
                            current = {}
                        elif (
                            isinstance(target, ast.Subscript)
                            and isinstance(target.value, ast.Name)
                            and target.value.id == name
                            and current is not None
                        ):
                            try:
                                current.pop(ast.literal_eval(target.slice), None)
                            except Exception:
                                continue
                elif (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "update"
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == name
                    and current is not None
                    and len(node.args) == 1
                    and not node.keywords
                ):
                    try:
                        additions = ast.literal_eval(node.args[0])
                    except Exception:
                        continue
                    if isinstance(additions, dict):
                        current.update(additions)
            return current

        if source_table is None and builder_call is not None:
            source_table = _source_before_the_builder(source_name)
        if source_table is None:
            source_table = _binding_at(source_name, builder_index)

        # Not the end of the probe: a row-only FP8 repo cannot be expressed through the
        # source table at all. A body that adds nothing is reported so, at the end.
        empty_base = not source_table
        tables = build_mappers(source_table or {})
        # Restored at the call below, which REPLACES all five tables.
        built = [dict(table) for table in tables]

        # Literal subscript, literal value, nothing called.
        by_name = {
            "INT_TO_FLOAT_MAPPER": tables[0],
            "FLOAT_TO_INT_MAPPER": tables[1],
            "MAP_TO_UNSLOTH_16bit": tables[2],
            "FLOAT_TO_FP8_BLOCK_MAPPER": tables[3],
            "FLOAT_TO_FP8_ROW_MAPPER": tables[4],
        }

        # Only the helper CALLS: the builder binds its own `INT_TO_FLOAT_MAPPER = {}`,
        # which the whole-name rule below would read as a clear.
        builder_additions = [
            (node, shadowed)
            for node, shadowed in _executed_nodes(builder_body)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in _MAPPER_HELPERS
        ]
        # In EXECUTION order, so a rebind after the call still empties the table.
        # `rebuilt` stands where the build assigns the exports; matched by identity.
        rebuilt = object()
        ordered = []
        for node, shadowed in _executed_nodes(tree.body):
            ordered.append((node, shadowed))
            # At the call that populates the EXPORTS, not at an earlier one.
            if node is builder_call or (builder_call is None and _calls_the_builder(node)):
                if builder_rebinds:
                    ordered.append((rebuilt, frozenset()))
                ordered.extend(builder_additions)
                builder_additions = []
        ordered.extend(builder_additions)
        for node, shadowed in ordered:
            if node is rebuilt:
                for table, original in zip(tables, built):
                    table.clear()
                    table.update(original)
            elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                # An annotated subscript assigns as the plain form does; a bare one
                # binds nothing.
                if isinstance(node, ast.AnnAssign):
                    if node.value is None:
                        continue
                    targets = [node.target]
                else:
                    targets = node.targets
                for target in targets:
                    if isinstance(target, ast.Name):
                        # A whole-name assignment REPLACES the exported table.
                        table = by_name.get(target.id)
                        if table is None or target.id in shadowed:
                            continue
                        try:
                            replacement = ast.literal_eval(node.value)
                        except ValueError:
                            continue
                        if isinstance(replacement, dict):
                            if not replacement and not builder_called:
                                # An INITIALISER, not a clear: a mapper.py with no
                                # `build_mappers` writes `X = {}` and fills all five
                                # from a module-scope loop the installed builder
                                # reproduces. Reading it as a clear emptied every
                                # table and the upgrade notice stopped firing.
                                continue
                            table.clear()
                            table.update(replacement)
                        continue
                    if not isinstance(target, ast.Subscript):
                        continue
                    if not isinstance(target.value, ast.Name):
                        continue
                    if target.value.id in shadowed:
                        continue
                    table = by_name.get(target.value.id)
                    if table is None:
                        continue
                    try:
                        table[ast.literal_eval(target.slice)] = ast.literal_eval(node.value)
                    except ValueError:
                        continue
                pass
            elif isinstance(node, ast.Delete):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        table = by_name.get(target.id)
                        if table is not None and target.id not in shadowed:
                            table.clear()
                        continue
                    if not isinstance(target, ast.Subscript):
                        continue
                    if not isinstance(target.value, ast.Name):
                        continue
                    if target.value.id in shadowed:
                        continue
                    table = by_name.get(target.value.id)
                    if table is None:
                        continue
                    try:
                        table.pop(ast.literal_eval(target.slice), None)
                    except ValueError:
                        continue
            elif (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "update"
            ):
                # Only a named receiver and a literal mapping; the keyword form cannot
                # express keys with a slash.
                if not isinstance(node.func.value, ast.Name):
                    continue
                if node.func.value.id in shadowed:
                    continue
                table = by_name.get(node.func.value.id)
                if table is None or len(node.args) != 1 or node.keywords:
                    continue
                try:
                    additions = ast.literal_eval(node.args[0])
                except ValueError:
                    continue
                if isinstance(additions, dict):
                    table.update(additions)
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                # An alias not derivable from the source table, applied with the
                # INSTALLED helper from literal arguments only.
                helper = _MAPPER_HELPERS.get(node.func.id)
                if helper is None:
                    continue
                supplied = dict(zip(("mapper", "key", "value"), node.args))
                for keyword in node.keywords:
                    if keyword.arg not in ("mapper", "key", "value"):
                        supplied = None
                        break
                    if keyword.arg in supplied:
                        supplied = None
                        break
                    supplied[keyword.arg] = keyword.value
                if supplied is None or set(supplied) != {"mapper", "key", "value"}:
                    continue
                destination = supplied["mapper"]
                key, value = supplied["key"], supplied["value"]
                if not isinstance(destination, ast.Name):
                    continue
                if destination.id in shadowed:
                    continue
                table = by_name.get(destination.id)
                if table is None:
                    continue
                try:
                    helper(table, ast.literal_eval(key), ast.literal_eval(value))
                except ValueError:
                    continue
            pass
        pass
        if empty_base and tables == build_mappers({}):
            # Nothing on top of the empty base, so the body carried no mapping at all.
            return {}, {}, {}, {}, {}
        return tables
    except:
        return {}, {}, {}, {}, {}


def _resolve_with_mappers(
    model_name,
    load_in_4bit,
    load_in_fp8,
    int_to_float,
    float_to_int,
    map_to_unsloth_16bit,
    fp8_block = None,
    fp8_row = None,
):
    # The probe passes the FETCHED fp8 tables, without rebinding the installed ones.
    return __get_model_name(
        model_name = model_name,
        load_in_4bit = load_in_4bit,
        INT_TO_FLOAT_MAPPER = int_to_float,
        FLOAT_TO_INT_MAPPER = float_to_int,
        MAP_TO_UNSLOTH_16bit = map_to_unsloth_16bit,
        load_in_fp8 = load_in_fp8,
        FLOAT_TO_FP8_BLOCK_MAPPER = FLOAT_TO_FP8_BLOCK_MAPPER if fp8_block is None else fp8_block,
        FLOAT_TO_FP8_ROW_MAPPER = FLOAT_TO_FP8_ROW_MAPPER if fp8_row is None else fp8_row,
    )


def get_model_name(
    model_name,
    load_in_4bit = True,
    load_in_fp8 = False,
    token = None,
    trust_remote_code = False,
):
    assert load_in_fp8 in (True, False, "block")
    new_model_name = _resolve_with_mappers(
        model_name = model_name,
        load_in_4bit = load_in_4bit,
        load_in_fp8 = load_in_fp8,
        int_to_float = INT_TO_FLOAT_MAPPER,
        float_to_int = FLOAT_TO_INT_MAPPER,
        map_to_unsloth_16bit = MAP_TO_UNSLOTH_16bit,
    )
    # Remap "bad" names (e.g. oversized dynamic quants or MoEs)
    if (
        new_model_name is not None
        and type(new_model_name) is str
        and new_model_name.lower() in BAD_MAPPINGS
    ):
        new_model_name = BAD_MAPPINGS[new_model_name.lower()]
    elif new_model_name is None and model_name.lower() in BAD_MAPPINGS:
        # Some bad names (e.g. the `-unsloth-bnb-4bit` dynamic quants) are keys
        # of the mappers, not values, so the resolver returns None for them and
        # the remap above is skipped; remap the input name directly instead.
        new_model_name = BAD_MAPPINGS[model_name.lower()]

    if (
        new_model_name is None
        and model_name.count("/") == 1
        and model_name[0].isalnum()
        and not _env_says_offline()  # offline: skip the remote (raw GitHub) mapper refresh
    ):
        # Try checking if a new Unsloth version allows it!
        (
            NEW_INT_TO_FLOAT_MAPPER,
            NEW_FLOAT_TO_INT_MAPPER,
            NEW_MAP_TO_UNSLOTH_16bit,
            NEW_FP8_BLOCK_MAPPER,
            NEW_FP8_ROW_MAPPER,
        ) = _get_new_mapper()
        upgraded_model_name = _resolve_with_mappers(
            model_name = model_name,
            load_in_4bit = load_in_4bit,
            load_in_fp8 = load_in_fp8,
            int_to_float = NEW_INT_TO_FLOAT_MAPPER,
            float_to_int = NEW_FLOAT_TO_INT_MAPPER,
            map_to_unsloth_16bit = NEW_MAP_TO_UNSLOTH_16bit,
            # the fp8 probe has to look at the FETCHED tables too, or a new fp8 repo would
            # miss both here and in the installed tables and skip the upgrade message
            fp8_block = NEW_FP8_BLOCK_MAPPER,
            fp8_row = NEW_FP8_ROW_MAPPER,
        )
        if upgraded_model_name is not None:
            raise NotImplementedError(
                f"Unsloth: {model_name} is not supported in your current Unsloth version! Please update Unsloth via:\n\n"
                "pip uninstall unsloth unsloth_zoo -y\n"
                'pip install --upgrade --no-cache-dir "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"\n'
                'pip install --upgrade --no-cache-dir "git+https://github.com/unslothai/unsloth-zoo.git"\n'
            )

    if new_model_name is None:
        new_model_name = model_name

    return new_model_name


def _offline_quantize_to_fp8(
    model_name: str,
    fp8_mode: str,
    *,
    text_only: bool = False,
    revision: str = None,
) -> str:
    """Quantize the model to fp8 via torchao, save to a temp dir, return its path.

    For vllm >= 0.12.0, prefer dynamic quantization in vllm instead (via
    hf_overrides={"quantization_config_file": "torchao_config.json"}).

    The caller's revision has to reach the source loads, and the cache name has to name it
    too: the returned path replaces model_name, so the revision gate downstream drops the
    pin, and two refs of one repo would otherwise share (and reuse) a single artifact.
    """
    from transformers import (
        AutoModelForCausalLM,
        AutoModelForImageTextToText,
        AutoTokenizer,
        AutoProcessor,
        TorchAoConfig,
        AutoConfig,
    )

    config = AutoConfig.from_pretrained(model_name, revision = revision)
    is_vlm = any(
        x.endswith(("ForConditionalGeneration", "ForVisionText2Text"))
        for x in (getattr(config, "architectures", None) or [])
    )
    is_vlm = is_vlm or hasattr(config, "vision_config")
    # Decide text-only before the cache name so the fp8 artifact and its path stay in sync. #5816
    text_config = None
    if text_only and hasattr(config, "vision_config"):
        from ._utils import (
            _get_text_only_config,
            resolve_model_class,
            _is_family_text_decoder,
        )

        candidate = _get_text_only_config(config, model_name)
        text_class = resolve_model_class(AutoModelForCausalLM, candidate)
        if text_class is not None and _is_family_text_decoder(
            getattr(config, "model_type", ""),
            getattr(candidate, "model_type", ""),
        ):
            text_config = candidate
            is_vlm = False

    temp_dir = tempfile.gettempdir()
    # Cache text-only and full-VLM artifacts separately so neither reuses the other. #5816
    cache_name = model_name.split("/")[-1] + "-fp8-" + fp8_mode
    if revision is not None:
        # Sanitizing is lossy (`release/v1` and `release.v1` collapse), so a digest of the
        # raw ref rides along and two refs never share an artifact.
        digest = hashlib.sha256(revision.encode("utf-8")).hexdigest()[:12]
        readable = re.sub(r"[^0-9A-Za-z_-]", "_", revision)[:40]
        cache_name += "-rev-" + readable + "-" + digest
    if text_config is not None:
        cache_name += "-text-only"
    new_model_name = os.path.join(temp_dir, cache_name)
    print(f"Unsloth: Quantizing '{model_name}' to fp8, using model_name='{new_model_name}' instead")

    if not os.path.isdir(new_model_name):
        from ._utils import _apply_text_only_key_mapping

        qconfig = _get_torchao_fp8_config(fp8_mode)
        qconfig = TorchAoConfig(qconfig)
        load_kwargs = dict(torch_dtype = "auto", device_map = "auto", quantization_config = qconfig)
        if text_config is not None:
            _apply_text_only_key_mapping(load_kwargs, config, text_config)
            config = text_config
        auto_model = AutoModelForImageTextToText if is_vlm else AutoModelForCausalLM
        auto_processor = AutoProcessor if is_vlm else AutoTokenizer
        model = auto_model.from_pretrained(
            model_name,
            config = config,
            revision = revision,
            **load_kwargs,
        )
        tokenizer = auto_processor.from_pretrained(model_name, revision = revision)
        model.save_pretrained(new_model_name, safe_serialization = False)
        del model
        for _ in range(2):
            torch.cuda.empty_cache()
            gc.collect()
        tokenizer.save_pretrained(new_model_name)
    return new_model_name


def _tag_model_with_fp8_torchao_config(model: torch.nn.Module, fp8_mode: str):
    """Tag a model with a `TorchAOConfig` so downstream callers know how to handle it."""
    try:
        base_config = _get_torchao_fp8_config(fp8_mode)
        model.torchao_config = TorchAOConfig(
            qat_scheme = None,
            base_config_and_filter_fns = [(base_config, None)],
        )
    except:
        pass


_FP8_DTYPES = tuple(
    dtype
    for dtype in (getattr(torch, "float8_e4m3fn", None), getattr(torch, "float8_e5m2", None))
    if dtype is not None
)


def _fp8_block_size_from_config(model):
    """Return the [block_out, block_in] block size of an fp8 checkpoint, or None if not block-fp8."""
    config = getattr(model, "config", None)
    quant = getattr(config, "quantization_config", None)
    if quant is None:
        return None
    if hasattr(quant, "to_dict"):
        quant = quant.to_dict()
    if not isinstance(quant, dict):
        return None
    if quant.get("quant_method") != "fp8":
        return None
    block = quant.get("weight_block_size")
    if not block:
        return None
    if isinstance(block, (int, float)):
        block = [block, block]
    elif isinstance(block, (list, tuple)):
        if len(block) == 1:
            block = [block[0], block[0]]
        elif len(block) < 2:
            return None
    else:
        return None
    return [int(block[0]), int(block[1])]


def _load_fp8_weight_map(
    model_name,
    local_files_only,
    token,
    revision = None,
    subfolder = None,
    cache_dir = None,
):
    """Return the checkpoint's tensor->file map, using the same snapshot the load used.

    Prefers the sharded `model.safetensors.index.json`; falls back to a single `model.safetensors`
    (every tensor maps to that one file) so unsharded checkpoints are covered too.
    """

    def _local_path(filename):
        return (
            os.path.join(model_name, subfolder, filename)
            if subfolder
            else os.path.join(model_name, filename)
        )

    def _remote_path(filename):
        from huggingface_hub import hf_hub_download
        return hf_hub_download(
            model_name,
            filename,
            revision = revision,
            subfolder = subfolder,
            cache_dir = cache_dir,
            local_files_only = local_files_only,
            token = token,
        )

    index_file = "model.safetensors.index.json"
    single_file = "model.safetensors"
    is_local = os.path.isdir(model_name)

    # Sharded checkpoint.
    if is_local and os.path.exists(_local_path(index_file)):
        index_path = _local_path(index_file)
    elif not is_local:
        try:
            index_path = _remote_path(index_file)
        except Exception:
            index_path = None
    else:
        index_path = None
    if index_path is not None:
        import json
        with open(index_path, "r", encoding = "utf-8") as f:
            return json.load(f).get("weight_map", None)

    # Unsharded single file: map every tensor to it.
    try:
        if is_local and os.path.exists(_local_path(single_file)):
            single_path = _local_path(single_file)
        elif not is_local:
            single_path = _remote_path(single_file)
        else:
            return None
        from safetensors import safe_open
        with safe_open(single_path, framework = "pt") as f:
            return {key: single_file for key in f.keys()}
    except Exception:
        return None


def _resolve_fp8_shard(
    model_name,
    shard,
    local_files_only,
    token,
    revision = None,
    subfolder = None,
    cache_dir = None,
):
    """Resolve a checkpoint shard filename to a local path (repo id or local dir)."""
    if os.path.isdir(model_name):
        return (
            os.path.join(model_name, subfolder, shard)
            if subfolder
            else os.path.join(model_name, shard)
        )
    from huggingface_hub import hf_hub_download

    return hf_hub_download(
        model_name,
        shard,
        revision = revision,
        subfolder = subfolder,
        cache_dir = cache_dir,
        local_files_only = local_files_only,
        token = token,
    )


def _match_fp8_module(module_by_name, base):
    """Resolve a checkpoint module name to a live module, allowing for VLM key remappings.

    VLM loads can name the text tower differently from the checkpoint keys: `text_only=True`
    strips the `language_model.` wrapper (so `model.language_model.layers.*` -> `model.layers.*`),
    and full VLM loads may expose `model.language_model.*` while the checkpoint stores
    `language_model.model.*`. Try the raw key first, then a few safe remappings.
    """
    if base in module_by_name:
        return module_by_name[base]
    candidates = []
    if "language_model." in base:
        candidates.append(base.replace("language_model.", "", 1))  # text-only: drop wrapper
    if "language_model.model." in base:
        candidates.append(base.replace("language_model.model.", "model.language_model.", 1))
    if base.startswith("language_model."):
        candidates.append("model." + base)  # add model. prefix
    for candidate in candidates:
        if candidate in module_by_name:
            return module_by_name[candidate]
    return None


def _restore_dropped_fp8_scales(
    model,
    model_name,
    *,
    local_files_only = False,
    token = None,
    revision = None,
    subfolder = None,
    cache_dir = None,
    variant = None,
):
    """Re-apply block-fp8 `weight_scale_inv` tensors that transformers dropped on load.

    On some block-scale fp8 checkpoints (e.g. Qwen3.6-27B-FP8, issue #6200) transformers fails to
    convert a Linear (such as `mlp.gate_proj`) to an fp8 module, loading the raw quantized values
    into a plain bf16 weight and discarding its `weight_scale_inv` as an unexpected key. The weight
    is then used un-scaled, producing a garbage model. For every checkpoint scale whose live weight
    is not fp8, dequantize the orphaned weight in place. Modules that were converted correctly keep
    an fp8 weight and are skipped, so a healthy checkpoint is a no-op. Returns (restored, skipped).
    """
    try:
        block = _fp8_block_size_from_config(model)
        if block is None or not _FP8_DTYPES:
            return (0, 0)
        # A variant load reads variant-named files; skip to avoid applying default scales to them.
        if variant:
            return (0, 0)
        # No fp8 params means the checkpoint was dequantized on purpose (e.g. load_in_16bit);
        # re-applying a scale would corrupt those already-correct 16bit weights, so do nothing.
        if not any(p.dtype in _FP8_DTYPES for p in model.parameters()):
            return (0, 0)
        weight_map = _load_fp8_weight_map(
            model_name, local_files_only, token, revision, subfolder, cache_dir
        )
        if not weight_map:
            return (0, 0)

        scale_keys = {k: v for k, v in weight_map.items() if k.endswith(".weight_scale_inv")}
        if not scale_keys:
            return (0, 0)

        module_by_name = dict(model.named_modules())
        bs0, bs1 = block
        restored = 0
        skipped = 0
        failed = 0
        offloaded = 0
        shard_cache = {}
        for scale_key, shard in scale_keys.items():
            base = scale_key[: -len(".weight_scale_inv")]
            module = _match_fp8_module(module_by_name, base)
            if module is None:
                continue
            weight = getattr(module, "weight", None)
            if not isinstance(weight, torch.Tensor) or weight.ndim != 2:
                continue
            if weight.device.type == "meta":
                # Disk-offloaded layer: weight lives on meta until forward, so it cannot be
                # scaled in place here. Count and warn rather than silently leave it unscaled.
                offloaded += 1
                continue
            if weight.dtype in _FP8_DTYPES:
                # Correctly converted fp8 module: the fp8 path already handles the scale.
                skipped += 1
                continue

            # Errors after this point are per-tensor: warn and continue, never abort or hide them.
            try:
                if shard not in shard_cache:
                    from safetensors import safe_open
                    shard_path = _resolve_fp8_shard(
                        model_name,
                        shard,
                        local_files_only,
                        token,
                        revision,
                        subfolder,
                        cache_dir,
                    )
                    shard_cache[shard] = safe_open(shard_path, framework = "pt")
                scale = shard_cache[shard].get_tensor(scale_key).to(torch.float32)

                out_features, in_features = weight.shape
                out_blocks = (out_features + bs0 - 1) // bs0
                in_blocks = (in_features + bs1 - 1) // bs1
                if tuple(scale.shape) == (out_blocks, in_blocks):
                    pass
                elif tuple(scale.shape) == (in_blocks, out_blocks) and out_blocks != in_blocks:
                    # Transposed block layout: same handling as the fp8 forward path.
                    scale = scale.t().contiguous()
                else:
                    # Shape does not match the block grid: skip rather than apply a wrong scale.
                    continue
                scale = scale.to(weight.device)
                with torch.no_grad():
                    if out_features % bs0 == 0 and in_features % bs1 == 0:
                        # Memory-frugal path: multiply block views in place against the broadcast
                        # fp32 scale, avoiding a full expanded scale and fp32 copy that could OOM.
                        # The in-place multiply promotes to fp32, matching the fallback exactly.
                        module.weight.data.view(out_blocks, bs0, in_blocks, bs1).mul_(
                            scale[:, None, :, None]
                        )
                    else:
                        scale_expanded = scale.repeat_interleave(bs0, dim = 0).repeat_interleave(
                            bs1, dim = 1
                        )[:out_features, :in_features]
                        module.weight.data = (weight.to(torch.float32) * scale_expanded).to(
                            weight.dtype
                        )
                restored += 1
            except Exception:
                failed += 1
                continue

        if restored > 0:
            print(f"Unsloth: Restored {restored} dropped FP8 weight_scale_inv tensor(s) on load")
        if failed > 0:
            print(f"Unsloth: {failed} dropped FP8 weight_scale_inv tensor(s) could not be restored")
        if offloaded > 0:
            print(
                f"Unsloth: {offloaded} dropped FP8 weight_scale_inv tensor(s) skipped because the "
                "layer is disk-offloaded; load without disk offload so the scales can be restored"
            )
        return (restored, skipped)
    except Exception:
        return (0, 0)


def check_and_disable_bitsandbytes_loading(
    model_config,
    load_in_4bit = True,
    load_in_8bit = False,
    verbose = True,
):
    """
    Check if we should disable bitsandbytes loading (load_in_4bit/load_in_8bit)
    because the model already has a non-bitsandbytes quantization config.
    If so, disable BOTH 4bit and 8bit loading and print a warning message.

    Args:
        model_config: The AutoConfig object from the model
        load_in_4bit: Whether load_in_4bit is currently enabled
        load_in_8bit: Whether load_in_8bit is currently enabled
        verbose: Whether to print warning messages

    Returns:
        tuple: (load_in_4bit, load_in_8bit, quant_method)
            load_in_4bit/load_in_8bit will be False if they were disabled
            quant_method is the detected quantization method or None
    """
    quant_method = get_quant_type(model_config)

    if quant_method is None or quant_method == "bitsandbytes":
        return load_in_4bit, load_in_8bit, quant_method

    # Model has a non-bitsandbytes quantization config (e.g., compressed-tensors, gptq, awq)
    # We should disable BOTH bitsandbytes loading to avoid config conflicts
    if load_in_4bit or load_in_8bit:
        if verbose:
            print(
                f"Unsloth: Model already quantized with {quant_method}. "
                f"Disabling `load_in_4bit` and `load_in_8bit` to avoid quantization config conflict."
            )
        load_in_4bit = False
        load_in_8bit = False

    return load_in_4bit, load_in_8bit, quant_method


def sync_unsloth_model_name_bnb_flags(load_in_4bit, load_in_8bit):
    """Make UNSLOTH_MODEL_NAME's `_load_in_4bit_`/`_load_in_8bit_` tokens match the EFFECTIVE bnb
    state (after get_model_name remap + check_and_disable). The per-load env is built from the
    pre-remap config (None for adapter-only PEFT repos), so its tokens can be wrong once the base
    resolves. Only the gpt-oss patch reads them, so this is gated to gpt-oss; no-op otherwise."""
    name = os.environ.get("UNSLOTH_MODEL_NAME", "")
    if "gpt_oss" not in name.replace("-", "_"):
        return
    for flag, present in (
        ("_load_in_4bit_", bool(load_in_4bit)),
        ("_load_in_8bit_", bool(load_in_8bit)),
    ):
        if present and flag not in name:
            name += flag
        elif not present and flag in name:
            name = name.replace(flag, "")
    os.environ["UNSLOTH_MODEL_NAME"] = name


def _get_fp8_mode_and_check_settings(
    load_in_fp8: Union[bool, str],
    fast_inference: bool,
    full_finetuning: bool = False,
    load_in_4bit: bool = False,
    load_in_8bit: bool = False,
    load_in_16bit: bool = False,
) -> str:
    """Validate `load_in_fp8` settings/environment and return the fp8 mode
    ("row" or "block"). Requires H100+, torchao 0.15.0+, torch 2.9.0+, and
    fbgemm_gpu_genai 1.4.1+ if installed.
    """
    assert load_in_fp8 is not False
    if load_in_fp8 is True:
        fp8_mode = "row"  # default
    else:
        fp8_mode = load_in_fp8

    # Check user settings
    if fp8_mode not in ["row", "block"]:
        raise ValueError(f"Unsloth: `load_in_fp8` can only be 'row' or 'block', got '{fp8_mode}'")
    if full_finetuning:
        raise ValueError("Unsloth: `load_in_fp8` is not compatible with full finetuning")
    if load_in_4bit or load_in_8bit or load_in_16bit:
        raise ValueError(
            "Unsloth: `load_in_fp8` is not compatible with `load_in_4bit`, `load_in_8bit` or `load_in_16bit`",
        )

    # Check if this is Hopper or above
    if not (
        torch.cuda.is_available()
        and torch.version.cuda
        and torch.cuda.get_device_capability() >= (9, 0)
    ):
        raise ValueError(
            "Unsloth: On the fly `load_in_fp8` requires H100 GPUs or after. Try `unsloth/Qwen3-8B` instead."
        )

    # Check if torch >= 2.9.0
    if Version(torch.__version__) < Version("2.9.0"):
        raise ValueError(
            "Unsloth: On the fly `load_in_fp8` requires torch 2.9.0+. Try `unsloth/Qwen3-8B` instead."
        )

    # Check if torchao has this PR: https://github.com/pytorch/ao/pull/3158,
    # which will be released in 0.15.0.
    if importlib.util.find_spec("torchao") is None:
        raise ValueError(
            "Unsloth: Please install torchao for on the fly float8 to work! Try `unsloth/Qwen3-8B` instead."
        )
    import torchao

    error_message = (
        "Unsloth: `load_in_fp8` requires torchao 0.15.0+ (or nightly).\n"
        f"You have torchao version={torchao.__version__}\n"
        "Use `pip install --upgrade --force-reinstall torchao`"
    )
    if Version(torchao.__version__) < Version("0.15.0"):
        raise ValueError(error_message)

    # If fbgemm_gpu_genai is installed and old, disable FBGEMM and use Triton instead
    if (
        importlib.util.find_spec("fbgemm_gpu") is not None
        and importlib.util.find_spec("fbgemm_gpu.experimental") is not None
    ):
        import fbgemm_gpu.experimental.gen_ai
        if Version(fbgemm_gpu.__version__) < Version("1.4.1"):
            # Old FBGEMM version - disable and use Triton kernels instead
            os.environ["UNSLOTH_HAS_FBGEMM"] = "0"
            from unsloth_zoo.log import logger
            logger.info(
                f"Unsloth: fbgemm_gpu_genai=={fbgemm_gpu.__version__} is old for FP8 loading. "
                f"Using Triton kernels instead."
            )
    return fp8_mode


# Rotary inv_freq buffers are deliberately kept on CPU - Unsloth pre-builds a
# cos/sin cache per GPU instead (see LlamaRotaryEmbedding.multi_gpu_cos_cached)
# so the GPU-resident lookup never needs to move the tiny inv_freq tensor itself.
# torch.nn.parallel.DistributedDataParallel ignores device entirely when it
# broadcasts buffers across ranks, so a CPU buffer crashes NCCL's
# _broadcast_coalesced with "No backend type associated with device type cpu".
# Telling DDP to skip these specific buffers avoids that crash without moving
# inv_freq to GPU (which would break the per-GPU cache design) and without
# disabling buffer broadcast for every other module (the user's workaround).
# Re-run this after wrapping with PEFT too - the buffers' fully qualified
# names change once they sit under a PeftModel (eg "base_model.model...").
# https://github.com/unslothai/unsloth/issues/6656
_ROTARY_INV_FREQ_BUFFER_NAMES = ("inv_freq", "short_inv_freq", "long_inv_freq")


def _exclude_rope_inv_freq_from_ddp(model):
    ignored = list(getattr(model, "_ddp_params_and_buffers_to_ignore", None) or [])
    for module_name, module in model.named_modules():
        for buffer_name, _ in module.named_buffers(recurse = False):
            if buffer_name in _ROTARY_INV_FREQ_BUFFER_NAMES:
                fqn = f"{module_name}.{buffer_name}" if module_name else buffer_name
                if fqn not in ignored:
                    ignored.append(fqn)
    if ignored:
        try:
            from torch.nn.parallel import DistributedDataParallel
            DistributedDataParallel._set_params_and_buffers_to_ignore_for_model(model, ignored)
        except Exception:
            # Private PyTorch API - fall back to setting the attribute DDP reads
            # directly if it ever moves or changes signature.
            model._ddp_params_and_buffers_to_ignore = ignored
    return model


# =============================================================================
# Offline loading - single source of truth (shared by vision.py, loader.py and
# the Unsloth exporter). Decide offline ONCE at the load boundary and force it
# ONCE around the whole load, so every nested HF call inherits it.
# =============================================================================

_OFFLINE_ENV_VALUES = {"1", "true", "yes", "on"}
_OFFLINE_ENV_KEYS = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")


def _env_says_offline():
    """True if an HF offline env var is set to a truthy value."""
    return any(
        os.environ.get(_k, "").strip().lower() in _OFFLINE_ENV_VALUES for _k in _OFFLINE_ENV_KEYS
    )


def _get_effective_local_files_only(kwargs):
    """Offline if local_files_only is truthy or an HF offline env var is set. Read-only."""
    if kwargs.get("local_files_only", None):
        return True
    return _env_says_offline()


# Attribute stamped on a tokenizer/processor that was loaded local-only, so a later
# save still knows. transformers takes local_files_only as an explicit from_pretrained
# parameter and never copies it into tokenizer.init_kwargs, and _offline_aware_load
# restores the offline env vars when the load window closes, so without this stamp an
# explicit local_files_only = True load is invisible by the time we save (issue #7481).
_LOCAL_FILES_ONLY_ATTR = "_unsloth_local_files_only"
# The load's cache_dir travels with it too: saving derives one from HF_HUB_CACHE /
# HF_HOME, which does not see a caller-supplied cache.
_LOADED_CACHE_DIR_ATTR = "_unsloth_loaded_cache_dir"
# So does the ref it was read at: saving restores sentencepiece assets from
# tokenizer.name_or_path, which names the repo but not the branch.
_LOADED_REVISION_ATTR = "_unsloth_loaded_revision"


def _mark_loaded_revision(result, revision):
    """Stamp the ref a tokenizer/processor was loaded at onto the returned objects."""
    if revision is None:
        return result
    for obj in result if isinstance(result, (tuple, list)) else (result,):
        try:
            targets = (obj, getattr(obj, "tokenizer", None))
        except Exception:
            targets = (obj,)
        for target in targets:
            if target is None:
                continue
            # Skip objects that reject new attributes (__slots__).
            try:
                setattr(target, _LOADED_REVISION_ATTR, str(revision))
            except Exception:
                pass
    return result


def _tokenizer_revision(tokenizer):
    """The ref this tokenizer was loaded at, or None for the default branch."""
    tokenizer = tokenizer.tokenizer if hasattr(tokenizer, "tokenizer") else tokenizer
    return getattr(tokenizer, _LOADED_REVISION_ATTR, None)


def _mark_loaded_local_files_only(result, cache_dir = None):
    """Stamp a load's local-only mode and cache_dir onto the returned objects."""
    for obj in result if isinstance(result, (tuple, list)) else (result,):
        try:
            # A processor keeps the tokenizer that _has_tokenizer_model unwraps to,
            # so stamp both (a wrapped model can raise from its own __getattr__).
            targets = (obj, getattr(obj, "tokenizer", None))
        except Exception:
            targets = (obj,)
        for target in targets:
            if target is None:
                continue
            # Objects that reject new attributes (__slots__) are skipped.
            try:
                setattr(target, _LOCAL_FILES_ONLY_ATTR, True)
                if cache_dir:
                    setattr(target, _LOADED_CACHE_DIR_ATTR, str(cache_dir))
            except Exception:
                pass
    return result


def _tokenizer_cache_dir(tokenizer):
    """The cache_dir the load used, when it was not the environment's."""
    tokenizer = tokenizer.tokenizer if hasattr(tokenizer, "tokenizer") else tokenizer
    return getattr(tokenizer, _LOADED_CACHE_DIR_ATTR, None)


def _tokenizer_wants_local_only(tokenizer):
    """True when Hub metadata probes should be skipped for this tokenizer."""
    if _env_says_offline():
        return True
    if getattr(tokenizer, _LOCAL_FILES_ONLY_ATTR, False):
        return True
    init_kwargs = getattr(tokenizer, "init_kwargs", None) or {}
    return bool(init_kwargs.get("local_files_only"))


def _is_offline_related_error(exc):
    """True if exc (or its cause/context chain) is a lost-connection error, not a
    missing file. Plain FileNotFoundError propagates; LocalEntryNotFoundError is offline."""
    import socket
    import ssl
    import urllib.error

    # Match network failures by type (locale independent), not just message wording.
    _net_types = [ConnectionError, TimeoutError, socket.gaierror, urllib.error.URLError]
    _offline_fnf_types = ()  # FileNotFoundError subclasses that count as offline
    # urllib HTTPError is a URLError subclass: judge by status (5xx offline, 4xx propagates).
    _http_types = (urllib.error.HTTPError,)
    # TLS/cert failures are security-sensitive (MITM, expired CA): never offline-retry them.
    _ssl_types = [ssl.SSLError]
    try:
        import requests

        _net_types += [requests.exceptions.ConnectionError, requests.exceptions.Timeout]
        _http_types += (requests.exceptions.HTTPError,)
        _ssl_types.append(requests.exceptions.SSLError)
    except Exception:
        pass
    try:
        from huggingface_hub.errors import (
            OfflineModeIsEnabled,
            HfHubHTTPError,
            LocalEntryNotFoundError,
        )

        _net_types += [OfflineModeIsEnabled, LocalEntryNotFoundError]
        _offline_fnf_types = (LocalEntryNotFoundError,)
        _http_types += (HfHubHTTPError,)
    except Exception:
        pass
    _net_types = tuple(_net_types)
    _ssl_types = tuple(_ssl_types)

    def _http_status(e):
        resp = getattr(e, "response", None)
        code = getattr(resp, "status_code", None)
        if code is None:
            code = getattr(e, "status_code", None)
        if code is None:
            code = getattr(e, "code", None)  # urllib.error.HTTPError uses .code
        try:
            return int(code)
        except (TypeError, ValueError):
            return None

    _wording = (
        "couldn't connect",
        "could not connect",
        "connection error",
        "connectionerror",
        "max retries",
        "offline",
        "timed out",
        "timeout",
        "couldn't reach",
        "could not reach",
        "failed to resolve",
        "getaddrinfo",
        "name resolution",
        "no address associated",
        "network is unreachable",
        "connection refused",
        "we couldn't connect to",
        "proxyerror",
        # Raw socket.gaierror DNS wording (Linux / macOS)
        "name or service not known",
        "temporary failure in name resolution",
        "nodename nor servname provided",
    )
    seen = set()
    cur = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        # TLS/cert failure (corporate MITM, expired CA): security-sensitive, never retry from
        # cache. Skip this node; a deeper cause in the chain may still be a genuine outage.
        if isinstance(cur, _ssl_types) or isinstance(getattr(cur, "reason", None), _ssl_types):
            cur = cur.__cause__ or cur.__context__
            continue
        is_fnf = isinstance(cur, FileNotFoundError) and not isinstance(cur, _offline_fnf_types)
        # urllib HTTPError is a URLError (net type) but must be judged by status code below,
        # unlike LocalEntryNotFoundError (an HfHubHTTPError that is always offline).
        if (
            isinstance(cur, _net_types)
            and not is_fnf
            and not isinstance(cur, urllib.error.HTTPError)
        ):
            return True
        if isinstance(cur, _http_types):
            code = _http_status(cur)
            if code is not None and 500 <= code < 600:
                return True
            # No status -> wording fallback (coded 4xx already decided above).
            if code is None and not is_fnf and any(w in str(cur).lower() for w in _wording):
                return True
        # OSError wording fallback (HTTP status already decided above).
        elif isinstance(cur, OSError) and not is_fnf:
            if any(w in str(cur).lower() for w in _wording):
                return True
        cur = cur.__cause__ or cur.__context__
    return False


# Process-wide HF offline state; the depth counter lets nested windows share one
# flip (first entrant saves originals, last exit restores). Lock guards flip/restore.
_force_offline_lock = _threading.RLock()
_force_offline_depth = 0
_force_offline_saved = []  # in-process module attributes
_force_offline_saved_env = {}  # HF offline env-var originals


def _reset_hf_sessions():
    """Clear hub's per-thread cached Sessions so the next rebuilds against the current
    offline flag. On hub 0.x the offline adapter is baked in at Session creation. Best-effort."""
    try:
        from huggingface_hub.utils._http import reset_sessions
    except Exception:
        try:
            from huggingface_hub.utils import reset_sessions
        except Exception:
            return
    try:
        reset_sessions()
    except Exception:
        pass


@contextlib.contextmanager
def _force_hf_offline():
    """Force HF offline for the window. local_files_only alone is not enough
    (transformers < 5 still pings /api/models), so set BOTH the env vars (cover
    subprocesses + raw urllib/requests) AND the in-process hub/transformers constants.
    Process-global; the refcount keeps restore correct under nesting / overlap."""
    global _force_offline_depth, _force_offline_saved, _force_offline_saved_env
    with _force_offline_lock:
        if _force_offline_depth == 0:
            saved = []
            saved_env = {}
            # Snapshot in-process constants BEFORE forcing the env: a module first imported
            # here would otherwise initialize its constant from the just-set "1" and we would
            # save (then restore) True, pinning the process offline after the window.
            try:
                import huggingface_hub.constants as _hfc
                if hasattr(_hfc, "HF_HUB_OFFLINE"):
                    saved.append((_hfc, "HF_HUB_OFFLINE", _hfc.HF_HUB_OFFLINE))
            except Exception:
                pass
            try:
                import transformers.utils.hub as _tuh
                for _attr in ("_is_offline_mode", "OFFLINE"):
                    if hasattr(_tuh, _attr):
                        saved.append((_tuh, _attr, getattr(_tuh, _attr)))
            except Exception:
                pass
            # Now force the env vars and flip the snapshotted constants to offline.
            for _k in _OFFLINE_ENV_KEYS:
                saved_env[_k] = os.environ.get(_k)
                os.environ[_k] = "1"
            for _obj, _attr, _ in saved:
                try:
                    setattr(_obj, _attr, True)
                except Exception:
                    pass
            _force_offline_saved = saved
            _force_offline_saved_env = saved_env
            # Rebuild cached sessions so they pick up the offline adapter.
            _reset_hf_sessions()
        _force_offline_depth += 1
    try:
        yield
    finally:
        with _force_offline_lock:
            _force_offline_depth -= 1
            if _force_offline_depth == 0:
                for obj, attr, val in _force_offline_saved:
                    try:
                        setattr(obj, attr, val)
                    except Exception:
                        pass
                _force_offline_saved = []
                for _k, _v in _force_offline_saved_env.items():
                    if _v is None:
                        os.environ.pop(_k, None)
                    else:
                        os.environ[_k] = _v
                _force_offline_saved_env = {}
                # Drop offline-mounted sessions so later online calls rebuild for the network.
                _reset_hf_sessions()


def _progress_bars_were_disabled():
    """Snapshot HF progress-bar state (None if unknown); pairs with _restore_progress_bars."""
    try:
        from huggingface_hub.utils import are_progress_bars_disabled
        return are_progress_bars_disabled()
    except Exception:
        return None


def _restore_progress_bars(were_disabled):
    """Re-enable HF progress bars only if a failed attempt left them disabled after they
    were enabled (a loader disables them around config probes and skips re-enabling on
    error). No-op if the user had them disabled or the state is unknown."""
    if were_disabled is False:
        try:
            from huggingface_hub.utils import enable_progress_bars
            enable_progress_bars()
        except Exception:
            pass


# Every way a cache miss reaches the caller once offline mode has skipped Transformers'
# own "does not appear to have a file named" raise: the resolved path stays None and the
# next line dereferences it, so the message names the None and never the cache. Same set
# the Unsloth training worker matches (studio/backend/core/training/worker.py, #7845):
# weights come out as `endswith`, tokenizers/processors as any of the other four.
_EMPTY_CACHE_ARTIFACTS = (
    "'nonetype' object has no attribute 'endswith'",
    "'nonetype' object has no attribute 'readlines'",
    "argument should be a str or an os.pathlike object",
    "expected str, bytes or os.pathlike object",
    "stat: path should be string, bytes, os.pathlike or integer",
    "can't find a vocabulary file at path 'none'",
)


def _empty_cache_artifact(exc):
    """True if exc, or something it wraps, is offline mode's empty-cache artifact.

    The one family of retry failure that says nothing useful.

    Named positively, rather than asking "is this an OOM": every other retry
    failure is real news and must reach the user, and enumerating the ways an
    accelerator spells OOM cannot be complete (accelerate re-raises it as a bare
    RuntimeError, XPU has its own class). Asking for the artifact instead is
    complete by construction.
    """
    seen = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        text = str(exc).lower()
        # Gate on the None: the same TypeError wording about a real path (`not 'int'`)
        # is a caller bug, not an empty cache.
        if ("nonetype" in text or "path 'none'" in text) and any(
            artifact in text for artifact in _EMPTY_CACHE_ARTIFACTS
        ):
            return True
        exc = exc.__cause__ or exc.__context__
    return False


def _release_traceback_locals(error):
    """Drop the frame locals along an exception chain, keeping file and line.

    An exception we keep past its handler keeps its frames alive, and a failed load's
    frames still own whatever the attempt had already built, so a retained error can pin
    a model's GPU tensors for as long as the caller holds it. Clearing the locals beats
    dropping the traceback: the memory goes either way, but the origin stays printable,
    which for a network failure inside `trust_remote_code` is the only clue there is.
    """
    seen = set()
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        # The frame we are running in cannot be cleared; clear_frames skips it for us.
        _traceback.clear_frames(error.__traceback__)
        error = error.__cause__ or error.__context__


def _note_offline_retry(error, retry_error):
    """Record the failed cache retry on the online error we are about to surface.

    A note is the only place it can go that survives: the online error usually
    already has a cause, and Python prints a cause INSTEAD of a context, so
    chaining the retry on would hide it (and would cost the chain that makes the
    online error classifiable). Notes are 3.11+; below that this is a no-op and
    the attribute is all there is."""
    text = f"Unsloth: retrying from the local cache also failed: {type(retry_error).__name__}: {retry_error}"
    try:
        error._unsloth_offline_retry_error = retry_error
    except Exception:
        pass
    add_note = getattr(error, "add_note", None)
    if add_note is None:
        return
    try:
        add_note(text)
    except Exception:
        pass


def _offline_aware_load(fn):
    """Decide offline ONCE (local_files_only kwarg or env) and force it around the
    whole load. If we started online and hit a network error, retry once forced-offline.
    Network-up online path is unchanged: no window, no retry."""

    @functools.wraps(fn)
    def _wrapper(*args, **kwargs):
        if _get_effective_local_files_only(kwargs):
            kwargs["local_files_only"] = True
            with _force_hf_offline():
                # Stamp inside the window: the env vars are restored on exit, so the
                # request has to travel on the objects themselves to reach saving.
                return _mark_loaded_local_files_only(fn(*args, **kwargs), kwargs.get("cache_dir"))
        _pb_were_disabled = _progress_bars_were_disabled()  # restore before any retry
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            # Skip if not network-related, or already retried by a nested decorator
            # (else outer layers reload the whole model again).
            if not _is_offline_related_error(e) or getattr(e, "_unsloth_offline_retried", False):
                raise
            # Holding `e` holds its frames, and those frames hold the half-built model,
            # so the collect below could not free it and a large VLM OOMed on the reload
            # the retry exists to make. A wrapper's cause/context carries tracebacks over
            # the SAME frames, so the whole chain has to be released, not just the top.
            online_error = e
            _release_traceback_locals(online_error)
        # Retry OUTSIDE the except so the failed attempt's traceback (a partial model)
        # is freed before reallocating, else a large VLM can OOM on the second load.
        try:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if hasattr(torch, "xpu") and torch.xpu.is_available():
                torch.xpu.empty_cache()
        except Exception:
            pass
        # A failed attempt may have left HF progress bars disabled; restore before retry.
        _restore_progress_bars(_pb_were_disabled)
        kwargs["local_files_only"] = True
        try:
            with _force_hf_offline():
                return fn(*args, **kwargs)
        except Exception as e:
            # A real retry failure (corrupt checkpoint, OOM) is news and goes out as
            # itself; only the empty-cache artifact is worth replacing.
            if not _empty_cache_artifact(e):
                # Tag so an enclosing _offline_aware_load skips its own redundant retry.
                try:
                    e._unsloth_offline_retried = True
                except Exception:
                    pass
                raise
            # The retry can load a whole cached model and only then trip over a missing
            # tokenizer file, and this error outlives the call on the online one, so its
            # frames would keep that model resident for as long as the caller holds it.
            _release_traceback_locals(e)
            retry_error = e
        # Report the ONLINE error: this retry only runs because of it, and its own
        # failure names an empty cache badly (offline mode skips Transformers'
        # "does not appear to have a file named" raise, so the user saw
        # `AttributeError: 'NoneType' ... 'endswith'`).
        try:
            online_error._unsloth_offline_retried = True
        except Exception:
            pass
        # Raise OUTSIDE the handler above: inside it, Python would overwrite
        # `__context__` with the cache miss, and that chain is often the only thing
        # that still makes the online error classifiable as network-related.
        if online_error.__cause__ is None and online_error.__context__ is None:
            # No chain to lose, so chain the retry on where it also prints.
            raise online_error from retry_error
        _note_offline_retry(online_error, retry_error)
        raise online_error

    return _wrapper


def _has_local_tokenizer_files(path):
    """True if a local dir has a loadable tokenizer (BPE vocab.json needs merges.txt;
    special_tokens_map.json is not required)."""
    return (
        os.path.exists(os.path.join(path, "tokenizer.json"))
        or os.path.exists(os.path.join(path, "tokenizer.model"))
        or (
            os.path.exists(os.path.join(path, "vocab.json"))
            and os.path.exists(os.path.join(path, "merges.txt"))
        )
        or os.path.exists(os.path.join(path, "vocab.txt"))
        or os.path.exists(os.path.join(path, "spiece.model"))
    )


def _has_local_processor_files(path):
    """True if a local dir ships a processor/image-processor config (a VLM needs this to
    build AutoProcessor; tokenizer files alone are not enough)."""
    return os.path.exists(os.path.join(path, "processor_config.json")) or os.path.exists(
        os.path.join(path, "preprocessor_config.json")
    )


def _resolve_hub_repo_local_dir(
    repo_id,
    *,
    token = None,
    cache_dir = None,
    revision = None,
    # Default closed: a "resolve local dir" helper must not download. False here
    # means five filenames each retried with backoff before it gives up.
    local_files_only = True,
    filenames = (
        "tokenizer_config.json",
        "config.json",
        "tokenizer.json",
        "preprocessor_config.json",
        "processor_config.json",
    ),
):
    """Return a local snapshot directory for a Hub repo id when files are cached.

    On transformers 4.57.2 through 5.5.4, ``PreTrainedTokenizerFast.from_pretrained``
    on a repo id can still call ``model_info()`` when ``local_files_only=True`` and
    no offline env var is set. Loading from the resolved snapshot dir avoids that
    Hub probe. Upstream fixed this in transformers 5.6.0 (huggingface/transformers#43603);
    this helper can be removed once the supported floor is past that version.
    """
    if not isinstance(repo_id, str) or not repo_id:
        return None
    if os.path.isdir(repo_id):
        return repo_id
    if cache_dir is None:
        cache_dir = os.environ.get("HF_HUB_CACHE")
    from huggingface_hub import hf_hub_download

    for filename in filenames:
        try:
            path = hf_hub_download(
                repo_id = repo_id,
                filename = filename,
                token = token,
                cache_dir = cache_dir,
                local_files_only = local_files_only,
                revision = revision,
            )
            if path and os.path.isfile(path):
                return os.path.dirname(path)
        except Exception:
            continue
    return None


def _resolve_hub_repo_cached_file(
    repo_id,
    filename,
    *,
    token = None,
    cache_dir = None,
    local_files_only = True,
    revision = None,
):
    """Return a cached file path under a Hub snapshot, or None if absent."""
    local_dir = _resolve_hub_repo_local_dir(
        repo_id,
        token = token,
        cache_dir = cache_dir,
        local_files_only = local_files_only,
        revision = revision,
        filenames = (filename,),
    )
    if local_dir is None:
        return None
    path = os.path.join(local_dir, filename)
    return path if os.path.isfile(path) else None


def _hub_repo_or_local_path(
    repo_id,
    *,
    token = None,
    cache_dir = None,
    local_files_only = False,
    filenames = None,
    revision = None,
):
    """Prefer a cached snapshot path over a Hub repo id when offline or ``local_files_only``."""
    if isinstance(repo_id, str) and os.path.isdir(repo_id):
        return repo_id
    lfo = bool(local_files_only) or _env_says_offline()
    if not lfo:
        return repo_id
    local_dir = _resolve_hub_repo_local_dir(
        repo_id,
        token = token,
        cache_dir = cache_dir,
        local_files_only = True,
        revision = revision,
        filenames = filenames
        or (
            "tokenizer_config.json",
            "config.json",
            "tokenizer.json",
            "preprocessor_config.json",
            "processor_config.json",
        ),
    )
    return local_dir if local_dir is not None else repo_id


def _load_pretrained_tokenizer_fast(
    tokenizer_name,
    *,
    padding_side = "left",
    token = None,
    trust_remote_code = False,
    cache_dir = None,
    local_files_only = False,
    revision = None,
):
    """Load ``PreTrainedTokenizerFast`` without Hub metadata probes when cached/offline.

    Needed on transformers 4.57.2-5.5.4; redundant once the floor is past 5.6.0.
    """
    from transformers import PreTrainedTokenizerFast

    lfo = bool(local_files_only) or _env_says_offline()
    load_path = _hub_repo_or_local_path(
        tokenizer_name,
        token = token,
        cache_dir = cache_dir,
        local_files_only = lfo,
        revision = revision,
        filenames = (
            "tokenizer_config.json",
            "tokenizer.json",
            "tokenizer.model",
        ),
    )
    return PreTrainedTokenizerFast.from_pretrained(
        load_path,
        padding_side = padding_side,
        token = token,
        trust_remote_code = trust_remote_code,
        cache_dir = cache_dir,
        local_files_only = lfo,
        revision = revision,
    )


def _resolve_checkpoint_tokenizer_name(
    old_model_name,
    kwargs,
    require_processor = False,
):
    """tokenizer_name for a PEFT/checkpoint load: caller override, else the local checkpoint
    dir if self-sufficient, else None (base repo). Always popped from kwargs (also passed
    explicitly downstream). For a VLM (require_processor), the dir must also ship processor
    files; otherwise fall back to the base repo whose cached processor still loads."""
    explicit = kwargs.pop("tokenizer_name", None)
    if explicit is not None:
        return explicit
    has_config = os.path.exists(os.path.join(old_model_name, "tokenizer_config.json"))
    if not (has_config and _has_local_tokenizer_files(old_model_name)):
        return None
    if require_processor and not _has_local_processor_files(old_model_name):
        return None
    return old_model_name
