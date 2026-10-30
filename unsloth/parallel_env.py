# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""One flag (or one line) for every parallel axis of an unmodified training script.

    UNSLOTH_PARALLEL="tp=2,cp=2" torchrun --nproc-per-node 8 train.py     # dp 2 (filled in) x cp 2 x tp 2
    UNSLOTH_PARALLEL="dp=2,pp=2,fsdp=1" torchrun --nproc-per-node 4 train.py

    import unsloth
    unsloth.parallelize(tp = 2, cp = 2)                                    # before from_pretrained

Keys: dp, tp, pp, cp (positive integers; dp may be omitted and then fills the world), cp_backend (auto, ulysses,
ring), fsdp (0 / 1: native FSDP2 over the data parallel ranks instead of DDP-style averaging), pp_schedule
(gpipe, 1f1b), pp_microbatches. The flag is parsed once, at `import unsloth`, and translated to the variables
the existing paths read: UNSLOTH_TP_SIZE, UNSLOTH_PP_SIZE, UNSLOTH_FSDP2, UNSLOTH_PP_SCHEDULE,
UNSLOTH_PP_MICROBATCHES, UNSLOTH_CP_SIZE and UNSLOTH_CP_BACKEND. Those variables keep working on their own; a
variable that disagrees with the flag is refused.

The world splits pp x dp x cp x tp, pp outermost and tp innermost (adjacent ranks share a tensor parallel group,
the fastest link). `root_mesh` builds that DeviceMesh once; tensor parallel takes mesh["tp"], pipeline parallel
mesh["pp"], context parallel mesh["cp"], and data parallel gradients (FSDP2 sharding or DDP-style averaging) the
flattened (dp, cp) mesh, because context parallel peers hold different tokens of the same rows. The data loader
splits over dp only. Size-1 axes are left out of the mesh, so they create no process group.

This module imports nothing heavy at module level: `import unsloth` translates the flag before transformers,
accelerate or torch.distributed read any of it.
"""

from __future__ import annotations

import math
import os

__all__ = [
    "ParallelSpec",
    "parse_parallel_spec",
    "apply_parallel_env",
    "parallelize",
    "env_cp_size",
    "env_cp_backend",
    "CP_BACKENDS",
    "PP_SCHEDULES",
]

FLAG = "UNSLOTH_PARALLEL"
CP_BACKENDS = ("auto", "ulysses", "ring")
PP_SCHEDULES = ("gpipe", "1f1b")
_SIZE_KEYS = ("dp", "tp", "pp", "cp")
_ALIASES = {
    "microbatches": "pp_microbatches",
    "mb": "pp_microbatches",
    "schedule": "pp_schedule",
    "backend": "cp_backend",
    "fsdp2": "fsdp",
}
_KEYS = _SIZE_KEYS + ("cp_backend", "fsdp", "pp_schedule", "pp_microbatches")
_TRUTHY = ("1", "true", "yes", "on")
_FALSY = ("0", "false", "no", "off")
# The variables each key writes, in the form the legacy readers expect.
_LEGACY = {
    "tp": "UNSLOTH_TP_SIZE",
    "pp": "UNSLOTH_PP_SIZE",
    "cp": "UNSLOTH_CP_SIZE",
    "cp_backend": "UNSLOTH_CP_BACKEND",
    "fsdp": "UNSLOTH_FSDP2",
    "pp_schedule": "UNSLOTH_PP_SCHEDULE",
    "pp_microbatches": "UNSLOTH_PP_MICROBATCHES",
}


class ParallelSpec(dict):
    """The parsed flag: {key: value} for the keys given (sizes as int, fsdp as bool)."""

    def sizes(self):
        return {k: self.get(k, 1) for k in ("pp", "tp", "cp")}

    def describe(self):
        return ",".join(f"{k}={_format(v)}" for k, v in self.items())


def _format(value):
    if isinstance(value, bool):
        return "1" if value else "0"
    return str(value)


def _positive(key, raw, source):
    try:
        value = int(str(raw).strip())
    except ValueError:
        raise ValueError(f"Unsloth: {source}: {key}={raw!r} must be a positive integer.") from None
    if value < 1:
        raise ValueError(f"Unsloth: {source}: {key}={value} must be >= 1.")
    return value


def parse_parallel_spec(raw, source = FLAG):
    """'dp=2,tp=2,cp=2,cp_backend=ulysses' -> ParallelSpec; ValueError naming the bad part otherwise."""
    spec = ParallelSpec()
    if raw is None or not str(raw).strip():
        return spec
    for part in str(raw).replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(
                f"Unsloth: {source}={raw!r}: {part!r} is not key=value (e.g. {FLAG}=\"dp=2,tp=2,cp=2\")."
            )
        key, value = (x.strip() for x in part.split("=", 1))
        key = _ALIASES.get(key.lower(), key.lower())
        if key not in _KEYS:
            raise ValueError(
                f"Unsloth: {source}={raw!r}: unknown key {key!r}; the keys are {', '.join(_KEYS)}."
            )
        if key in spec:
            raise ValueError(f"Unsloth: {source}={raw!r} gives {key} twice.")
        if key in _SIZE_KEYS or key == "pp_microbatches":
            spec[key] = _positive(key, value, source)
        elif key == "fsdp":
            low = value.lower()
            if low not in _TRUTHY + _FALSY:
                raise ValueError(f"Unsloth: {source}={raw!r}: fsdp={value!r} must be 0 or 1.")
            spec[key] = low in _TRUTHY
        elif key == "cp_backend":
            low = value.lower()
            if low not in CP_BACKENDS:
                raise ValueError(
                    f"Unsloth: {source}={raw!r}: cp_backend={value!r} must be one of {', '.join(CP_BACKENDS)}."
                )
            spec[key] = low
        else:  # pp_schedule
            low = value.lower().replace("-", "").replace("_", "")
            if low not in PP_SCHEDULES:
                hint = (
                    " (interleaved virtual stages are not supported yet)"
                    if low.startswith("interleav")
                    else ""
                )
                raise ValueError(
                    f"Unsloth: {source}={raw!r}: pp_schedule={value!r} must be one of "
                    f"{', '.join(PP_SCHEDULES)}{hint}."
                )
            spec[key] = low
    if "cp_backend" in spec and spec.get("cp", 1) == 1:
        raise ValueError(f"Unsloth: {source}={raw!r}: cp_backend needs cp > 1.")
    for key in ("pp_schedule", "pp_microbatches"):
        if key in spec and spec.get("pp", 1) == 1:
            raise ValueError(f"Unsloth: {source}={raw!r}: {key} needs pp > 1.")
    return spec


def _world(environ):
    raw = environ.get("WORLD_SIZE")
    if raw is None or not str(raw).strip():
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def _legacy_value(key, environ):
    name = _LEGACY[key]
    raw = environ.get(name)
    if raw is None or not str(raw).strip():
        return None
    raw = str(raw).strip()
    if key == "fsdp":
        low = raw.lower()
        return True if (low in _TRUTHY or low == "native") else (False if low in _FALSY else raw)
    if key == "cp_backend":
        return raw.lower()
    if key == "pp_schedule":
        return raw.lower().replace("-", "").replace("_", "")
    try:
        return int(raw)
    except ValueError:
        return raw


def translate(spec, environ = None):
    """{legacy variable: value} for a parsed spec, after checking the world size and the legacy variables
    already set (ValueError on a conflict). Pure: the environment is only read."""
    environ = os.environ if environ is None else environ
    world = _world(environ)
    sizes = spec.sizes()
    model_ranks = math.prod(sizes.values())
    given = spec.describe()
    if "dp" in spec:
        need = spec["dp"] * model_ranks
        if world is not None and world != need:
            raise ValueError(
                f"Unsloth: {FLAG}=\"{given}\" needs dp x pp x cp x tp = {need} ranks, but WORLD_SIZE={world}; "
                f"launch with torchrun --nproc-per-node {need}, or drop dp= to let it fill the world."
            )
    elif world is not None and world % model_ranks:
        raise ValueError(
            f"Unsloth: {FLAG}=\"{given}\": pp x cp x tp = {model_ranks} must divide the number of ranks "
            f"(WORLD_SIZE={world}); launch with torchrun --nproc-per-node {model_ranks} (or a multiple of it)."
        )
    out = {}
    wanted = {
        "tp": sizes["tp"],
        "pp": sizes["pp"],
        "cp": sizes["cp"],
    }
    for key in ("cp_backend", "fsdp", "pp_schedule", "pp_microbatches"):
        if key in spec:
            wanted[key] = spec[key]
    for key, value in wanted.items():
        current = _legacy_value(key, environ)
        if key in _SIZE_KEYS and current is None and value == 1:
            continue
        if current is not None and current != value and not (key in _SIZE_KEYS and current == 1 and value == 1):
            raise ValueError(
                f"Unsloth: {FLAG}=\"{given}\" sets {key}={_format(value)}, but {_LEGACY[key]}="
                f"{environ.get(_LEGACY[key])} asks for something else. Use one of the two forms: unset "
                f"{_LEGACY[key]}, or drop {FLAG}."
            )
        out[_LEGACY[key]] = _format(value)
    return out


def apply_parallel_env(environ = None):
    """`import unsloth` (and `parallelize`): translate UNSLOTH_PARALLEL into the legacy variables. Returns the
    spec, or None when the flag is not set. Idempotent: a second call sees its own values and no conflict."""
    environ = os.environ if environ is None else environ
    raw = environ.get(FLAG)
    if raw is None or not str(raw).strip():
        return None
    spec = parse_parallel_spec(raw)
    environ.update(translate(spec, environ))
    return spec


def env_cp_size(environ = None):
    """UNSLOTH_CP_SIZE (1 when unset); ValueError on a non-positive / non-integer value."""
    environ = os.environ if environ is None else environ
    raw = environ.get("UNSLOTH_CP_SIZE")
    if raw is None or not str(raw).strip():
        return 1
    return _positive("UNSLOTH_CP_SIZE", raw, "UNSLOTH_CP_SIZE")


def env_cp_backend(environ = None):
    """UNSLOTH_CP_BACKEND (None when unset); ValueError outside auto / ulysses / ring."""
    environ = os.environ if environ is None else environ
    raw = environ.get("UNSLOTH_CP_BACKEND")
    if raw is None or not str(raw).strip():
        return None
    low = str(raw).strip().lower()
    if low not in CP_BACKENDS:
        raise ValueError(
            f"Unsloth: UNSLOTH_CP_BACKEND={raw!r} must be one of {', '.join(CP_BACKENDS)}."
        )
    return low


def parallelize(
    dp = None,
    tp = 1,
    pp = 1,
    cp = 1,
    cp_backend = None,
    fsdp = None,
    pp_schedule = None,
    pp_microbatches = None,
):
    """`UNSLOTH_PARALLEL` from Python: call after `import unsloth` and before `from_pretrained`, on every rank.

        import unsloth
        unsloth.parallelize(tp = 2, cp = 2)

    Returns the resolved {legacy variable: value}. Refused once a model has been loaded under another layout."""
    parts = []
    if dp is not None:
        parts.append(f"dp={dp}")
    for key, value in (("tp", tp), ("pp", pp), ("cp", cp)):
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"Unsloth: parallelize({key} = {value!r}): must be a positive integer.")
        if value > 1:
            parts.append(f"{key}={value}")
    if dp is not None and (isinstance(dp, bool) or not isinstance(dp, int) or dp < 1):
        raise ValueError(f"Unsloth: parallelize(dp = {dp!r}): must be a positive integer.")
    if cp_backend is not None:
        parts.append(f"cp_backend={cp_backend}")
    if fsdp is not None:
        parts.append(f"fsdp={1 if fsdp else 0}")
    if pp_schedule is not None:
        parts.append(f"pp_schedule={pp_schedule}")
    if pp_microbatches is not None:
        parts.append(f"pp_microbatches={pp_microbatches}")
    raw = ",".join(parts)
    spec = parse_parallel_spec(raw, source = "unsloth.parallelize")
    current = os.environ.get(FLAG)
    ours = bool(current) and current == _SET_BY_PARALLELIZE[0]
    from_env = bool(current and current.strip()) and not ours
    if from_env and parse_parallel_spec(current) != spec:
        raise ValueError(
            f"Unsloth: unsloth.parallelize({raw}) disagrees with {FLAG}=\"{current}\" from the environment; "
            "use one of the two."
        )
    environ = dict(os.environ)
    if ours:
        # Our own earlier call: its variables are not a conflict, the layout check below decides.
        for name in _LEGACY.values():
            environ.pop(name, None)
    resolved = translate(spec, environ)
    _refuse_after_layout_read(resolved)
    if ours:
        for name in _LEGACY.values():
            os.environ.pop(name, None)
    os.environ.update(resolved)
    os.environ[FLAG] = raw
    _SET_BY_PARALLELIZE[0] = raw
    _reinstall_after_import()
    return resolved


_SET_BY_PARALLELIZE = [None]


def _refuse_after_layout_read(resolved):
    """A layout already read (a model loaded) cannot change under it."""
    import sys

    tp_env = sys.modules.get("unsloth.models.tp_env")
    if tp_env is None:
        return
    layout = getattr(tp_env, "_LAYOUT", None)
    if layout is getattr(tp_env, "_UNSET", None):
        return
    before = (1, 1) if layout is None else (layout.pp, layout.tp)
    after = (int(resolved.get("UNSLOTH_PP_SIZE", 1)), int(resolved.get("UNSLOTH_TP_SIZE", 1)))
    if before != after:
        raise RuntimeError(
            "Unsloth: unsloth.parallelize(...) must run before from_pretrained: a model was already loaded "
            f"with pp {before[0]} x tp {before[1]}."
        )
    tp_env._reset_env_layout()


def _reinstall_after_import():
    """What `import unsloth` installs from these variables (it already ran when parallelize is called)."""
    import sys

    if "unsloth" not in sys.modules:
        return
    from .fsdp2_launch import apply_unsloth_fsdp2_launch_env, install_native_fsdp2_trainer_init

    apply_unsloth_fsdp2_launch_env()
    install_native_fsdp2_trainer_init()
    from .models.tp_env import _reset_env_layout, install_tensor_parallel_env_trainer_init

    _reset_env_layout()
    install_tensor_parallel_env_trainer_init()


def _int_env(name):
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return None
    try:
        value = int(raw.strip())
    except ValueError:
        raise ValueError(f"Unsloth: {name}={raw!r} must be an integer.") from None
    if value < 1:
        raise ValueError(f"Unsloth: {name}={value} must be >= 1.")
    return value


class ParallelLayout:
    """world = pp x dp x cp x tp; `mesh()` builds (once) the root device mesh, initializing the process group
    if needed. (models/tp_env.TPEnvLayout is this class.)"""

    def __init__(
        self,
        tp,
        dp,
        world,
        fsdp2,
        pp = 1,
        microbatches = None,
        schedule = "gpipe",
        cp = 1,
        cp_backend = None,
    ):
        self.tp, self.dp, self.world, self.fsdp2 = tp, dp, world, fsdp2
        self.pp, self.microbatches, self.schedule = pp, microbatches, schedule
        self.cp, self.cp_backend = cp, cp_backend
        self._mesh = None
        self._grad_mesh = None

    def __repr__(self):
        parts = [f"pp {self.pp}"] if self.pp > 1 else []
        if self.dp > 1:
            parts.append(f"dp {self.dp} ({'native FSDP2' if self.fsdp2 else 'DDP'})")
        if self.cp > 1:
            parts.append(f"cp {self.cp}")
        if self.tp > 1:
            parts.append(f"tp {self.tp}")
        return " x ".join(parts)

    @property
    def env_vars(self):
        """The environment variables that asked for this layout, e.g. "UNSLOTH_PP_SIZE=2 UNSLOTH_TP_SIZE=2"."""
        out = [f"UNSLOTH_PP_SIZE={self.pp}"] if self.pp > 1 else []
        if self.tp > 1:
            out.append(f"UNSLOTH_TP_SIZE={self.tp}")
        if self.cp > 1:
            out.append(f"UNSLOTH_CP_SIZE={self.cp}")
        return " ".join(out)

    @property
    def sharded_loading(self):
        if self.pp > 1:
            return False  # the pipeline split comes first, on whole layers (models/pp_env.py)
        return os.environ.get("UNSLOTH_TP_SHARDED_LOAD", "1").strip().lower() not in _FALSY

    @property
    def dims(self):
        """[(name, size)] of the root mesh, pp outermost and tp innermost; size-1 axes left out."""
        dp_name = "dp_shard" if self.fsdp2 else "dp_replicate"
        dims = [("pp", self.pp), (dp_name, self.dp), ("cp", self.cp), ("tp", self.tp)]
        return [(name, size) for name, size in dims if size > 1]

    def mesh(self):
        if self._mesh is not None:
            return self._mesh
        import torch
        import torch.distributed as dist
        from torch.distributed.device_mesh import init_device_mesh

        cuda = torch.cuda.is_available()
        if cuda:
            torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
        if not dist.is_initialized():
            # What the Trainer's accelerate state would create; it reuses an existing group.
            dist.init_process_group("nccl" if cuda else "gloo")
        if dist.get_world_size() != self.world:
            raise RuntimeError(
                f"Unsloth: {self.env_vars} read WORLD_SIZE={self.world}, but the process group has "
                f"{dist.get_world_size()} ranks."
            )
        device = "cuda" if cuda else "cpu"
        dims = self.dims or [("tp", self.tp)]
        mesh = init_device_mesh(
            device, tuple(size for _, size in dims), mesh_dim_names = tuple(name for name, _ in dims)
        )
        if len(dims) > 1:
            from .models.pp_native import check_mesh_composition

            check_mesh_composition({name: mesh[name] for name, _ in dims}, f"{self.env_vars} ({self})")
        self._mesh = mesh
        if dist.get_rank() == 0:
            what = "pipeline" if self.pp > 1 else "tensor"
            if self.pp > 1 and self.tp > 1:
                what = "pipeline x tensor"
            print(f"Unsloth: {what} parallel from {self.env_vars}: {self} over {self.world} ranks.")
        return mesh

    def _dim(self, name):
        return self.mesh()[name] if any(n == name for n, _ in self.dims) else None

    @property
    def tp_mesh(self):
        return self._dim("tp")

    @property
    def pp_mesh(self):
        return self._dim("pp")

    @property
    def cp_mesh(self):
        return self._dim("cp")

    @property
    def data_mesh(self):
        """The data parallel ranks: they read different rows (cp / tp / pp peers read the same ones)."""
        return self._dim("dp_shard" if self.fsdp2 else "dp_replicate")

    @property
    def grad_size(self):
        return self.dp * self.cp

    @property
    def dp_mesh(self):
        """The mesh data parallel gradients reduce over (FSDP2 sharding / DDP-style averaging): dp, and cp
        flattened in when context parallel is on (cp peers hold different tokens of the same rows)."""
        if self.cp == 1:
            return self.data_mesh
        if self.dp == 1:
            return self.cp_mesh
        if self._grad_mesh is None:
            dp_name = "dp_shard" if self.fsdp2 else "dp_replicate"
            self._grad_mesh = self.mesh()[(dp_name, "cp")]._flatten(f"{dp_name}_cp")
        return self._grad_mesh


def read_layout():
    """The layout UNSLOTH_TP_SIZE / UNSLOTH_PP_SIZE (and UNSLOTH_CP_SIZE beside them) ask for, or None."""
    pp = _int_env("UNSLOTH_PP_SIZE") or 1
    tp = _int_env("UNSLOTH_TP_SIZE") or 1
    if pp == 1 and tp == 1:
        return None  # context parallel alone is the CP manager's (context_parallel.py)
    cp = env_cp_size()
    cp_backend = env_cp_backend()
    try:
        world = int(os.environ.get("WORLD_SIZE", "1"))
    except ValueError:
        world = 1
    sizes = (("UNSLOTH_PP_SIZE", pp), ("UNSLOTH_CP_SIZE", cp), ("UNSLOTH_TP_SIZE", tp))
    names = " x ".join(name for name, size in sizes if size > 1)
    given = " ".join(f"{name}={size}" for name, size in sizes if size > 1)
    if world % (pp * tp * cp):
        what = (
            "data parallel x tensor parallel"
            if pp == 1
            else "data parallel x pipeline parallel" + (" x tensor parallel" if tp > 1 else "")
        )
        raise ValueError(
            f"Unsloth: {given}: {names} = {pp * tp * cp} must divide the number of ranks (WORLD_SIZE={world}); "
            f"launch with torchrun --nproc-per-node {pp * tp * cp} (or a multiple of {pp * tp * cp} for {what}), or "
            f"unset {' / '.join(name for name in names.split(' x '))}."
        )
    microbatches, schedule = None, "gpipe"
    if pp > 1:
        microbatches = _int_env("UNSLOTH_PP_MICROBATCHES")
        schedule = (os.environ.get("UNSLOTH_PP_SCHEDULE") or "gpipe").strip().lower().replace("-", "")
        schedule = schedule.replace("_", "")
        if schedule not in PP_SCHEDULES:
            raise ValueError(
                f"Unsloth: UNSLOTH_PP_SCHEDULE={os.environ.get('UNSLOTH_PP_SCHEDULE')!r} must be one of "
                f"{PP_SCHEDULES}."
            )
        if schedule == "1f1b" and microbatches is not None and microbatches < pp:
            raise ValueError(
                f"Unsloth: UNSLOTH_PP_SCHEDULE=1f1b needs UNSLOTH_PP_MICROBATCHES >= UNSLOTH_PP_SIZE "
                f"({microbatches} < {pp})."
            )
        from .models.pp_native import _check_torch_pipelining

        try:
            _check_torch_pipelining()
        except RuntimeError as exc:
            raise RuntimeError(f"{exc} (UNSLOTH_PP_SIZE={pp}: unset it to run without pipeline parallel.)") from None
    from .fsdp2_launch import unsloth_fsdp2_requested

    fsdp2 = bool(unsloth_fsdp2_requested())
    backend = os.environ.get("UNSLOTH_FSDP2_BACKEND", "").strip().lower()
    if fsdp2 and backend in ("accelerate", "hf", "transformers"):
        what = "pipeline" if pp > 1 else "tensor"
        raise ValueError(
            f"Unsloth: {given} composes with native FSDP2 only (UNSLOTH_FSDP2=1); "
            f"UNSLOTH_FSDP2_BACKEND={backend} would hand the {what} parallel model to accelerate's FSDP. "
            "Unset UNSLOTH_FSDP2_BACKEND."
        )
    layout = ParallelLayout(
        tp, world // (pp * tp * cp), world, fsdp2, pp, microbatches, schedule, cp = cp, cp_backend = cp_backend
    )
    from .parallel_capabilities import check_layout

    check_layout(layout, cells = False)  # the layout-only rules, before any model is loaded under it
    return layout


def describe_trainer_layout(trainer, layout = None, cp_manager = None):
    """One line: the resolved mesh (pp, dp_replicate, dp_shard, cp, tp), schedule and micro-batches, effective
    batch, context parallel backend, attention backend and gradient checkpointing mode."""
    world = _dist_value("get_world_size", 1)
    cp = cp_manager.size if cp_manager is not None else getattr(layout, "cp", 1)
    pp = getattr(layout, "pp", 1)
    tp = getattr(layout, "tp", 1)
    dp = getattr(layout, "dp", None) or max(world // max(pp * tp * cp, 1), 1)
    fsdp2 = getattr(layout, "fsdp2", None)
    if fsdp2 is None:
        from .fsdp2_launch import unsloth_fsdp2_native_requested

        try:
            fsdp2 = bool(unsloth_fsdp2_native_requested())
        except Exception:
            fsdp2 = False
    mesh = {
        "pp": pp,
        "dp_replicate": 1 if fsdp2 else dp,
        "dp_shard": dp if fsdp2 else 1,
        "cp": cp,
        "tp": tp,
    }
    parts = [f"mesh ({', '.join(f'{k} {v}' for k, v in mesh.items())}) over {world} ranks"]
    if pp > 1:
        mb = getattr(layout, "microbatches", None)
        parts.append(f"pp schedule {getattr(layout, 'schedule', 'gpipe')}, {mb or 'auto'} micro-batches")
    args = getattr(trainer, "args", None)
    if args is not None:
        bs = getattr(args, "per_device_train_batch_size", 1)
        ga = getattr(args, "gradient_accumulation_steps", 1)
        parts.append(f"effective batch {bs} x {ga} accumulation x {dp} data parallel = {bs * ga * dp} rows")
    if cp > 1:
        backend = getattr(cp_manager, "backend", None) or getattr(layout, "cp_backend", None) or "auto"
        parts.append(f"cp backend {backend}")
    model = getattr(trainer, "model", None)
    config = getattr(model, "config", None)
    attention = getattr(config, "_attn_implementation", None)
    if attention:
        parts.append(f"attention {attention}")
    gc = getattr(args, "gradient_checkpointing", None) if args is not None else None
    gc_mode = getattr(model, "_unsloth_gradient_checkpointing", None) or (
        "on" if gc else "off" if gc is not None else "unknown"
    )
    parts.append(f"gradient checkpointing {gc_mode}")
    return "; ".join(parts)


def announce(trainer, layout = None, cp_manager = None):
    """Rank 0 prints the resolved layout once per trainer."""
    if getattr(trainer, "__dict__", {}).get("_unsloth_parallel_announced"):
        return None
    line = "Unsloth: parallel layout: " + describe_trainer_layout(trainer, layout, cp_manager) + "."
    try:
        trainer.__dict__["_unsloth_parallel_announced"] = line
    except Exception:
        pass
    if _dist_value("get_rank", 0) == 0:
        print(line)
    return line


def _dist_value(name, default):
    try:
        import torch.distributed as dist

        return getattr(dist, name)() if dist.is_available() and dist.is_initialized() else default
    except Exception:
        return default
