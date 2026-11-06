# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""QLoRA under FSDP2: packed 4-bit bytes stored as float dtype to shard, QuantState kept; mixed
precision never casts them (foreach copy rewrites NaN bit patterns on CUDA); float storage only
when patchable."""

__all__ = [
    "is_params4bit",
    "float_storage_survives_wrap",
    "fsdp2_quant_storage_dtype",
    "set_fsdp2_quant_storage",
    "params4bit_to_float_storage",
    "params4bit_to_uint8_storage",
    "patch_fsdp2_params4bit",
    "nonfloat_param_guard",
    "prepare_fsdp2_qlora",
    "fsdp2_qlora_wrap",
]

import contextlib
import inspect
import sys

import torch

_FLOAT_STORAGE = (torch.float16, torch.bfloat16, torch.float32)
_QUANT_ATTRS = (
    "quant_state",
    "blocksize",
    "compress_statistics",
    "quant_type",
    "quant_storage",
    "bnb_quantized",
    # A transformers v5 3D expert stack (zoo moe_utils_bnb4bit) records its logical shape here.
    "_original_shape",
)


# Marks an FSDP2 all-gathered 4-bit weight (read by unsloth-zoo's MoE pointer-table caches).
_GATHERED_MARK = "_unsloth_fsdp2_gathered"


def is_params4bit(tensor):
    return any(cls.__name__ == "Params4bit" for cls in type(tensor).__mro__)


def float_storage_survives_wrap(prepare_fn = None, native = False):
    """Whether 4-bit weights may be packed in a float dtype for FSDP2. False when this torch's FSDP2
    cannot be patched (`patch_fsdp2_params4bit`), checks frozen dtypes per unit (torch < 2.8), or when accelerate's FSDP2 wrap ends with
    `model.to(torch.float32)` (accelerate < 1.13 with mixed precision), which converts float-stored
    packed bytes as if they were weights. uint8 storage is safe from both, so 4-bit weights then stay
    uint8 (left replicated by accelerate, but correct). `native` is Unsloth's own fully_shard wrap
    (`fsdp2_native`), which never upcasts the model: only the two FSDP2 conditions apply."""
    if native:
        return not _missing_fsdp2_anchors() and not _fsdp2_checks_frozen_dtypes()
    if prepare_fn is None:
        try:
            from accelerate.utils.fsdp_utils import fsdp2_prepare_model as prepare_fn
        except Exception:
            prepare_fn = None
    if _missing_fsdp2_anchors():
        # Unpatched FSDP2 would scramble float-stored packed bytes (see 3 above).
        return False
    if _fsdp2_checks_frozen_dtypes():
        # torch < 2.8 wants one dtype per unit, frozen weights included: packed bytes in the compute
        # dtype next to float32 LoRA / upcast weights would fail it, while uint8 is left out by accelerate.
        return False
    if prepare_fn is None:
        return True
    prepare_fn = inspect.unwrap(prepare_fn)
    try:
        source = inspect.getsource(prepare_fn)
    except (OSError, TypeError):
        return False
    return "model.to(torch.float32)" not in source


# FSDP2's private modules, by static import on either layout: torch.distributed.fsdp._fully_shard.<name>
# (torch >= 2.6) or torch.distributed._composable.fsdp.<name> (2.4 / 2.5).


def _load_fsdp_param():
    try:
        from torch.distributed.fsdp._fully_shard import _fsdp_param as module
    except ImportError:  # torch 2.4 / 2.5
        from torch.distributed._composable.fsdp import _fsdp_param as module
    return module


def _load_fsdp_state():
    try:
        from torch.distributed.fsdp._fully_shard import _fsdp_state as module
    except ImportError:  # torch 2.4 / 2.5
        from torch.distributed._composable.fsdp import _fsdp_state as module
    return module


def _load_fsdp_common():
    try:
        from torch.distributed.fsdp._fully_shard import _fsdp_common as module
    except ImportError:  # torch 2.4 / 2.5
        from torch.distributed._composable.fsdp import _fsdp_common as module
    return module


def _load_fsdp_param_group():
    try:
        from torch.distributed.fsdp._fully_shard import _fsdp_param_group as module
    except ImportError:  # torch 2.4 / 2.5
        from torch.distributed._composable.fsdp import _fsdp_param_group as module
    return module


def _load_fsdp_init():
    try:
        from torch.distributed.fsdp._fully_shard import _fsdp_init as module
    except ImportError:  # torch 2.4 / 2.5
        from torch.distributed._composable.fsdp import _fsdp_init as module
    return module


def _load_fsdp_collectives():
    try:
        from torch.distributed.fsdp._fully_shard import _fsdp_collectives as module
    except ImportError:  # torch 2.4 / 2.5
        from torch.distributed._composable.fsdp import _fsdp_collectives as module
    return module


def _load_fully_shard():
    try:
        from torch.distributed.fsdp._fully_shard import _fully_shard as module
    except (
        ImportError
    ):  # torch 2.4 / 2.5: the module is `fully_shard` (the package attribute is the function)
        import torch.distributed._composable.fsdp.fully_shard  # noqa: F401
        module = sys.modules["torch.distributed._composable.fsdp.fully_shard"]
    return module


_FSDP_PRIVATE_LOADERS = {
    "_fsdp_param": _load_fsdp_param,
    "_fsdp_state": _load_fsdp_state,
    "_fsdp_common": _load_fsdp_common,
    "_fsdp_param_group": _load_fsdp_param_group,
    "_fsdp_init": _load_fsdp_init,
    "_fsdp_collectives": _load_fsdp_collectives,
    "_fully_shard": _load_fully_shard,
}


def _fsdp_private(submodule):
    """FSDP2's private module `submodule` (_fsdp_param, _fsdp_state, _fsdp_common, _fsdp_param_group,
    _fsdp_init, _fsdp_collectives, _fully_shard) on either layout (see _FSDP_PRIVATE_LOADERS)."""
    loader = _FSDP_PRIVATE_LOADERS.get(submodule)
    if loader is None:
        raise ImportError(f"no FSDP2 module {submodule}")
    return loader()


def _fsdp2_checks_frozen_dtypes():
    try:
        FSDPParamGroup = _fsdp_private("_fsdp_param_group").FSDPParamGroup
        return "requires_grad" not in inspect.getsource(FSDPParamGroup._init_mp_dtypes)
    except Exception:
        return True


def fsdp2_quant_storage_dtype(dtype):
    """The `bnb_4bit_quant_storage` FSDP2 can shard: the compute dtype, so the mixed precision cast
    of the all-gather is a no-op on the packed bytes. None for a dtype bitsandbytes cannot store in."""
    if isinstance(dtype, str):
        dtype = getattr(torch, dtype, None)
    return dtype if dtype in _FLOAT_STORAGE else None


def set_fsdp2_quant_storage(quantization_config, dtype):
    """Store the packed 4-bit bytes of a BitsAndBytesConfig in `dtype`. Returns True when changed.
    A caller's explicit float storage is kept; only the uint8 default is replaced, and only when the
    FSDP2 wrap will not upcast the whole model (`float_storage_survives_wrap`)."""
    storage = fsdp2_quant_storage_dtype(dtype)
    if quantization_config is None or storage is None or not float_storage_survives_wrap():
        return False
    if not getattr(quantization_config, "load_in_4bit", False):
        return False
    current = getattr(quantization_config, "bnb_4bit_quant_storage", None)
    if isinstance(current, str):
        current = getattr(torch, current, None)
    if current is not None and current != torch.uint8:
        return False
    quantization_config.bnb_4bit_quant_storage = storage
    return True


def params4bit_to_float_storage(model, dtype):
    """Reinterpret every uint8 Params4bit (a pre-quantized checkpoint) as `dtype` storage in place.
    The bytes, the QuantState and the Parameter object are unchanged; only the element type of the
    packed buffer differs, exactly what `bnb_4bit_quant_storage = dtype` would have produced.
    Returns the number of weights converted."""
    storage = fsdp2_quant_storage_dtype(dtype)
    if storage is None:
        return 0
    itemsize = torch.empty((), dtype = storage).element_size()
    n_converted = 0
    for module in model.modules():
        for name, param in list(module._parameters.items()):
            if param is None or not is_params4bit(param) or param.dtype != torch.uint8:
                continue
            nbytes = param.numel()
            if not param.is_contiguous() or nbytes % itemsize != 0 or param.device.type == "meta":
                continue
            param.data = param.data.reshape(-1).view(storage).reshape(-1, 1)
            param.quant_storage = storage
            if name == "weight" and hasattr(module, "quant_storage"):
                # Linear4bit rebuilds a lost Params4bit with it (fix_4bit_weight_quant_state_from_module).
                module.quant_storage = storage
            n_converted += 1
    return n_converted


def params4bit_to_uint8_storage(model):
    """Inverse of `params4bit_to_float_storage`: reinterpret float-packed Params4bit as the uint8
    bitsandbytes stores by default. Returns (converted, left), left = those that could not be viewed."""
    converted = left = 0
    for module in model.modules():
        for name, param in list(module._parameters.items()):
            if param is None or not is_params4bit(param) or not param.is_floating_point():
                continue
            if (
                not getattr(param, "bnb_quantized", False)
                or getattr(param, "quant_state", None) is None
            ):
                # Still dense values bitsandbytes quantizes on the next device move, not packed bytes.
                continue
            if not param.is_contiguous() or param.device.type == "meta":
                left += 1
                continue
            param.data = param.data.reshape(-1).view(torch.uint8).reshape(-1, 1)
            param.quant_storage = torch.uint8
            if name == "weight" and hasattr(module, "quant_storage"):
                module.quant_storage = torch.uint8
            converted += 1
    return converted, left


@contextlib.contextmanager
def packed_4bit_as_uint8(model, enabled = True):
    """Show float-packed 4-bit weights as the uint8 bitsandbytes stores by default for the duration.
    Loader dtype passes (`module.to(module._pre_set_compute_dtype)`, forced float32) cast every
    floating parameter of a module, so packed bytes held in float16 / bfloat16 would be converted as
    if they were weights; as uint8 they are skipped, as they are without FSDP2. Restored afterwards."""
    packed = []
    if enabled:
        for module in model.modules():
            for name, param in module._parameters.items():
                if (
                    param is None
                    or not is_params4bit(param)
                    or not param.is_floating_point()
                    or not getattr(param, "bnb_quantized", False)
                    or getattr(param, "quant_state", None) is None
                    or not param.is_contiguous()
                    or param.device.type == "meta"
                ):
                    continue
                packed.append((module, name, param.dtype))
    for module, name, _ in packed:
        param = module._parameters[name]
        param.data = param.data.reshape(-1).view(torch.uint8).reshape(-1, 1)
        param.quant_storage = torch.uint8
    try:
        yield len(packed)
    finally:
        for module, name, dtype in packed:
            param = module._parameters.get(name)
            if param is None or param.dtype != torch.uint8:
                continue
            param.data = param.data.reshape(-1).view(dtype).reshape(-1, 1)
            param.quant_storage = dtype


def _packed_numel(param):
    if is_params4bit(param) and param.is_floating_point():
        return param.numel() * param.element_size()
    return param.numel()


def patch_size_based_wrap_policy():
    """accelerate's FSDP2 SIZE_BASED_WRAP sums `p.numel()`, which float packing divides by the storage
    itemsize; count those weights as the uint8 bytes they hold so the same blocks get wrapped."""
    try:
        import accelerate.utils.fsdp_utils as fsdp_utils
        from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
    except Exception:
        return False
    original = getattr(fsdp_utils, "fsdp2_prepare_auto_wrap_policy", None)
    if original is None:
        return False
    if getattr(original, "_unsloth_original", None) is not None:
        return True

    def fsdp2_prepare_auto_wrap_policy(fsdp2_plugin, *args, **kwargs):
        policy = original(fsdp2_plugin, *args, **kwargs)
        fn = getattr(fsdp2_plugin, "auto_wrap_policy", None)
        fn = getattr(fn, "func", fn)
        min_num_params = getattr(fsdp2_plugin, "min_num_params", None)
        if policy is None or fn is not size_based_auto_wrap_policy or min_num_params is None:
            return policy

        def size_policy(module):
            return sum(_packed_numel(p) for p in module.parameters()) > min_num_params

        return size_policy

    fsdp2_prepare_auto_wrap_policy.__name__ = original.__name__
    fsdp2_prepare_auto_wrap_policy.__doc__ = original.__doc__
    fsdp2_prepare_auto_wrap_policy._unsloth_original = original
    fsdp_utils.fsdp2_prepare_auto_wrap_policy = fsdp2_prepare_auto_wrap_policy
    return True


@contextlib.contextmanager
def nonfloat_frozen_unmanaged():
    """While FSDP2 shards on torch < 2.8: leave frozen non-float parameters (uint8 Params4bit) out of
    every unit, as accelerate >= 1.14 does with `ignored_params` on torch >= 2.7. torch < 2.8 asserts one
    original dtype over all of a unit's parameters, so uint8 next to float LoRA weights cannot be sharded."""
    modules = []
    for name in ("_fully_shard", "_fsdp_init"):
        try:
            modules.append(_fsdp_private(name))
        except Exception:
            pass

    patched = []
    for module in modules:
        original = getattr(module, "_get_managed_states", None)
        if original is None:
            continue

        def _get_managed_states(
            *args,
            _original = original,
            **kwargs,
        ):
            params, buffers = _original(*args, **kwargs)
            params = [
                p for p in params if p.requires_grad or p.is_floating_point() or p.is_complex()
            ]
            return params, buffers

        module._get_managed_states = _get_managed_states
        patched.append((module, original))
    try:
        yield len(patched)
    finally:
        for module, original in patched:
            module._get_managed_states = original


def _quant_meta(param):
    if not is_params4bit(param) or getattr(param, "quant_state", None) is None:
        return None
    return {k: getattr(param, k) for k in _QUANT_ATTRS if hasattr(param, k)}


def _restore_param_class(
    unsharded,
    cls,
    module = None,
):
    """FSDP2 rebuilds the all-gathered weight as a plain nn.Parameter. Give it back the class it was
    sharded from (bitsandbytes Params4bit), so the `isinstance(weight, Params4bit)` checks of the
    unsloth-zoo 4-bit MoE paths (and of bitsandbytes) see what they see without FSDP2. Same object,
    same storage: only the Python type changes, plus the `module` Params4bit.__torch_function__ reads
    on torch.chunk / torch.split. A DTensor (HSDP / TP) or another subclass is left as it is."""
    if unsharded is None or cls is None or type(unsharded) is cls:
        return
    if type(unsharded) is not torch.nn.Parameter or not issubclass(cls, torch.nn.Parameter):
        return
    try:
        unsharded.__class__ = cls
    except TypeError:
        return
    if not hasattr(unsharded, "module"):
        try:
            unsharded.module = module
        except Exception:
            pass


def _attach_quant_meta(tensor, meta):
    if tensor is None:
        return
    for k, v in meta.items():
        try:
            setattr(tensor, k, v)
        except Exception:
            pass


def _check_fsdp_param_anchors(FSDPParam):
    """The FSDPParam surface the wrappers rely on, in torch 2.6 .. 2.14. A string per missing anchor."""
    missing = []
    methods = {}
    for name in ("_init_sharded_param", "init_unsharded_param", "init_dtype_attrs"):
        method = getattr(FSDPParam, name, None)
        if not callable(method):
            missing.append(f"FSDPParam.{name}")
        methods[name] = getattr(method, "_unsloth_original", method)
    if missing:
        return missing
    try:
        params = list(inspect.signature(methods["_init_sharded_param"]).parameters)
        if len(params) < 2 or params[1] != "param":
            missing.append("FSDPParam._init_sharded_param(self, param, ...)")
    except (TypeError, ValueError):
        missing.append("FSDPParam._init_sharded_param signature")
    try:
        if "self._unsharded_param" not in inspect.getsource(methods["init_unsharded_param"]):
            missing.append("FSDPParam.init_unsharded_param sets self._unsharded_param")
        if "self.param_dtype" not in inspect.getsource(methods["init_dtype_attrs"]):
            missing.append("FSDPParam.init_dtype_attrs sets self.param_dtype")
    except (OSError, TypeError):
        missing.append("FSDPParam source")
    return missing


def _check_collectives_anchors(collectives):
    """foreach_all_gather picks the collective dtype in `_get_all_gather_input_metadatas` (uint8 when
    the inputs' dtypes differ) from what `_get_param_all_gather_inputs` returned, in torch 2.6 .. 2.14."""
    missing = []
    for name in (
        "foreach_all_gather",
        "_get_param_all_gather_inputs",
        "_get_all_gather_input_metadatas",
    ):
        if not callable(getattr(collectives, name, None)):
            missing.append(f"_fsdp_collectives.{name}")
    if missing:
        return missing
    try:
        source = inspect.getsource(collectives.foreach_all_gather)
    except (OSError, TypeError):
        return ["_fsdp_collectives source"]
    for anchor in (
        "_get_param_all_gather_inputs(fsdp_params)",
        "_get_all_gather_input_metadatas(",
        "if dtype == torch.uint8:",
    ):
        if anchor not in source:
            missing.append(f"foreach_all_gather: {anchor}")
    return missing


def _missing_fsdp2_anchors():
    try:
        _fsdp_collectives = _fsdp_private("_fsdp_collectives")
        FSDPParam = _fsdp_private("_fsdp_param").FSDPParam
    except Exception as error:
        return [f"FSDP2 ({type(error).__name__})"]
    return _check_fsdp_param_anchors(FSDPParam) + _check_collectives_anchors(_fsdp_collectives)


def _is_packed_fsdp_param(fsdp_param):
    return getattr(fsdp_param, "_unsloth_quant_meta", None) is not None


# Set by _get_param_all_gather_inputs, read by the _get_all_gather_input_metadatas call right after it
# in the same foreach_all_gather.
_BYTE_ALL_GATHER = [False]
_PATCH_STATE = {"patched": False, "reason": None}


def patch_fsdp2_params4bit():
    """Wrap FSDPParam (and the all-gather's dtype choice) so bitsandbytes 4-bit weights keep their
    QuantState and their packed bytes through FSDP2. Idempotent. Returns (ok, reason): ok is False, with the missing anchors, when this
    torch has no FSDP2 or a FSDPParam this was not written against; nothing is patched then."""
    if _PATCH_STATE["patched"]:
        return True, None
    missing = _missing_fsdp2_anchors()
    if missing:
        return False, f"torch {torch.__version__} FSDP2 lacks " + ", ".join(missing)
    _fsdp_collectives = _fsdp_private("_fsdp_collectives")
    FSDPParam = _fsdp_private("_fsdp_param").FSDPParam

    if getattr(FSDPParam.init_dtype_attrs, "_unsloth_original", None) is not None:
        _PATCH_STATE["patched"] = True
        return True, None

    original_init_sharded_param = FSDPParam._init_sharded_param
    original_init_unsharded_param = FSDPParam.init_unsharded_param
    original_init_dtype_attrs = FSDPParam.init_dtype_attrs

    def _init_sharded_param(self, param, *args, **kwargs):
        meta = _quant_meta(param)
        out = original_init_sharded_param(self, param, *args, **kwargs)
        if meta is not None:
            self._unsloth_quant_meta = meta
            self._unsloth_param_cls = type(param)
            self._unsloth_param_module = getattr(param, "module", None)
            _attach_quant_meta(getattr(self, "sharded_param", None), meta)
        return out

    def init_unsharded_param(self, *args, **kwargs):
        out = original_init_unsharded_param(self, *args, **kwargs)
        meta = getattr(self, "_unsloth_quant_meta", None)
        if meta is not None:
            unsharded = getattr(self, "_unsharded_param", None)
            if (
                unsharded is not None
                and getattr(unsharded, "quant_state", None) is not meta["quant_state"]
            ):
                _attach_quant_meta(unsharded, meta)
            _restore_param_class(
                unsharded,
                getattr(self, "_unsloth_param_cls", None),
                getattr(self, "_unsloth_param_module", None),
            )
            if unsharded is not None and not getattr(unsharded, _GATHERED_MARK, False):
                # unsloth-zoo's 4-bit MoE paths cache NF4 pointer tables by weight address; FSDP2
                # frees and re-allocates this storage around every forward and backward, so they
                # re-key it per call instead (zoo moe_ready_epoch.FSDP2_GATHERED).
                try:
                    setattr(unsharded, _GATHERED_MARK, True)
                except Exception:
                    pass
        return out

    def init_dtype_attrs(self, mp_policy, *args, **kwargs):
        out = original_init_dtype_attrs(self, mp_policy, *args, **kwargs)
        if getattr(self, "param_dtype", None) is None:
            return out
        # A cast reinterprets nothing: it converts values, which destroys packed 4-bit bytes. Leave
        # tensor subclasses with their own all-gather hooks (torchao NF4Tensor) to those hooks.
        sharded = getattr(self, "sharded_param", None)
        local = getattr(sharded, "_local_tensor", sharded)
        packed = getattr(self, "_unsloth_quant_meta", None) is not None or (
            sharded is not None and not sharded.is_floating_point()
        )
        if packed and not hasattr(local, "fsdp_pre_all_gather"):
            self.param_dtype = None
        return out

    original_get_inputs = _fsdp_collectives._get_param_all_gather_inputs
    original_get_metadatas = _fsdp_collectives._get_all_gather_input_metadatas

    def _get_param_all_gather_inputs(fsdp_params, *args, **kwargs):
        out = original_get_inputs(fsdp_params, *args, **kwargs)
        _BYTE_ALL_GATHER[0] = any(_is_packed_fsdp_param(p) for p in fsdp_params)
        return out

    def _get_all_gather_input_metadatas(*args, **kwargs):
        dtypes, numels, dtype = original_get_metadatas(*args, **kwargs)
        if _BYTE_ALL_GATHER[0] and dtype != torch.uint8:
            dtype = torch.uint8
        _BYTE_ALL_GATHER[0] = False
        return dtypes, numels, dtype

    for fn, original in (
        (_init_sharded_param, original_init_sharded_param),
        (init_unsharded_param, original_init_unsharded_param),
        (init_dtype_attrs, original_init_dtype_attrs),
        (_get_param_all_gather_inputs, original_get_inputs),
        (_get_all_gather_input_metadatas, original_get_metadatas),
    ):
        fn.__name__ = original.__name__
        fn.__qualname__ = getattr(original, "__qualname__", original.__name__)
        fn.__doc__ = original.__doc__
        fn._unsloth_original = original

    FSDPParam._init_sharded_param = _init_sharded_param
    FSDPParam.init_unsharded_param = init_unsharded_param
    FSDPParam.init_dtype_attrs = init_dtype_attrs
    _fsdp_collectives._get_param_all_gather_inputs = _get_param_all_gather_inputs
    _fsdp_collectives._get_all_gather_input_metadatas = _get_all_gather_input_metadatas
    _PATCH_STATE["patched"] = True
    return True, None


@contextlib.contextmanager
def nonfloat_param_guard(model = None):
    """While FSDP2 shards: freeze non-float parameters, and make `nn.Parameter(x)` default to
    requires_grad = False for non-float `x`. torch < 2.12 builds the sharded parameter with the default
    requires_grad = True before copying the flag over, which raises on uint8. Restored on exit."""
    if model is not None:
        for p in model.parameters():
            if p.requires_grad and not (p.is_floating_point() or p.is_complex()):
                p.requires_grad_(False)
    Parameter = torch.nn.Parameter
    original_new = Parameter.__dict__.get("__new__")
    # Called through the class: a raw staticmethod object is not callable before Python 3.10.
    call_new = Parameter.__new__

    def __new__(
        cls,
        data = None,
        requires_grad = True,
    ):
        if data is not None and not (data.is_floating_point() or data.is_complex()):
            requires_grad = False
        return call_new(cls, data, requires_grad)

    Parameter.__new__ = __new__
    try:
        yield
    finally:
        if original_new is None:
            del Parameter.__new__
        else:
            Parameter.__new__ = original_new


def prepare_fsdp2_qlora(
    model,
    dtype = None,
    float_storage = None,
):
    """Make a bitsandbytes 4-bit model shardable by FSDP2: float packed storage (unless
    `float_storage` is False, default `float_storage_survives_wrap()`), QuantState kept through
    sharding, no mixed precision cast of packed bytes. A no-op on a model without Params4bit.
    Returns a dict (`params4bit`, `converted`, `patched`, `reason`) for logging and tests."""
    info = {"params4bit": 0, "converted": 0, "patched": False, "reason": None}
    params4bit = [p for p in model.parameters() if is_params4bit(p)]
    info["params4bit"] = len(params4bit)
    if not params4bit:
        return info
    if dtype is None:
        dtype = next(
            (
                getattr(p.quant_state, "dtype", None)
                for p in params4bit
                if getattr(p, "quant_state", None) is not None
            ),
            None,
        )
    if float_storage is None:
        float_storage = float_storage_survives_wrap()
    if float_storage:
        info["converted"] = params4bit_to_float_storage(model, dtype)
    info["patched"], info["reason"] = patch_fsdp2_params4bit()
    patch_size_based_wrap_policy()
    return info


@contextlib.contextmanager
def fsdp2_qlora_wrap(
    model,
    prepare_fn = None,
    native = False,
    dtype = None,
):
    """Around an FSDP2 wrap (accelerate's `fsdp2_prepare_model` is `prepare_fn`, or Unsloth's own
    fully_shard calls with `native = True`): prepare the 4-bit weights, and let uint8 ones still left
    shard on torch < 2.12. Yields the `prepare_fsdp2_qlora` info, or None for a model without
    bitsandbytes 4-bit weights (nothing is touched then). `dtype` is the float storage (default: the
    weights' compute dtype)."""
    if not any(is_params4bit(p) for p in model.parameters()):
        yield None
        return
    survives = float_storage_survives_wrap(prepare_fn, native = native)
    info = prepare_fsdp2_qlora(model, dtype = dtype, float_storage = survives)
    info["reverted"] = 0
    if not survives or not info["patched"]:
        # Float storage here comes only from the caller's own BitsAndBytesConfig. An unpatched FSDP2
        # all-gather, accelerate < 1.13's `model.to(torch.float32)` or torch < 2.8's one dtype per unit
        # would rewrite or refuse it; the same bytes as uint8 take the path uint8 loads take.
        info["reverted"], left = params4bit_to_uint8_storage(model)
        if left:
            raise RuntimeError(
                f"Unsloth: FSDP2 cannot keep {left} float-packed 4-bit weights intact here "
                f"({info['reason'] or 'the wrap would convert them'}). Load them with the default uint8 "
                "bnb_4bit_quant_storage, or train with DDP."
            )
    info["storage"] = sorted(
        {str(p.dtype).replace("torch.", "") for p in model.parameters() if is_params4bit(p)}
    )
    nonfloat = any(not (p.is_floating_point() or p.is_complex()) for p in model.parameters())
    with contextlib.ExitStack() as stack:
        if nonfloat:
            stack.enter_context(nonfloat_param_guard(model))
            if _fsdp2_checks_frozen_dtypes():
                stack.enter_context(nonfloat_frozen_unmanaged())
        yield info
