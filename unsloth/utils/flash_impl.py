# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""Run torch's flash kernels as FA4 (flash_attn.cute) or FA3 (flash_attn_interface) while a packed /
padding-free trainer trains, on Hopper / datacenter Blackwell, when the user installed the kernels.

torch (2.11+) swaps the implementation behind its flash ops process-wide, so this also changes dense
SDPA calls that pick flash (on sm100 only decode-shaped calls and dropout: cuDNN takes the rest). FA4
raises instead of falling back on dropout and on head dims it lacks, and its Python launch costs about
0.2 ms per call, so it is turned on only inside ``trainer.train()`` and only when the model has no
attention dropout, every head dim it can send to flash is one the kernel has, and a probe on the
model's own shapes matches torch's bundled flash. Anything uncertain leaves torch untouched.

UNSLOTH_FLASH_ATTENTION_IMPL = off (default) | auto (FA4 on sm100 when every gate passes) | fa4 | fa3 (sm90).
Opt-in: FA4 helps long documents (about 13% per step at 16k on B200) and is neutral to slower on short
ones, where its launch cost dominates.
"""

import contextlib
import functools
import logging
import os

import torch

__all__ = [
    "maybe_activate_flash_attention_impl",
    "deactivate_flash_attention_impl",
    "scope_flash_attention_impl_to_train",
    "flash_impl_allows",
    "FLASH_IMPL_STATE",
]

_ENV = "UNSLOTH_FLASH_ATTENTION_IMPL"
_MODULES = {"FA4": "flash_attn.cute.interface", "FA3": "flash_attn_interface"}
_DROPOUT_KEYS = (
    "attention_dropout",
    "attn_dropout",
    "attn_pdrop",
    "attention_probs_dropout_prob",
    "attention_dropout_prob",
)
_SUBCONFIG_KEYS = ("text_config", "vision_config", "audio_config", "encoder", "decoder")
# active: impl Unsloth turned on (None otherwise); reason: last decision; verdicts: probe results per shape key.
FLASH_IMPL_STATE = {"active": None, "reason": None, "verdicts": {}, "logged": set()}
_logger = logging.getLogger(__name__)


def _requested():
    value = os.environ.get(_ENV, "off").strip().lower()
    if value in ("auto", "1", "on", "true"):
        return "auto"
    if value in ("fa4", "fa3"):
        return value.upper()
    if value not in ("", "0", "off", "false", "no", "none"):
        _log_once(f"Unsloth: {_ENV}={value!r} is not auto / fa4 / fa3 / off; leaving torch's flash kernels as they are.")
    return None


def _log_once(message, level = logging.INFO):
    if message not in FLASH_IMPL_STATE["logged"]:
        FLASH_IMPL_STATE["logged"].add(message)
        _logger.log(level, message)


def _registry():
    """(activate, restore, list, current) from torch >= 2.11, else None."""
    try:
        from torch.nn.attention import (
            activate_flash_attention_impl,
            current_flash_attention_impl,
            list_flash_attention_impls,
            restore_flash_attention_impl,
        )
    except Exception:
        return None
    return (
        activate_flash_attention_impl,
        restore_flash_attention_impl,
        list_flash_attention_impls,
        current_flash_attention_impl,
    )


def _current_impl():
    registry = _registry()
    if registry is None:
        return None
    try:
        return registry[3]()
    except Exception:
        return None


def _device_majors():
    if torch.version.hip is not None or not torch.cuda.is_available():
        return None
    try:
        return {torch.cuda.get_device_capability(i)[0] for i in range(torch.cuda.device_count())}
    except Exception:
        return None


def _candidates(requested, majors):
    """Impls worth trying on these devices. torch's FA4 hook takes majors 9 / 10, its FA3 hook 9.
    auto only turns FA4 on for sm100 (measured); Hopper (FA3 / FA4) is opt-in until measured there."""
    if not majors or len(majors) != 1:
        return []
    major = next(iter(majors))
    if major == 10:
        return ["FA4"] if requested in ("auto", "FA4") else []
    if major == 9 and requested in ("FA3", "FA4"):
        return [requested]
    return []


def _package_error(impl):
    try:
        import importlib

        importlib.import_module(_MODULES[impl])
        if impl == "FA3":
            ops = getattr(torch.ops, "flash_attn_3", None)
            if ops is None or not hasattr(ops, "fwd") or not hasattr(ops, "bwd"):
                return "flash_attn_interface has no torch.ops.flash_attn_3 fwd / bwd"
    except Exception as exc:
        return f"{_MODULES[impl]} not importable ({type(exc).__name__})"
    return None


def head_dim_supported(impl, major, head_dim, head_dim_v = None, window = None) -> bool:
    """Shapes the impl's kernels take (flash_attn.cute interface._validate_head_dims and the sm100
    hd256 backward, which has no local / sliding window). torch pads dense SDPA heads to a multiple of 8."""
    head_dim_v = head_dim if head_dim_v is None else head_dim_v
    if head_dim % 8 or head_dim_v % 8 or min(head_dim, head_dim_v) < 8:
        return False
    if impl == "FA3" or major == 9:
        return head_dim <= 256 and head_dim_v <= 256
    if head_dim <= 128 and head_dim_v <= 128:
        return True
    if (head_dim, head_dim_v) == (192, 128):
        return True
    return (head_dim, head_dim_v) == (256, 256) and not window


try:
    from torch.nn.attention import _registry as _TORCH_FLASH_REGISTRY
except Exception:
    _TORCH_FLASH_REGISTRY = None


def flash_impl_allows(head_dim, window, device) -> bool:
    """Per varlen call: False when a non-default flash impl is active and lacks this shape, so the caller
    keeps its per-segment / wrapped SDPA route instead of an assertion from inside the kernel. Plain
    attribute reads, so a compiled caller guards on the active impl instead of breaking its graph."""
    active = getattr(_TORCH_FLASH_REGISTRY, "_FLASH_ATTENTION_ACTIVE", None)
    if active is None:
        return True
    major = torch.cuda.get_device_capability(device)[0]
    return head_dim_supported(active[0], major, head_dim, window = window)


def _configs(model):
    config = getattr(model, "config", None)
    seen, out, stack = set(), [], [config] if config is not None else []
    while stack:
        cfg = stack.pop()
        if cfg is None or id(cfg) in seen:
            continue
        seen.add(id(cfg))
        out.append(cfg)
        for key in _SUBCONFIG_KEYS:
            stack.append(getattr(cfg, key, None))
        sub = getattr(cfg, "sub_configs", None)
        if isinstance(sub, dict):
            stack.extend(getattr(cfg, key, None) for key in sub)
    return out


def _ceil8(value):
    return -(-int(value) // 8) * 8


def model_facts(model):
    """(attention dropout > 0, [(head_dim, head_dim_v)], heads, kv heads, sliding window, dtype) from the
    model's config and attention modules; None fields when unknown."""
    dropout, dims, n_heads, n_kv, window = False, [], None, None, None
    for cfg in _configs(model):
        for key in _DROPOUT_KEYS:
            value = getattr(cfg, key, None)
            if isinstance(value, (int, float)) and value > 0:
                dropout = True
        heads = getattr(cfg, "num_attention_heads", None)
        hidden = getattr(cfg, "hidden_size", None)
        head_dim = getattr(cfg, "head_dim", None)
        if not isinstance(head_dim, int) and isinstance(heads, int) and isinstance(hidden, int) and heads:
            head_dim = hidden // heads
        qk = [getattr(cfg, k, None) for k in ("qk_nope_head_dim", "qk_rope_head_dim")]
        v_dim = getattr(cfg, "v_head_dim", None)
        if all(isinstance(x, int) for x in qk) and isinstance(v_dim, int):
            dims.append((qk[0] + qk[1], v_dim))
        elif isinstance(head_dim, int) and head_dim > 0:
            dims.append((head_dim, head_dim))
        for key in ("global_head_dim", "local_head_dim"):
            value = getattr(cfg, key, None)
            if isinstance(value, int) and value > 0:
                dims.append((value, value))
        if n_heads is None and isinstance(heads, int):
            n_heads = heads
            kv = getattr(cfg, "num_key_value_heads", None)
            n_kv = kv if isinstance(kv, int) and kv > 0 else heads
            sliding = getattr(cfg, "sliding_window", None)
            layer_types = getattr(cfg, "layer_types", None) or ()
            if isinstance(sliding, int) and sliding > 0 and (
                getattr(cfg, "use_sliding_window", True) is not False
                or any("sliding" in str(t) for t in layer_types)
            ):
                window = sliding
    modules = getattr(model, "modules", None)
    if callable(modules):
        try:
            for module in modules():
                if "attention" not in type(module).__name__.lower():
                    continue
                for key in ("attention_dropout", "dropout", "attn_dropout", "dropout_p"):
                    value = getattr(module, key, None)
                    p = getattr(value, "p", value)
                    if isinstance(p, (int, float)) and not isinstance(p, bool) and p > 0:
                        dropout = True
        except Exception:
            pass
    dtype = getattr(model, "dtype", None)
    return dropout, dims, n_heads, n_kv, window, dtype


def _model_gate(impl, major, facts):
    dropout, dims, *_ = facts
    if dropout:
        return "the model has attention dropout > 0 (FA4 / FA3 raise on dropout instead of falling back)"
    for head_dim, head_dim_v in dims:
        # Heads above 256 never reach a flash kernel (torch and the varlen sites refuse them first);
        # torch only calls flash with equal q / v head dims.
        if max(head_dim, head_dim_v) > 256 or head_dim != head_dim_v:
            continue
        if not head_dim_supported(impl, major, _ceil8(head_dim)):
            return f"head_dim {head_dim} is not one {impl} has on sm{major}x"
    return None


def _probe_cases(facts):
    _, dims, n_heads, n_kv, window, dtype = facts
    dtype = dtype if dtype in (torch.float16, torch.bfloat16) else torch.bfloat16
    ratio = n_heads // n_kv if n_heads and n_kv and n_heads % n_kv == 0 else 4
    head_dims = sorted({d for d, v in dims if d == v and 8 <= d <= 256 and d % 8 == 0}) or [128]
    cases = []
    for head_dim in head_dims[:2]:
        cases.append(("varlen", head_dim, ratio, None, dtype))
        if window:
            cases.append(("varlen", head_dim, ratio, min(int(window), 64), dtype))
        cases.append(("decode", head_dim, ratio, None, dtype))
    return cases


def _run_case(case, device):
    kind, head_dim, ratio, window, dtype = case
    g = torch.Generator(device = device).manual_seed(1234)
    n_kv = 2
    n_heads = n_kv * ratio

    def randn(*shape):
        return torch.randn(*shape, device = device, dtype = dtype, generator = g)

    if kind == "decode":
        q, k = randn(1, n_heads, 1, head_dim), randn(1, n_kv, 77, head_dim)
        v = randn(1, n_kv, 77, head_dim)
        with torch.no_grad():
            return (torch.nn.functional.scaled_dot_product_attention(q, k, v, enable_gqa = True),)
    from torch.nn.attention.varlen import varlen_attn

    lengths = [200, 1, 333, 64, 130]
    total = sum(lengths)
    q = randn(total, n_heads, head_dim).requires_grad_()
    k = randn(total, n_kv, head_dim).requires_grad_()
    v = randn(total, n_kv, head_dim).requires_grad_()
    cu = torch.tensor([0] + torch.tensor(lengths).cumsum(0).tolist(), device = device, dtype = torch.int32)
    with torch.enable_grad():
        out = varlen_attn(
            q, k, v, cu, cu, max(lengths), max(lengths),
            window_size = (window - 1, 0) if window else (-1, 0),
            enable_gqa = True,
        )
        out.backward(randn(*out.shape))
    return out.detach(), q.grad, k.grad, v.grad


@contextlib.contextmanager
def _count_impl_calls(impl):
    """Count torch's {impl} forward / backward runner calls (None when torch names them differently)."""
    name = impl.lower()
    try:
        import importlib

        module = importlib.import_module(f"torch.nn.attention._{name}")
        fwd, bwd = getattr(module, f"_{name}_run_forward"), getattr(module, f"_{name}_run_backward")
    except Exception:
        yield None
        return
    calls = {"fwd": 0, "bwd": 0}

    def counted(key, fn):
        def run(*args, **kwargs):
            calls[key] += 1
            return fn(*args, **kwargs)

        return run

    setattr(module, f"_{name}_run_forward", counted("fwd", fwd))
    setattr(module, f"_{name}_run_backward", counted("bwd", bwd))
    try:
        yield calls
    finally:
        setattr(module, f"_{name}_run_forward", fwd)
        setattr(module, f"_{name}_run_backward", bwd)


def _probe_error(impl, cases, activate, restore):
    """Reference cases on torch's bundled flash, then activate impl and compare; restore on mismatch."""
    device = torch.device("cuda", torch.cuda.current_device())
    try:
        refs = [_run_case(case, device) for case in cases]
    except Exception as exc:
        return f"reference flash probe failed ({type(exc).__name__}: {exc})"
    try:
        activate(impl)
    except Exception as exc:
        return f"activate_flash_attention_impl({impl!r}) failed ({type(exc).__name__}: {exc})"
    try:
        for case, ref in zip(cases, refs):
            with _count_impl_calls(impl) as calls:
                got = _run_case(case, device)
                torch.cuda.synchronize(device)
            # A torch whose varlen_attn picks cuDNN for causal rows never reaches the impl: nothing to gain.
            if case[0] == "varlen" and calls is not None and not (calls["fwd"] and calls["bwd"]):
                raise RuntimeError(f"varlen_attn did not run on {impl} ({calls})")
            tol = 2e-2 if case[-1] == torch.bfloat16 else 1e-2
            for a, b in zip(got, ref):
                if not torch.isfinite(a).all():
                    raise RuntimeError(f"non-finite output for {case[:4]}")
                err = (a.float() - b.float()).abs().max().item()
                if err > tol * max(1.0, b.float().abs().max().item()):
                    raise RuntimeError(f"{case[:4]} differs from bundled flash by {err:.3g}")
    except Exception as exc:
        try:
            restore(_raise_warn = False)
        except Exception:
            pass
        return f"probe failed ({type(exc).__name__}: {exc})"
    return None


def maybe_activate_flash_attention_impl(model = None):
    """Turn on FA4 / FA3 for torch's flash kernels when every gate passes. Returns the impl now active
    because of this call or an earlier one, else None. Never call while a backward is pending."""
    requested = _requested()
    registry = _registry()
    if requested is None or registry is None:
        FLASH_IMPL_STATE["reason"] = "off" if requested is None else "torch has no flash attention impl registry (< 2.11)"
        return None
    if torch.compiler.is_compiling():
        return FLASH_IMPL_STATE["active"]
    activate, restore, list_impls, current = registry
    current_impl = current()
    if current_impl is not None and FLASH_IMPL_STATE["active"] is None:
        FLASH_IMPL_STATE["reason"] = f"{current_impl} was activated outside Unsloth; left as is"
        return None
    majors = _device_majors()
    candidates = [impl for impl in _candidates(requested, majors) if impl in list_impls()]
    if not candidates:
        FLASH_IMPL_STATE["reason"] = f"no supported impl for devices sm{sorted(majors or [])}"
        deactivate_flash_attention_impl()
        return None
    impl, major = candidates[0], next(iter(majors))
    error = _package_error(impl)
    facts = model_facts(model) if model is not None else (False, [], None, None, None, None)
    error = error or _model_gate(impl, major, facts)
    cases = _probe_cases(facts)
    key = (impl, tuple(cases))
    if error is None:
        if key not in FLASH_IMPL_STATE["verdicts"]:
            # The reference must run on the bundled kernels.
            deactivate_flash_attention_impl()
            error = _probe_error(impl, cases, activate, restore)
            FLASH_IMPL_STATE["verdicts"][key] = error
        else:
            error = FLASH_IMPL_STATE["verdicts"][key]
            if error is None and current() != impl:
                try:
                    activate(impl)
                except Exception as exc:
                    error = f"activate_flash_attention_impl({impl!r}) failed ({type(exc).__name__}: {exc})"
    if error is not None:
        FLASH_IMPL_STATE["reason"] = error
        deactivate_flash_attention_impl()
        if requested != "auto":
            _log_once(f"Unsloth: {_ENV}={requested.lower()} not used: {error}", logging.WARNING)
        return None
    FLASH_IMPL_STATE["active"] = impl
    FLASH_IMPL_STATE["reason"] = f"{impl} active"
    message = f"Unsloth: packed attention trains on {impl} ({_MODULES[impl]}); {_ENV}=off keeps torch's bundled flash."
    if message not in FLASH_IMPL_STATE["logged"]:
        FLASH_IMPL_STATE["logged"].add(message)
        print(message)
    return impl


def deactivate_flash_attention_impl():
    """Put torch's bundled flash back if Unsloth activated another impl."""
    if FLASH_IMPL_STATE["active"] is None:
        return
    FLASH_IMPL_STATE["active"] = None
    registry = _registry()
    if registry is not None:
        try:
            registry[1](_raise_warn = False)
        except Exception:
            pass


def scope_flash_attention_impl_to_train(trainer, model) -> None:
    """Wrap trainer.train once so the impl is active only while it trains: inference after training
    keeps torch's bundled flash (FA4's Python launch makes each decode step's attention ~6x slower)."""
    if trainer is None or getattr(trainer, "_unsloth_flash_impl_wrapped", False):
        return
    if _requested() is None or _registry() is None:
        return
    original = getattr(trainer, "train", None)
    if not callable(original):
        return

    @functools.wraps(original)
    def train(*args, **kwargs):
        activated = None
        try:
            activated = maybe_activate_flash_attention_impl(getattr(trainer, "model", None) or model)
        except Exception as exc:
            FLASH_IMPL_STATE["reason"] = f"activation error ({type(exc).__name__}: {exc})"
            deactivate_flash_attention_impl()
        try:
            return original(*args, **kwargs)
        finally:
            if activated is not None:
                deactivate_flash_attention_impl()

    try:
        trainer.train = train
        trainer._unsloth_flash_impl_wrapped = True
    except Exception:
        pass
