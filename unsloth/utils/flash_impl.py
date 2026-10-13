# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""Run torch's flash kernels as FA4 / FA3 while a packed / padding-free trainer trains.

torch (2.11+) swaps the flash impl process-wide, and FA4 raises (not falls back) on dropout and missing
head dims and costs ~0.2 ms per launch, so it is on only inside ``trainer.train()``, with no dropout,
supported head dims, and a probe matching torch's bundled flash.
UNSLOTH_FLASH_ATTENTION_IMPL = auto (default: FA4 on sm100; on sm90 FA3 then FA4, only for models with
hidden_size >= _HOPPER_AUTO_MIN_HIDDEN) | fa4 | fa3 | off.
"""

import contextlib
import functools
import inspect
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
# llm_config / language_config: InternVL / Nemotron-style wrappers expose the decoder only there.
_SUBCONFIG_KEYS = (
    "text_config",
    "vision_config",
    "audio_config",
    "encoder",
    "decoder",
    "llm_config",
    "language_config",
)
FLASH_IMPL_STATE = {"active": None, "reason": None, "verdicts": {}, "logged": set()}
# H100: FA3 / FA4 kernels are faster but cost ~60 / ~110 us more host time per forward; at 8K tokens / step
# hidden 1024 trains 2 to 7% slower on FA3, hidden 2048 to 4096 0.3 to 1.8% faster.
_HOPPER_AUTO_MIN_HIDDEN = 2048
_logger = logging.getLogger(__name__)

# B200/FA4: ~0.1 ms extra host time per call makes short-document packed SFT 8-11% slower than bundled flash.
# 1024 matches attention_dispatch._VARLEN_CUDNN_MAX_MEAN_LEN.
_BLACKWELL_AUTO_MIN_MEAN_DOC_LEN = 1024


def _requested():
    value = os.environ.get(_ENV, "auto").strip().lower()
    if value in ("", "auto", "1", "on", "true"):
        return "auto"
    if value in ("fa4", "fa3"):
        return value.upper()
    if value not in ("0", "off", "false", "no", "none"):
        _log_once(
            f"Unsloth: {_ENV}={value!r} is not auto / fa4 / fa3 / off; leaving torch's flash kernels as they are."
        )
    return None


def _log_once(message, level = logging.INFO):
    if message not in FLASH_IMPL_STATE["logged"]:
        FLASH_IMPL_STATE["logged"].add(message)
        _logger.log(level, message)


def _registry():
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
    """auto: FA4 on sm100; FA3 (the mature Hopper kernel) then FA4 on sm90, for wide models only."""
    if not majors or len(majors) != 1:
        return []
    major = next(iter(majors))
    if major == 10:
        return ["FA4"] if requested in ("auto", "FA4") else []
    if major == 9:
        return ["FA3", "FA4"] if requested == "auto" else [requested]
    return []


def _package_error(impl):
    try:
        if impl == "FA4":
            import flash_attn.cute.interface  # noqa: F401
        else:
            import flash_attn_interface  # noqa: F401
        if impl == "FA3":
            ops = getattr(torch.ops, "flash_attn_3", None)
            if ops is None or not hasattr(ops, "fwd") or not hasattr(ops, "bwd"):
                return "flash_attn_interface has no torch.ops.flash_attn_3 fwd / bwd"
    except Exception as exc:
        return f"{_MODULES[impl]} not importable ({type(exc).__name__})"
    return None


def head_dim_supported(
    impl,
    major,
    head_dim,
    head_dim_v = None,
    window = None,
) -> bool:
    """Mirrors flash_attn.cute _validate_head_dims; the sm100 hd256 backward has no sliding window."""
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
    """False when the active impl lacks this shape. Plain attribute reads so compiled callers do not graph break."""
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
        try:
            stack.append(cfg.get_text_config())
        except Exception:
            pass
        sub = getattr(cfg, "sub_configs", None)
        if isinstance(sub, dict):
            stack.extend(getattr(cfg, key, None) for key in sub)
    return out


def _hidden_size(model):
    sizes = [getattr(cfg, "hidden_size", None) for cfg in _configs(model)]
    sizes = [s for s in sizes if isinstance(s, int) and not isinstance(s, bool) and s > 0]
    return max(sizes) if sizes else None


def _ceil8(value):
    return -(-int(value) // 8) * 8


def model_facts(model):
    """(dropout > 0, [(head_dim, head_dim_v)], heads, kv heads, window, dtype); None when unknown."""
    dropout, dims, n_heads, n_kv, window = False, [], None, None, None
    for cfg in _configs(model):
        for key in _DROPOUT_KEYS:
            value = getattr(cfg, key, None)
            if isinstance(value, (int, float)) and value > 0:
                dropout = True
        heads = getattr(cfg, "num_attention_heads", None)
        hidden = getattr(cfg, "hidden_size", None)
        head_dim = getattr(cfg, "head_dim", None)
        if (
            not isinstance(head_dim, int)
            and isinstance(heads, int)
            and isinstance(hidden, int)
            and heads
        ):
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
            if (
                isinstance(sliding, int)
                and sliding > 0
                and (
                    getattr(cfg, "use_sliding_window", True) is not False
                    or any("sliding" in str(t) for t in layer_types)
                )
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
        # Heads > 256 or unequal q / v dims never reach flash.
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
    cu = torch.tensor(
        [0] + torch.tensor(lengths).cumsum(0).tolist(), device = device, dtype = torch.int32
    )
    # torch 2.11's varlen_attn has no enable_gqa: repeat the K/V heads as the packed path does there.
    if "enable_gqa" in inspect.signature(varlen_attn).parameters:
        gqa, k_in, v_in = {"enable_gqa": True}, k, v
    else:
        gqa, k_in, v_in = {}, k.repeat_interleave(ratio, dim = 1), v.repeat_interleave(ratio, dim = 1)
    with torch.enable_grad():
        out = varlen_attn(
            q,
            k_in,
            v_in,
            cu,
            cu,
            max(lengths),
            max(lengths),
            window_size = (window - 1, 0) if window else (-1, 0),
            **gqa,
        )
        out.backward(randn(*out.shape))
    return out.detach(), q.grad, k.grad, v.grad


@contextlib.contextmanager
def _count_impl_calls(impl):
    name = impl.lower()
    try:
        if impl == "FA4":
            from torch.nn.attention import _fa4 as module
        else:
            from torch.nn.attention import _fa3 as module
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
            # varlen_attn on cuDNN never reaches the impl.
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


def maybe_activate_flash_attention_impl(model = None, mean_document_length = None):
    """Returns the active impl or None. Never call while a backward is pending.
    mean_document_length: float, lazy callable (sm100 auto only) or None if unknown."""
    requested = _requested()
    registry = _registry()
    if requested is None or registry is None:
        FLASH_IMPL_STATE["reason"] = (
            "off" if requested is None else "torch has no flash attention impl registry (< 2.11)"
        )
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
    major = next(iter(majors))
    if requested == "auto" and major == 9:
        hidden = _hidden_size(model) if model is not None else None
        if hidden is None or hidden < _HOPPER_AUTO_MIN_HIDDEN:
            FLASH_IMPL_STATE["reason"] = (
                f"auto keeps torch's bundled flash on sm90 below hidden_size {_HOPPER_AUTO_MIN_HIDDEN} "
                f"(model: {hidden}); {_ENV}=fa3 or fa4 opts in"
            )
            deactivate_flash_attention_impl()
            return None
    if requested == "auto" and major == 10 and callable(mean_document_length):
        mean_document_length = mean_document_length()
    if (
        requested == "auto"
        and major == 10
        and mean_document_length is not None
        and mean_document_length < _BLACKWELL_AUTO_MIN_MEAN_DOC_LEN
    ):
        FLASH_IMPL_STATE["reason"] = (
            f"auto keeps torch's bundled flash on sm100 for documents averaging under "
            f"{_BLACKWELL_AUTO_MIN_MEAN_DOC_LEN} tokens ({mean_document_length:.0f}); {_ENV}=fa4 opts in"
        )
        deactivate_flash_attention_impl()
        return None
    facts = model_facts(model) if model is not None else (False, [], None, None, None, None)
    cases = _probe_cases(facts)
    # auto falls through to the next impl when one is missing, refuses this model or fails its probe.
    for impl in candidates:
        error = _package_error(impl) or _model_gate(impl, major, facts)
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
        if error is None:
            break
        deactivate_flash_attention_impl()
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
    if FLASH_IMPL_STATE["active"] is None:
        return
    FLASH_IMPL_STATE["active"] = None
    registry = _registry()
    if registry is not None:
        try:
            registry[1](_raise_warn = False)
        except Exception:
            pass


def _sampled_rows(dataset, indices, key):
    # Custom transform = unknown: rows may differ from storage and running it early advances its RNG.
    format = getattr(dataset, "format", None) or {}
    if format.get("type") == "custom":
        return None
    columns = set(dataset.column_names)
    if format.get("columns") is not None and not format.get("output_all_columns", False):
        columns &= set(format["columns"])
    wanted = [c for c in dict.fromkeys((key, "input_ids")) if c in columns]
    if not wanted:
        return None
    return list(dataset.with_format(None).select_columns(wanted).select(indices))


def _mean_document_length(dataset, rows = 128, key = "seq_lengths"):
    # Rows spread across the dataset: TRL BFD puts the longest documents first.
    try:
        total = len(dataset)
        if total <= 0:
            return None
        n = min(rows, total)
        indices = sorted({i * (total - 1) // (n - 1) for i in range(n)}) if n > 1 else [0]
        sample = _sampled_rows(dataset, indices, key)
        if sample is None:
            return None
        lengths = []
        for row in sample:
            per_document = row.get(key)
            if per_document is None:
                ids = row.get("input_ids")
                if ids is None:
                    continue
                per_document = [len(ids)]
            lengths.extend(int(x) for x in per_document)
        if not lengths:
            lengths = [len(row["input_ids"]) for row in sample if row.get("input_ids") is not None]
    except Exception:
        return None
    lengths = [x for x in lengths if x > 0]
    return sum(lengths) / len(lengths) if lengths else None


def scope_flash_attention_impl_to_train(trainer, model, sequence_lengths_key = "seq_lengths") -> None:
    """Active only during trainer.train: FA4's launch cost makes decode attention ~6x slower."""
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
            activated = maybe_activate_flash_attention_impl(
                getattr(trainer, "model", None) or model,
                mean_document_length = lambda: _mean_document_length(
                    getattr(trainer, "train_dataset", None), key = sequence_lengths_key
                ),
            )
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
