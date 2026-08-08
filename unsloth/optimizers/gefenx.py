"""Gefen-X optimizer integration for Unsloth.

Two optimizers are exposed through Unsloth's config-object pattern (mirroring
``QGaloreConfig`` / the Muon integration):

* ``gefenx``       — ``gefen.Gefen``: an AdamW-family optimizer that keeps its
  optimizer state at roughly 1 byte per parameter (8-bit quantized momentum +
  factored / block-shared second moment).
* ``gefenx_muon``  — ``gefen.GefenMuonHybrid``: Muon (Newton-Schulz
  orthogonalization) on the 2D hidden weight matrices, plain Gefen on
  embeddings / heads / norms / biases, at the same ≈1 B/param footprint.

The optimizers themselves live entirely in the upstream ``gefen`` (``gefen-x``)
package; this module only maps Unsloth's ``GefenXConfig`` / ``GefenXMuonConfig``
to the right constructor plus the parameter routing Unsloth already uses for its
other optimizers. ``gefen`` is imported lazily so ``import unsloth`` does not
require the package unless a Gefen-X optimizer is actually requested.

The dataclasses live in ``unsloth.trainer`` (next to ``QGaloreConfig``); this
module reads their fields by attribute, so it stays decoupled from their exact
definition.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Tuple

# Keys the builders pass explicitly (positionally or by keyword) to the gefen
# constructors, or that Unsloth consumes elsewhere (embedding LR / param routing).
# Allowing them through the ``extra_kwargs`` escape hatch would duplicate a keyword
# argument (TypeError), shadow ``learning_rate``, or land as an unexpected keyword,
# so they are dropped from ``extra_kwargs`` with a warning — set them via the
# dedicated config field / ``learning_rate`` / ``weight_decay`` /
# ``embedding_learning_rate`` instead.
_RESERVED_EXTRA_KWARGS = frozenset(
    {
        "params",
        "model",
        "lr",
        "weight_decay",
        "backup_substrings",
        "backup_lr_scale",
        "embedding_lr",
        "embedding_learning_rate",
    }
)


def _require_nvidia_cuda() -> None:
    """Gate the Gefen-X optimizers to NVIDIA CUDA.

    ``gefen`` ships CUDA-only kernels (``*_cuda`` ops, ``device.type != "cuda"``
    guards), so AMD/ROCm (HIP) and Intel XPU are unsupported. On ROCm PyTorch
    ``torch.cuda.is_available()`` is ``True`` and device type reports ``"cuda"``,
    which would silently route into kernels compiled for NVIDIA — so detect HIP by
    build tag and fail fast with a clear message instead. (Apple MLX never reaches
    here: Unsloth's MLX build uses a separate trainer that has no Gefen-X path.)

    A CPU-only box (no CUDA, no HIP/XPU) is allowed — ``gefen`` transparently
    downgrades its fused path to the pure-PyTorch step there, which keeps the
    optimizers usable for tests and small CPU experiments.
    """
    import torch

    hip = getattr(getattr(torch, "version", None), "hip", None)
    if hip:
        raise RuntimeError(
            "Unsloth: the Gefen-X optimizers require NVIDIA CUDA, but this is an "
            f"AMD/ROCm (HIP {hip}) build of PyTorch. gefen's CUDA kernels do not "
            "support ROCm."
        )
    if (
        hasattr(torch, "xpu")
        and torch.xpu.is_available()
        and not (hasattr(torch, "cuda") and torch.cuda.is_available())
    ):
        raise RuntimeError(
            "Unsloth: the Gefen-X optimizers require NVIDIA CUDA; Intel XPU is "
            "unsupported by gefen's kernels."
        )


def coerce_optim_arg(value: Any) -> Any:
    """Restore native types on string-form optimizer args.

    String ``optim_args`` (``key=value``) arrive as strings; turn the obvious
    ``true``/``false``/``none``/int/float literals back into real values. Mirrors
    the axolotl Gefen-X integration so a config authored either way behaves the
    same.
    """
    if not isinstance(value, str):
        return value
    lowered = value.strip().lower()
    if lowered in ("true", "false"):
        return lowered == "true"
    if lowered in ("none", "null"):
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


# Fields on GefenXConfig that map 1:1 onto gefen.Gefen keyword arguments.
_GEFENX_FIELDS: Tuple[str, ...] = (
    "fused",
    "factored_v_2d",
    "force_1d_period_one",
    "force_2d_period_one",
    "period_one_substrings",
    "codebook_refresh_every",
    "stochastic_round",
    "capturable",
)

# Fields on GefenXMuonConfig that map 1:1 onto gefen.GefenMuonHybrid kwargs.
# `backup_lr_scale`, `backup_substrings`, `betas` and `eps` are handled specially.
_GEFENX_MUON_FIELDS: Tuple[str, ...] = (
    "fused",
    "adjust_lr_fn",
    "muon_lr",
    "backup_lr",
    "muon_weight_decay",
    "backup_weight_decay",
    "backup_1d_period_one",
    "backup_2d_period_one",
    "momentum",
    "nesterov",
    "ns_steps",
    "ns_schedule",
    "sharded_mode",
    "fp8_ns",
    "stochastic_round",
    "normuon",
    "cautious",
    "capturable",
)


def _collect_kwargs(config, fields: Tuple[str, ...]) -> Dict[str, Any]:
    """Pull the named attributes off `config`, plus its `extra_kwargs` escape hatch."""
    kwargs: Dict[str, Any] = {}
    for name in fields:
        if hasattr(config, name):
            value = getattr(config, name)
            # An empty period_one_substrings means "use the gefen default"; drop it
            # so we don't override with () when the user never set it.
            if name == "period_one_substrings" and not value:
                continue
            # None means "unset" in our configs (e.g. muon_lr / *_weight_decay) —
            # skip it so gefen applies its own runtime default instead of being
            # handed an explicit None.
            if value is None:
                continue
            kwargs[name] = value
    extra = getattr(config, "extra_kwargs", None)
    if extra:
        for key, value in extra.items():
            if key in _RESERVED_EXTRA_KWARGS:
                warnings.warn(
                    f"Unsloth: ignoring reserved key {key!r} in Gefen-X extra_kwargs; "
                    f"set it via the config field / learning_rate / weight_decay instead."
                )
                continue
            kwargs[key] = coerce_optim_arg(value)
    return kwargs


def make_gefenx_param_groups(
    model,
    lr: float,
    weight_decay: float,
    embedding_lr: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """Group trainable params for plain ``gefen.Gefen``.

    Matches Unsloth's ``_create_unsloth_optimizer`` embedding split: params saved
    via PEFT ``modules_to_save`` (embeddings / heads trained at full rank) get the
    dedicated ``embedding_lr`` when one is provided; everything else shares ``lr``.

    Params are emitted as ``(name, param)`` pairs so gefen keeps the real
    parameter names — otherwise it synthesises ``group_i_param_j`` names and
    ``GefenXConfig.period_one_substrings`` (matched against names) never fires.
    Only non-empty buckets become groups: gefen rejects an empty parameter group,
    so an embeddings-only run (all trainable params are ``modules_to_save``) would
    otherwise crash on the empty ``non_embeddings`` group.
    """
    non_embeddings: List[Any] = []
    embeddings: List[Any] = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if embedding_lr is not None and name.endswith("modules_to_save.default.weight"):
            partial_name = name[: -len(".modules_to_save.default.weight")]
            partial_name = partial_name[partial_name.rfind(".") + 1 :]
            print(
                f"Unsloth: Setting lr = {embedding_lr:.2e} instead of {lr:.2e} for {partial_name}."
            )
            embeddings.append((name, param))
        else:
            non_embeddings.append((name, param))

    param_groups: List[Dict[str, Any]] = []
    if non_embeddings:
        param_groups.append({"params": non_embeddings, "weight_decay": weight_decay, "lr": lr})
    if embeddings:
        param_groups.append(
            {"params": embeddings, "weight_decay": weight_decay, "lr": embedding_lr}
        )
    if not param_groups:
        # No trainable params at all — hand gefen a single (empty) group so it
        # raises its own clear "empty parameter list" error.
        param_groups.append({"params": non_embeddings, "weight_decay": weight_decay, "lr": lr})
    return param_groups


def build_gefenx_optimizer(
    model,
    config,
    *,
    lr: float,
    weight_decay: float,
    betas: Tuple[float, float],
    eps: float,
    embedding_lr: Optional[float] = None,
):
    """Construct ``gefen.Gefen`` from a ``GefenXConfig`` and the trainer's hyperparams."""
    _require_nvidia_cuda()
    from gefen import Gefen

    kwargs = _collect_kwargs(config, _GEFENX_FIELDS)
    # Trainer's AdamW betas/eps are the fallback; a config override wins.
    config_betas = getattr(config, "betas", None)
    config_eps = getattr(config, "eps", None)
    kwargs.setdefault("betas", tuple(config_betas) if config_betas is not None else betas)
    kwargs.setdefault("eps", config_eps if config_eps is not None else eps)

    param_groups = make_gefenx_param_groups(model, lr, weight_decay, embedding_lr)
    return Gefen(param_groups, lr = lr, weight_decay = weight_decay, **kwargs)


def build_gefenx_muon_optimizer(
    model, config, *, lr: float, weight_decay: float, betas: Tuple[float, float], eps: float
):
    """Construct ``gefen.GefenMuonHybrid`` from a ``GefenXMuonConfig``.

    ``GefenMuonHybrid.from_model`` splits the model's params (2D hidden weights →
    Muon, everything else → Gefen backup), validates the split covers every
    trainable parameter exactly once, and constructs the hybrid. The Gefen-X
    recommended recipe (``backup_1d_period_one`` / ``adjust_lr_fn`` /
    ``fused`` / ``backup_lr = backup_lr_scale * lr``) is applied as defaults, all
    overridable through the config.
    """
    _require_nvidia_cuda()
    from gefen import GefenMuonHybrid

    kwargs = _collect_kwargs(config, _GEFENX_MUON_FIELDS)

    # Recommended recipe defaults (only fill in what the config left unset).
    kwargs.setdefault("backup_1d_period_one", True)
    kwargs.setdefault("adjust_lr_fn", "match_rms_adamw")
    kwargs.setdefault("fused", True)

    # backup_lr defaults to backup_lr_scale * lr (validated loss lever) unless the
    # config set an explicit backup_lr.
    if kwargs.get("backup_lr") is None:
        scale = getattr(config, "backup_lr_scale", 0.5)
        if scale is not None:
            kwargs["backup_lr"] = scale * lr

    config_betas = getattr(config, "betas", None)
    config_eps = getattr(config, "eps", None)
    if config_betas is not None:
        kwargs["betas"] = tuple(config_betas)
    else:
        kwargs.setdefault("betas", betas)
    if config_eps is not None:
        kwargs["eps"] = config_eps
    else:
        kwargs.setdefault("eps", eps)

    backup_substrings = getattr(config, "backup_substrings", None)

    return GefenMuonHybrid.from_model(
        model,
        lr = lr,
        weight_decay = weight_decay,
        backup_substrings = backup_substrings,
        **kwargs,
    )
