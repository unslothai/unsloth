# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Hosted bf16 VAE for a bf16 diffusers pipeline load whose base repo stores its VAE in fp32.

``DiffusionFamily.hosted_bf16_vae`` names, per upstream base, a hosted single-file bf16 copy of ``vae/``. A bf16
``from_pretrained`` casts the fp32 file to bf16 on load, so reading the bf16 copy gives the same module from half the
bytes. Used only when the load dtype is bf16 and the pick is a full pipeline; fp16 / fp32 loads keep ``vae/``.

The download plan drops the base's dense ``vae/`` weights only when the hosted file resolves on the Hub
(``hosted_vae_plan_files``). The load (``hosted_vae_pipe_kwargs``) builds the VAE strictly from the base's own
``vae/config.json`` and checks every tensor is bf16; any failure logs the reason and loads the base ``vae/`` instead,
fetching it first when the plan left it out. ``UNSLOTH_DIFFUSION_HOSTED_VAE=0`` turns the whole path off.
"""

from __future__ import annotations

import os
from typing import Any, Optional

HOSTED_VAE_ENV = "UNSLOTH_DIFFUSION_HOSTED_VAE"
HOSTED_VAE_COMPONENT = "vae"


def hosted_vae_enabled() -> bool:
    return (os.environ.get(HOSTED_VAE_ENV) or "").strip().lower() not in ("0", "off", "false", "no")


def _is_bf16(dtype: Any) -> bool:
    return str(dtype) in ("torch.bfloat16", "bfloat16")


def hosted_vae_source(fam: Any, base: Optional[str]) -> Optional[tuple[str, str]]:
    """``(repo_id, filename)`` of the hosted bf16 VAE for ``base`` in ``fam``, or None. Pure, never raises."""
    try:
        from .diffusion_families import canonical_base

        key = canonical_base(str(base or "")).strip().lower()
        if not key:
            return None
        for row in getattr(fam, "hosted_bf16_vae", ()) or ():
            if isinstance(row, (tuple, list)) and len(row) == 3 and str(row[0]).lower() == key:
                return str(row[1]), str(row[2])
    except Exception:  # noqa: BLE001 -- a bad table row is "no hosted VAE"
        return None
    return None


def hosted_vae_wanted(
    fam: Any, base: Optional[str], *, kind: Optional[str], dtype: Any
) -> Optional[tuple[str, str]]:
    """The hosted VAE this load would read, or None: switch on, full pipeline, bf16 load dtype, a row for the base."""
    if not hosted_vae_enabled() or kind != "pipeline" or not _is_bf16(dtype):
        return None
    return hosted_vae_source(fam, base)


def hosted_vae_plan_files(
    fam: Any,
    base: Optional[str],
    *,
    kind: Optional[str],
    dtype: Any,
    hf_token: Optional[str] = None,
    local_files_only: bool = False,
    logger: Any = None,
) -> dict[str, tuple[str, list[tuple[str, int]]]]:
    """``{"vae": (repo_id, [(filename, size)])}`` when the plan may drop the base's dense ``vae/`` weights, else {}.

    Only a file the Hub lists with a size earns the drop, so a missing or unreachable artifact keeps ``vae/`` in the
    plan. Offline, nothing is dropped (the load still prefers a cached hosted file). Never raises."""
    src = hosted_vae_wanted(fam, base, kind = kind, dtype = dtype)
    if src is None or local_files_only:
        return {}
    repo_id, filename = src
    try:
        from huggingface_hub import HfApi
        infos = HfApi(token = hf_token or None).get_paths_info(repo_id, [filename])
        size = next((int(getattr(i, "size", 0) or 0) for i in infos if i.path == filename), 0)
    except Exception as exc:  # noqa: BLE001 -- unresolvable: keep the dense VAE
        if logger is not None:
            logger.info("diffusion.hosted_vae: %s/%s not resolvable (%s)", repo_id, filename, exc)
        return {}
    if size <= 0:
        return {}
    return {HOSTED_VAE_COMPONENT: (repo_id, [(filename, size)])}


def _vae_config(
    source: str, *, hf_token: Optional[str], local_files_only: bool, cache_dir: Optional[str]
) -> dict:
    """The base's ``vae/config.json``, from a staged / local folder or the Hub cache."""
    import json

    local = os.path.join(os.path.expanduser(source), "vae", "config.json")
    if not os.path.isfile(local):
        from huggingface_hub import hf_hub_download
        local = hf_hub_download(
            source,
            "vae/config.json",
            cache_dir = cache_dir,
            token = hf_token or None,
            local_files_only = local_files_only,
        )
    with open(local, encoding = "utf-8") as fh:
        return json.load(fh)


def _build_hosted_vae(path: str, config: dict, dtype: Any, logger: Any) -> Any:
    import diffusers

    from .diffusion_comfy_components import (
        ComponentFileError,
        build_vae,
        read_safetensors_header,
    )

    header, _meta = read_safetensors_header(path)
    dtypes = {
        str(v.get("dtype")) for k, v in header.items() if isinstance(v, dict) and "dtype" in v
    }
    if dtypes != {"BF16"}:
        raise ComponentFileError(
            f"'{os.path.basename(path)}' is not an all-bf16 VAE ({sorted(dtypes)})"
        )
    vae_cls = getattr(diffusers, str(config.get("_class_name") or ""), None)
    if vae_cls is None:
        raise ComponentFileError(
            f"unknown VAE class {config.get('_class_name')!r} in the base config"
        )
    return build_vae(path, vae_cls = vae_cls, config = config, dtype = dtype, logger = logger)


def hosted_vae_pipe_kwargs(
    fam: Any,
    base: Optional[str],
    *,
    kind: Optional[str],
    dtype: Any,
    hf_token: Optional[str] = None,
    local_files_only: bool = False,
    config_source: Optional[str] = None,
    cache_dir: Optional[str] = None,
    dense_staged: bool = True,
    logger: Any = None,
) -> dict[str, Any]:
    """``{"vae": module}`` for ``from_pretrained``, or {} to let it read the base's ``vae/``.

    ``dense_staged`` False means the plan dropped the dense ``vae/`` weights for the hosted file, so a failure here
    loads the base VAE itself (from the Hub cache, downloading it if needed) rather than leave ``from_pretrained``
    without weights."""
    src = hosted_vae_wanted(fam, base, kind = kind, dtype = dtype)
    source = config_source or str(base)
    if src is None:
        # Planned at a bf16 target but loading at another dtype: the dropped vae/ still has to come from somewhere.
        if dense_staged:
            return {}
        if logger is not None:
            logger.info(
                "diffusion.hosted_vae: %s load does not take the bf16 VAE; loading the base vae/",
                str(dtype).replace("torch.", ""),
            )
        return _dense_vae_kwargs(
            base,
            source,
            dtype,
            hf_token = hf_token,
            local_files_only = local_files_only,
            cache_dir = cache_dir,
        )
    repo_id, filename = src
    try:
        from .diffusion_te_prequant import te_prequant_mirror_path

        # The plan leaves a mirrored file out of staging, so read it in place.
        path = te_prequant_mirror_path(repo_id, filename)
        if path is None:
            from huggingface_hub import hf_hub_download
            path = hf_hub_download(
                repo_id,
                filename,
                cache_dir = cache_dir,
                token = hf_token or None,
                local_files_only = local_files_only,
            )
        config = _vae_config(
            source, hf_token = hf_token, local_files_only = local_files_only, cache_dir = cache_dir
        )
        vae = _build_hosted_vae(path, config, dtype, logger)
        if logger is not None:
            logger.info(
                "diffusion.hosted_vae: reading %s/%s (bf16, %d bytes) instead of the base vae/ weights",
                repo_id,
                filename,
                os.path.getsize(path),
            )
        return {HOSTED_VAE_COMPONENT: vae}
    except Exception as exc:  # noqa: BLE001 -- fall back to the base repo's VAE
        if logger is not None:
            logger.warning(
                "diffusion.hosted_vae: %s/%s not used (%s); loading the base vae/ instead",
                repo_id,
                filename,
                exc,
            )
    if dense_staged:
        return {}
    return _dense_vae_kwargs(
        base,
        source,
        dtype,
        hf_token = hf_token,
        local_files_only = local_files_only,
        cache_dir = cache_dir,
    )


def _dense_vae_kwargs(
    base: Optional[str],
    source: str,
    dtype: Any,
    *,
    hf_token: Optional[str],
    local_files_only: bool,
    cache_dir: Optional[str],
) -> dict[str, Any]:
    """The base's own ``vae/`` read by id, so the Hub cache supplies (or downloads) the weights the plan left out."""
    import diffusers

    config = _vae_config(
        source, hf_token = hf_token, local_files_only = local_files_only, cache_dir = cache_dir
    )
    vae_cls = getattr(diffusers, str(config.get("_class_name") or ""))
    kwargs: dict[str, Any] = {
        "subfolder": "vae",
        "torch_dtype": dtype,
        "local_files_only": local_files_only,
    }
    if cache_dir:
        kwargs["cache_dir"] = cache_dir
    if hf_token:
        kwargs["token"] = hf_token
    return {HOSTED_VAE_COMPONENT: vae_cls.from_pretrained(str(base), **kwargs)}
