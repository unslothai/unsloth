# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Freeze every resolved per-family table value, so moving the tables into family data cannot change them.

The int8 exclude / ConvRot / NVFP4 policy sets are baked into published prequant artifacts and checked at load
(``diffusion_prequant._validate_checkpoint`` requires the artifact's exclude tokens to be a superset of
``exclude_tokens_for_scheme``), so a silent change here invalidates hosted checkpoints. The rest (preview factor keys,
untiled VAE coefficients, labels, defaults, trusted repos, ...) is frozen with them so a data move stays a pure move.

A deliberate change regenerates the fixture: ``UNSLOTH_FREEZE_UPDATE=1 pytest tests/test_diffusion_family_data_freeze.py``
and the diff of ``tests/fixtures/diffusion_family_data_freeze.json`` is the review artifact.
"""

from __future__ import annotations

import dataclasses
import json
import os
from pathlib import Path

import pytest

FIXTURE = Path(__file__).parent / "fixtures" / "diffusion_family_data_freeze.json"

_EXTRA_NAMES = ("", "unknown-family", "QWEN-IMAGE", " qwen-image ")


def _jsonable(value):
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {f.name: _jsonable(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in sorted(value.items(), key = lambda kv: str(kv[0]))}
    if isinstance(value, (frozenset, set)):
        return sorted(_jsonable(v) for v in value)
    if isinstance(value, (tuple, list)):
        return [_jsonable(v) for v in value]
    return value


def _snapshot() -> dict:
    from core.inference import diffusion_transformer_quant as tq
    from core.inference import diffusion_precision as prec
    from core.inference import video_vae_untiled as vu
    from core.inference import diffusion_cache as dcache
    from core.inference import diffusion_content as dcontent
    from core.inference.diffusion_families import _FAMILIES as IMAGE, default_generation_params
    from core.inference.video_families import _FAMILIES as VIDEO, default_video_generation_params
    from core.inference.diffusion_preview_factors import FAMILY_FACTORS
    from core.inference import diffusion as dimg
    from core.inference import video as dvid
    from core.training import diffusion_train_common as dtc

    image_names = [f.name for f in IMAGE]
    video_names = [f.name for f in VIDEO]
    names = image_names + video_names + list(_EXTRA_NAMES)
    schemes = list(tq.TQ_SCHEMES) + ["auto", "bf16"]

    # Every repo id the registry names, as candidates for the per-repo ConvRot variant check.
    repos = set()
    for fam in (*IMAGE, *VIDEO):
        repos.add(fam.base_repo)
        for row in getattr(fam, "prequant_repos", ()):
            repos.add(row[-1])
        for row in getattr(fam, "prequant_variant_repos", ()):
            repos.update((row[0], row[-1]))
    repos.update(
        ("unsloth/Qwen-Image-2.1-Turbo-FP8", "unsloth/qwen-image-2.1-turbo-fp8", "", "x/y")
    )
    repos = sorted(repos)

    per_family = {}
    for name in names:
        row = {}
        for scheme in schemes:
            variant_repos = [r for r in repos if tq.convrot_prequant_variant_repo(scheme, name, r)]
            row[scheme] = {
                "exclude": list(tq.exclude_tokens_for_scheme(scheme, name)),
                "pad": list(tq.pad_tokens_for_scheme(scheme, name)),
                "zero_row": list(tq.zero_row_tokens_for_scheme(scheme, name)),
                "convrot_spec": _jsonable(tq.convrot_spec_for_scheme(scheme, name)),
                "convrot_filename": tq.convrot_prequant_filename(scheme, name),
                "convrot_repo": tq.convrot_prequant_repo(scheme, name),
                "convrot_variant_repos": variant_repos,
            }
        row["convrot_default_on"] = tq.int8_convrot_enabled(name)
        row["auto_bf16_when_resident"] = tq.auto_bf16_when_resident_reason(name)
        row["te_int8_effective"] = prec.effective_te_quant("int8", name)
        row["preview_key"] = FAMILY_FACTORS.get(str(name).strip().lower())
        row["page"] = dcontent.family_page(name)
        per_family[name] = row
    # None as a family (the historical default) on every accessor.
    per_family["<None>"] = {
        scheme: {
            "exclude": list(tq.exclude_tokens_for_scheme(scheme)),
            "pad": list(tq.pad_tokens_for_scheme(scheme)),
            "zero_row": list(tq.zero_row_tokens_for_scheme(scheme)),
            "convrot_spec": _jsonable(tq.convrot_spec_for_scheme(scheme)),
        }
        for scheme in schemes
    }

    untiled = {}
    for name in video_names + list(_EXTRA_NAMES):
        untiled[name] = [
            vu.untiled_decode_bytes(name, shape, itemsize)
            for shape in ((1, 16, 5, 60, 104), (2, 48, 31, 44, 80), (1, 128, 16, 22, 40))
            for itemsize in (2, 4)
        ]

    static_skip = {}
    for fam in (*IMAGE, *VIDEO):
        bases = [fam.base_repo, *getattr(fam, "named_variant_bases", ())]
        static_skip[fam.name] = {b: dcache.auto_static_skip_entry(b) for b in bases}

    flags = {}
    for fam in (*IMAGE, *VIDEO):
        flags[fam.name] = {
            "cudnn_benchmark": getattr(fam, "cudnn_benchmark", True),
            "filter_reduction_configs": bool(getattr(fam, "filter_reduction_configs", False)),
            "filter_reduction_configs_archs": _jsonable(
                tuple(getattr(fam, "filter_reduction_configs_archs", None) or ())
            ),
        }

    defaults = {}
    for fam in IMAGE:
        defaults[fam.name] = list(default_generation_params(fam.base_repo))
    for fam in VIDEO:
        defaults[fam.name] = {
            "fields": [fam.default_steps, fam.default_guidance],
            "resolved": list(
                default_video_generation_params(
                    fam.base_repo,
                    fallback = (fam.default_steps, fam.default_guidance),
                    family = fam.name,
                )
            ),
        }

    return _jsonable(
        {
            "per_family": per_family,
            "tables": {
                "int8_exclude_base": tq._INT8_EXCLUDE_NAME_TOKENS,
                "int8_family_exclude": tq._INT8_FAMILY_EXCLUDE_NAME_TOKENS,
                "int8_family_pad": tq._INT8_FAMILY_PAD_NAME_TOKENS,
                "int8_family_convrot": tq._INT8_FAMILY_CONVROT,
                "int8_family_convrot_filename": tq._INT8_FAMILY_CONVROT_FILENAME,
                "int8_family_convrot_repo": tq._INT8_FAMILY_CONVROT_REPO,
                "int8_family_convrot_variant_repos": tq._INT8_FAMILY_CONVROT_VARIANT_REPOS,
                "int8_family_convrot_default_on": tq._INT8_FAMILY_CONVROT_DEFAULT_ON,
                "nvfp4_family_zero_row": tq._NVFP4_FAMILY_ZERO_ROW_NAME_TOKENS,
                "family_scheme_deny": tq._FAMILY_SCHEME_DENY,
                "family_auto_prefer": tq._FAMILY_AUTO_PREFER,
                "family_auto_bf16_when_resident": tq._FAMILY_AUTO_BF16_WHEN_RESIDENT,
                "family_train_scheme_deny": tq._FAMILY_TRAIN_SCHEME_DENY,
                "te_int8_skip": prec._TE_INT8_SKIP,
                "untiled_bytes_per_latent_pixel": vu._BYTES_PER_LATENT_PIXEL,
                "untiled_output_scale": vu._OUTPUT_SCALE,
                "untiled_bytes_per_frame_latent_pixel": vu._BYTES_PER_FRAME_LATENT_PIXEL,
                "untiled_temporal_scale": vu._TEMPORAL_SCALE,
                "preview_family_factors": FAMILY_FACTORS,
                "auto_static_skip": dcache.AUTO_STATIC_SKIP,
                "train_family_labels": dtc._FAMILY_LABELS,
                "trainable_video_families": dtc.TRAINABLE_VIDEO_FAMILIES,
                "content_video_families": dcontent._VIDEO_FAMILIES,
                "trusted_image_repos": dimg._TRUSTED_NON_GGUF_REPOS,
                "trusted_video_repos": dvid._TRUSTED_NON_GGUF_VIDEO_REPOS,
            },
            "untiled_decode_bytes": untiled,
            "auto_static_skip_by_base": static_skip,
            "flags": flags,
            "defaults": defaults,
        }
    )


@pytest.fixture
def _clean_env(monkeypatch):
    for var in (
        "UNSLOTH_DIFFUSION_INT8_CONVROT",
        "UNSLOTH_DIFFUSION_NVFP4",
        "UNSLOTH_DIFFUSION_AUTO_STEP_SKIP",
        "UNSLOTH_VIDEO_VAE_UNTILED",
    ):
        monkeypatch.delenv(var, raising = False)


def test_family_data_resolves_identically(_clean_env):
    snapshot = json.loads(json.dumps(_snapshot()))
    if os.environ.get("UNSLOTH_FREEZE_UPDATE") == "1":
        FIXTURE.write_text(json.dumps(snapshot, indent = 1, sort_keys = True) + "\n")
    frozen = json.loads(FIXTURE.read_text())
    for section in sorted(set(frozen) | set(snapshot)):
        assert snapshot.get(section) == frozen.get(section), section
