# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Contract: the diffusion TRAINING precision menu never gains an INFERENCE-only scheme.

Inference and training share a vocabulary of quantisation names, but not the same set.
Inference offers ``nvfp4`` (torchao 4-bit weight-only, Blackwell) and ``fp8_dynamic``;
the DiT trainer offers neither -- there is no training path for them, so a UI that
advertised one would evict every resident model, start a run, and then fail. The chain
that has to stay honest is:

    train_precision_modes()  ->  family_train_infos()  ->  GET /diffusion/info
                                                              |
                                       diffusion-train-panel.tsx `precisionModes`
                                                              |
                              DiffusionTrainingStartRequest.base_precision (422 gate)

These assertions read BOTH ends -- the live Python probe over a simulated GPU matrix, and
the frontend source -- so adding ``nvfp4`` to either one reddens. They are deliberately
paired with a positive check that ``nvfp4`` really is a supported INFERENCE scheme, so the
suite cannot pass by the name having quietly disappeared everywhere.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import get_args

import pytest

import core.training.diffusion_train_common as common
from core.inference.diffusion_lora import _DIFFUSERS_LORA_BLOCKED_QUANT
from core.inference.diffusion_precision import TE_QUANT_MODES, TE_QUANT_NVFP4
from core.training.diffusion_train_common import DiffusionLoraConfig, train_precision_modes
from models.inference import DiffusionLoadRequest, VideoLoadRequest
from models.training import DiffusionTrainingStartRequest

_BACKEND = Path(__file__).resolve().parent.parent
_FRONTEND = _BACKEND.parent / "frontend" / "src"

# The base_precision wire contract: anything outside this is a 422 before a GPU is touched.
_TRAIN_PRECISIONS: frozenset[str] = frozenset(
    get_args(DiffusionTrainingStartRequest.model_fields["base_precision"].annotation)
)

# "off"/"none"/"auto" are request sentinels, not schemes; strip them before diffing the two vocabularies.
_SENTINELS: frozenset[str] = frozenset({"auto", "none", "off"})


def _literal_names(model, field: str) -> frozenset[str]:
    """The Literal member names of an ``Optional[Literal[...]]`` field."""
    annotation = model.model_fields[field].annotation
    names = {a for a in get_args(annotation) if isinstance(a, str)}
    for arg in get_args(annotation):
        names |= {a for a in get_args(arg) if isinstance(a, str)}
    return frozenset(names)


# Every quantisation name inference can be asked for, across the transformer and the text encoders.
_INFERENCE_SCHEMES: frozenset[str] = (
    _literal_names(DiffusionLoadRequest, "transformer_quant")
    | _literal_names(DiffusionLoadRequest, "text_encoder_quant")
    | frozenset(TE_QUANT_MODES)
) - _SENTINELS

# Schemes inference supports that training has no path for. Derived, not hardcoded, so a new
# inference-only scheme is covered the day it lands; the guard below pins nvfp4 into it so the
# derivation cannot silently empty out (which would make every assertion here vacuous).
_INFERENCE_ONLY: frozenset[str] = _INFERENCE_SCHEMES - _TRAIN_PRECISIONS

# (major, minor) capabilities spanning every branch of the probe: pre-Ampere, Ampere, Ada, Hopper, Blackwell, and newer.
_CAPABILITIES = ((7, 5), (8, 0), (8, 6), (8, 9), (9, 0), (10, 0), (12, 0))


def _probe(
    monkeypatch,
    capability,
    *,
    cuda = True,
    torchao = True,
) -> tuple[list[str], str]:
    """(modes, recommended) as train_precision_modes() would answer on the given machine.

    The recommendation is returned, not discarded: the Train panel seeds basePrecision from it,
    so a recommendation outside the reported list is an option the user starts on and the select
    never offered."""
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda *a, **k: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: capability)
    monkeypatch.setattr(common, "has_functional_torchao", lambda: torchao)
    return train_precision_modes()


def _every_advertisable_mode(monkeypatch) -> frozenset[str]:
    """The union of everything the probe can EVER put in front of a user, over the whole
    GPU x torchao matrix. The UI can only ever render a subset of this."""
    seen: set[str] = set()
    for capability in _CAPABILITIES:
        for torchao in (True, False):
            seen.update(_probe(monkeypatch, capability, torchao = torchao)[0])
    seen.update(_probe(monkeypatch, (10, 0), cuda = False)[0])
    return frozenset(seen)


# ── the vocabularies really do differ ─────────────────────────────────────────


def test_nvfp4_is_a_real_inference_scheme_and_not_a_training_one():
    """Anchors the rest of the file: nvfp4 must exist on the inference side, or every
    "nvfp4 is absent" assertion below would pass for the wrong reason."""
    assert TE_QUANT_NVFP4 == "nvfp4"
    assert TE_QUANT_NVFP4 in TE_QUANT_MODES
    assert "nvfp4" in _literal_names(DiffusionLoadRequest, "transformer_quant")
    assert "nvfp4" in _literal_names(DiffusionLoadRequest, "text_encoder_quant")
    # It is also the scheme the diffusers LoRA path refuses to attach to, so it is genuinely live.
    assert "nvfp4" in _DIFFUSERS_LORA_BLOCKED_QUANT
    # ...and it is inference-only.
    assert "nvfp4" in _INFERENCE_ONLY
    assert "nvfp4" not in _TRAIN_PRECISIONS


# ── backend: the probe ────────────────────────────────────────────────────────


@pytest.mark.parametrize("capability", _CAPABILITIES)
@pytest.mark.parametrize("torchao", (True, False))
def test_train_precision_modes_never_offers_an_inference_only_scheme(
    monkeypatch, capability, torchao
):
    modes, recommended = _probe(monkeypatch, capability, torchao = torchao)
    leaked = sorted(_INFERENCE_ONLY.intersection(modes))
    assert not leaked, (
        f"train_precision_modes() on sm{capability[0]}{capability[1]} (torchao={torchao}) "
        f"advertises inference-only scheme(s) {leaked}; the DiT trainer has no path for them, "
        "so /diffusion/info would offer a start that evicts resident models and then fails"
    )
    # Anything advertised must also clear the request schema, or the UI offers a guaranteed 422.
    assert set(modes) <= _TRAIN_PRECISIONS, sorted(set(modes) - _TRAIN_PRECISIONS)
    assert "nf4" in modes  # the floor is always available
    # The recommendation is what the panel seeds basePrecision with, so one outside the reported
    # list is an option the user starts on and the select never offered -- and one outside the
    # schema is a guaranteed 422 on the first start.
    assert recommended in modes, (
        f"the recommendation {recommended!r} is not in the modes reported for "
        f"sm{capability[0]}{capability[1]} (torchao={torchao}): {sorted(modes)}"
    )
    assert recommended in _TRAIN_PRECISIONS


def test_the_advertisable_vocabulary_is_exactly_the_schema_vocabulary(monkeypatch):
    """Across every GPU the probe can meet, the modes it emits are exactly the request
    Literal -- no more (a 422 the UI could hit) and no fewer (a dead schema member)."""
    assert _every_advertisable_mode(monkeypatch) == _TRAIN_PRECISIONS


def test_family_train_infos_never_advertises_an_inference_only_scheme(monkeypatch, dit_train_host):
    """The /diffusion/info payload itself, on a Blackwell host where every scheme is live."""
    _probe(monkeypatch, (10, 0))
    for info in common.family_train_infos():
        modes = info["precision_modes"]
        assert not _INFERENCE_ONLY.intersection(
            modes
        ), f"family {info['name']!r} advertises {sorted(_INFERENCE_ONLY.intersection(modes))}"
        assert set(modes) <= _TRAIN_PRECISIONS
        assert info["recommended_precision"] in _TRAIN_PRECISIONS


# ── schema: the 422 gate ──────────────────────────────────────────────────────


@pytest.mark.parametrize("scheme", sorted(_INFERENCE_ONLY))
def test_the_start_request_rejects_an_inference_only_precision(scheme):
    with pytest.raises(Exception) as excinfo:
        DiffusionTrainingStartRequest(
            base_model = "black-forest-labs/FLUX.1-dev",
            data_dir = "d",
            output_dir = "o",
            base_precision = scheme,
        )
    assert "base_precision" in str(excinfo.value)


def test_the_trainer_accepts_exactly_what_the_schema_advertises():
    """The one link the rest of this file cannot supply. Every set above is derived from the
    request schema and the probe, so a mode added to BOTH of those disappears from
    ``_INFERENCE_ONLY`` and every assertion here passes -- while
    ``DiffusionLoraConfig.normalized()`` keeps its own hardcoded tuple and rejects the run after
    it has already evicted the resident model. Asked of the trainer directly rather than parsed
    out of it, so a refactor of that tuple cannot fool the check.
    """
    accepted, refused = set(), {}
    for mode in sorted(_TRAIN_PRECISIONS):
        config = DiffusionLoraConfig(
            base_model = "black-forest-labs/FLUX.1-dev",
            data_dir = "d",
            output_dir = "o",
            base_precision = mode,
        )
        try:
            config.normalized()
        except Exception as exc:  # noqa: BLE001 - anything that stops a start counts as a refusal
            # The mode-name message is the expected shape, but it is not the only way a start
            # dies: a dense-base check, a mixed-precision check or a new dataset-path check would
            # block the same advertised mode just as completely. Swallowing those would keep this
            # green for a precision the UI offers and the trainer refuses.
            refused[mode] = f"{type(exc).__name__}: {exc}"
            continue
        accepted.add(mode)

    assert not refused, (
        f"the request schema advertises {sorted(refused)}, which DiffusionLoraConfig.normalized() "
        "rejects; a start would evict the resident model and then fail"
    )
    assert accepted == set(_TRAIN_PRECISIONS)

    # ...and the tuple is not simply permissive: an inference-only scheme still has to bounce,
    # or the assertion above would hold for a trainer that accepts everything.
    for scheme in sorted(_INFERENCE_ONLY):
        bogus = DiffusionLoraConfig(
            base_model = "black-forest-labs/FLUX.1-dev",
            data_dir = "d",
            output_dir = "o",
            base_precision = scheme,
        )
        with pytest.raises(ValueError, match = "base_precision must be one of"):
            bogus.normalized()


def test_every_training_precision_is_accepted_by_the_start_request():
    for mode in sorted(_TRAIN_PRECISIONS):
        req = DiffusionTrainingStartRequest(
            base_model = "black-forest-labs/FLUX.1-dev",
            data_dir = "d",
            output_dir = "o",
            base_precision = mode,
        )
        assert req.base_precision == mode


# ── frontend: the Train panel's precision selector ────────────────────────────


def _precision_memo_block() -> str:
    """The body of ``diffusion-train-panel.tsx``'s ``precisionModes`` useMemo: the type
    annotation, the reported-mode filter, and the no-backend fallback array."""
    src = (_FRONTEND / "features" / "images" / "train" / "diffusion-train-panel.tsx").read_text(
        encoding = "utf-8"
    )
    start = src.index("const precisionModes = useMemo<")
    close = src.index("\n  }, [", start)
    block = src[start : src.index(");", close) + 2]
    # Guard the extraction itself: a refactor that moves the memo must not silently shrink this to nothing.
    assert "familyUntrainable" in block and "return [" in block, block
    return block


# A TS/TSX string literal in any of the three quotings. Prettier normalizes this file to
# double quotes, but the guard must not depend on that: a hand-edit or a merge that spelled a
# scheme 'nvfp4' or `nvfp4` would otherwise slip past every assertion below while rendering
# exactly the same option. Verified by mutation -- a single-quoted arm used to pass clean.
_STRING_LITERAL = re.compile(r"""["'`]([^"'`\\\n]*)["'`]""")
_M_EQUALS = re.compile(r"""m === ["'`]([^"'`]+)["'`]""")


def _strip_comments(block: str) -> str:
    """Line and block comments removed, so a scheme merely NAMED in prose is not read as an
    offered option (and, the other way round, so a commented-out arm cannot mask a real one)."""
    return re.sub(r"//[^\n]*", "", re.sub(r"/\*.*?\*/", "", block, flags = re.S))


def _memo_string_literals(block: str) -> list[str]:
    return _STRING_LITERAL.findall(_strip_comments(block))


def test_the_precision_selector_names_only_training_precisions():
    """Every string literal inside the memo -- the TS union, the runtime filter whitelist and
    the fallback array -- must be a real training precision. This is the assertion that
    reddens if someone drops "nvfp4" anywhere into the Train panel's precision list."""
    literals = set(_memo_string_literals(_precision_memo_block()))
    assert literals, "parsed no string literals out of the precisionModes memo"
    assert literals <= _TRAIN_PRECISIONS, (
        f"the Train precision selector names {sorted(literals - _TRAIN_PRECISIONS)}, which "
        f"{'is' if len(literals - _TRAIN_PRECISIONS) == 1 else 'are'} not accepted by "
        "DiffusionTrainingStartRequest.base_precision"
    )
    assert not _INFERENCE_ONLY.intersection(literals)


def test_the_precision_selector_fallback_is_a_subset_of_what_the_backend_can_report(monkeypatch):
    """With no /diffusion/info report the panel falls back to a hardcoded array. It must stay
    inside what the backend could actually have said, or the first paint offers a dead option."""
    returns = re.findall(r"return\s*\[([^\]]*)\]", _strip_comments(_precision_memo_block()))
    fallback = _STRING_LITERAL.findall(returns[-1])
    assert len(fallback) >= 2, f"failed to parse the fallback array: {returns[-1]!r}"
    advertisable = _every_advertisable_mode(monkeypatch)
    assert set(fallback) <= advertisable, (
        f"the Train panel's offline fallback offers {sorted(set(fallback) - advertisable)}, "
        "which train_precision_modes() never reports on any GPU"
    )
    assert not _INFERENCE_ONLY.intersection(fallback)


def test_the_reported_mode_filter_is_a_subset_of_what_the_backend_can_report(monkeypatch):
    """The panel narrows the backend's list through an explicit ``m === "..."`` whitelist.
    Every arm of it must be a mode the backend can actually emit."""
    block = _strip_comments(_precision_memo_block())
    predicate = block[block.index(".filter(") : block.index("return [", block.index(".filter("))]
    whitelist = set(_M_EQUALS.findall(predicate))
    assert whitelist, f"parsed no whitelist arms out of {predicate!r}"
    advertisable = _every_advertisable_mode(monkeypatch)
    # Equality, not containment. A subset check passes just as happily when an arm is DELETED,
    # and the effect of deleting one is that the backend keeps reporting the mode while the panel
    # silently drops it from the select -- a mode the user can never pick and no error anywhere.
    assert whitelist == advertisable - _SENTINELS, (
        f"the panel filters to {sorted(whitelist)} but the backend can report "
        f'{sorted(advertisable - _SENTINELS)}; "auto" is prepended separately, so the filter '
        "has to name every other advertisable mode exactly"
    )
    assert not _INFERENCE_ONLY.intersection(whitelist)
    # ...and Auto has to survive the memo. Subtracting it above is only sound while the memo
    # prepends it: change `return ["auto", ...reported]` to `return reported` and this file
    # would still pass while the user loses the backend-recommended option the moment a report
    # arrives. Both return paths, since the fallback is what a backendless first paint renders.
    # `return []` for an untrainable family offers nothing at all, deliberately; every return
    # that offers anything has to lead with Auto.
    returns = [
        line
        for line in block.splitlines()
        if "return [" in line and line.strip() not in ("return [];", "if (familyUntrainable) return [];")
    ]
    assert returns, f"parsed no return arrays out of the memo: {block!r}"
    for line in returns:
        assert '"auto"' in line, f"the memo returns a list with no Auto option: {line.strip()!r}"


# ── inference keeps NVFP4 ─────────────────────────────────────────────────────


def test_inference_still_offers_nvfp4_end_to_end():
    """The training guard must not be "fixed" by deleting NVFP4 from inference, where it is a
    legitimate Blackwell option on both the image and the video load forms."""
    req = DiffusionLoadRequest(
        model_path = "unsloth/Z-Image-Turbo-GGUF",
        transformer_quant = "nvfp4",
        text_encoder_quant = "nvfp4",
    )
    assert req.transformer_quant == "nvfp4" and req.text_encoder_quant == "nvfp4"

    # The video load form too, through the schema rather than its source: video-page.tsx can
    # keep offering NVFP4 long after VideoLoadRequest stopped accepting it, and the only symptom
    # is a 422 from /video/load.
    video = VideoLoadRequest(model_path = "unsloth/Wan2.2-TI2V-5B", transformer_quant = "nvfp4")
    assert video.transformer_quant == "nvfp4"

    for rel in (
        ("features", "images", "images-page.tsx"),
        ("features", "video", "video-page.tsx"),
    ):
        src = (_FRONTEND.joinpath(*rel)).read_text(encoding = "utf-8")
        assert '["nvfp4", "NVFP4 (Blackwell)"]' in src, f"{rel[-1]} no longer offers NVFP4"
    exports = (_FRONTEND / "features" / "export" / "constants.ts").read_text(encoding = "utf-8")
    assert 'value: "nvfp4"' in exports
