# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""What a family needs to put several requests' images in one denoiser forward.

A family says whether its pipeline call fuses a list of prompts into one forward through ``batched_forward`` (bool) on
its family object; unset keeps today's answer (``_DEFAULT_SERIAL`` and the pipe checks below). A family that needs its
own row layout sets ``batch_adapter`` to a ``BatchAdapter``; every other batched family uses ``DefaultAdapter``, the
diffusers list call."""

from __future__ import annotations

from typing import Any, Awaitable, Callable, Hashable, Optional, Sequence

BATCHED_FORWARD_ATTR = "batched_forward"
BATCH_ADAPTER_ATTR = "batch_adapter"


def _default_serial_names() -> frozenset:
    from .diffusion_anima import ANIMA_FAMILY_NAME
    from .diffusion_families import MINIMAX_H3_IMAGE_FAMILY_NAME, MOSS_IMAGE_FAMILY_NAME
    from .diffusion_flux3 import FLUX3_FAMILY_NAMES

    # One image per forward unless a G2c verdict (Anima) or a measured layer-major load (MOSS) says otherwise.
    return frozenset(
        (
            *FLUX3_FAMILY_NAMES,
            ANIMA_FAMILY_NAME,
            MINIMAX_H3_IMAGE_FAMILY_NAME,
            MOSS_IMAGE_FAMILY_NAME,
        )
    )


def _moss_layer_major(family: Any, pipe: Any) -> bool:
    """A MOSS load that runs a batch layer-major on a card with a measured win (diffusion_moss_image.coalesce_key)."""
    from .diffusion_families import MOSS_IMAGE_FAMILY_NAME

    if (
        getattr(family, "name", None) != MOSS_IMAGE_FAMILY_NAME
        or getattr(pipe, "transformer", None) is None
    ):
        return False
    try:
        from .diffusion_moss_image import coalesce_key
        return coalesce_key(pipe) is not None
    except Exception:  # noqa: BLE001 - an unreadable load renders alone
        return False


def _qwen21_denoiser(pipe: Any) -> bool:
    from .diffusion_qwenimage21 import _CLASS
    return type(getattr(pipe, "transformer", None)).__name__ == _CLASS


def _card_verdict_families() -> frozenset:
    from .diffusion_anima import ANIMA_FAMILY_NAME

    # Batch-shape GEMM tiling moves a row by IEEE noise: fuse only where a stored G2c PASS bounds it.
    return frozenset((ANIMA_FAMILY_NAME,))


def _card_measured(family: Any, pipe: Any) -> bool:
    if pipe is None or getattr(pipe, "transformer", None) is None:
        return False
    from .diffusion_batch_quality import family_card_ok
    return family_card_ok(getattr(family, "name", None), pipe)


def default_batched_forward(family: Any, pipe: Any = None) -> bool:
    """Today's answer for a family that does not say: whether its list call runs the rows in one forward."""
    if getattr(family, "layer_count", 0):
        return False  # the layered pipeline's per-prompt layer stride is only right for one input
    if getattr(family, "name", None) in _card_verdict_families():
        return _card_measured(family, pipe)
    if getattr(family, "name", None) in _default_serial_names():
        return _moss_layer_major(family, pipe)
    if pipe is not None and _qwen21_denoiser(pipe):
        return False  # lays out a batch from row 0, so it splits by text length and gains nothing
    return True


def admits_without_pipe(family: Any) -> bool:
    """The route's family check, which has no pipe: refuse only when the answer cannot depend on the load. MOSS fuses
    rows only on a measured layer-major load and Anima only on a card with a G2c verdict, which the engine decides
    with its pipe (coalesce_plan)."""
    from .diffusion_families import MOSS_IMAGE_FAMILY_NAME

    if batched_forward(family):
        return True
    return getattr(family, BATCHED_FORWARD_ATTR, None) is None and (
        getattr(family, "name", None) == MOSS_IMAGE_FAMILY_NAME
        or getattr(family, "name", None) in _card_verdict_families()
    )


def batched_forward(family: Any, pipe: Any = None) -> bool:
    """The family's ``batched_forward`` when it sets one, else ``default_batched_forward``."""
    said = getattr(family, BATCHED_FORWARD_ATTR, None)
    if said is not None:
        return bool(said)
    return default_batched_forward(family, pipe)


class BatchAdapter:
    """Per-family row contract. ``prepare`` returns the layout key a row must share with its batch neighbours (token
    bucket, reference layout, canvas, CFG structure), or None when the row must render alone."""

    name = "base"

    def prepare(self, family: Any, pipe: Any, prompt: str) -> Optional[tuple]:
        raise NotImplementedError

    def compatible(self, a: Optional[tuple], b: Optional[tuple]) -> bool:
        return a is not None and b is not None and a == b

    async def denoise_batched(
        self,
        run: Callable[..., Awaitable[dict]],
        prompts: list[str],
        seeds: list[int],
        owners: Optional[list[Hashable]] = None,
    ) -> dict:
        return await run(prompts, seeds, owners = owners)

    def split(
        self,
        result: dict,
        members: Sequence[Any],
        *,
        prefix: bool = False,
    ) -> list[dict]:
        from .image_request_batching import _split
        return _split(result, list(members), prefix = prefix)


class DefaultAdapter(BatchAdapter):
    """The diffusers list call: one prompt list, one generator per row, images back in row order."""

    name = "default"

    def prepare(self, family: Any, pipe: Any, prompt: str) -> Optional[tuple]:
        from .diffusion_prompt_weighting import has_prompt_syntax
        from .diffusion_text_length import FLUX_T5_FAMILIES, flux_t5_sequence_length

        name = getattr(family, "name", None)
        if name in FLUX_T5_FAMILIES:
            # FLUX.1 pads T5 to the call's longest prompt bucket.
            bucket = flux_t5_sequence_length(getattr(pipe, "tokenizer_2", None), [prompt])
            return None if bucket is None else ("t5 bucket", bucket)
        if getattr(family, "textual_inversion", None) is not None:
            # SD / SDXL chunk the CLIP prompt per call: members must match chunk counts and syntax.
            from .diffusion import _clip_chunk_counts
            chunks = _clip_chunk_counts(pipe, prompt)
            return None if chunks is None else ("clip", has_prompt_syntax(prompt), chunks)
        return ()


DEFAULT_ADAPTER = DefaultAdapter()


def adapter_for(family: Any, pipe: Any = None) -> Optional[BatchAdapter]:
    """The adapter that batches this family's rows, or None when its requests render one at a time."""
    if not batched_forward(family, pipe):
        return None
    own = getattr(family, BATCH_ADAPTER_ATTR, None)
    return own if isinstance(own, BatchAdapter) else DEFAULT_ADAPTER


def key_tokenizes(family: Any) -> bool:
    """Whether ``DefaultAdapter.prepare`` runs a tokenizer for this family (the caller then needs private copies)."""
    from .diffusion_text_length import FLUX_T5_FAMILIES
    return (
        getattr(family, "name", None) in FLUX_T5_FAMILIES
        or getattr(family, "textual_inversion", None) is not None
    )
