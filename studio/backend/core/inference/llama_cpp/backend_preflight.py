# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""LlamaCppBackend: lifecycle, media architecture preflight."""

import os

from typing import Optional
from utils.gguf_archs import (
    is_audio_cpp_gguf_architecture,
    is_speech_gguf_architecture,
    SPEECH_GGUF_ARCHS as _SPEECH_GGUF_ARCHS,
)

from .log import logger


class _PreflightMixin:
    # ── Lifecycle ─────────────────────────────────────────────────

    # GGUF ``general.architecture`` values for diffusion / image models.
    # llama.cpp has no such architectures, so loading one as a chat model dies
    # with "unknown model architecture: '<arch>'". These match the patched
    # stable-diffusion.cpp / ComfyUI-GGUF enums. Unsloth publishes FLUX and
    # Qwen-Image GGUFs under
    # https://huggingface.co/collections/unsloth/unsloth-diffusion-ggufs.
    # Matched exactly (not a substring) so a chat arch containing "wan"/"sd1"
    # (e.g. "taiwan") isn't misrouted to Images.
    _IMAGE_ARCHES = frozenset(
        (
            "qwen_image",
            "qwen_image21",
            "flux",
        )
    )
    # Archs shared by more than one family, so the header alone does NOT say whether the
    # Images page can run the file: Z-Image's DiT is a Lumina2 derivative, so its GGUFs
    # declare "lumina2" too (measured on unsloth/Z-Image-Turbo-GGUF/z-image-turbo-Q2_K.gguf
    # and on calcuis/lumina-gguf). ``routes.models._arch_to_task`` resolves these from the
    # repo id and the filename and tags the rest ``image-diffusion-unsupported``, hiding
    # them from the Images picker, so the refusal must run the SAME resolution before it
    # names that page -- e.g. ``neta-art/neta-lumina-gguf/checkpoint-e3_s9658-*.gguf``
    # declares "lumina2" and resolves to no family ("lumina-2" is not aliased to bare
    # "lumina"). Mirrors ``routes.models._AMBIGUOUS_DIFFUSION_GGUF_ARCHS``.
    _AMBIGUOUS_IMAGE_ARCHES = frozenset(("lumina2",))
    # Text-to-video archs the Video page can offer. Split from the image set so the refusal
    # names Video: sending a Wan / LTX-2 GGUF to Images is a second dead end.
    _VIDEO_ARCHES = frozenset(
        (
            "ltxv",
            "wan",
        )
    )
    # Media archs Unsloth recognises but NO page can run, so the refusal must not promise
    # one. Mirrors ``routes.models._UNSUPPORTED_DIFFUSION_GGUF_ARCHS``, which tags these
    # ``image-diffusion-unsupported`` and so hides them from the Images AND Video pickers.
    #   * "cosmos": no VideoFamily exists at all (``detect_video_family`` returns None for
    #     every cosmos repo id and for override="cosmos"), and the published cosmos GGUFs
    #     are Cosmos-Predict2-14B-Text2Image, which Images cannot assemble either.
    #   * "hyvid": HunyuanVideo 1.0 has no VideoFamily, and the 1.5 GGUFs that do map to
    #     one are still tagged unsupported (that family ships no ``gguf_repo``).
    #   * "sd1" / "sd3" / "sdxl" / "aura" / "hidream": unsupported on the image side for
    #     the same reason, so "use the Images page" was always an empty promise.
    # Kept inside _DIFFUSION_ARCHES so the pre-launch refusal still fires, it just names no
    # destination. Mirrors the whole set: a partial mirror leaves the same dead end.
    _UNRUNNABLE_MEDIA_ARCHES = frozenset(
        (
            "cosmos",
            "hyvid",
            "sd1",
            "sd3",
            "sdxl",
            "aura",
            "hidream",
        )
    )
    _DIFFUSION_ARCHES = (
        _IMAGE_ARCHES | _AMBIGUOUS_IMAGE_ARCHES | _VIDEO_ARCHES | _UNRUNNABLE_MEDIA_ARCHES
    )
    # TTS archs llama.cpp cannot load (CSM support is only on an unmerged upstream branch). One
    # shared definition, so this gate and the listing classifier cannot drift.
    _SPEECH_ARCHES = _SPEECH_GGUF_ARCHS

    # Not architectures: the literal placeholders gguf-connector writes into
    # general.architecture for its diffusion GGUFs (gguf-org/flux2-dev-gguf and
    # calcuis/cosmos-predict2-gguf both declare "pig"); ComfyUI-GGUF special cases the same
    # two. llama.cpp knows neither, so without this they slip past the arch sets and die in
    # llama-server as the opaque failure this preflight exists to prevent.
    _PLACEHOLDER_ARCHES = frozenset(("pig", "cow"))

    @staticmethod
    def _video_arch_is_pickable(
        arch: Optional[str], gguf_path: Optional[str], model_identifier: Optional[str]
    ) -> bool:
        """Whether the Video page would actually OFFER this GGUF.

        Same fixed order ``_arch_to_task`` asks in: the ARCHITECTURE first, since some map
        straight to a family (``detect_video_family("", override = "ltxv")`` resolves LTX-2
        with no name to go on), then the repo id, then the filename. Whatever resolves must
        be an MoE only when the file names one expert of a high/low noise pair (the loader
        assembles the pair) and must be buildable by the video engine -- bare "wan" resolves
        nothing, and an unpaired QuantStack/Wan2.2-T2V-A14B-GGUF file is not promised the page."""
        try:
            from core.inference.video_families import detect_video_family

            from routes.models import _video_family_buildable

            fam = detect_video_family("", override = (arch or "").strip().lower() or None)
            if fam is None:
                for needle in (model_identifier, os.path.basename(gguf_path or "")):
                    if not needle:
                        continue
                    fam = detect_video_family(needle)
                    if fam is not None:
                        break
            if fam is None:
                return False
            from core.inference.video_moe_pair import moe_pick_pairs

            # An MoE expert is offered when its name pairs the partner the loader assembles it with.
            return moe_pick_pairs(fam, gguf_path) and _video_family_buildable(fam)
        except Exception as e:  # noqa: BLE001 -- never lose the page over a probe failure
            logger.debug("Family probe failed for video arch: %s", e)
            return True

    @staticmethod
    def _image_arch_is_pickable(gguf_path: Optional[str], model_identifier: Optional[str]) -> bool:
        """The Images picker's fail-open buildability gate (Qwen-Image-2.1 needs diffusers 0.41)."""
        try:
            from routes.models import _gguf_family_buildable
            return _gguf_family_buildable((model_identifier, os.path.basename(gguf_path or "")))
        except Exception as e:  # noqa: BLE001 -- never lose the page over a probe failure
            logger.debug("Family probe failed for image arch: %s", e)
            return True

    @staticmethod
    def _ambiguous_image_arch_is_pickable(
        gguf_path: Optional[str], model_identifier: Optional[str]
    ) -> bool:
        """Whether the Images page would actually OFFER this shared-arch GGUF.

        The same three questions ``routes.models._arch_to_task`` asks in its
        ``_AMBIGUOUS_DIFFUSION_GGUF_ARCHS`` branch, over the same evidence (repo id, then
        filename): does a family resolve, can a GGUF transformer be assembled for it, can an
        engine on THIS host build it. Both sides must answer alike -- the picker's answer is
        what decides whether the row the refusal points at exists.

        Imported lazily inside the function since ``routes`` imports this module, so the
        canonical classifier cannot be imported from here.

        Fails OPEN on a probe error, like every family probe on the listing side: naming the
        page is a nicety, not worth losing over a broken import."""
        try:
            from core.inference.diffusion_engine_router import family_buildable_here
            from core.inference.diffusion_families import (
                detect_family_for_pick,
                family_gguf_loadable,
            )

            for needle in (model_identifier, os.path.basename(gguf_path or "")):
                if not needle:
                    continue
                fam = detect_family_for_pick(needle)
                if fam is not None:
                    return family_gguf_loadable(fam) and family_buildable_here(
                        fam, model_kind = "gguf"
                    )
            return False
        except Exception as e:  # noqa: BLE001 -- never lose the page over a probe failure
            logger.debug("Family probe failed for ambiguous image arch: %s", e)
            return True

    def _non_chat_gguf_refusal(self, gguf_path: str) -> Optional[str]:
        """Why this GGUF cannot be a chat model, decided from its own header BEFORE
        llama-server is launched; None when there is no such verdict.

        Otherwise a media GGUF opened in a chat runs the whole download-and-launch path only
        to die in llama-server as "llama-server failed to start", with no hint that the file
        was never a chat model (the reported Video-model case).

        The header alone is enough to be certain, in two ways:

        * A declared architecture in the image / video sets. llama.cpp has no such
          architectures, so this is exactly the failure it would hit, one launch earlier.
        * NO ``general.architecture`` at all -- a mandatory key for anything llama.cpp can
          load, and Unsloth's video GGUFs (MiniMax-H3) carry a bare tensor header with zero
          KV pairs, so they never reach the arch test. Gated on ``_gguf_header_parsed`` so a
          read that FAILED still gets its chance in llama-server.

        Naming the destination page is the point, so when the architecture does not say
        which it is, the family detectors are asked about the repo id and the filename."""
        arch = (self._architecture or "").strip().lower()
        # A declared media arch is the whole answer, and it is KV #0 in every GGUF measured,
        # so the remote probe's prefix can carry it even where bulkier later metadata leaves
        # the walk unfinished. The walk still gates the no-architecture fallback below, the
        # one case where an incomplete read is indistinguishable from declaring nothing.
        if (
            arch not in self._DIFFUSION_ARCHES
            and not is_speech_gguf_architecture(arch)
            and not is_audio_cpp_gguf_architecture(arch)
            and not getattr(self, "_gguf_header_parsed", False)
        ):
            return None
        if arch in self._PLACEHOLDER_ARCHES:
            # Names no architecture, so treat it like a GGUF that declares none and let the
            # name-based branch below name a page.
            arch = ""
        if arch:
            if is_audio_cpp_gguf_architecture(arch):
                return (
                    "This is an audio GGUF (speech, music or transcription), which cannot run as "
                    "a chat model. Open it from the Audio page instead."
                )
            if is_speech_gguf_architecture(arch):
                # Points at the Transformers build, not this file on the Audio page: that page
                # cannot list a speech GGUF either, so it would promise a row that is not there.
                return (
                    "This is a text-to-speech GGUF, which llama.cpp cannot load: it has no "
                    "CSM decoder. Run the model's Transformers build (for example "
                    "unsloth/csm-1b) from the Audio page instead."
                )
            if arch in self._UNRUNNABLE_MEDIA_ARCHES:
                return (
                    f"This is an image / video generation GGUF (architecture '{arch}'), "
                    "which cannot run as a chat model. Unsloth Studio cannot run this "
                    "architecture at all: neither the Images page nor the Video page "
                    "accepts it."
                )
            if arch in self._VIDEO_ARCHES:
                # Shared like the image archs: the canonical classifier resolves a
                # VideoFamily by name and rejects an MoE the GGUF loader cannot assemble,
                # so "wan" alone does not mean the Video page will list it.
                if self._video_arch_is_pickable(arch, gguf_path, self._model_identifier):
                    return (
                        f"This is a text-to-video GGUF (architecture '{arch}'), which "
                        "cannot run as a chat model. Open it from the Video page instead."
                    )
                return (
                    f"This is a text-to-video GGUF (architecture '{arch}'), which cannot "
                    "run as a chat model. Unsloth cannot assemble this particular model "
                    "from a GGUF, so the Video page does not list it either."
                )
            if arch in self._IMAGE_ARCHES:
                if self._image_arch_is_pickable(gguf_path, self._model_identifier):
                    return (
                        f"This is an image-generation GGUF (architecture '{arch}'), which "
                        "cannot run as a chat model. Open it from the Images page instead."
                    )
                return (
                    f"This is an image-generation GGUF (architecture '{arch}'), which "
                    "cannot run as a chat model. No image engine installed here can build "
                    "this model family, so the Images page does not list it either. "
                    "Updating Unsloth may add support for it."
                )
            if arch in self._AMBIGUOUS_IMAGE_ARCHES:
                if self._ambiguous_image_arch_is_pickable(gguf_path, self._model_identifier):
                    return (
                        f"This is an image-generation GGUF (architecture '{arch}'), which "
                        "cannot run as a chat model. Open it from the Images page instead."
                    )
                return (
                    f"This is an image-generation GGUF (architecture '{arch}'), which "
                    "cannot run as a chat model. Unsloth could not tell which model family "
                    f"this file belongs to -- '{arch}' is shared by several -- so the "
                    "Images page cannot assemble it either."
                )
            return None

        # A shard past the first of a gguf-split set declares no architecture (shard 1 holds
        # it) but is no media GGUF, and llama.cpp's own "model must be loaded with the first
        # split" is more use than "carries no metadata". Needs split.no > 0 AND nothing
        # declared: shard 1 carries the split keys too, and a placeholder arch is blanked
        # above, so a looser test would wave through the very files the placeholder handling
        # exists to catch. Only reachable with a hand-written variant anyway: every producer
        # here strips the shard suffix and sorts, landing on shard 1.
        if self._architecture is None and (self._gguf_split_index or 0) > 0:
            return None

        # No architecture declared: not loadable by llama-server whatever it is. Say which
        # page runs it when the identifiers make that clear.
        needles = [n for n in (self._model_identifier, os.path.basename(gguf_path or "")) if n]
        # Naming a page answers to the same predicates as the arch branches above: a family
        # that resolves is not enough, the picker also drops an MoE it cannot assemble and a
        # family no installed engine can build. So detect first, then ask whether the page
        # would list it, rather than pointing at a page the model is missing from.
        try:
            from core.inference.video_families import detect_video_family
            if any(detect_video_family(n) is not None for n in needles):
                if self._video_arch_is_pickable(None, gguf_path, self._model_identifier):
                    return (
                        "This is a text-to-video model, not a chat model: its GGUF carries "
                        "no llama.cpp model metadata, so llama-server cannot load it. Open "
                        "it from the Video page instead."
                    )
                return (
                    "This is a text-to-video model, not a chat model, and its GGUF carries "
                    "no llama.cpp model metadata. Unsloth cannot assemble this particular "
                    "model from a GGUF either, so the Video page does not list it."
                )
        except Exception as e:  # noqa: BLE001 -- naming the page is a nicety, never a gate
            logger.debug("Video family probe failed during non-chat GGUF preflight: %s", e)
        try:
            from core.inference.diffusion_families import detect_family
            if any(detect_family(n) is not None for n in needles):
                if self._ambiguous_image_arch_is_pickable(gguf_path, self._model_identifier):
                    return (
                        "This is an image-generation model, not a chat model: its GGUF "
                        "carries no llama.cpp model metadata, so llama-server cannot load "
                        "it. Open it from the Images page instead."
                    )
                return (
                    "This is an image-generation model, not a chat model, and its GGUF "
                    "carries no llama.cpp model metadata. Unsloth cannot assemble this "
                    "particular model from a GGUF either, so the Images page does not list "
                    "it."
                )
        except Exception as e:  # noqa: BLE001
            logger.debug("Image family probe failed during non-chat GGUF preflight: %s", e)
        return (
            "This GGUF carries no llama.cpp model metadata (no general.architecture), so "
            "llama-server cannot load it as a chat model. Image and video GGUFs are "
            "packaged this way -- try the Images or Video page, or pick a chat GGUF."
        )

    # Distro package names for the shared libraries the Linux prebuilt links,
    # so the loader failure below can say what to install.
    _SHARED_LIB_PACKAGES = {
        "libgomp.so.1": " (Debian/Ubuntu: libgomp1, Fedora/RHEL: libgomp)",
    }

    # Shared objects Unsloth ships itself, next to llama-server in build/bin
    # (see runtime_payload_health_groups in install_llama_prebuilt.py, and
    # _llama_server_env_for_binary which puts that dir on the platform's
    # loader search path -- DYLD_LIBRARY_PATH on macOS). No
    # distro packages these, so a loader failure naming one means the managed
    # runtime is incomplete or mismatched, not that something must be installed.
    _BUNDLED_LIB_PREFIXES = ("libllama", "libggml", "libmtmd")
