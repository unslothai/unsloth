# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""CPU-only unit tests for the pre-launch non-chat GGUF refusal.

Reported: a text-to-video model opened from the Model Hub landed in a chat, ran the whole
download-and-launch path, and died as "llama-server failed to start" with no hint that the file
was never a chat model. ``_non_chat_gguf_refusal`` decides that from the GGUF header instead,
before llama-server is launched, and names the page that does run it.

The Unsloth video GGUFs (MiniMax-H3) carry a bare tensor header with ZERO KV pairs, so there is
no ``general.architecture`` to match -- which is why "declares no architecture" is a verdict here
and why it has to be told apart from "the header could not be read".
"""

from __future__ import annotations

import struct
from pathlib import Path

import pytest

from core.inference.llama_cpp import LlamaCppBackend

GGUF_MAGIC = 0x46554747


def _write_gguf(
    path: Path,
    *,
    arch: str | None,
    tensor_count: int = 1,
) -> Path:
    """A minimal GGUF header: magic, version, counts, then one optional
    ``general.architecture`` string KV. Enough for the header parser, which never reads
    the tensor data."""
    body = b""
    kv_count = 0
    if arch is not None:
        key = b"general.architecture"
        val = arch.encode()
        body += struct.pack("<Q", len(key)) + key
        body += struct.pack("<I", 8)  # STRING
        body += struct.pack("<Q", len(val)) + val
        kv_count = 1
    header = struct.pack("<II", GGUF_MAGIC, 3) + struct.pack("<QQ", tensor_count, kv_count)
    path.write_bytes(header + body)
    return path


def _refusal(
    tmp_path: Path,
    *,
    arch: str | None,
    name: str,
    identifier: str | None = None,
):
    gguf = _write_gguf(tmp_path / name, arch = arch)
    backend = LlamaCppBackend()
    backend._model_identifier = identifier
    backend._read_gguf_metadata(str(gguf))
    return backend, backend._non_chat_gguf_refusal(str(gguf))


def test_chat_arch_is_not_refused(tmp_path):
    backend, message = _refusal(tmp_path, arch = "llama", name = "chat.gguf")
    assert backend._architecture == "llama"
    assert backend._gguf_header_parsed is True
    assert message is None


def test_image_arch_names_the_images_page(tmp_path):
    _, message = _refusal(tmp_path, arch = "flux", name = "flux.gguf")
    assert message is not None
    assert "Images page" in message
    assert "Video page" not in message


def test_video_arch_names_the_video_page(tmp_path):
    _, message = _refusal(tmp_path, arch = "ltxv", name = "ltx.gguf")
    assert message is not None
    assert "Video page" in message
    assert "Images page" not in message


def test_diffusiongemma_arch_is_left_to_the_diffusion_runner(tmp_path):
    # A block-diffusion LANGUAGE model IS servable (by the diffusion runner), so the
    # preflight must not intercept it on its way there.
    backend, message = _refusal(tmp_path, arch = "diffusiongemma", name = "dg.gguf")
    assert backend._is_diffusion is True
    assert message is None


def test_metadata_less_video_gguf_is_refused_and_named(tmp_path):
    # The shape of unsloth/MiniMax-H3-GGUF: valid GGUF, zero KV pairs, no architecture.
    backend, message = _refusal(
        tmp_path,
        arch = None,
        name = "minimax_h3_fl2va_pruned-Q2_K.gguf",
        identifier = "unsloth/MiniMax-H3-GGUF",
    )
    assert backend._architecture is None
    assert backend._gguf_header_parsed is True
    assert message is not None
    assert "Video page" in message


def test_metadata_less_gguf_of_unknown_family_still_refuses(tmp_path):
    # No architecture means llama-server cannot load it whatever it is; the message just
    # cannot promise which page runs it.
    _, message = _refusal(tmp_path, arch = None, name = "mystery.gguf", identifier = "x/mystery")
    assert message is not None
    assert "general.architecture" in message


def test_unreadable_header_yields_no_verdict(tmp_path):
    # A non-GGUF file, and a GGUF truncated mid-KV, must both fall through to llama-server
    # rather than being refused as "declares no architecture".
    not_gguf = tmp_path / "notes.txt"
    not_gguf.write_bytes(b"this is not a gguf file at all")
    backend = LlamaCppBackend()
    backend._read_gguf_metadata(str(not_gguf))
    assert backend._gguf_header_parsed is False
    assert backend._non_chat_gguf_refusal(str(not_gguf)) is None

    full = _write_gguf(tmp_path / "chat.gguf", arch = "llama")
    truncated = tmp_path / "truncated.gguf"
    # Keep the counts (which promise a KV pair) but cut the KV itself away.
    truncated.write_bytes(full.read_bytes()[:24])
    backend2 = LlamaCppBackend()
    backend2._read_gguf_metadata(str(truncated))
    assert backend2._architecture is None
    assert backend2._gguf_header_parsed is False
    assert backend2._non_chat_gguf_refusal(str(truncated)) is None


def test_verdict_requires_a_parsed_header_even_if_called_directly(tmp_path):
    # Defensive: a caller that skipped _read_gguf_metadata gets no verdict, not a refusal
    # built on a never-populated architecture.
    backend = LlamaCppBackend()
    assert backend._non_chat_gguf_refusal(str(tmp_path / "unread.gguf")) is None


@pytest.mark.parametrize("arch", sorted(LlamaCppBackend._DIFFUSION_ARCHES))
def test_every_media_arch_is_refused_before_launch(tmp_path, arch):
    _, message = _refusal(tmp_path, arch = arch, name = f"{arch}.gguf")
    assert message is not None
    assert "cannot run as a chat model" in message


def test_the_path_probe_does_not_touch_the_live_backend(tmp_path):
    # The refusal has to run BEFORE Phase 1 kills the resident model, otherwise it still
    # costs the user their chat model to answer a question the header already answered.
    # Reading the header into `self` that early would overwrite the LIVE model's metadata
    # with the file being rejected, so the pre-teardown probe uses its own instance.
    live = LlamaCppBackend()
    live._model_identifier = "unsloth/Qwen3-0.6B-GGUF"
    live._architecture = "qwen3"
    live._context_length = 40960
    live._gguf_header_parsed = True

    video = _write_gguf(tmp_path / "minimax_h3-Q2_K.gguf", arch = None)
    message = LlamaCppBackend._non_chat_gguf_refusal_for_path(str(video), "unsloth/MiniMax-H3-GGUF")
    assert message is not None and "Video page" in message
    # The resident model's metadata is untouched.
    assert live._architecture == "qwen3"
    assert live._context_length == 40960
    assert live._model_identifier == "unsloth/Qwen3-0.6B-GGUF"


def test_the_path_probe_agrees_with_the_instance_check(tmp_path):
    # One verdict, two entry points: the pre-teardown probe and the post-download check
    # must never disagree, or a load could be refused on one path and launched on the other.
    for arch, name in (("llama", "chat.gguf"), ("flux", "flux.gguf"), ("ltxv", "ltx.gguf")):
        gguf = _write_gguf(tmp_path / name, arch = arch)
        backend = LlamaCppBackend()
        backend._model_identifier = None
        backend._read_gguf_metadata(str(gguf))
        assert backend._non_chat_gguf_refusal(str(gguf)) == (
            LlamaCppBackend._non_chat_gguf_refusal_for_path(str(gguf), None)
        )


@pytest.mark.parametrize(
    "identifier,name",
    [
        # Measured headers (byte-range read of the live repos): both declare "lumina2".
        ("unsloth/Z-Image-Turbo-GGUF", "z-image-turbo-Q2_K.gguf"),
        ("Immac/NetaYume-Lumina-Image-2.0-GGUF", "NetaYumev2plus_unet_q2_k.gguf"),
        # The family keyword lives only in the filename, as it does for a local copy.
        (None, "lumina2-q4_k_m.gguf"),
    ],
)
def test_a_resolvable_shared_arch_still_names_the_images_page(tmp_path, identifier, name):
    _, message = _refusal(tmp_path, arch = "lumina2", name = name, identifier = identifier)
    assert message is not None
    assert "Open it from the Images page" in message


def test_an_unresolvable_shared_arch_promises_no_page(tmp_path):
    # "lumina2" is shared: Z-Image's DiT is a Lumina2 derivative, so the header alone does
    # not say which family a file is. neta-art/neta-lumina-gguf ships REAL lumina2 GGUFs
    # (header read from the live repo: general.architecture = "lumina2", 400 tensors) whose
    # repo id and filename resolve to no family -- "lumina-2" is deliberately not aliased
    # to bare "lumina" -- so routes.models._arch_to_task tags them
    # image-diffusion-unsupported and the Images picker never lists them. Sending the user
    # there is the empty promise the runnable/unrunnable split exists to remove.
    _, message = _refusal(
        tmp_path,
        arch = "lumina2",
        name = "checkpoint-e3_s9658-Q2_K.gguf",
        identifier = "neta-art/neta-lumina-gguf",
    )
    assert message is not None
    assert "cannot run as a chat model" in message
    assert "Open it from the" not in message


def test_the_shared_arch_verdict_matches_the_pickers(tmp_path):
    # The one invariant: the refusal may name the Images page exactly when
    # routes.models._arch_to_task says the row is selectable there. Same evidence, same
    # answer, or the message points at a list the model is not in.
    from routes.models import _AMBIGUOUS_DIFFUSION_GGUF_ARCHS, _arch_to_task
    assert LlamaCppBackend._AMBIGUOUS_IMAGE_ARCHES == _AMBIGUOUS_DIFFUSION_GGUF_ARCHS
    for identifier, name in (
        ("unsloth/Z-Image-Turbo-GGUF", "z-image-turbo-Q2_K.gguf"),
        ("neta-art/neta-lumina-gguf", "checkpoint-e3_s9658-Q2_K.gguf"),
        ("Immac/NetaYume-Lumina-Image-2.0-GGUF", "NetaYumev2plus_unet_q2_k.gguf"),
        (None, "some-random-denoiser.gguf"),
    ):
        for arch in sorted(LlamaCppBackend._AMBIGUOUS_IMAGE_ARCHES):
            _, message = _refusal(tmp_path, arch = arch, name = name, identifier = identifier)
            task = _arch_to_task(arch, name_hints = (identifier, name))
            picker_lists_it = task == "text-to-image"
            assert ("Open it from the Images page" in message) is picker_lists_it, (
                identifier,
                name,
                message,
            )


def test_the_unrunnable_set_mirrors_the_canonical_one():
    # Two places must not drift: routes.models tags these GGUFs
    # image-diffusion-unsupported, which hides them from the Images AND Video pickers, so
    # naming either page here would send the user to an empty list.
    from routes.models import _UNSUPPORTED_DIFFUSION_GGUF_ARCHS

    assert LlamaCppBackend._UNRUNNABLE_MEDIA_ARCHES == _UNSUPPORTED_DIFFUSION_GGUF_ARCHS
    # ...and a runnable set may never overlap it, or the destination is a promise again.
    assert not (LlamaCppBackend._IMAGE_ARCHES & _UNSUPPORTED_DIFFUSION_GGUF_ARCHS)
    assert not (LlamaCppBackend._VIDEO_ARCHES & _UNSUPPORTED_DIFFUSION_GGUF_ARCHS)


@pytest.mark.parametrize("arch", sorted(LlamaCppBackend._UNRUNNABLE_MEDIA_ARCHES))
def test_an_unrunnable_media_arch_promises_no_page(tmp_path, arch):
    _, message = _refusal(tmp_path, arch = arch, name = f"{arch}.gguf")
    assert message is not None
    # Refused, but WITHOUT being sent anywhere: no page can run these, so naming one is
    # the empty promise the whole item was about.
    assert "neither the Images page nor the Video page" in message
    assert "Open it from the" not in message


@pytest.mark.parametrize("repo", ["gguf-org/flux2-dev-gguf", "calcuis/cosmos-predict2-gguf"])
def test_a_placeholder_architecture_still_refuses(tmp_path, repo):
    # gguf-connector writes a literal "pig" into general.architecture rather than an
    # architecture; measured on gguf-org/flux2-dev-gguf/flux2-dev-iq4_nl.gguf. llama.cpp
    # knows no such arch, so without normalising it the file slips past every set and dies
    # in llama-server as the opaque failure this preflight exists to prevent.
    _, message = _refusal(tmp_path, arch = "pig", name = "flux2-dev-iq4_nl.gguf", identifier = repo)
    assert message is not None
    assert "cannot" in message


def test_a_placeholder_architecture_matches_the_picker_verdict(tmp_path):
    # The placeholder carries no family of its own, so both sides have to fall back to the
    # repo id and filename, and they have to agree: naming the Images page for a file the
    # Images picker drops is the empty promise the runnable split exists to remove.
    from routes.models import _arch_to_task
    for identifier, name, page_named in (
        ("gguf-org/flux2-dev-gguf", "flux2-dev-iq4_nl.gguf", True),
        ("calcuis/cosmos-predict2-gguf", "cosmos-predict2-q4_0.gguf", False),
        ("someone/mystery-gguf", "mystery-q4_0.gguf", False),
    ):
        for arch in sorted(LlamaCppBackend._PLACEHOLDER_ARCHES):
            _, message = _refusal(tmp_path, arch = arch, name = name, identifier = identifier)
            assert message is not None
            task = _arch_to_task(arch, name_hints = (identifier, name))
            assert (task == "text-to-image") is page_named, (identifier, arch, task)
            assert ("Open it from the Images page" in message) is page_named, message


def test_an_unassemblable_video_arch_promises_no_page(tmp_path):
    # Wan 2.2 A14B is a two-expert MoE the Video backend cannot assemble, and Wan 2.1's repo
    # ids resolve to no family at all, so routes.models._arch_to_task tags both
    # image-diffusion-unsupported and the Video picker never lists them. The header says
    # "wan" for all three, so the refusal has to consult the same family resolution rather
    # than trusting the arch.
    from routes.models import _arch_to_task
    for identifier, name, page_named in (
        ("QuantStack/Wan2.2-TI2V-5B-GGUF", "Wan2.2-TI2V-5B-Q4_K_M.gguf", True),
        ("QuantStack/Wan2.2-T2V-A14B-GGUF", "Wan2.2-T2V-A14B-HighNoise-Q4_K_M.gguf", False),
        ("QuantStack/Wan2.1-T2V-14B-GGUF", "Wan2.1-T2V-14B-Q4_K_M.gguf", False),
    ):
        _, message = _refusal(tmp_path, arch = "wan", name = name, identifier = identifier)
        assert message is not None
        task = _arch_to_task("wan", name_hints = (identifier, name))
        assert (task == "text-to-video") is page_named, (identifier, task)
        assert ("Open it from the Video page" in message) is page_named, message
