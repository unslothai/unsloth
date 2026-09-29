# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""CPU-only unit tests for the non-chat GGUF refusal on a REPO load.

The local-file refusal already runs before Phase 1; a repo load did not, because the file
only exists after Phase 2, so the resident chat model had already been killed for a load
that was never going to start. That is the path the Model Hub takes (the route resolves
every Hub model to hf_repo), so it was the reported case, not an API-only corner.

The fix decides it from a header-sized byte range instead of the download. Measured over the
Hub, media GGUFs finish their KV walk in 144-3,987 bytes while a chat model's
tokenizer.ggml.tokens array pushes its KV block past 5 MB, so a 256 KiB prefix is decisive
for the first and truncated for the second -- and truncated means no verdict, which is the
direction it has to fail in.
"""

from __future__ import annotations

import contextlib
import inspect
import struct
from dataclasses import replace
from pathlib import Path

import pytest

import core.inference.diffusion_compat as diffusion_compat
import core.inference.llama_cpp as llama_cpp_module
from core.inference.llama_cpp import GgufLoadIntent, LlamaCppBackend

GGUF_MAGIC = 0x46554747


def _gguf_bytes(
    *,
    arch: str | None,
    tensor_count: int = 1,
    declared_kv: int | None = None,
) -> bytes:
    """A GGUF header as BYTES. ``declared_kv`` overstates the KV count without writing the
    pairs -- the shape a byte-range prefix of a chat model has: the counts say more is
    coming and the buffer ends first."""
    body = b""
    kv_count = 0
    if arch is not None:
        key = b"general.architecture"
        val = arch.encode()
        body += struct.pack("<Q", len(key)) + key
        body += struct.pack("<I", 8)  # STRING
        body += struct.pack("<Q", len(val)) + val
        kv_count = 1
    if declared_kv is not None:
        kv_count = declared_kv
    return struct.pack("<II", GGUF_MAGIC, 3) + struct.pack("<QQ", tensor_count, kv_count) + body


def _probe(
    monkeypatch,
    *,
    header: bytes,
    filename: str = "model-Q4_K_M.gguf",
    identifier = None,
    local: str | None = None,
    patch_cache: bool = True,
):
    """Run the remote probe against a canned header, recording whether it went to the Hub."""
    requests: list[tuple[str, str]] = []

    def _fake_remote(repo_id, gguf_filename, hf_token):
        requests.append((repo_id, gguf_filename))
        return header

    monkeypatch.setattr(
        llama_cpp_module, "_resolve_variant_gguf_files", lambda *_a, **_k: (filename, [])
    )
    # The verified cache lookup is the only way a local copy enters the probe; a test with
    # its own stub for it passes patch_cache = False.
    if patch_cache:
        monkeypatch.setattr(llama_cpp_module, "cached_gguf_for_load", lambda *_a, **_k: local)
    monkeypatch.setattr(diffusion_compat, "_read_gguf_header", _fake_remote)
    message = LlamaCppBackend._remote_non_chat_gguf_refusal(
        hf_repo = "owner/model",
        hf_variant = "Q4_K_M",
        hf_token = None,
        model_identifier = identifier,
    )
    return message, requests


# ── the verdict itself ────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "arch, page",
    [
        ("flux", "Images page"),
        ("ltxv", "Video page"),
        ("wan", "Video page"),
        ("sd3", "Images page"),
    ],
)
def test_a_media_repo_gguf_is_refused_from_its_header_alone(monkeypatch, arch, page):
    # 144-145 bytes on the Hub for every one of these: three KV pairs and no vocabulary.
    message, requests = _probe(monkeypatch, header = _gguf_bytes(arch = arch))
    assert message is not None
    assert page in message
    assert requests == [("owner/model", "model-Q4_K_M.gguf")]


def test_a_placeholder_arch_repo_gguf_is_refused(monkeypatch):
    # gguf-connector writes a literal "pig"; the probe must normalise it like the on-disk
    # check does, or a 20 GB flux2 download still costs the resident model.
    message, _ = _probe(
        monkeypatch,
        header = _gguf_bytes(arch = "pig"),
        filename = "flux2-dev-q4_k_m.gguf",
        identifier = "gguf-org/flux2-dev-gguf",
    )
    assert message is not None
    assert "Images page" in message


def test_the_filename_survives_into_the_verdict(monkeypatch):
    # With no general.architecture the page is named off the file name, so the prefix is
    # spooled under the REAL name and not a temp one.
    message, _ = _probe(
        monkeypatch,
        header = _gguf_bytes(arch = None),
        filename = "wan2.2-ti2v-5b-Q4_K_M.gguf",
    )
    assert message is not None
    assert "Video page" in message


def test_an_ambiguous_arch_is_resolved_from_the_real_filename(monkeypatch):
    # A shared architecture is resolved from the repo id and the FILE NAME: the other reason
    # the prefix is spooled under its real name.
    if not getattr(LlamaCppBackend, "_AMBIGUOUS_IMAGE_ARCHES", None):
        pytest.skip("no ambiguous image archs on this build")
    message, _ = _probe(
        monkeypatch,
        header = _gguf_bytes(arch = sorted(LlamaCppBackend._AMBIGUOUS_IMAGE_ARCHES)[0]),
        filename = "z-image-turbo-Q2_K.gguf",
        identifier = "unsloth/Z-Image-Turbo-GGUF",
    )
    assert message is not None
    assert "Open it from the Images page" in message


def test_a_chat_repo_gguf_is_not_refused(monkeypatch):
    message, _ = _probe(monkeypatch, header = _gguf_bytes(arch = "qwen3"))
    assert message is None


# ── fail-open, in every direction ─────────────────────────────────────────────────


def test_a_prefix_cut_mid_kv_yields_no_verdict(monkeypatch):
    # The chat-model case for real: 256 KiB of a 5.9 MB KV block. The counts promise 32
    # pairs, one arrives, the buffer ends. _gguf_header_parsed stays False, so "declares no
    # architecture" must not be read out of a read that simply stopped early.
    message, _ = _probe(monkeypatch, header = _gguf_bytes(arch = "llama", declared_kv = 32))
    assert message is None


def test_an_unreachable_hub_yields_no_verdict(monkeypatch):
    # _read_gguf_header returns b"" for offline, a 401, a deadline, or a proxy that
    # answered 200 instead of 206.
    message, _ = _probe(monkeypatch, header = b"")
    assert message is None


def test_an_unresolvable_filename_yields_no_verdict(monkeypatch):
    monkeypatch.setattr(
        llama_cpp_module, "_resolve_variant_gguf_files", lambda *_a, **_k: (None, [])
    )
    called = []
    monkeypatch.setattr(
        diffusion_compat, "_read_gguf_header", lambda *_a, **_k: called.append(True) or b""
    )
    assert (
        LlamaCppBackend._remote_non_chat_gguf_refusal(
            hf_repo = "owner/model", hf_variant = None, hf_token = None, model_identifier = None
        )
        is None
    )
    # No filename means no request either: there is nothing to ask for.
    assert called == []


def test_a_probe_that_raises_yields_no_verdict(monkeypatch):
    def _boom(*_a, **_k):
        raise ConnectionError("hub unreachable")

    monkeypatch.setattr(
        llama_cpp_module, "_resolve_variant_gguf_files", lambda *_a, **_k: ("m.gguf", [])
    )
    monkeypatch.setattr(diffusion_compat, "_local_gguf_path", lambda *_a, **_k: None)
    monkeypatch.setattr(diffusion_compat, "_read_gguf_header", _boom)
    assert (
        LlamaCppBackend._remote_non_chat_gguf_refusal(
            hf_repo = "owner/model",
            hf_variant = "Q4_K_M",
            hf_token = None,
            model_identifier = None,
        )
        is None
    )


def test_a_cached_copy_is_read_off_disk_instead_of_the_hub(monkeypatch, tmp_path):
    # A cached file answers this with no request, the only way the verdict survives an
    # unreachable Hub.
    cached = tmp_path / "ltx-2-19b-dev-Q4_K_M.gguf"
    cached.write_bytes(_gguf_bytes(arch = "ltxv"))
    message, requests = _probe(
        monkeypatch,
        header = b"",  # would fail open if the probe went to the network
        filename = "ltx-2-19b-dev-Q4_K_M.gguf",
        local = str(cached),
    )
    assert message is not None and "Video page" in message
    assert requests == []


# ── the invariant the whole item is about ─────────────────────────────────────────


def test_the_remote_probe_agrees_with_the_other_two_entry_points(tmp_path, monkeypatch):
    # Three entry points, one verdict: otherwise a load is refused on one path and launched
    # on another.
    for arch, name in (("llama", "chat.gguf"), ("flux", "flux.gguf"), ("ltxv", "ltx.gguf")):
        header = _gguf_bytes(arch = arch)
        gguf = tmp_path / name
        gguf.write_bytes(header)
        backend = LlamaCppBackend()
        backend._model_identifier = None
        backend._read_gguf_metadata(str(gguf))
        instance = backend._non_chat_gguf_refusal(str(gguf))
        by_path = LlamaCppBackend._non_chat_gguf_refusal_for_path(str(gguf), None)
        remote, _ = _probe(monkeypatch, header = header, filename = name)
        assert instance == by_path == remote, arch


def test_the_repo_refusal_sits_above_the_teardown_in_source():
    src = inspect.getsource(llama_cpp_module.LlamaCppBackend.load_model)
    remote = src.index("self._remote_non_chat_gguf_refusal(")
    teardown = src.index("# ── Phase 1: kill old process")
    assert remote < teardown
    # And the post-download check stays as the backstop for what the probe fails open on.
    assert src.index("non_chat = self._non_chat_gguf_refusal(model_path)") > teardown


def _repo_load(monkeypatch, backend, order):
    monkeypatch.setattr(backend, "_find_llama_server_binary", lambda **_kwargs: "/bin/llama")
    monkeypatch.setattr(backend, "_is_vulkan_backend", lambda _binary = None: False)
    monkeypatch.setattr(backend, "_backend_lacks_gpu_lib", lambda _binary = None: False)
    monkeypatch.setattr(backend, "_kill_process", lambda: order.append("kill"))
    monkeypatch.setattr(llama_cpp_module, "_resolve_repo_id_casing", lambda repo: repo)
    monkeypatch.setattr(
        llama_cpp_module, "_hf_offline_if_unreachable", lambda: contextlib.nullcontext()
    )
    monkeypatch.setattr(
        llama_cpp_module, "_resolve_variant_gguf_files", lambda *_a, **_k: ("m-Q4_K_M.gguf", [])
    )
    monkeypatch.setattr(diffusion_compat, "_local_gguf_path", lambda *_a, **_k: None)


def test_a_media_repo_load_keeps_the_resident_model_and_never_downloads(monkeypatch):
    # The regression. Before the fix this killed the live server, downloaded the whole
    # media GGUF, and only then raised.
    backend = LlamaCppBackend()
    order: list[str] = []
    _repo_load(monkeypatch, backend, order)
    monkeypatch.setattr(
        backend, "_download_gguf", lambda **_kwargs: order.append("download") or "/cache/m.gguf"
    )
    monkeypatch.setattr(
        diffusion_compat, "_read_gguf_header", lambda *_a, **_k: _gguf_bytes(arch = "wan")
    )

    with pytest.raises(ValueError, match = "Video page"):
        backend.load_model(
            GgufLoadIntent(
                hf_repo = "unsloth/Wan2.2-TI2V-5B-GGUF",
                hf_variant = "Q4_K_M",
                model_identifier = "unsloth/Wan2.2-TI2V-5B-GGUF",
            )
        )

    # Neither the teardown nor the multi-GB download happened.
    assert order == []


def test_a_chat_repo_load_still_proceeds_past_the_teardown(monkeypatch):
    # A prefix the probe cannot finish must not become a refusal: the load carries on.
    backend = LlamaCppBackend()
    order: list[str] = []
    _repo_load(monkeypatch, backend, order)

    def _download(**_kwargs):
        order.append("download")
        raise RuntimeError("stop here")

    monkeypatch.setattr(backend, "_download_gguf", _download)
    monkeypatch.setattr(
        diffusion_compat,
        "_read_gguf_header",
        lambda *_a, **_k: _gguf_bytes(arch = "qwen3", declared_kv = 32),
    )

    with pytest.raises(RuntimeError, match = "stop here"):
        backend.load_model(
            GgufLoadIntent(
                hf_repo = "unsloth/Qwen3-0.6B-GGUF",
                hf_variant = "Q4_K_M",
                model_identifier = "unsloth/Qwen3-0.6B-GGUF",
            )
        )

    assert order == ["kill", "download"]


def test_a_preflighted_file_is_judged_without_a_second_request(monkeypatch, tmp_path):
    # When a GPU-pin preflight already fetched the file the verdict is free, and that path
    # used to skip the refusal entirely.
    media = tmp_path / "flux1-dev-Q4_K_S.gguf"
    media.write_bytes(_gguf_bytes(arch = "flux"))
    backend = LlamaCppBackend()
    order: list[str] = []
    _repo_load(monkeypatch, backend, order)
    monkeypatch.setattr(backend, "_is_vulkan_backend", lambda _binary = None: True)
    monkeypatch.setattr(backend, "_get_gpu_memory", lambda _binary = None: [(0, 1024, 2048)])
    monkeypatch.setattr(
        backend, "_download_gguf", lambda **_kwargs: order.append("download") or str(media)
    )
    monkeypatch.setattr(backend, "_gguf_path_is_diffusion", lambda *_args: False)
    monkeypatch.setattr(
        diffusion_compat,
        "_read_gguf_header",
        lambda *_a, **_k: pytest.fail("the preflighted file must be judged directly"),
    )

    with pytest.raises(ValueError, match = "Images page"):
        backend.load_model(
            GgufLoadIntent(
                hf_repo = "city96/FLUX.1-dev-gguf",
                hf_variant = "Q4_K_S",
                model_identifier = "city96/FLUX.1-dev-gguf",
                gpu_ids = [0],
            )
        )

    assert order == ["download"]
    assert Path(media).is_file()


def test_the_cached_file_the_load_will_open_is_the_one_judged(monkeypatch, tmp_path):
    # _download_gguf resolves its local copy through cached_gguf_for_load, which walks the
    # cached variant candidates rather than the current listing's filename. The probe follows
    # it, or a repo that renamed the file for a quant would let the probe judge one GGUF and
    # the load open another. The listing name here is deliberately not the cached one.
    cached = tmp_path / "renamed-ltx-2-19b-dev-Q4_K_M.gguf"
    cached.write_bytes(_gguf_bytes(arch = "ltxv"))
    monkeypatch.setattr(llama_cpp_module, "cached_gguf_for_load", lambda *_a, **_k: str(cached))
    message, requests = _probe(
        monkeypatch,
        header = b"",
        filename = "listing-Q4_K_M.gguf",
        patch_cache = False,
    )
    assert message is not None and "Video page" in message
    assert requests == []


def test_nothing_cached_means_the_hub_is_asked(monkeypatch, tmp_path):
    # With no verified cache hit the probe range-reads the Hub rather than an unverified
    # path: a candidate the verified lookup rejected is one _download_gguf will skip anyway.
    message, requests = _probe(
        monkeypatch,
        header = _gguf_bytes(arch = "ltxv"),
        filename = "ltx-2-19b-dev-Q4_K_M.gguf",
        local = None,
    )
    assert message is not None and "Video page" in message
    assert requests == [("owner/model", "ltx-2-19b-dev-Q4_K_M.gguf")]


def test_a_refused_load_leaves_the_resident_process_state_alone(monkeypatch):
    # The refusal spares the resident server, so it has to spare what describes it. Resetting
    # the launch revision would say the live process launched from the binary installed NOW,
    # so an Apply after `unsloth studio update` would dedupe against it instead of
    # relaunching; clearing the DFlash verdict loses a retry the same way.
    backend = LlamaCppBackend()
    order: list[str] = []
    _repo_load(monkeypatch, backend, order)
    backend._launch_binary_revision = ("llama-server", 1, 111.0)
    backend._dflash_retry_needed = True
    monkeypatch.setattr(
        backend, "_download_gguf", lambda **_kwargs: order.append("download") or "/cache/m.gguf"
    )
    monkeypatch.setattr(
        diffusion_compat, "_read_gguf_header", lambda *_a, **_k: _gguf_bytes(arch = "wan")
    )

    with pytest.raises(ValueError, match = "Video page"):
        backend.load_model(
            GgufLoadIntent(
                hf_repo = "unsloth/Wan2.2-TI2V-5B-GGUF",
                hf_variant = "Q4_K_M",
                model_identifier = "unsloth/Wan2.2-TI2V-5B-GGUF",
            )
        )

    assert order == []
    assert backend._launch_binary_revision == ("llama-server", 1, 111.0)
    assert backend._dflash_retry_needed is True


def test_the_per_load_resets_sit_below_the_teardown_in_source():
    src = inspect.getsource(llama_cpp_module.LlamaCppBackend.load_model)
    teardown = src.index("# ── Phase 1: kill old process")
    assert src.index("self._launch_binary_revision = self._binary_revision(binary)") > teardown
    assert src.index("self._dflash_retry_needed = False") > teardown


def test_the_cached_probe_is_verified_the_way_the_loader_verifies_it(monkeypatch, tmp_path):
    # cached_gguf_for_load(verify_sizes = True) is what _download_gguf uses, so a snapshot
    # truncated after its header is skipped there. Judging it here would classify bytes the
    # load never opens, refusing a valid chat repo off a stale snapshot.
    cached = tmp_path / "ltx-2-19b-dev-Q4_K_M.gguf"
    cached.write_bytes(_gguf_bytes(arch = "ltxv"))
    seen: list[bool] = []

    def _cached(
        _repo,
        _variant,
        *,
        verify_sizes = False,
        hf_token = None,
    ):
        seen.append(verify_sizes)
        return str(cached) if verify_sizes else "/should/not/be/used.gguf"

    monkeypatch.setattr(llama_cpp_module, "cached_gguf_for_load", _cached)
    message, requests = _probe(
        monkeypatch, header = b"", filename = "listing-Q4_K_M.gguf", patch_cache = False
    )
    assert seen == [True]
    assert message is not None and "Video page" in message
    assert requests == []


def test_a_declared_media_arch_is_acted_on_before_the_walk_finishes(monkeypatch, tmp_path):
    # general.architecture is KV #0 in every GGUF measured, so a repo with bulky later
    # metadata can declare a media arch inside the 256 KiB prefix and still leave the KV walk
    # unfinished. Discarding that verdict put the teardown and the full download back.
    header = _gguf_bytes(arch = "flux", declared_kv = 4096)
    message, _requests = _probe(monkeypatch, header = header, filename = "flux1-dev-Q4_K_M.gguf")
    assert message is not None and "Images page" in message

    # The no-architecture fallback still needs the complete walk: an unfinished read is
    # indistinguishable from a file that declares nothing.
    quiet = _probe(
        monkeypatch, header = _gguf_bytes(arch = None, declared_kv = 4096), filename = "mystery-Q4_K_M.gguf"
    )[0]
    assert quiet is None


def test_the_probe_runs_under_the_same_offline_guard_as_the_download(monkeypatch):
    # With the Hub unreachable and the model cached, the probe's two Hub calls (the file
    # listing, then the cached candidate's revision sizes) would each wait out their retry
    # backoff before any local header is read. The download has always run under the guard;
    # the probe has to as well, or a cached load pays two timeouts for a free verdict.
    backend = LlamaCppBackend()
    order: list[str] = []
    _repo_load(monkeypatch, backend, order)
    entered: list[str] = []

    @contextlib.contextmanager
    def _guard():
        entered.append("enter")
        yield True

    monkeypatch.setattr(llama_cpp_module, "_hf_offline_if_unreachable", _guard)

    guarded: list[bool] = []

    def _probe_call(**_kwargs):
        # Recorded, not asserted: an assert here would be swallowed by the raises() below.
        guarded.append(bool(entered))
        return None

    monkeypatch.setattr(backend, "_remote_non_chat_gguf_refusal", _probe_call)
    monkeypatch.setattr(
        backend, "_download_gguf", lambda **_kwargs: order.append("download") or "/cache/m.gguf"
    )

    with pytest.raises(Exception):
        backend.load_model(
            GgufLoadIntent(
                hf_repo = "unsloth/Qwen3-0.6B-GGUF",
                hf_variant = "Q4_K_M",
                model_identifier = "unsloth/Qwen3-0.6B-GGUF",
            )
        )
    assert guarded == [True], guarded


def test_the_probe_guard_sits_above_the_teardown_in_source():
    src = inspect.getsource(llama_cpp_module.LlamaCppBackend.load_model)
    guarded = src.index("with _hf_offline_if_unreachable():\n                    _early_non_chat")
    assert guarded < src.index("# ── Phase 1: kill old process")


def test_the_route_entry_point_judges_an_intent_before_the_arbiter(monkeypatch, tmp_path):
    # What the route calls. load_model's own copy runs before ITS teardown but after
    # acquire_for has evicted a resident pipeline and cancelled the running chats.
    media = tmp_path / "ltx-2-19b-dev-Q4_K_M.gguf"
    media.write_bytes(_gguf_bytes(arch = "ltxv"))
    intent = GgufLoadIntent(
        model_identifier = "unsloth/LTX-2-GGUF",
        gguf_path = str(media),
    )
    message = LlamaCppBackend.non_chat_gguf_refusal_for_intent(intent)
    assert message is not None and "Video page" in message

    chat = tmp_path / "qwen3-0.6b-Q4_K_M.gguf"
    chat.write_bytes(_gguf_bytes(arch = "qwen3"))
    assert (
        LlamaCppBackend.non_chat_gguf_refusal_for_intent(
            GgufLoadIntent(model_identifier = "unsloth/Qwen3-0.6B-GGUF", gguf_path = str(chat))
        )
        is None
    )


def test_the_route_entry_point_fails_open(monkeypatch, tmp_path):
    # A probe that raises is not a verdict: the load proceeds, its own checks back it up.
    def _boom(*_a, **_k):
        raise RuntimeError("hub down")

    monkeypatch.setattr(LlamaCppBackend, "_remote_non_chat_gguf_refusal", _boom)
    assert (
        LlamaCppBackend.non_chat_gguf_refusal_for_intent(
            GgufLoadIntent(model_identifier = "owner/model", hf_repo = "owner/model")
        )
        is None
    )


def test_the_route_asks_before_it_takes_the_gpu():
    # Source order, since the eviction it protects is the arbiter's, not the loader's.
    import inspect as _inspect

    from routes import inference as inference_routes

    src = _inspect.getsource(inference_routes)
    refusal = src.index("non_chat_gguf_refusal_for_intent")
    acquire = src.index("lambda: gguf_load_stack.enter_context(chat_load_in_flight())")
    assert refusal < acquire


def test_the_route_hands_its_verdict_to_the_load(monkeypatch, tmp_path):
    # The route and the loader ask for the same intent seconds apart, and each ask is a
    # listing, a cache verification and a range request. The second takes the first's answer.
    calls: list[str] = []

    def _verdict(**kwargs):
        calls.append(kwargs["hf_repo"])
        return "This is a text-to-video GGUF. Open it from the Video page instead."

    monkeypatch.setattr(LlamaCppBackend, "_remote_non_chat_gguf_verdict", _verdict)
    monkeypatch.setattr(LlamaCppBackend, "_route_verdict_handoff", None)
    intent = GgufLoadIntent(
        model_identifier = "unsloth/LTX-2-GGUF",
        hf_repo = "unsloth/LTX-2-GGUF",
        hf_variant = "Q4_K_M",
    )
    assert LlamaCppBackend.non_chat_gguf_refusal_for_intent(intent) is not None
    assert calls == ["unsloth/LTX-2-GGUF"]

    # The load that follows reuses it...
    reused = LlamaCppBackend._remote_non_chat_gguf_refusal(
        hf_repo = "unsloth/LTX-2-GGUF",
        hf_variant = "Q4_K_M",
        hf_token = None,
        model_identifier = "unsloth/LTX-2-GGUF",
    )
    assert reused is not None
    assert calls == ["unsloth/LTX-2-GGUF"]

    # ...exactly once. A second load probes for itself rather than trusting a stale answer.
    LlamaCppBackend._remote_non_chat_gguf_refusal(
        hf_repo = "unsloth/LTX-2-GGUF",
        hf_variant = "Q4_K_M",
        hf_token = None,
        model_identifier = "unsloth/LTX-2-GGUF",
    )
    assert calls == ["unsloth/LTX-2-GGUF"] * 2


def test_a_handoff_for_another_model_is_not_taken(monkeypatch):
    calls: list[str] = []

    def _verdict(**kwargs):
        calls.append(kwargs["hf_repo"])
        return None

    monkeypatch.setattr(LlamaCppBackend, "_remote_non_chat_gguf_verdict", _verdict)
    monkeypatch.setattr(LlamaCppBackend, "_route_verdict_handoff", None)
    LlamaCppBackend.non_chat_gguf_refusal_for_intent(
        GgufLoadIntent(model_identifier = "owner/a", hf_repo = "owner/a", hf_variant = "Q4_K_M")
    )
    LlamaCppBackend._remote_non_chat_gguf_refusal(
        hf_repo = "owner/b", hf_variant = "Q4_K_M", hf_token = None, model_identifier = "owner/b"
    )
    assert calls == ["owner/a", "owner/b"]
    # A different variant of the same repo is a different file, so it is not taken either.
    LlamaCppBackend._remote_non_chat_gguf_refusal(
        hf_repo = "owner/a", hf_variant = "Q8_0", hf_token = None, model_identifier = "owner/a"
    )
    assert calls == ["owner/a", "owner/b", "owner/a"]


# Cached GGUF reuse


def _intent(**changes) -> GgufLoadIntent:
    base = dict(
        model_identifier = "owner/model",
        hf_repo = "owner/model",
        hf_variant = "Q4_K_M",
    )
    base.update(changes)
    return GgufLoadIntent(**base)


def _hub_cache(monkeypatch, root):
    """Point the active Hub cache at ``root``."""
    import types as _types

    import utils.hf_cache_settings as hf_cache_settings

    monkeypatch.setattr(
        hf_cache_settings,
        "get_hf_cache_paths",
        lambda: _types.SimpleNamespace(hub_cache = Path(root)),
    )


def _cached_gguf(root: Path, name: str, payload: bytes) -> Path:
    """Create a cached GGUF under ``root``."""
    snapshot = root / "models--owner--model" / "snapshots" / ("a" * 40)
    snapshot.mkdir(parents = True, exist_ok = True)
    path = snapshot / name
    path.write_bytes(payload)
    return path


def _verified(
    path: Path,
    repo: str = "owner/model",
    variant: str = "Q4_K_M",
):
    """What config resolution carries for a cached copy it just verified."""
    return (
        repo,
        variant,
        str(path),
        llama_cpp_module._cached_variant_sizes(repo, variant, str(path)),
    )


def test_a_verified_cached_file_is_judged_without_resolving_it_again(monkeypatch, tmp_path):
    """The probe reuses the file verified during config resolution."""
    _hub_cache(monkeypatch, tmp_path)
    cached = _cached_gguf(tmp_path, "ltx-2-19b-dev-Q4_K_M.gguf", _gguf_bytes(arch = "ltxv"))

    def _no_hub(*_a, **_k):
        raise AssertionError("the probe went back to the Hub for a file it was handed")

    monkeypatch.setattr(llama_cpp_module, "_resolve_variant_gguf_files", _no_hub)
    monkeypatch.setattr(llama_cpp_module, "cached_gguf_for_load", _no_hub)
    monkeypatch.setattr(diffusion_compat, "_read_gguf_header", _no_hub)

    intent = _intent(verified_gguf = _verified(cached))
    verdict = LlamaCppBackend.non_chat_gguf_refusal_for_intent(intent)
    assert verdict is not None and "Video page" in verdict

    # The verdict is also handed to the in-load probe.
    assert (
        LlamaCppBackend._remote_non_chat_gguf_refusal(
            hf_repo = "owner/model",
            hf_variant = "Q4_K_M",
            hf_token = None,
            model_identifier = None,
        )
        == verdict
    )


def test_a_chat_gguf_carried_the_same_way_is_still_not_refused(monkeypatch, tmp_path):
    _hub_cache(monkeypatch, tmp_path)
    cached = _cached_gguf(tmp_path, "chat-Q4_K_M.gguf", _gguf_bytes(arch = "llama"))

    def _no_hub(*_a, **_k):
        raise AssertionError("the probe went back to the Hub for a file it was handed")

    monkeypatch.setattr(llama_cpp_module, "_resolve_variant_gguf_files", _no_hub)
    monkeypatch.setattr(llama_cpp_module, "cached_gguf_for_load", _no_hub)

    intent = _intent(verified_gguf = _verified(cached))
    assert LlamaCppBackend.non_chat_gguf_refusal_for_intent(intent) is None


def test_a_carried_path_is_only_used_for_the_repo_and_variant_it_was_verified_for(
    monkeypatch, tmp_path
):
    """A carried path is valid only for its verified repo and variant."""
    _hub_cache(monkeypatch, tmp_path)
    cached = _cached_gguf(tmp_path, "model-Q4_K_M.gguf", b"GGUF")
    verified = _verified(cached)
    take = LlamaCppBackend._verified_cached_gguf

    assert take(_intent(verified_gguf = verified), "owner/model", "Q4_K_M") == str(cached)
    # Repo and variant matching is case-insensitive.
    assert take(_intent(verified_gguf = verified), "Owner/Model", "Q4_K_M") == str(cached)
    assert take(_intent(verified_gguf = verified), "owner/model", "q4_k_m") == str(cached)

    # Other repos and variants fall back to normal resolution.
    assert take(_intent(verified_gguf = verified), "owner/model", "Q8_0") is None
    assert take(_intent(verified_gguf = verified), "other/model", "Q4_K_M") is None
    assert take(_intent(verified_gguf = verified), "owner/model", None) is None

    # Missing or malformed values are ignored.
    assert take(_intent(), "owner/model", "Q4_K_M") is None
    assert take(_intent(verified_gguf = ("owner/model",)), "owner/model", "Q4_K_M") is None
    assert (
        take(
            _intent(verified_gguf = ("owner/model", "Q4_K_M", str(cached))),
            "owner/model",
            "Q4_K_M",
        )
        is None
    )
    assert take(_intent(verified_gguf = "just-a-path"), "owner/model", "Q4_K_M") is None


def test_a_file_deleted_between_the_request_and_the_launch_is_not_reused(monkeypatch, tmp_path):
    """A deleted carried file must not be reused."""
    _hub_cache(monkeypatch, tmp_path)
    cached = _cached_gguf(tmp_path, "model-Q4_K_M.gguf", b"GGUF")
    intent = _intent(verified_gguf = _verified(cached))
    assert LlamaCppBackend._verified_cached_gguf(intent, "owner/model", "Q4_K_M") == str(cached)

    cached.unlink()
    assert LlamaCppBackend._verified_cached_gguf(intent, "owner/model", "Q4_K_M") is None

    # A directory at the path is not a model either.
    cached.mkdir()
    assert LlamaCppBackend._verified_cached_gguf(intent, "owner/model", "Q4_K_M") is None


def test_a_file_truncated_after_config_resolution_is_not_reused(monkeypatch, tmp_path):
    _hub_cache(monkeypatch, tmp_path)
    cached = _cached_gguf(tmp_path, "model-Q4_K_M.gguf", b"GGUF payload")
    intent = _intent(verified_gguf = _verified(cached))

    cached.write_bytes(b"GGUF")

    assert LlamaCppBackend._verified_cached_gguf(intent, "owner/model", "Q4_K_M") is None


def test_an_incomplete_shard_set_after_config_resolution_is_not_reused(monkeypatch, tmp_path):
    _hub_cache(monkeypatch, tmp_path)
    main = _cached_gguf(
        tmp_path,
        "model-Q4_K_M-00001-of-00002.gguf",
        b"first shard",
    )
    sibling = _cached_gguf(
        tmp_path,
        "model-Q4_K_M-00002-of-00002.gguf",
        b"second shard",
    )
    intent = _intent(verified_gguf = _verified(main))
    assert LlamaCppBackend._verified_cached_gguf(intent, "owner/model", "Q4_K_M") == str(main)

    sibling.unlink()

    assert LlamaCppBackend._verified_cached_gguf(intent, "owner/model", "Q4_K_M") is None


def test_a_shard_truncated_after_config_resolution_is_not_reused(monkeypatch, tmp_path):
    """A shard that shrinks is as unusable as one that disappears.

    The shard set alone only proves the siblings exist, so the carried value records
    every shard's byte count and Phase 2 rechecks all of them.
    """
    _hub_cache(monkeypatch, tmp_path)
    main = _cached_gguf(tmp_path, "model-Q4_K_M-00001-of-00003.gguf", b"first shard")
    second = _cached_gguf(tmp_path, "model-Q4_K_M-00002-of-00003.gguf", b"second shard")
    _cached_gguf(tmp_path, "model-Q4_K_M-00003-of-00003.gguf", b"third shard")
    intent = _intent(verified_gguf = _verified(main))
    assert LlamaCppBackend._verified_cached_gguf(intent, "owner/model", "Q4_K_M") == str(main)

    second.write_bytes(b"sec")

    assert LlamaCppBackend._verified_cached_gguf(intent, "owner/model", "Q4_K_M") is None

    # A shard that grows past its recorded size is a different set too.
    second.write_bytes(b"second shard and more")
    assert LlamaCppBackend._verified_cached_gguf(intent, "owner/model", "Q4_K_M") is None


def test_the_probe_falls_back_to_resolving_when_nothing_usable_is_carried(monkeypatch, tmp_path):
    """An unusable carried file falls back to normal resolution."""
    missing = tmp_path / "gone-Q4_K_M.gguf"
    header = _gguf_bytes(arch = "flux")
    requests: list[tuple[str, str]] = []

    monkeypatch.setattr(
        llama_cpp_module,
        "_resolve_variant_gguf_files",
        lambda *_a, **_k: ("model-Q4_K_M.gguf", []),
    )
    monkeypatch.setattr(llama_cpp_module, "cached_gguf_for_load", lambda *_a, **_k: None)
    monkeypatch.setattr(
        diffusion_compat,
        "_read_gguf_header",
        lambda repo_id, gguf_filename, hf_token: (
            requests.append((repo_id, gguf_filename)) or header
        ),
    )

    _hub_cache(monkeypatch, tmp_path)
    intent = _intent(verified_gguf = ("owner/model", "Q4_K_M", str(missing), 123))
    verdict = LlamaCppBackend.non_chat_gguf_refusal_for_intent(intent)
    assert verdict is not None and "Images page" in verdict
    assert requests == [("owner/model", "model-Q4_K_M.gguf")]


def test_the_launch_reuses_the_carried_file_instead_of_resolving_it_again():
    """The launch checks the carried file before downloading."""
    src = inspect.getsource(llama_cpp_module.LlamaCppBackend.load_model)
    teardown = src.index("# ── Phase 1: kill old process")
    reuse = src.index("self._verified_cached_gguf(intent, hf_repo, hf_variant)")
    # Search after teardown to skip the placement preflight download.
    download = src.index("model_path = self._download_gguf(", teardown)
    assert teardown < reuse < download, "the carried file is not consulted before the download"


def test_a_carried_path_outside_the_active_cache_is_not_reused(monkeypatch, tmp_path):
    """A carried path must remain inside the active Hub cache."""
    old_root, new_root = tmp_path / "old", tmp_path / "new"
    _hub_cache(monkeypatch, old_root)
    cached = _cached_gguf(old_root, "model-Q4_K_M.gguf", b"GGUF")
    intent = _intent(verified_gguf = _verified(cached))

    assert LlamaCppBackend._verified_cached_gguf(intent, "owner/model", "Q4_K_M") == str(cached)

    # Moving the cache leaves the old file readable but no longer active.
    new_root.mkdir()
    _hub_cache(monkeypatch, new_root)
    assert cached.is_file()
    assert LlamaCppBackend._verified_cached_gguf(intent, "owner/model", "Q4_K_M") is None

    # Restoring the cache root makes the file valid again.
    _hub_cache(monkeypatch, old_root)
    assert LlamaCppBackend._verified_cached_gguf(intent, "owner/model", "Q4_K_M") == str(cached)

    # Reject paths outside the cache and unreadable cache settings.
    loose = tmp_path / "loose-Q4_K_M.gguf"
    loose.write_bytes(b"GGUF")
    loose_intent = _intent(verified_gguf = _verified(loose))
    assert LlamaCppBackend._verified_cached_gguf(loose_intent, "owner/model", "Q4_K_M") is None

    import utils.hf_cache_settings as hf_cache_settings

    def _unreadable():
        raise RuntimeError("cache settings unavailable")

    monkeypatch.setattr(hf_cache_settings, "get_hf_cache_paths", _unreadable)
    assert LlamaCppBackend._verified_cached_gguf(intent, "owner/model", "Q4_K_M") is None


def test_the_probe_resolves_normally_when_the_cache_moved_under_it(monkeypatch, tmp_path):
    """A moved cache falls back to normal resolution."""
    old_root, new_root = tmp_path / "old", tmp_path / "new"
    new_root.mkdir()
    cached = _cached_gguf(old_root, "model-Q4_K_M.gguf", _gguf_bytes(arch = "ltxv"))
    _hub_cache(monkeypatch, new_root)

    requests: list[tuple[str, str]] = []
    monkeypatch.setattr(
        llama_cpp_module,
        "_resolve_variant_gguf_files",
        lambda *_a, **_k: ("model-Q4_K_M.gguf", []),
    )
    monkeypatch.setattr(llama_cpp_module, "cached_gguf_for_load", lambda *_a, **_k: None)
    monkeypatch.setattr(
        diffusion_compat,
        "_read_gguf_header",
        lambda repo_id, gguf_filename, hf_token: (
            requests.append((repo_id, gguf_filename)) or _gguf_bytes(arch = "ltxv")
        ),
    )

    intent = _intent(verified_gguf = _verified(cached))
    verdict = LlamaCppBackend.non_chat_gguf_refusal_for_intent(intent)
    assert verdict is not None and "Video page" in verdict
    assert requests == [("owner/model", "model-Q4_K_M.gguf")]


def test_the_launch_opens_the_carried_file_instead_of_resolving_it(monkeypatch, tmp_path):
    """Phase 2 opens the carried file; with nothing carried it resolves and verifies again."""
    _hub_cache(monkeypatch, tmp_path)
    cached = _cached_gguf(tmp_path, "m-Q4_K_M.gguf", _gguf_bytes(arch = "llama"))

    def _load(verified) -> list[str]:
        order: list[str] = []
        backend = LlamaCppBackend()
        # A route ask no load came for is still handed over for this key.
        monkeypatch.setattr(LlamaCppBackend, "_route_verdict_handoff", None)
        _repo_load(monkeypatch, backend, order)
        monkeypatch.setattr(
            backend, "_download_gguf", lambda **_k: order.append("download") or str(cached)
        )
        monkeypatch.setattr(
            diffusion_compat, "_read_gguf_header", lambda *_a, **_k: _gguf_bytes(arch = "llama")
        )

        def _stop(**_kwargs):
            raise RuntimeError("stop here")

        # The first companion fetch past the download decision, so the load stops there.
        monkeypatch.setattr(backend, "_download_mtp", _stop)
        with pytest.raises(RuntimeError, match = "stop here"):
            backend.load_model(_intent(verified_gguf = verified))
        return order

    assert _load(_verified(cached)) == ["kill"]
    assert _load(None) == ["kill", "download"]


def _launchable_backend(monkeypatch) -> LlamaCppBackend:
    """A backend whose launch succeeds without a real llama-server."""

    class _Process:
        pid = 123
        stdout = ()
        returncode = None

        def poll(self):
            return None

        def terminate(self):
            return None

        def wait(self, timeout = None):
            return None

        def kill(self):
            return None

    backend = LlamaCppBackend()
    backend._find_llama_server_binary = lambda include_denied = False: "/fake/llama-server"
    backend._get_gpu_memory = lambda _binary = None, **_kw: []
    backend._get_gpu_free_memory = lambda _binary = None, **_kw: []
    backend._read_gguf_metadata = lambda _path: None
    backend._can_estimate_kv = lambda: False
    backend._get_gguf_size_bytes = lambda _path: 1024
    backend._resolve_launch_mmproj_path = lambda **_kwargs: None
    backend._record_server_pid = lambda _pid: None
    backend._clear_server_pid = lambda: None
    backend._llama_server_env_for_binary = lambda _binary: {}
    backend._wait_for_health = lambda timeout, **_kw: True
    backend.probe_server_capabilities = lambda _binary: {"found": True}
    monkeypatch.setattr(
        LlamaCppBackend, "_is_vulkan_backend", staticmethod(lambda _binary = None: False)
    )
    monkeypatch.setattr(llama_cpp_module.subprocess, "Popen", lambda cmd, **_kw: _Process())
    return backend


def test_the_snapshot_kept_for_a_respawn_carries_no_verified_file(monkeypatch, tmp_path):
    """_respawn_if_dead replays this snapshot arbitrarily later, so it holds no path this
    request happened to verify: recovery resolves and verifies the file again."""
    gguf = tmp_path / "m-Q4_K_M.gguf"
    gguf.write_bytes(_gguf_bytes(arch = "llama"))
    intent = GgufLoadIntent(
        model_identifier = "owner/model",
        gguf_path = str(gguf),
        hf_variant = "Q4_K_M",
        verified_gguf = _verified(gguf),
    )

    backend = _launchable_backend(monkeypatch)
    assert backend.load_model(intent) is True
    # Only the hint is dropped; everything else the replay needs survives.
    assert backend.last_load_intent == replace(intent, verified_gguf = None)
