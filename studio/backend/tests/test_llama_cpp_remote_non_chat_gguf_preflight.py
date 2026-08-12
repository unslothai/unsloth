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
    # The verified cache lookup is the only way a local copy enters the probe now. A test
    # that installs its own stub for it passes patch_cache = False and keeps that stub.
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
    # cached variant candidates rather than trusting the current listing's filename. The
    # probe follows it: a repo that renamed the file for a quant while a complete older
    # snapshot is cached would otherwise let the probe judge one GGUF and the load open
    # another. The listing name here is deliberately not the cached one.
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
    # With no verified cache hit the probe range-reads the Hub rather than reaching for an
    # unverified path: a candidate the verified lookup rejected is one _download_gguf is
    # about to skip, so judging it would judge a file the load will not open.
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
    # With the Hub unreachable and the model already cached, the probe's two Hub calls (the
    # file listing, then the cached candidate's revision sizes) would each wait out their own
    # retry backoff before any local header is read. The download below has always run under
    # the guard; the probe has to as well, or a cached load that needs no network pays two
    # timeouts for a verdict that is free off disk.
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
    # acquire_for has evicted a resident Images/Video pipeline and the reload confirmation
    # has cancelled the running chats, so the route needs a verdict of its own.
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
    # A probe that raises is not a verdict: the load proceeds and the checks inside it stay
    # as the backstop.
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
    # listing, a cache verification and a range request, all bounded at 15s. The second one
    # takes the first one's answer instead of paying again.
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
