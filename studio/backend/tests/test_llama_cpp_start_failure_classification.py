# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for LlamaCppBackend._classify_llama_start_failure.

When llama-server exits before becoming healthy, load_model turns its
captured stdout/stderr into a user-facing reason. A diffusion/image GGUF
(FLUX, Qwen-Image, ...) is a valid file with plenty of memory, so the
generic "invalid file or out of memory" message is misleading (issue
#5842). These tests pin the classification.
"""

from __future__ import annotations

import os
import sys
import types as _types
from pathlib import Path

import pytest

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

# Match sibling tests' stubbing so the module imports in a lightweight
# env without fastapi.
_loggers_stub = _types.ModuleType("loggers")
_loggers_stub.get_logger = lambda name: __import__("logging").getLogger(name)
sys.modules.setdefault("loggers", _loggers_stub)
# Give the structlog stub a real get_logger: a bare ModuleType poisons
# sys.modules for later tests that call structlog.get_logger at import time.
_structlog_stub = _types.ModuleType("structlog")
_structlog_stub.get_logger = lambda *a, **k: __import__("logging").getLogger("structlog")
sys.modules.setdefault("structlog", _structlog_stub)
if not hasattr(sys.modules["structlog"], "get_logger"):
    sys.modules["structlog"].get_logger = _structlog_stub.get_logger

from core.inference.llama_cpp import LlamaCppBackend  # noqa: E402

_classify = LlamaCppBackend._classify_llama_start_failure

# Real llama-server failure lines (lower-cased downstream anyway).
_QWEN_IMAGE_OUT = (
    "load_model: loading model 'qwen-image-edit-2511-Q4_K_M.gguf'\n"
    "llama_model_load: error loading model: unknown model architecture: 'qwen_image'\n"
    "llama_model_load_from_file_impl: failed to load model"
)
_OOM_OUT = (
    "ggml_backend_cuda_buffer_type_alloc_buffer: allocating 12000.00 MiB on "
    "device 0: cudaMalloc failed: out of memory"
)


class TestDiffusionArchitectures:
    def test_qwen_image_routes_to_images_page(self):
        msg = _classify(_QWEN_IMAGE_OUT, "/models/qwen-image.gguf", "local/qwen-image")
        assert "diffusion" in msg.lower()
        assert "Images page" in msg
        assert "qwen_image" in msg
        # Must NOT keep blaming memory / file validity.
        assert "out of memory" not in msg.lower()
        assert "enough memory" not in msg.lower()

    # Parametrize over the production set so new arches are auto-covered.
    @pytest.mark.parametrize("arch", sorted(LlamaCppBackend._IMAGE_ARCHES))
    def test_every_image_arch_is_recognised(self, arch):
        out = f"error loading model: unknown model architecture: '{arch}'"
        msg = _classify(out, f"/models/{arch}.gguf", f"local/{arch}")
        assert "diffusion" in msg.lower()
        assert "Images page" in msg
        assert arch in msg

    # A video arch must name the VIDEO page: Images cannot run it either.
    @pytest.mark.parametrize("arch", sorted(LlamaCppBackend._VIDEO_ARCHES))
    def test_every_video_arch_routes_to_the_video_page(self, arch):
        out = f"error loading model: unknown model architecture: '{arch}'"
        msg = _classify(out, f"/models/{arch}.gguf", f"local/{arch}")
        assert "text-to-video" in msg.lower()
        assert "Video page" in msg
        assert "Images page" not in msg
        assert arch in msg
        assert "enough memory" not in msg.lower()

    # An arch no page can run must promise NEITHER page: the picker tags these
    # ``image-diffusion-unsupported``, hiding them from the Images and Video lists alike.
    @pytest.mark.parametrize("arch", sorted(LlamaCppBackend._UNRUNNABLE_MEDIA_ARCHES))
    def test_unrunnable_media_arch_names_no_page(self, arch):
        out = f"error loading model: unknown model architecture: '{arch}'"
        msg = _classify(out, f"/models/{arch}.gguf", f"local/{arch}")
        assert arch in msg
        assert "neither the Images page nor the Video page" in msg
        assert "Use Unsloth's image generation page" not in msg
        assert "Open it from" not in msg
        assert "cannot run" in msg.lower()
        assert "enough memory" not in msg.lower()

    @pytest.mark.parametrize("arch", sorted(LlamaCppBackend._SPEECH_ARCHES))
    def test_every_speech_arch_routes_to_the_audio_page(self, arch):
        out = f"error loading model: unknown model architecture: '{arch}'"
        msg = _classify(out, f"/models/{arch}.gguf", f"local/{arch}")
        assert "text-to-speech" in msg.lower()
        assert "Audio page" in msg
        assert "Images page" not in msg
        assert arch in msg

    def test_media_arch_sets_are_disjoint_and_cover_the_union(self):
        sets = (
            LlamaCppBackend._IMAGE_ARCHES,
            LlamaCppBackend._AMBIGUOUS_IMAGE_ARCHES,
            LlamaCppBackend._VIDEO_ARCHES,
            LlamaCppBackend._UNRUNNABLE_MEDIA_ARCHES,
        )
        assert sum(len(s) for s in sets) == len(set().union(*sets))
        assert set().union(*sets) == LlamaCppBackend._DIFFUSION_ARCHES

    # The video set must stay inside what the Video picker offers: an arch routes.models
    # tags unsupported can never be picked on the page we name. Video only -- the image half
    # (sd1/sd3/sdxl/aura/hidream) predates the video split and is left to its own change.
    def test_no_runnable_video_arch_is_tagged_unsupported_by_the_picker(self):
        from routes.models import _UNSUPPORTED_DIFFUSION_GGUF_ARCHS
        assert not (LlamaCppBackend._VIDEO_ARCHES & _UNSUPPORTED_DIFFUSION_GGUF_ARCHS)


class TestUnsupportedNonDiffusionArchitecture:
    def test_unknown_llm_arch_says_unsupported_not_oom(self):
        out = "error loading model: unknown model architecture: 'some_new_llm'"
        msg = _classify(out, "/models/x.gguf", "local/x")
        assert "some_new_llm" in msg
        assert "architecture" in msg.lower()
        # Specific, not the misleading memory message.
        assert "enough memory" not in msg.lower()
        assert "diffusion" not in msg.lower()

    # Exact match: a chat arch merely containing a diffusion token (wan,
    # sd1, flux, ...) must not be routed to the Images page.
    @pytest.mark.parametrize(
        "arch",
        [
            "taiwan",  # contains "wan"
            "swan_llm",  # contains "wan"
            "fluxion",  # contains "flux"
            "sd1234",  # contains "sd1"
            "sd3_chat",  # contains "sd3"
            "aura2_text",  # contains "aura"
            "cosmos_reason",  # contains "cosmos"
            "qwen_image_text",  # contains "qwen_image"
        ],
    )
    def test_arch_containing_diffusion_token_is_not_misrouted(self, arch):
        out = f"error loading model: unknown model architecture: '{arch}'"
        msg = _classify(out, f"/models/{arch}.gguf", f"local/{arch}")
        assert arch in msg
        assert "does not support" in msg.lower()
        assert "diffusion" not in msg.lower()
        assert "Images page" not in msg


class TestOllamaAndFallback:
    _OLLAMA_GGUF = f"/home/u/.ollama{__import__('os').sep}ollama_links{__import__('os').sep}m.gguf"

    def test_ollama_compat_message_still_works(self):
        out = "llama_model_load: error loading model: key not found"
        msg = _classify(out, self._OLLAMA_GGUF, "ollama/llama3")
        assert "Ollama" in msg

    def test_ollama_unknown_arch_keeps_ollama_guidance(self):
        # Ollama + non-diffusion unknown arch keeps the Ollama hint, not the
        # generic llama.cpp "unsupported" message.
        out = "error loading model: unknown model architecture: 'some_new_llm'"
        msg = _classify(out, self._OLLAMA_GGUF, "ollama/some-new")
        assert "Ollama" in msg
        assert "directly through Ollama" in msg
        assert "does not support" not in msg.lower()

    def test_ollama_diffusion_arch_still_routes_to_images(self):
        # Diffusion routing wins over the Ollama hint.
        out = "error loading model: unknown model architecture: 'flux'"
        msg = _classify(out, self._OLLAMA_GGUF, "ollama/flux")
        assert "diffusion" in msg.lower()
        assert "Images page" in msg

    def test_generic_oom_keeps_memory_message(self):
        msg = _classify(_OOM_OUT, "/models/big.gguf", "local/big")
        assert "enough memory" in msg.lower()
        assert "diffusion" not in msg.lower()

    def test_empty_output_is_safe(self):
        msg = _classify("", None, None)
        assert "llama-server failed to start" in msg

    def test_health_timeout_names_probe_not_generic(self):
        # A live server that never returns 200 on /health must name the probe and
        # proxy/context causes, not blame a bad GGUF (#5740).
        msg = _classify(
            "llama-server health check timed out after 600.0s", "/models/x.gguf", "local/x"
        )
        assert "/health" in msg
        assert "NO_PROXY" in msg
        assert "GGUF file is valid" not in msg


class TestOsKillReturncode:
    """SIGKILL (-9) with no diagnostic output is the OOM killer and gets a named,
    actionable message; SIGTERM (-15) is also unload/cancel/supervisor stop, so it
    stays neutral; a recognized output still wins; a hard fault (-11) keeps the
    generic fallback."""

    def test_sigkill_with_no_output_names_oom(self, monkeypatch):
        # Pin the platform: macOS SIGKILLs an invalid code signature the same
        # way, so the message there names both readings (see
        # TestMacOSLoaderFailures) and this wording is the non-Darwin one.
        monkeypatch.setattr(sys, "platform", "linux")
        msg = _classify("", "/models/big-bf16.gguf", "local/big", -9)
        assert "signal 9" in msg
        assert "out of memory" in msg.lower()
        assert ".wslconfig" in msg
        assert "code signature" not in msg.lower()
        assert "GGUF file is valid" not in msg

    def test_sigterm_is_neutral_not_oom(self):
        msg = _classify("", "/models/big-bf16.gguf", "local/big", -15)
        assert "signal 15" in msg
        assert "terminated" in msg.lower()
        assert "out of memory" not in msg.lower()

    def test_specific_output_wins_over_os_kill_code(self):
        msg = _classify(_QWEN_IMAGE_OUT, "/models/qwen-image.gguf", "local/qwen-image", -9)
        assert "diffusion" in msg.lower()
        assert "out of memory" not in msg.lower()

    def test_signal_crash_code_keeps_generic_message(self):
        # -11 is handled by the retry ladder; if it reaches here with no output
        # it gets the generic fallback, not the OOM message.
        msg = _classify("", "/models/x.gguf", "local/x", -11)
        assert "GGUF file is valid" in msg
        assert "out of memory" not in msg.lower()


class TestMissingSharedLibrary:
    """The dynamic loader stops llama-server before it prints anything of its
    own, so a stock container missing libgomp.so.1 used to be reported as an
    invalid GGUF or too little memory."""

    _LOADER_OUT = (
        "/home/tester/.unsloth/llama.cpp/llama-server: error while loading "
        "shared libraries: libgomp.so.1: cannot open shared object file: "
        "No such file or directory"
    )

    def test_missing_libgomp_is_named_with_its_packages(self):
        msg = _classify(self._LOADER_OUT, "/models/x.gguf", "local/x", 127)
        assert "libgomp.so.1" in msg
        assert "libgomp1" in msg
        assert "Fedora/RHEL" in msg
        assert "GGUF file is valid" not in msg
        assert "enough memory" not in msg.lower()

    def test_unknown_library_is_still_named(self):
        out = "llama-server: error while loading shared libraries: libfoo.so.7: cannot open"
        msg = _classify(out, "/models/x.gguf", "local/x", 127)
        assert "libfoo.so.7" in msg
        assert "package manager" in msg
        assert "libgomp1" not in msg

    def test_exit_127_with_no_output_names_both_causes(self):
        # 127 is also a shell-wrapper entrypoint whose exec target is gone, so
        # it must not claim a distro package is missing. The generic
        # file/memory message is still wrong.
        msg = _classify("", "/models/x.gguf", "local/x", 127)
        assert "could not be found or run" in msg
        assert "shared libraries" in msg
        assert "package manager" not in msg
        assert "GGUF file is valid" not in msg
        assert "enough memory" not in msg.lower()

    def test_exit_127_on_a_pinned_binary_does_not_send_it_to_the_updater(self, monkeypatch):
        # A wrapper whose exec target is gone exits 127 with no loader line, and
        # the updater refuses to touch a LLAMA_SERVER_PATH pin, so the managed
        # remedy is a dead end there too.
        monkeypatch.setenv("LLAMA_SERVER_PATH", "/opt/custom/llama-server")
        msg = _classify("", "/models/x.gguf", "local/x", 127, "/opt/custom/llama-server")
        assert "unsloth studio update" not in msg
        assert "custom llama.cpp" in msg

    def test_exit_127_on_a_managed_binary_still_points_at_the_updater(self):
        msg = _classify("", "/models/x.gguf", "local/x", 127)
        assert "unsloth studio update" in msg

    def test_wrapper_exec_failure_is_not_called_a_system_library(self):
        # write_exec_wrapper's entrypoint: /bin/sh reports a missing exec
        # target as "not found" and exits 127.
        out = (
            "/home/t/.unsloth/llama.cpp/llama-server: 2: exec: ./build/bin/llama-server: not found"
        )
        msg = _classify(out, "/models/x.gguf", "local/x", 127)
        assert "package manager" not in msg
        assert "could not be found or run" in msg

    def test_symbol_lookup_error_is_not_called_a_system_library(self):
        # A mismatched bundled runtime exits 127 with this, not a loader line.
        out = "llama-server: symbol lookup error: llama-server: undefined symbol: ggml_backend_init"
        msg = _classify(out, "/models/x.gguf", "local/x", 127)
        assert "package manager" not in msg
        # Must not steal the ROCm branch: the object here is llama-server.
        assert "HIP/ROCR" not in msg
        assert "hsa_amd_queue_create" not in msg

    def test_bundled_runtime_library_points_at_the_installer(self):
        # libggml/libllama/libmtmd ship in build/bin (runtime_payload_health_groups)
        # and no package manager can supply them.
        out = (
            "/home/t/.unsloth/llama.cpp/build/bin/llama-server: error while loading "
            "shared libraries: libggml.so.0: cannot open shared object file: "
            "No such file or directory"
        )
        msg = _classify(out, "/models/x.gguf", "local/x", 127)
        assert "libggml.so.0" in msg
        assert "unsloth studio update" in msg
        assert "package manager" not in msg

    def test_corrupt_library_is_not_reported_as_missing(self):
        # glibc reuses the same prefix for a present-but-unusable library.
        out = (
            "/opt/llama/llama-server: error while loading shared libraries: "
            "/usr/lib/x86_64-linux-gnu/libgomp.so.1: file too short"
        )
        msg = _classify(out, "/models/x.gguf", "local/x", 127)
        assert "file too short" in msg
        assert "is missing" not in msg
        assert "Install it with your package manager" not in msg

    def test_corrupt_bundled_library_points_at_the_installer(self):
        out = (
            "llama-server: error while loading shared libraries: "
            "/home/t/.unsloth/llama.cpp/build/bin/libggml-cuda.so: invalid ELF header"
        )
        msg = _classify(out, "/models/x.gguf", "local/x", 127)
        assert "invalid ELF header" in msg
        assert "unsloth studio update" in msg
        assert "is missing" not in msg

    # Verified on glibc 2.39: an absolute DT_NEEDED dependency that exists but
    # cannot be opened exits 127 with the EACCES strerror appended.
    def test_permission_denied_library_is_not_reported_as_missing(self):
        out = (
            "/opt/llama/llama-server: error while loading shared libraries: "
            "/opt/llama/lib/libgomp.so.1: cannot open shared object file: "
            "Permission denied"
        )
        msg = _classify(out, "/models/x.gguf", "local/x", 127)
        assert "Permission denied" in msg
        assert "is missing" not in msg
        assert "package manager" not in msg

    def test_permission_denied_bundled_library_is_not_reported_as_missing(self):
        out = (
            "/home/t/.unsloth/llama.cpp/build/bin/llama-server: error while loading "
            "shared libraries: /home/t/.unsloth/llama.cpp/build/bin/libggml.so: "
            "cannot open shared object file: Permission denied"
        )
        msg = _classify(out, "/models/x.gguf", "local/x", 127)
        assert "Permission denied" in msg
        assert "is missing" not in msg

    # glibc echoes the object name verbatim, so a path with spaces must not be
    # truncated at the first space (verified on glibc 2.39).
    def test_library_path_with_spaces_is_named_in_full(self):
        out = (
            "/opt/llama/llama-server: error while loading shared libraries: "
            "/opt/My Runtime/libfoo.so: file too short"
        )
        msg = _classify(out, "/models/x.gguf", "local/x", 127)
        assert "/opt/My Runtime/libfoo.so" in msg
        assert "file too short" in msg

    def test_missing_library_path_with_spaces_is_named_in_full(self):
        out = (
            "/opt/llama/llama-server: error while loading shared libraries: "
            "/opt/My Runtime/libbar.so: cannot open shared object file: "
            "No such file or directory"
        )
        msg = _classify(out, "/models/x.gguf", "local/x", 127)
        assert "/opt/My Runtime/libbar.so" in msg
        assert "is missing" in msg

    def test_an_absolute_path_is_not_offered_to_a_package_manager(self):
        # An absolute DT_NEEDED names one exact file. No package puts a file at
        # /opt/vendor, so apt/dnf is the wrong instruction whoever owns the
        # binary.
        out = (
            "llama-server: error while loading shared libraries: "
            "/opt/vendor/libaccelerator.so: cannot open shared object file: "
            "No such file or directory"
        )
        msg = _classify(out, "/models/x.gguf", "local/x", 127)
        assert "/opt/vendor/libaccelerator.so" in msg
        assert "package manager" not in msg
        assert "that exact location" in msg

    def test_an_absolute_path_on_a_pinned_binary_names_the_custom_runtime(self, monkeypatch):
        monkeypatch.setenv("LLAMA_SERVER_PATH", "/opt/custom/llama-server")
        out = (
            "llama-server: error while loading shared libraries: "
            "/opt/vendor/libaccelerator.so: cannot open shared object file: "
            "No such file or directory"
        )
        msg = _classify(out, "/models/x.gguf", "local/x", 127, "/opt/custom/llama-server")
        assert "custom llama.cpp" in msg
        assert "unsloth studio update" not in msg
        assert "package manager" not in msg

    def test_a_bare_soname_keeps_package_advice_even_on_a_pinned_binary(self, monkeypatch):
        # The counter-case that stops the rule from being "unmanaged means never
        # mention a package": a custom-built llama.cpp on a bare-bones host is
        # still missing a distro library, and libgomp1 is exactly what fixes it.
        monkeypatch.setenv("LLAMA_SERVER_PATH", "/opt/custom/llama-server")
        out = (
            "llama-server: error while loading shared libraries: "
            "libgomp.so.1: cannot open shared object file: No such file or directory"
        )
        msg = _classify(out, "/models/x.gguf", "local/x", 127, "/opt/custom/llama-server")
        assert "libgomp1" in msg
        assert "package manager" in msg

    def test_a_relative_dt_needed_is_an_exact_path_too(self):
        # glibc's rule is `strchr (name, '/') == NULL`: a slash anywhere means
        # no search happened, so subdir/libfoo.so names one exact file just as
        # an absolute path does. Reproduced on glibc 2.39 with a SONAME-less .so
        # linked by relative path; it takes both to get here, so this is about
        # matching the loader's rule rather than a case users hit.
        out = (
            "llama-server: error while loading shared libraries: "
            "subdir/libvendor.so: cannot open shared object file: "
            "No such file or directory"
        )
        msg = _classify(out, "/models/x.gguf", "local/x", 127)
        assert "subdir/libvendor.so" in msg
        assert "package manager" not in msg
        assert "that exact location" in msg

    def test_a_windows_absolute_path_is_recognised_too(self):
        out = (
            "llama-server: error while loading shared libraries: "
            "C:\\vendor\\accel.dll: cannot open shared object file: "
            "No such file or directory"
        )
        msg = _classify(out, "/models/x.gguf", "local/x", 127)
        assert "package manager" not in msg
        assert "that exact location" in msg

    def test_bundled_library_under_a_spaced_path_still_points_at_the_installer(self):
        out = (
            "llama-server: error while loading shared libraries: "
            "/home/My User/.unsloth/llama.cpp/build/bin/libggml.so: invalid ELF header"
        )
        msg = _classify(out, "/models/x.gguf", "local/x", 127)
        assert "/home/My User/.unsloth/llama.cpp/build/bin/libggml.so" in msg
        assert "unsloth studio update" in msg

    def test_pinned_custom_binary_is_not_called_unsloths_runtime(self, monkeypatch):
        # LLAMA_SERVER_PATH pins an install update_flow.managed_install_root
        # refuses to manage, so `unsloth studio update` cannot repair it.
        monkeypatch.setenv("LLAMA_SERVER_PATH", "/opt/mybuild/bin/llama-server")
        out = (
            "/opt/mybuild/bin/llama-server: error while loading shared libraries: "
            "libggml.so: cannot open shared object file: No such file or directory"
        )
        msg = _classify(out, "/models/x.gguf", "local/x", 127, "/opt/mybuild/bin/llama-server")
        assert "libggml.so" in msg
        assert "unsloth studio update" not in msg
        assert "package manager" not in msg
        assert "custom install" in msg

    def test_managed_binary_still_points_at_the_installer(self, monkeypatch, tmp_path):
        monkeypatch.delenv("LLAMA_SERVER_PATH", raising = False)
        binary = tmp_path / "llama.cpp" / "build" / "bin" / "llama-server"
        binary.parent.mkdir(parents = True)
        binary.write_text("")
        monkeypatch.setenv("UNSLOTH_LLAMA_CPP_PATH", str(tmp_path / "llama.cpp"))
        out = (
            f"{binary}: error while loading shared libraries: libggml.so: "
            "cannot open shared object file: No such file or directory"
        )
        msg = _classify(out, "/models/x.gguf", "local/x", 127, str(binary))
        assert "unsloth studio update" in msg

    def test_nameless_loader_error_does_not_invent_a_library(self):
        # glibc's own allocation failures pass an empty object name, so the
        # text right after the colon is prose, not a soname.
        out = "llama-server: error while loading shared libraries: cannot create search path array"
        msg = _classify(out, "/models/x.gguf", "local/x", 127)
        assert "cannot create search path array" in msg
        assert "the library cannot" not in msg
        assert "is missing" not in msg

    def test_loader_error_wins_without_a_returncode(self):
        msg = _classify(self._LOADER_OUT, "/models/x.gguf", "local/x")
        assert "libgomp.so.1" in msg

    def test_a_normal_failure_is_untouched(self):
        msg = _classify(_OOM_OUT, "/models/big.gguf", "local/big", 1)
        assert "enough memory" in msg.lower()
        assert "system library" not in msg

    def test_a_named_arch_wins_over_exit_127(self):
        # The bare code is only a fallback, so it must not mask a diagnosis the
        # output already gives.
        msg = _classify(_QWEN_IMAGE_OUT, "/models/qwen-image.gguf", "local/qwen-image", 127)
        assert "Images page" in msg
        assert "system library" not in msg


class TestBundledHipRocrMismatch:
    """Unsloth prepends system ROCm, the prebuilt still binds its bundled HIP,
    and glibc exits 127 on the symbol lookup (#8998). That used to read as a
    missing llama-server and get retried as a VRAM miss. Neither is true.
    """

    _FIELD_OUT = (
        "0.00.018.048 I srv    load_model: loading model '/models/x.gguf'\n"
        "/home/t/.unsloth/llama.cpp/llama-server: symbol lookup error: "
        "/home/t/.unsloth/llama.cpp/build/bin/libamdhip64.so.7: "
        "undefined symbol: hsa_amd_queue_create, version ROCR_1"
    )

    def test_field_log_is_the_hip_rocr_mix(self):
        assert LlamaCppBackend._is_bundled_hip_rocr_mismatch(self._FIELD_OUT)

    def test_ggml_symbol_lookup_is_not_the_hip_rocr_mix(self):
        out = (
            "llama-server: symbol lookup error: llama-server: "
            "undefined symbol: ggml_backend_init"
        )
        assert not LlamaCppBackend._is_bundled_hip_rocr_mismatch(out)

    def test_missing_libamdhip64_is_not_the_hip_rocr_mix(self):
        # Absent file is the glibc loader line, not a symbol lookup.
        out = (
            "llama-server: error while loading shared libraries: "
            "libamdhip64.so.7: cannot open shared object file: "
            "No such file or directory"
        )
        assert not LlamaCppBackend._is_bundled_hip_rocr_mismatch(out)

    def test_empty_and_none_are_not_the_mix(self):
        assert not LlamaCppBackend._is_bundled_hip_rocr_mismatch("")
        assert not LlamaCppBackend._is_bundled_hip_rocr_mismatch(None)

    def test_another_lib_from_the_prepended_dir_is_the_same_mix(self):
        # The prepend covers the whole system ROCm dir, so rocBLAS against a
        # different-version HIP fails the same way and wants the same retry.
        out = (
            "llama-server: symbol lookup error: "
            "/home/t/.unsloth/llama.cpp/build/bin/librocblas.so.4: "
            "undefined symbol: hipGraphicsResourceGetMappedPointer"
        )
        assert LlamaCppBackend._is_bundled_hip_rocr_mismatch(out)
        msg = _classify(out, "/models/x.gguf", "local/x", 127)
        assert "librocblas.so.4" in msg
        assert "hipGraphicsResourceGetMappedPointer" in msg
        # The field log's symbol is not this crash; do not name it.
        assert "hsa_amd_queue_create" not in msg

    def test_an_oversized_loader_token_is_bounded_in_the_message(self):
        # Straight from the child, and _drain_stdout keeps an unterminated line
        # whole, so the message has to bound both captures.
        out = (
            "llama-server: symbol lookup error: "
            f"/b/libamdhip64.so.{'9' * 3000}: undefined symbol: {'s' * 8192}"
        )
        msg = _classify(out, "/models/x.gguf", "local/x", 127)
        assert "HIP/ROCR" in msg
        assert len(msg) < 1000

    def test_an_object_longer_than_a_path_is_not_the_mix(self):
        # The object capture stops at PATH_MAX: no such path can exist, and an
        # unbounded one lets a single hostile line drive the scan quadratically.
        out = (
            "llama-server: symbol lookup error: "
            f"/b/libamdhip64.so.{'9' * 8192}: undefined symbol: hsa_amd_queue_create"
        )
        assert not LlamaCppBackend._is_bundled_hip_rocr_mismatch(out)

    def test_a_bundle_under_a_path_with_spaces_is_still_the_mix(self):
        # glibc echoes the object verbatim, and a custom LLAMA_SERVER_PATH can
        # sit under a directory with spaces. Splitting on whitespace dropped
        # those into the generic 127 text instead of the retry.
        out = (
            "llama-server: symbol lookup error: "
            "/opt/My Runtime/build/bin/libamdhip64.so.7: "
            "undefined symbol: hsa_amd_queue_create, version ROCR_1"
        )
        assert LlamaCppBackend._is_bundled_hip_rocr_mismatch(out)
        assert LlamaCppBackend._bundled_hip_symbol_miss(out)[0].endswith(
            "/opt/My Runtime/build/bin/libamdhip64.so.7"
        )
        assert "libamdhip64.so.7" in _classify(out, "/models/x.gguf", "local/x", 127)

    def test_a_path_component_does_not_stand_in_for_the_object(self):
        out = (
            "llama-server: symbol lookup error: "
            "/opt/librocm-vendor/lib/libfoo.so.1: undefined symbol: foo_init"
        )
        assert not LlamaCppBackend._is_bundled_hip_rocr_mismatch(out)

    def test_classify_names_the_mix_not_a_missing_binary(self):
        msg = _classify(self._FIELD_OUT, "/models/x.gguf", "local/x", 127)
        assert "HIP/ROCR" in msg
        assert "hsa_amd_queue_create" in msg
        assert "Vulkan" in msg
        assert "could not be found or run" not in msg
        assert "not out of VRAM" in msg
        assert "GGUF file is valid" not in msg
        assert "enough memory" not in msg.lower()

    def test_classify_on_a_pinned_binary_does_not_send_it_to_the_updater(self, monkeypatch):
        monkeypatch.setenv("LLAMA_SERVER_PATH", "/opt/custom/llama-server")
        msg = _classify(
            self._FIELD_OUT, "/models/x.gguf", "local/x", 127, "/opt/custom/llama-server"
        )
        assert "unsloth studio update" not in msg
        assert "custom llama.cpp" in msg

    def test_hip_rocr_retry_is_checked_before_the_fit_on_retry(self):
        # --fit cannot load a missing symbol, so the library check has to come
        # first. The launch sequence itself is asserted behaviourally in
        # test_gpu_init_crash_message.py::TestHipRocrRetryKeepsFitBudget.
        src = Path(__file__).resolve().parent.parent / "core" / "inference" / "llama_cpp.py"
        text = src.read_text(encoding = "utf-8")
        spawn_start = text.index("def _spawn_and_wait(")
        spawn_end = text.index("def _raise_terminal_load_failure", spawn_start)
        body = text[spawn_start:spawn_end]
        assert body.index("_is_bundled_hip_rocr_mismatch") < body.index(
            "retrying once with --fit on so it can offload"
        )
        assert "use_system_rocm = False" in body


# Real dyld output. macOS says none of the things glibc says, so before #8566
# every one of these fell through to "invalid GGUF or not enough memory" on a
# Mac that had neither problem. Classification is pure text matching, so these
# run on any host.
_DYLD_MISSING_OUT = (
    "dyld[54231]: Library not loaded: @rpath/libllama.dylib\n"
    "  Referenced from: <A1B2C3D4> /Users/me/.unsloth/studio/llama.cpp/build/bin/llama-server\n"
    "  Reason: tried: '/Users/me/.unsloth/studio/llama.cpp/build/bin/libllama.dylib' "
    "(no such file), '/usr/local/lib/libllama.dylib' (no such file)"
)
_DYLD_OLD_MACOS_OUT = (
    "dyld: Library not loaded: @rpath/libggml-metal.dylib\n"
    "  Referenced from: /Users/me/.unsloth/studio/llama.cpp/build/bin/llama-server\n"
    "  Reason: image not found"
)
_DYLD_SIGNATURE_OUT = (
    "dyld[54231]: Library not loaded: @rpath/libggml-base.dylib\n"
    "  Reason: tried: '/Users/me/.unsloth/studio/llama.cpp/build/bin/libggml-base.dylib' "
    "(code signature in <A1B2> '.../libggml-base.dylib' not valid for use in process: "
    "mapped file has no cdhash, completely unsigned? Code has to be at least ad-hoc signed.)"
)
_DYLD_ARCH_OUT = (
    "dyld[54231]: Library not loaded: @rpath/libmtmd.dylib\n"
    "  Reason: tried: '/Users/me/.unsloth/studio/llama.cpp/build/bin/libmtmd.dylib' "
    "(mach-o file, but is an incompatible architecture (have 'x86_64', need 'arm64'))"
)
_DYLD_TOO_NEW_OUT = (
    "dyld[54231]: Library not loaded: @rpath/libggml-metal.dylib\n"
    "  Reason: tried: '.../libggml-metal.dylib' (built for macOS 26.0 which is "
    "newer than running OS)"
)
_DYLD_SYMBOL_OUT = (
    "dyld[54231]: Symbol not found: __ZN4ggml7backend6deviceEv\n"
    "  Referenced from: <A1B2> /Users/me/.unsloth/studio/llama.cpp/build/bin/libmtmd.dylib\n"
    "  Expected in: <C3D4> /Users/me/.unsloth/studio/llama.cpp/build/bin/libllama.dylib"
)


def _blames_the_gguf_or_memory(msg: str) -> bool:
    lowered = msg.lower()
    return "gguf file is valid" in lowered or "enough memory" in lowered


class TestMacOSLoaderFailures:
    def test_missing_dylib_names_the_library_and_the_installer(self):
        msg = _classify(_DYLD_MISSING_OUT, "/models/x.gguf", "local/x", 1)
        assert "libllama.dylib" in msg
        assert "unsloth studio update" in msg
        assert not _blames_the_gguf_or_memory(msg)

    def test_old_macos_image_not_found_is_recognised(self):
        msg = _classify(_DYLD_OLD_MACOS_OUT, "/models/x.gguf", "local/x", 1)
        assert "libggml-metal.dylib" in msg
        assert not _blames_the_gguf_or_memory(msg)

    def test_invalid_code_signature_is_not_reported_as_a_bad_file(self):
        msg = _classify(_DYLD_SIGNATURE_OUT, "/models/x.gguf", "local/x", 1)
        assert "code signature" in msg.lower()
        assert not _blames_the_gguf_or_memory(msg)

    def test_wrong_architecture_is_named(self):
        msg = _classify(_DYLD_ARCH_OUT, "/models/x.gguf", "local/x", 1)
        assert "architecture" in msg.lower()
        assert not _blames_the_gguf_or_memory(msg)

    def test_runtime_built_for_a_newer_macos(self):
        msg = _classify(_DYLD_TOO_NEW_OUT, "/models/x.gguf", "local/x", 1)
        assert "newer version of macOS" in msg
        assert not _blames_the_gguf_or_memory(msg)

    def test_mtlresidency_symbol_is_read_as_a_too_new_macos_build(self):
        # What a Tahoe-built libggml-metal actually reports on macOS 15; the
        # installer's looks_like_macos_incompatibility keys on the same symbol.
        out = (
            "dyld[1]: Symbol not found: _MTLResidencySetDescriptor\n"
            "  Referenced from: .../libggml-metal.dylib"
        )
        msg = _classify(out, "/models/x.gguf", "local/x", 1)
        assert "newer version of macOS" in msg

    def test_symbol_mismatch_reports_a_mixed_install(self):
        msg = _classify(_DYLD_SYMBOL_OUT, "/models/x.gguf", "local/x", 1)
        assert "different build" in msg
        assert not _blames_the_gguf_or_memory(msg)

    def test_dyld_output_outranks_signal_9(self):
        # -9 alone reads as the OOM killer; a dyld diagnostic is a fact and
        # must win, or the user is sent to free memory they already have.
        msg = _classify(_DYLD_MISSING_OUT, "/models/x.gguf", "local/x", -9)
        assert "libllama.dylib" in msg
        assert "out of memory" not in msg.lower()

    def test_custom_binary_is_not_sent_to_the_unsloth_updater(self, monkeypatch):
        monkeypatch.setenv("LLAMA_SERVER_PATH", "/opt/mybuild/bin/llama-server")
        msg = _classify(
            _DYLD_ARCH_OUT, "/models/x.gguf", "local/x", 1, "/opt/mybuild/bin/llama-server"
        )
        assert "unsloth studio update" not in msg
        assert "custom llama.cpp" in msg

    def test_linux_loader_wording_is_untouched(self):
        # The glibc branch must keep winning; the macOS branch runs after it.
        out = (
            "llama-server: error while loading shared libraries: libgomp.so.1: "
            "cannot open shared object file: No such file or directory"
        )
        msg = _classify(out, "/models/x.gguf", "local/x", 127)
        assert "libgomp.so.1" in msg
        assert "macOS" not in msg

    def test_silent_signal_9_on_macos_also_names_the_code_signature(self, monkeypatch):
        # macOS kills an unsigned or altered Mach-O with SIGKILL before it can
        # print anything, which is indistinguishable from the OOM killer by
        # returncode alone. Offer both readings there instead of only memory.
        monkeypatch.setattr(sys, "platform", "darwin")
        msg = _classify("", "/models/x.gguf", "local/x", -9)
        assert "code signature" in msg.lower()
        assert "out of memory" in msg.lower()

    def test_ordinary_output_mentioning_reason_is_not_a_loader_failure(self):
        out = "llama_model_load: error loading model: Reason: something went wrong"
        msg = _classify(out, "/models/x.gguf", "local/x", 1)
        assert "llama-server failed to start." in msg


class TestANonGgufFile:
    # What llama.cpp prints when the bytes are not a GGUF. It formats the four it found with %c,
    # so an AppleDouble sidecar's 0x00051607 arrives as unprintable characters (#8566).
    _OUT = (
        "build: 9415 (06d26dfd) with Apple clang version 17.0.0 for arm64-apple-darwin24.6.0\n"
        "gguf_init_from_reader: invalid magic characters: '\ufffd\ufffd\ufffd\ufffd', "
        "expected 'GGUF'\n"
        "llama_server: exiting due to model loading error"
    )

    def test_it_is_reported_as_not_a_gguf(self, tmp_path):
        log = tmp_path / "llama-1-port-8080.log"
        msg = _classify(self._OUT, "/models/._muse-UD-Q2_K_XL.gguf", "local/muse", 1, None, log)

        assert "not a GGUF" in msg
        # The two things the generic fallback used to blame, neither of which is the cause.
        assert "enough memory" not in msg.lower()
        assert not msg.startswith("llama-server failed to start.")
        # The echoed bytes are unreadable, so the path is the only usable identifier.
        assert "._muse-UD-Q2_K_XL.gguf" in msg
        assert "companion" in msg
        # The remedy for the volume, not the generic re-download (#8566).
        assert "dot_clean -m" in msg
        assert "llama-server output:" in msg
        assert f"Full log: {log}" in msg

    def test_it_does_not_call_an_ordinary_main_model_the_bad_file(self):
        # The message must not settle which file was invalid, nor promise the output does.
        assert "is not a GGUF" in _classify(self._OUT, None, "local/muse", 1)
        msg = _classify(self._OUT, "/models/model-Q4_K_M.gguf", "local/m", 1)
        assert "model-Q4_K_M.gguf" in msg and "companion" in msg
        assert "names it" not in msg
        # An ordinary path is no evidence about the volume, so it keeps the generic remedy.
        assert "dot_clean" not in msg and "Re-download the model" in msg

    def test_a_dyld_failure_still_outranks_it(self):
        # Ordering only matters when both appear; the loader diagnosis is the more specific one.
        out = (
            "dyld[1]: Library not loaded: @rpath/libggml.dylib\n"
            "gguf_init_from_reader: invalid magic characters: '????', expected 'GGUF'"
        )
        msg = _classify(out, "/models/x.gguf", "local/x", 1)
        assert "not a GGUF" not in msg


class TestStartupDiagnostics:
    _UNKNOWN_OUT = (
        "build: 9415 (06d26dfd) with Apple clang version 17.0.0 for arm64-apple-darwin24.6.0\n"
        "ggml_metal_init: error: failed to allocate buffer\n"
        "GGML_ASSERT(ctx->device != nil) failed"
    )

    def test_unknown_failure_carries_the_output_tail(self):
        msg = _classify(self._UNKNOWN_OUT, "/models/x.gguf", "local/x", 1)
        assert msg.startswith("llama-server failed to start.")
        assert "GGML_ASSERT" in msg
        assert "llama-server output:" in msg

    def test_unknown_failure_carries_the_log_path(self, tmp_path):
        log = tmp_path / "llama-1-port-8080.log"
        msg = _classify(self._UNKNOWN_OUT, "/models/x.gguf", "local/x", 1, None, log)
        assert f"Full log: {log}" in msg

    def test_the_tail_is_bounded(self):
        out = "x" * 50_000 + "\nfinal line"
        msg = _classify(out, "/models/x.gguf", "local/x", 1)
        assert "final line" in msg
        assert len(msg) < 4000

    def test_control_characters_are_stripped(self):
        out = "loading \x1b[32mmodel\x1b[0m \x00done"
        msg = _classify(out, "/models/x.gguf", "local/x", 1)
        assert "\x00" not in msg
        assert "\x1b" not in msg
        assert "done" in msg

    def test_no_output_and_no_log_keeps_the_old_message_exactly(self):
        assert _classify("", "/models/x.gguf", "local/x", 1) == (
            "llama-server failed to start. "
            "Check that the GGUF file is valid and you have enough memory."
        )

    def test_a_classified_message_gains_no_diagnostics(self, tmp_path):
        # Only the unknown fallback carries evidence; the specific messages are
        # already actionable and must keep their exact text.
        log = tmp_path / "llama-1-port-8080.log"
        msg = _classify(_QWEN_IMAGE_OUT, "/models/q.gguf", "local/q", 1, None, log)
        assert "Images page" in msg
        assert "Full log" not in msg
        assert "llama-server output:" not in msg


class TestMacOSLoaderEdgeCases:
    """dyld's real output shape: a mixed "tried:" list, unrelated text above it,
    and a reason that can run for kilobytes."""

    def test_a_mixed_tried_list_judges_our_own_copy(self, monkeypatch, tmp_path):
        # An Apple Silicon Mac with a leftover Intel Homebrew tree lists our
        # missing dylib AND an x86_64 one under /usr/local. Scanning the whole
        # list would report "wrong CPU architecture" for a file that is simply
        # absent from our install.
        monkeypatch.delenv("LLAMA_SERVER_PATH", raising = False)
        binary = tmp_path / "llama.cpp" / "build" / "bin" / "llama-server"
        binary.parent.mkdir(parents = True)
        binary.write_text("")
        out = (
            "dyld[4210]: Library not loaded: @rpath/libmtmd.dylib\n"
            f"  Reason: tried: '{binary.parent}/libmtmd.dylib' (no such file), "
            "'/usr/local/lib/libmtmd.dylib' (mach-o file, but is an incompatible "
            "architecture (have 'x86_64', need 'arm64')), "
            "'/usr/lib/libmtmd.dylib' (no such file, not in dyld cache)"
        )
        msg = _classify(out, "/models/x.gguf", "local/x", 1, str(binary))
        assert "libmtmd.dylib" in msg
        assert "architecture" not in msg.lower()
        assert "missing" in msg.lower()

    def test_a_policy_blocked_sibling_does_not_accuse_the_user_of_tampering(
        self, monkeypatch, tmp_path
    ):
        monkeypatch.delenv("LLAMA_SERVER_PATH", raising = False)
        binary = tmp_path / "llama.cpp" / "build" / "bin" / "llama-server"
        binary.parent.mkdir(parents = True)
        binary.write_text("")
        out = (
            "dyld[4210]: Library not loaded: @rpath/libllama.dylib\n"
            f"  Reason: tried: '{binary.parent}/libllama.dylib' (no such file), "
            "'/usr/local/lib/libllama.dylib' (code signature not valid for use in "
            "process: library load disallowed by system policy)"
        )
        msg = _classify(out, "/models/x.gguf", "local/x", 1, str(binary))
        assert "code signature" not in msg.lower()
        assert "was modified" not in msg

    def test_our_own_copy_really_is_the_signature_failure(self, monkeypatch, tmp_path):
        monkeypatch.delenv("LLAMA_SERVER_PATH", raising = False)
        binary = tmp_path / "llama.cpp" / "build" / "bin" / "llama-server"
        binary.parent.mkdir(parents = True)
        binary.write_text("")
        out = (
            "dyld[4210]: Library not loaded: @rpath/libllama.dylib\n"
            f"  Reason: tried: '{binary.parent}/libllama.dylib' (code signature in "
            "<A1B2> not valid for use in process: mapped file has no cdhash)"
        )
        msg = _classify(out, "/models/x.gguf", "local/x", 1, str(binary))
        assert "code signature" in msg.lower()

    def test_an_earlier_unrelated_reason_is_not_read_as_dylds(self):
        # llama.cpp prints its own "Reason:" lines; only the one after the
        # "Library not loaded:" explains it.
        out = (
            "main: loading model\n"
            "llama_model_load: error loading model: Reason: unknown\n"
            "ggml_metal_init: skipping device, incompatible architecture\n"
            "dyld[9]: Library not loaded: @rpath/libggml-base.dylib\n"
            "  Reason: tried: '/x/libggml-base.dylib' (no such file)"
        )
        msg = _classify(out, "/models/x.gguf", "local/x", 1)
        assert "libggml-base.dylib" in msg
        assert "architecture" not in msg.lower()

    def test_a_stray_reason_does_not_disable_symbol_classification(self):
        # A "reason:" anywhere used to swallow the symbol-mismatch branch.
        out = (
            "srv load: Reason: retrying\n"
            "dyld[1]: Symbol not found: __ZN4ggml7backendE\n"
            "  Referenced from: /x/libmtmd.dylib\n"
            "  Expected in: /x/libllama.dylib"
        )
        msg = _classify(out, "/models/x.gguf", "local/x", 1)
        assert "different build" in msg

    def test_the_classified_reason_is_bounded(self):
        out = (
            "dyld[1]: Library not loaded: @rpath/libllama.dylib\n"
            "  Reason: no suitable image found. Did find: " + "x" * 20_000
        )
        msg = _classify(out, "/models/x.gguf", "local/x", 1)
        assert "libllama.dylib" in msg
        assert len(msg) < 1000

    def test_the_health_timeout_marker_is_not_absorbed_into_a_dyld_reason(self):
        # Unsloth appends its own marker to the captured output; it must not be
        # quoted back to the user as part of dyld's diagnosis.
        out = (
            "dyld[1]: Library not loaded: @rpath/libllama.dylib\n"
            "  Reason: tried: '/x/libllama.dylib' (no such file)\n"
            "llama-server health check timed out after 600.0s"
        )
        msg = _classify(out, "/models/x.gguf", "local/x", 1)
        assert "health check timed out" not in msg

    def test_signal_9_on_macos_respects_a_custom_binary(self, monkeypatch):
        monkeypatch.setattr(sys, "platform", "darwin")
        monkeypatch.setenv("LLAMA_SERVER_PATH", "/opt/mybuild/bin/llama-server")
        msg = _classify("", "/models/x.gguf", "local/x", -9, "/opt/mybuild/bin/llama-server")
        assert "unsloth studio update" not in msg
        assert "custom llama.cpp" in msg


class TestDiagnosticsDoNotLeak:
    """The output tail is llama-server's own stdout, and llama-server inherits
    nearly all of Unsloth's environment."""

    _OUT = "build: 9415\nenv dump: OPENAI_API_KEY=sk-owner-secret-1234567890\nabort"

    def test_a_secret_env_value_is_redacted(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-owner-secret-1234567890")
        msg = _classify(self._OUT, "/models/x.gguf", "local/x", 1)
        assert "sk-owner-secret-1234567890" not in msg
        assert "***" in msg

    def test_a_non_secret_env_value_is_left_alone(self, monkeypatch):
        monkeypatch.setenv("UNSLOTH_MODEL_DIR", "/models/mymodels")
        msg = _classify("loading from /models/mymodels failed", "/models/x.gguf", "local/x", 1)
        assert "/models/mymodels" in msg

    def test_a_credential_url_is_redacted_whatever_it_is_named(self, monkeypatch):
        # DATABASE_URL / REDIS_URL match no name marker, so the value has to
        # carry the verdict: scheme://user:secret@host is a credential.
        monkeypatch.setenv("DATABASE_URL", "postgres://admin:hunter2secret@db:5432/prod")
        out = "dump: DATABASE_URL=postgres://admin:hunter2secret@db:5432/prod"
        msg = _classify(out, "/models/x.gguf", "local/x", 1)
        assert "hunter2secret" not in msg

    def test_a_token_shape_is_redacted_under_an_unremarkable_name(self, monkeypatch):
        monkeypatch.setenv("GITHUB_PAT", "github_pat_11ABCDEFG0123456789abcdefghij")
        out = "dump: GITHUB_PAT=github_pat_11ABCDEFG0123456789abcdefghij"
        msg = _classify(out, "/models/x.gguf", "local/x", 1)
        assert "github_pat_11ABCDEFG0123456789abcdefghij" not in msg

    def test_a_bare_hf_token_shape_is_redacted(self):
        out = "curl -H 'Authorization: Bearer hf_" + "a" * 34 + "' failed"
        msg = _classify(out, "/models/x.gguf", "local/x", 1)
        assert "hf_" + "a" * 34 not in msg

    def test_a_huge_unterminated_line_is_cheap(self):
        # _drain_stdout keeps an unterminated line whole; the tail must be
        # sliced before it is filtered character by character.
        #
        # Proven by BEHAVIOUR rather than by a stopwatch. The bound is observable: an
        # argument error further back than the tail cannot be reported unless
        # something read it. A wall-clock budget tests the same property by proxy and
        # measures the runner instead, which is why this exact assertion goes red on
        # the Windows runner for main as well as for a branch. The stopwatch stays
        # only as a catastrophic guard, loose enough that no runner can trip it.
        import time

        buried = "error: invalid argument: --nope\n" + "x" * 10_000_000 + "\nggml_metal_init: error"
        start = time.perf_counter()
        msg = _classify(buried, "/models/x.gguf", "local/x", 1)
        elapsed = time.perf_counter() - start

        # 10 MB back, so out of the scanned tail: reporting it would mean the whole
        # capture was walked.
        assert "--nope" not in msg
        # And what IS in the tail still surfaces.
        assert "ggml_metal_init: error" in msg
        assert elapsed < 5, f"{elapsed:.1f}s to classify one 10 MB line"

    def test_an_argument_error_inside_the_tail_is_still_reported(self):
        # The other half of the bound: near the end is where llama-server actually
        # prints it, immediately before exiting.
        out = "x" * 10_000_000 + "\nerror: invalid argument: --nope"
        msg = _classify(out, "/models/x.gguf", "local/x", 1)

        assert "--nope" in msg


class TestDyldInstallNames:
    def test_an_rpath_placeholder_is_not_reported_as_a_location(self, monkeypatch):
        # "@rpath/libfoo.dylib is missing from that exact location" is
        # unusable advice: @rpath is a search directive, not a directory.
        monkeypatch.setenv("LLAMA_SERVER_PATH", "/opt/mybuild/bin/llama-server")
        out = (
            "dyld[1]: Library not loaded: @rpath/libomp.dylib\n"
            "  Reason: tried: '/opt/mybuild/bin/libomp.dylib' (no such file)"
        )
        msg = _classify(out, "/models/x.gguf", "local/x", 1, "/opt/mybuild/bin/llama-server")
        assert "@rpath" not in msg
        assert "libomp.dylib" in msg
        assert "that exact location" not in msg

    def test_a_bundled_dylib_behind_rpath_still_points_at_the_installer(self, monkeypatch):
        monkeypatch.delenv("LLAMA_SERVER_PATH", raising = False)
        out = (
            "dyld[1]: Library not loaded: @rpath/libggml-metal.dylib\n"
            "  Reason: tried: '/x/libggml-metal.dylib' (no such file)"
        )
        msg = _classify(out, "/models/x.gguf", "local/x", 1)
        assert "unsloth studio update" in msg


class TestMacOSSymbolProvenance:
    """Where dyld expected the symbol decides the remedy: reinstalling the same
    build cannot make the OS export a symbol it does not have."""

    def test_a_system_framework_symbol_reads_as_a_too_new_build(self):
        out = (
            "dyld[1]: Symbol not found: _MTLDeviceNewCommandQueueWithDescriptor\n"
            "  Referenced from: /Users/me/.unsloth/llama.cpp/build/bin/libggml-metal.dylib\n"
            "  Expected in: /System/Library/Frameworks/Metal.framework/Versions/A/Metal"
        )
        msg = _classify(out, "/models/x.gguf", "local/x", 1)
        assert "newer macOS" in msg
        assert "different build than the llama-server binary" not in msg

    def test_a_llama_dylib_symbol_still_reads_as_a_mixed_install(self):
        out = (
            "dyld[1]: Symbol not found: __ZN4ggml7backendE\n"
            "  Referenced from: /Users/me/.unsloth/llama.cpp/build/bin/libmtmd.dylib\n"
            "  Expected in: /Users/me/.unsloth/llama.cpp/build/bin/libllama.dylib"
        )
        msg = _classify(out, "/models/x.gguf", "local/x", 1)
        assert "different build" in msg
        assert "newer macOS" not in msg

    def test_a_usr_lib_symbol_is_read_the_same_way_as_a_framework(self):
        out = (
            "dyld[1]: Symbol not found: _os_signpost_emit\n"
            "  Expected in: /usr/lib/libSystem.B.dylib"
        )
        msg = _classify(out, "/models/x.gguf", "local/x", 1)
        assert "newer macOS" in msg


class TestCommandLineSecrets:
    """--api-key is minted per launch with secrets.token_urlsafe(32), so it is
    in no environment variable and matches no token shape."""

    _KEY = "Xy9_kQ2mLp7RtVw4NbZc8FgHjKdSaEuI0oPqWxYz1A"

    def test_the_api_key_is_redacted_by_its_flag(self):
        out = f"exec: llama-server -m x.gguf --api-key {self._KEY} --port 8080\nabort"
        msg = _classify(out, "/models/x.gguf", "local/x", 1)
        assert self._KEY not in msg
        assert "--api-key" in msg  # the flag stays; only the value goes

    def test_the_equals_form_is_redacted_too(self):
        out = f"argv: --api-key={self._KEY}"
        msg = _classify(out, "/models/x.gguf", "local/x", 1)
        assert self._KEY not in msg

    def test_a_known_secret_is_redacted_wherever_it_appears(self):
        # Passed by the caller, so it goes even without the flag next to it.
        out = f"client sent token {self._KEY} and was refused"
        msg = _classify(out, "/models/x.gguf", "local/x", 1, None, None, (self._KEY,))
        assert self._KEY not in msg

    def test_no_known_secret_is_harmless(self):
        msg = _classify("plain failure", "/models/x.gguf", "local/x", 1, None, None, (None,))
        assert "plain failure" in msg


class TestClassifierScoping:
    """The classifier is pure text and runs on every platform, so a macOS
    diagnosis needs macOS evidence, not an English phrase."""

    def test_a_bare_symbol_not_found_is_not_a_macos_diagnosis(self):
        # A Linux wrapper or a plugin loader saying this is not dyld.
        out = "plugin loader: Symbol not found in module registry\nexiting"
        msg = _classify(out, "/models/x.gguf", "local/x", 1)
        assert "llama.cpp libraries are from" not in msg
        assert "llama-server failed to start." in msg
        assert "plugin loader" in msg  # the tail survives instead

    def test_dyld_framing_still_classifies(self):
        out = (
            "dyld[1]: Symbol not found: __ZN4ggml7backendE\n"
            "  Expected in: /Users/me/.unsloth/llama.cpp/build/bin/libllama.dylib"
        )
        msg = _classify(out, "/models/x.gguf", "local/x", 1)
        assert "different build" in msg


class TestNestedDyldPlaceholders:
    def test_a_nested_placeholder_is_reduced_to_the_library_name(self, monkeypatch):
        monkeypatch.setenv("LLAMA_SERVER_PATH", "/opt/mybuild/bin/llama-server")
        out = (
            "dyld[1]: Library not loaded: @loader_path/../Frameworks/libomp.dylib\n"
            "  Reason: tried: '/opt/mybuild/Frameworks/libomp.dylib' (no such file)"
        )
        msg = _classify(out, "/models/x.gguf", "local/x", 1, "/opt/mybuild/bin/llama-server")
        assert "libomp.dylib" in msg
        assert "@loader_path" not in msg
        assert "../Frameworks" not in msg
        assert "that exact location" not in msg


class TestShortSecrets:
    def test_a_short_password_is_redacted_beside_its_name(self, monkeypatch):
        monkeypatch.setenv("DATABASE_PASSWORD", "hunter2")
        out = "env dump: DATABASE_PASSWORD=hunter2 PORT=8080"
        msg = _classify(out, "/models/x.gguf", "local/x", 1)
        assert "hunter2" not in msg
        assert "PORT=8080" in msg  # non-secret short values are untouched

    def test_a_short_non_secret_value_is_not_replaced_globally(self, monkeypatch):
        monkeypatch.setenv("UNSLOTH_PORT", "8080")
        msg = _classify("bound 8080 then failed", "/models/x.gguf", "local/x", 1)
        assert "8080" in msg


class TestDiagnosticsAreAFixedPoint:
    """Decorating an already-decorated message must be a no-op.

    Only one call site exists today, so none of this can fire yet. It is pinned
    because the failure mode is silent: a second caller would double the tail
    and the log line, and nothing else would notice.
    """

    _OUT = "some unrecognised startup noise"
    _LOG = "/Users/me/.unsloth/studio/logs/llama-server/llama-1-port-8080.log"

    def test_appending_twice_adds_nothing(self):
        once = LlamaCppBackend._with_startup_diagnostics("base", self._OUT, self._LOG)
        assert once.count("llama-server output:") == 1
        assert once.count("Full log: ") == 1
        assert LlamaCppBackend._with_startup_diagnostics(once, self._OUT, self._LOG) == once

    def test_a_classified_message_fed_back_in_is_not_decorated_twice(self):
        once = _classify(self._OUT, "/models/x.gguf", "local/x", 1, None, log_path = self._LOG)
        twice = LlamaCppBackend._with_startup_diagnostics(once, self._OUT, self._LOG)
        assert twice == once

    def test_a_tail_that_merely_mentions_the_label_is_still_decorated(self):
        # The guard keys on our own two-newline framing, so a server that
        # printed the words itself does not suppress the diagnostics.
        out = "llama-server output: 3 tokens/s\nthen it died"
        msg = LlamaCppBackend._with_startup_diagnostics("base", out, None)
        assert msg.startswith("base\n\nllama-server output:\n")


class TestTheDyldReasonIsBounded:
    def test_a_pathological_reason_does_not_stall_the_classifier(self):
        # 100KB of "'a' (" drove the candidate scan quadratic: 6.3s measured
        # before the cap, against 0.0s on main, on the thread serving the load.
        import time

        out = (
            "dyld[1]: Library not loaded: @rpath/libllama.dylib\n"
            "  Reason: tried: " + "'a' (" * 20000
        )
        start = time.perf_counter()
        msg = _classify(
            out,
            "/models/x.gguf",
            "local/x",
            1,
            "/Users/me/.unsloth/llama.cpp/build/bin/llama-server",
        )
        assert time.perf_counter() - start < 1.0
        assert "libllama.dylib" in msg

    def test_a_real_reason_is_not_truncated(self):
        out = (
            "dyld[1]: Library not loaded: @rpath/libggml-base.dylib\n"
            "  Reason: tried: '/Users/me/.unsloth/llama.cpp/build/bin/libggml-base.dylib' "
            "(mach-o file, but is an incompatible architecture (have 'x86_64', need 'arm64'))"
        )
        msg = _classify(
            out,
            "/models/x.gguf",
            "local/x",
            1,
            "/Users/me/.unsloth/llama.cpp/build/bin/llama-server",
        )
        assert "architecture" in msg


class TestOnlyDyldsOwnOutputIsReadAsDyld:
    """llama.cpp echoes GGUF metadata while loading.

    A model whose general.name contains "Library not loaded:" or "Symbol not
    found" put those words in llama-server's stderr, and every case below was
    answered with library advice instead of the classification it had before.
    Two of them outranked branches that were already correct.
    """

    _BIN = "/Users/me/.unsloth/llama.cpp/build/bin/llama-server"

    def test_metadata_does_not_beat_the_tensor_parallel_branch(self):
        out = (
            "llama_model_loader: - kv 2: general.name str = Symbol not found: MTLResidency\n"
            "split_mode_tensor not implemented"
        )
        assert "Tensor parallelism" in _classify(out, "/m.gguf", "u/x", 1, self._BIN)

    def test_metadata_does_not_beat_the_diffusion_branch(self):
        out = (
            "llama_model_loader: - kv 2: general.name str = Library not loaded: libfake.so\n"
            "llama_model_load: error loading model architecture: "
            "unknown model architecture: 'qwen_image'"
        )
        assert "diffusion" in _classify(out, "/m.gguf", "u/x", 1, self._BIN)

    def test_metadata_does_not_beat_status_127(self):
        msg = _classify(
            "wrapper says Library not loaded: libx.so", "/m.gguf", "u/x", 127, self._BIN
        )
        assert "status 127" in msg

    def test_metadata_does_not_beat_signal_15(self):
        msg = _classify(
            "wrapper says Library not loaded: libx.so", "/m.gguf", "u/x", -15, self._BIN
        )
        assert "signal 15" in msg

    def test_metadata_does_not_claim_a_too_new_macos(self):
        out = (
            "llama_model_loader: - kv 9: general.description str = "
            "built for macOS 26.0 which is newer than running OS"
        )
        msg = _classify(out, "/m.gguf", "u/x", 1, self._BIN)
        assert "newer version of macOS" not in msg

    def test_real_dyld_framing_still_classifies(self):
        out = (
            "dyld[4321]: Library not loaded: @rpath/libllama.dylib\n"
            "  Referenced from: <A> /Users/me/.unsloth/llama.cpp/build/bin/llama-server\n"
            "  Reason: tried: '/Users/me/.unsloth/llama.cpp/build/bin/libllama.dylib' (no such file)\n"
        )
        assert "libllama.dylib" in _classify(out, "/m.gguf", "u/x", 1, self._BIN)

    def test_the_older_bare_dyld_prefix_is_framing_too(self):
        out = "dyld: Library not loaded: @rpath/libllama.dylib\n  Reason: image not found\n"
        assert "libllama.dylib" in _classify(out, "/m.gguf", "u/x", 1, self._BIN)


class TestArchitectureAndQuotedPaths:
    _BIN = "/Users/me/.unsloth/llama.cpp/build/bin/llama-server"

    def test_a_quoted_system_framework_reads_as_a_too_new_build(self):
        out = (
            "dyld[5]: Symbol not found: _MTLFoo\n"
            "  Referenced from: <A> /Users/me/.unsloth/llama.cpp/build/bin/llama-server\n"
            "  Expected in: '/System/Library/Frameworks/Metal.framework/Versions/A/Metal'\n"
        )
        msg = _classify(out, "/m.gguf", "u/x", 1, self._BIN)
        assert "newer macOS" in msg
        assert "different build" not in msg

    def test_the_older_wrong_architecture_wording_is_recognised(self):
        out = (
            "dyld[5]: Library not loaded: @rpath/libggml.dylib\n"
            "  Reason: tried: '/Users/me/.unsloth/llama.cpp/build/bin/libggml.dylib' "
            "(mach-o, but wrong architecture)\n"
        )
        assert "different CPU architecture" in _classify(out, "/m.gguf", "u/x", 1, self._BIN)

    def test_bad_cpu_type_never_reaches_dyld_and_is_still_named(self):
        out = "/bin/sh: /Users/me/.unsloth/llama.cpp/build/bin/llama-server: Bad CPU type in executable"
        msg = _classify(out, "/m.gguf", "u/x", 126, self._BIN)
        assert "different CPU architecture" in msg
        assert "GGUF file is valid" not in msg

    def test_a_linux_exec_failure_is_not_read_as_a_mac_one(self):
        out = "bash: /opt/x/llama-server: cannot execute binary file: Exec format error"
        msg = _classify(out, "/m.gguf", "u/x", 126, "/opt/x/llama-server")
        assert "this Mac" not in msg


class TestHttpCredentialsInTheTail:
    def test_a_basic_credential_is_redacted(self):
        msg = _classify("Authorization: Basic QWxhZGRpbjpvcGVuIHNlc2FtZQ==", "/m.gguf", "u/x", 1)
        assert "QWxhZGRpbg" not in msg

    def test_a_bearer_token_is_redacted_including_its_base64_tail(self):
        msg = _classify("Authorization: Bearer QUFB+U0VDUkVUL1RBSUw=", "/m.gguf", "u/x", 1)
        assert "U0VDUkVU" not in msg


class TestOutputIsNeverTrustedForBeingOurOwnFraming:
    """Child stdout gets the full treatment however it is worded.

    An earlier revision short-circuited when the OUTPUT carried our framing, to
    keep re-classification a fixed point. That handed a wrapper the whole error
    message: printing "llama-server output:" as its first line returned its
    stdout verbatim, past the redaction and past the 2000-character cap. The
    fixed point was for a caller that does not exist; the bypass was reachable
    by anything Unsloth launches.
    """

    _LOG = "/Users/me/.unsloth/studio/logs/llama-server/llama-1-port-8080.log"

    @pytest.mark.parametrize(
        "prefix",
        [
            "llama-server output:\n",
            "Full log: /wherever\n",
            "starting\n\nllama-server output:\n",
            "starting\n\nFull log: /wherever\n",
        ],
    )
    def test_output_wearing_our_heading_is_still_capped_and_scrubbed(self, prefix, monkeypatch):
        monkeypatch.setenv("MY_API_TOKEN", "sk-super-secret-value-1234567890")
        out = prefix + "MY_API_TOKEN=sk-super-secret-value-1234567890\n" + "X" * 50000
        msg = _classify(out, "/m.gguf", "u/x", 1, None, log_path = self._LOG)
        assert "sk-super-secret-value-1234567890" not in msg
        assert len(msg) < 3000, len(msg)
        assert msg != out
        # The log path is the whole point of the fallback, and the bypass
        # dropped it along with everything else.
        assert self._LOG in msg

    def test_repeated_passes_stay_bounded_by_the_tail_cap(self):
        # Re-classifying a result is not a fixed point: each pass wraps the
        # previous message in a fresh tail, growing ~136 characters at a time.
        # Bounded is the property that matters, and the cap supplies it.
        cur = "some unrecognised startup noise"
        seen = []
        for _ in range(60):
            cur = _classify(cur, "/m.gguf", "u/x", 1, None, log_path = self._LOG)
            seen.append(len(cur))
        assert max(seen) < 3000, max(seen)
        assert seen[-1] == seen[-2], seen[-4:]

    def test_real_output_that_merely_says_the_words_is_still_classified(self):
        out = "llama-server output: 3 t/s\nerror while loading shared libraries: libgomp.so.1: cannot open shared object file"
        msg = _classify(out, "/m.gguf", "u/x", 1)
        assert "libgomp.so.1" in msg

    def test_a_dyld_diagnosis_survives_the_heading(self):
        out = (
            "llama-server output:\n"
            "dyld[1]: Library not loaded: @rpath/libllama.dylib\n"
            "  Reason: tried: '/a/libllama.dylib' (no such file)\n"
        )
        binary = "/Users/me/.unsloth/llama.cpp/build/bin/llama-server"
        assert "libllama.dylib" in _classify(out, "/m.gguf", "u/x", 1, binary)


class TestAnEncodedSecretIsStillRedacted:
    """Redacting on the value alone only works if the child prints it verbatim.

    A wrapper that dumps its environment as JSON prints pa"ss\\word as
    pa\\"ss\\\\word. That is a different string, so the literal replacement
    missed it and the credential reached the API error fully reconstructible.
    Redacting whatever sits beside a secret-looking NAME does not care how the
    value was encoded.
    """

    SECRET = 'pa"ss\\word-12345'

    @pytest.fixture(autouse = True)
    def _env(self, monkeypatch):
        monkeypatch.setenv("DB_PASSWORD", self.SECRET)
        monkeypatch.setenv("MY_API_TOKEN", "sk-abcdefghijklmnopqrstuvwxyz")

    @pytest.mark.parametrize(
        "dump",
        [
            '{"DB_PASSWORD": "pa\\"ss\\\\word-12345"}',  # JSON escaped
            'DB_PASSWORD=pa"ss\\word-12345',  # bare
            "DB_PASSWORD='pa\"ss\\word-12345'",  # shell quoted
            'DB_PASSWORD: pa"ss\\word-12345',  # yaml-ish
            "DB_PASSWORD=pa%22ss%5Cword-12345",  # url encoded
            "DB_PASSWORD=secret123456,PORT=8080",  # one of several pairs
        ],
    )
    def test_no_recognisable_fragment_survives(self, dump):
        msg = _classify(dump, "/m.gguf", "u/x", 1, None, log_path = "/tmp/l.log")
        for fragment in ("ss\\word", "pa%22", "secret123456", 'pa\\"ss'):
            assert fragment not in msg, msg
        assert "***" in msg

    def test_a_value_we_never_set_is_redacted_by_its_name(self):
        """The point of the name pass: we cannot match a value we never saw."""
        msg = _classify('{"DB_PASSWORD": "never-set-98765"}', "/m.gguf", "u/x", 1)
        assert "never-set-98765" not in msg

    @pytest.mark.parametrize(
        "line",
        [
            "PATH=/usr/bin:/bin",
            '{"port": 8080}',
            "model_path=/models/x.gguf",
            "n_ctx: 4096",
            "load time = 1234.5 ms",
            "llama_model_loader: n_layers = 32",
        ],
    )
    def test_ordinary_llama_cpp_output_is_not_redacted(self, line):
        """Over-redaction would destroy the diagnostics this PR exists to add."""
        assert "***" not in _classify(line, "/m.gguf", "u/x", 1)

    @pytest.mark.parametrize(
        "blob",
        [
            '"A' + '\\"' * 20000 + 'B"',
            "K=" + "x" * 100000,
            "A=1," * 25000,
            'TOKEN="' + "y" * 100000,
        ],
        # Named, because pytest puts the whole parameter in the node id and then in
        # PYTEST_CURRENT_TEST. Windows caps an environment variable at 32767
        # characters, so a 100 KB id errors the test in setup on that OS alone.
        ids = ["escaped-quotes", "long-value", "many-pairs", "unterminated-quote"],
    )
    def test_the_name_pass_stays_linear(self, blob):
        """No nested quantifier: a crafted line must not be able to stall it."""
        import time

        start = time.monotonic()
        _classify(blob, "/m.gguf", "u/x", 1)
        assert time.monotonic() - start < 2.0


class TestQuotedShortSecrets:
    """A short secret survived quoting.

    The name-adjacent rule for values under eight characters matched only the
    bare NAME=value form, so a wrapper dumping its environment as shell-ish or
    JSON put the credential straight into the startup-output tail.
    """

    @pytest.mark.parametrize(
        "dump",
        [
            "env dump: DB_PASSWORD=hunter2 PORT=8080",
            "env dump: DB_PASSWORD='hunter2'",
            'env dump: DB_PASSWORD="hunter2"',
            '{"DB_PASSWORD": "hunter2", "port": 8080}',
            "DB_PASSWORD: hunter2",
            "  'DB_PASSWORD' => 'hunter2'".replace(" => ", ": "),
        ],
    )
    def test_the_value_never_reaches_the_message(self, monkeypatch, dump):
        monkeypatch.setenv("DB_PASSWORD", "hunter2")
        assert "hunter2" not in _classify(dump, "/m.gguf", "u/x", 1)

    def test_a_non_secret_short_value_is_still_untouched(self, monkeypatch):
        monkeypatch.setenv("UNSLOTH_PORT", "8080")
        assert "8080" in _classify('{"UNSLOTH_PORT": "8080"}', "/m.gguf", "u/x", 1)


class TestASecretLongerThanTheTailWindow:
    """The pre-slice window cut long credentials in half.

    _scrub_secret_values matches a whole value, so a secret longer than the
    8000-character prefilter lost its head to the slice and the surviving
    suffix matched neither the literal nor any token shape. A secret SHORTER
    than the window cannot straddle it: if its end is inside a window wider
    than the secret, so is its start.
    """

    def test_a_long_key_does_not_reach_the_message(self, monkeypatch):
        secret = "A" * 12000 + "TAILMARKER"
        monkeypatch.setenv("MY_PRIVATE_KEY", secret)
        out = "noise " * 100 + "dump: MY_PRIVATE_KEY=" + secret + "\nfatal: boom"
        msg = _classify(out, "/m.gguf", "u/x", 1)
        assert "TAILMARKER" not in msg
        assert "AAAA" not in msg

    def test_a_long_minted_api_key_does_not_reach_the_message(self, monkeypatch):
        key = "k" * 9000 + "MINTEDMARK"
        out = "x" * 5000 + " --api-key " + key + "\nfatal: boom"
        msg = _classify(out, "/m.gguf", "u/x", 1, None, secrets = (key,))
        assert "MINTEDMARK" not in msg

    def test_the_tail_is_still_bounded(self, monkeypatch):
        monkeypatch.setenv("MY_PRIVATE_KEY", "A" * 12000)
        msg = _classify("z" * 400000 + "\nfatal: boom", "/m.gguf", "u/x", 1)
        assert len(msg) < 4000

    def test_no_secret_means_the_window_is_unchanged(self, monkeypatch):
        for name in list(os.environ):
            if "SECRET" in name or "PRIVATE" in name or "PASSWORD" in name:
                monkeypatch.delenv(name, raising = False)
        assert LlamaCppBackend._max_secret_len(()) >= 0


class TestTheLibraryNameIsBounded:
    """The install name is quoted from untrusted output into an HTTP error.

    A real dyld install name is a path, and macOS PATH_MAX is 1024, but nothing
    stopped a wrapper from printing a longer one: a 200000-character input
    produced a 100000-character API response.
    """

    def test_an_absurd_install_name_is_truncated(self):
        huge = "lib" + "A" * 100000 + ".dylib"
        out = (
            f"dyld[9]: Library not loaded: @rpath/{huge}\n"
            f"  Reason: tried: '/a/{huge}' (no such file)\n"
        )
        msg = _classify(out, "/m.gguf", "u/x", 1, "/i/bin/llama-server")
        assert len(msg) < 4000, len(msg)
        assert "..." in msg

    def test_a_real_install_name_is_untouched(self):
        out = (
            "dyld[9]: Library not loaded: @rpath/libllama.dylib\n"
            "  Reason: tried: '/a/libllama.dylib' (no such file)\n"
        )
        msg = _classify(out, "/m.gguf", "u/x", 1, "/i/bin/llama-server")
        assert "libllama.dylib" in msg
        assert "..." not in msg


class TestEveryClassifiedBranchIsRedacted:
    """Redaction covered the tail only, so the other branches leaked.

    A dyld message names the library it could not load, and that name comes
    from the child. A credential appearing there went out in the API error
    while the same credential in the startup-output tail was starred out.
    """

    def test_a_secret_in_a_dyld_library_name_is_redacted(self, monkeypatch):
        token = "sk-supersecret-abcdefghijk"
        monkeypatch.setenv("MY_API_TOKEN", token)
        out = (
            f"dyld[1]: Library not loaded: @rpath/{token}.dylib\n"
            f"  Reason: tried: '/a/{token}.dylib' (no such file)\n"
        )
        msg = _classify(
            out, "/m.gguf", "u/x", 1, "/i/bin/llama-server", log_path = "/l.log", secrets = (token,)
        )
        assert token not in msg
        assert "***" in msg

    def test_a_per_launch_secret_is_redacted_in_a_classified_branch(self):
        minted = "unsloth-launch-key-abcdefghij"
        out = f"dyld[1]: Library not loaded: @rpath/{minted}.dylib\n"
        msg = _classify(out, "/m.gguf", "u/x", 1, "/i/bin/llama-server", secrets = (minted,))
        assert minted not in msg

    def test_an_ordinary_diagnosis_is_unchanged_by_the_extra_pass(self):
        out = (
            "dyld[1]: Library not loaded: @rpath/libllama.dylib\n"
            "  Reason: tried: '/a/libllama.dylib' (no such file)\n"
        )
        msg = _classify(out, "/m.gguf", "u/x", 1, "/i/bin/llama-server")
        assert "libllama.dylib" in msg
        assert "***" not in msg


class TestAQuotedValueEndsOnItsOwnDelimiter:
    """A JSON value may contain an apostrophe, and a shell one a quote.

    Rejecting both quote characters inside the value made the quoted arm fail,
    so the bare arm took over and stopped at the first whitespace, leaving the
    rest of the credential standing in the API error.
    """

    @pytest.mark.parametrize(
        "dump,leak",
        [
            ('{"DB_PASSWORD": "prefix\' supersecret"}', "supersecret"),
            ("{'DB_PASSWORD': 'has \" quote inside'}", "quote inside"),
            ('{"DB_PASSWORD": "trailing \'"}', "trailing"),
        ],
    )
    def test_the_whole_quoted_value_is_replaced(self, dump, leak):
        cleaned = LlamaCppBackend._scrub_secret_values(dump, ())
        assert leak not in cleaned, cleaned
        assert "***" in cleaned


class TestTheRedactionHolesCodexFound:
    """Three ways a credential got past the name pass.

    Each was reachable by a wrapper or crash handler printing its own
    configuration rather than the environment we could match on.
    """

    @pytest.mark.parametrize(
        "dump",
        [
            'DB_PASSWORD="prefix supersecret',  # truncated output
            "DB_PASSWORD='prefix supersecret",
            'DB_PASSWORD="prefix supersecret\nnext=1',  # runs to end of line only
        ],
    )
    def test_an_unterminated_quoted_value_is_redacted_to_the_line_end(self, dump):
        cleaned = LlamaCppBackend._scrub_secret_values(dump, ())
        assert "supersecret" not in cleaned, cleaned
        if "\n" in dump:
            assert "next=1" in cleaned, "only the value's line should be consumed"

    @pytest.mark.parametrize(
        "dump,secret",
        [
            ('{"api-key": "abc123def456"}', "abc123def456"),
            ('{"db-password": "correct horse"}', "correct horse"),
            ("client.secret: hunter2value", "hunter2value"),
            ('{"private-key": "MIIEvQIBADAN"}', "MIIEvQIBADAN"),
            ("auth.token = deadbeefcafe", "deadbeefcafe"),
        ],
    )
    def test_a_config_style_key_is_recognised(self, dump, secret):
        """Hyphens and dots spell the same names the predicate underscores."""
        assert secret not in LlamaCppBackend._scrub_secret_values(dump, ())

    @pytest.mark.parametrize(
        "line",
        [
            "model-path=/models/x.gguf",
            "ggml.backend: metal",
            "n-gpu-layers=99",
            "cache.type-k: q8_0",
            "llama_model_loader: n_layers = 32",
            "load time = 1234.5 ms",
        ],
    )
    def test_a_widened_name_does_not_redact_ordinary_output(self, line):
        assert "***" not in LlamaCppBackend._scrub_secret_values(line, ())

    @pytest.mark.parametrize(
        "order", [("abcdefgh", "abcdefgh VERYSECRET"), ("abcdefgh VERYSECRET", "abcdefgh")]
    )
    def test_overlapping_secrets_are_replaced_longest_first(self, order):
        """The short one first rewrote the long one so it no longer matched."""
        text = "child printed abcdefgh VERYSECRET plainly"
        cleaned = LlamaCppBackend._scrub_secret_values(text, order)
        assert "VERYSECRET" not in cleaned, cleaned

    @pytest.mark.parametrize(
        "blob",
        ['TOKEN="' + "y" * 100000, "a.b.c.d=" * 12000, "A=1," * 25000],
        # Same reason as above: the parameter is the node id, and the node id
        # becomes an environment variable.
        ids = ["unterminated-quote", "dotted-names", "many-pairs"],
    )
    def test_the_widened_pattern_stays_linear(self, blob):
        import time

        start = time.monotonic()
        LlamaCppBackend._scrub_secret_values(blob, ())
        assert time.monotonic() - start < 2.0


class TestRejectedArguments:
    """Argument parsing runs before the model is touched, so these are never a
    bad GGUF or an OOM. The strings are what the bundled llama-server actually
    prints, captured from it directly rather than written from memory."""

    def test_an_unknown_flag_is_named(self):
        msg = _classify("error: invalid argument: --tempp", "/models/x.gguf", "local/x", 1)
        assert "--tempp" in msg
        assert "extra arguments" in msg
        # The generic diagnosis must not survive: the file and the memory are fine.
        assert "memory" not in msg.lower()

    def test_a_flag_unsloth_set_itself_is_covered_by_the_same_message(self):
        # Nothing reaching the classifier says whose flag it was, and Unsloth emits
        # its own conditionally on the capability probe, so a binary swapped under a
        # cached probe lands here too. The message has to serve that reader as well
        # as the one who mistyped something in the box.
        msg = _classify("error: invalid argument: --flash-attn", "/models/x.gguf", "local/x", 1)
        assert "--flash-attn" in msg
        assert "reinstall llama.cpp" in msg

    def test_a_rejected_value_is_told_apart_from_an_unknown_flag(self):
        # Different fix for the reader: the flag is right, the value is not.
        msg = _classify(
            'error while handling argument "--numa": invalid value',
            "/models/x.gguf",
            "local/x",
            1,
        )
        assert "--numa" in msg
        assert "invalid value" in msg
        assert "does not recognise" not in msg

    def test_a_missing_value_keeps_llama_cpps_own_reason(self):
        msg = _classify(
            'error while handling argument "--top-k": expected value for argument',
            "/models/x.gguf",
            "local/x",
            1,
        )
        assert "--top-k" in msg
        assert "expected value" in msg

    def test_a_std_stoi_failure_is_translated(self):
        # llama.cpp surfaces the C++ standard library's exception name verbatim.
        # "stoi" is not an error message anyone outside libstdc++ can act on.
        msg = _classify(
            'error while handling argument "--top-k": stoi', "/models/x.gguf", "local/x", 1
        )
        assert "not a number" in msg
        assert "stoi" not in msg

    def test_a_value_error_on_a_flag_the_user_did_not_set_stays_neutral(self):
        # Unsloth emits its own options conditionally on the capability probe, so a
        # build that reads "--flash-attn on" differently rejects a value the box
        # never held. Sending that reader to edit their extra arguments points them
        # at a setting they cannot use to fix it.
        msg = _classify(
            'error while handling argument "--flash-attn": invalid value',
            "/models/x.gguf",
            "local/x",
            1,
        )
        assert "--flash-attn" in msg
        assert "reinstall llama.cpp" in msg

    def test_a_value_error_on_a_flag_the_user_did_set_names_the_box(self):
        # Ownership established: the extras really do carry the flag, so the box is
        # where the fix is.
        msg = LlamaCppBackend._classify_llama_start_failure(
            'error while handling argument "--numa": invalid value',
            "/models/x.gguf",
            "local/x",
            1,
            None,
            None,
            (),
            ["--numa", "wherever"],
        )
        assert "--numa" in msg
        assert "Fix it in the extra arguments" in msg
        assert "reinstall" not in msg

    def test_an_alias_spelling_falls_back_to_the_neutral_wording(self):
        # -fa and --flash-attn are the same option to llama.cpp but not to this
        # comparison, and a wrong "you set this" is worse than a neutral one.
        msg = LlamaCppBackend._classify_llama_start_failure(
            'error while handling argument "--flash-attn": invalid value',
            "/models/x.gguf",
            "local/x",
            1,
            None,
            None,
            (),
            ["-fa", "on"],
        )
        assert "reinstall llama.cpp" in msg

    def test_an_ordinary_failure_is_untouched(self):
        # The two new branches sit ahead of the generic diagnosis, so this pins
        # that they do not swallow it.
        msg = _classify(_OOM_OUT, "/models/big.gguf", "local/big", 1)
        assert "enough memory" in msg.lower()
        assert "argument" not in msg.lower()

    def test_the_argument_scan_reads_only_the_tail(self):
        # The bound exists because _drain_stdout keeps an unterminated line whole,
        # and scanning a 10 MB one twice puts the classifier past any sane budget.
        # Asserted by BEHAVIOUR rather than by the clock: a wall-clock budget on a
        # shared CI runner measures the runner, and a Windows one failed this at
        # 204ms against 200 while the bound it was meant to prove was in place.
        from core.inference.llama_cpp import _FAILURE_SCAN_TAIL_CHARS

        # Inside the tail: found, and the whole capture is enormous either way.
        within = "x" * 10_000_000 + "\nerror: invalid argument: --tempp"
        assert "--tempp" in _classify(within, "/models/x.gguf", "local/x", 1)
        # Before it: not found, which is only possible if the scan stopped short of
        # the head. Reported as an ordinary failure instead.
        buried = "error: invalid argument: --tempp\n" + "x" * (_FAILURE_SCAN_TAIL_CHARS * 2)
        assert "--tempp" not in _classify(buried, "/models/x.gguf", "local/x", 1)

    def test_a_model_load_error_mentioning_arguments_is_not_misread(self):
        # "invalid argument" as an errno string (EINVAL) is not llama.cpp's
        # argument parser, and the anchored "error: invalid argument:" prefix is
        # what keeps them apart.
        out = "llama_model_load: error loading model: invalid argument (22)"
        msg = _classify(out, "/models/x.gguf", "local/x", 1)
        assert "does not recognise" not in msg


class TestArgumentErrorsAreQuotedShort:
    """What a failed start copies out of the child's own output."""

    def test_a_pathological_argument_is_truncated(self):
        # A wrapper on LLAMA_SERVER_PATH can print anything, and the capture is a run
        # of non-whitespace, so without a bound the API error becomes 64 KiB of it.
        out = "error: invalid argument: " + "x" * 50_000
        msg = _classify(out, "/models/x.gguf", "local/x", 1)

        assert len(msg) < 1_000, len(msg)
        assert "..." in msg
        # Still says what happened, and still names the beginning of the argument.
        assert "does not recognise the argument" in msg
        assert "xxxx" in msg

    def test_a_pathological_reason_is_truncated(self):
        out = 'error while handling argument "--top-k": ' + "y" * 50_000
        msg = _classify(out, "/models/x.gguf", "local/x", 1)

        assert len(msg) < 1_000, len(msg)
        assert "--top-k" in msg

    def test_an_ordinary_argument_error_is_untouched(self):
        out = "error: invalid argument: --tempp"
        msg = _classify(out, "/models/x.gguf", "local/x", 1)

        assert "--tempp" in msg
        assert "..." not in msg


class TestTensorSplitQuantizedKvUnsupported:
    """llama.cpp before ggml-org/llama.cpp#23792 (b9455) refused a quantized KV
    cache under --split-mode tensor. Unsloth no longer pre-empts that refusal, so
    the message has to name the remedy: the generic invalid-GGUF/OOM fallback sends
    the user to check their file or buy VRAM, neither of which is the problem."""

    # Verbatim from the guard #23792 deleted in src/llama-context.cpp.
    _OUT = (
        "llama_init_from_model: simultaneous use of SPLIT_MODE_TENSOR and "
        "KV cache quantization not implemented\n"
    )

    def test_the_legacy_refusal_names_the_build_and_the_remedies(self):
        msg = _classify(self._OUT, "/models/x.gguf", "local/x", 2)

        assert "b9455" in msg
        assert "quantized KV cache" in msg
        # All three ways out, because which one is available depends on whether the
        # user controls the binary.
        assert "Update" in msg
        assert "f16" in msg
        assert "Tensor Parallelism" in msg
        # Not the fallback it used to get.
        assert "GGUF file is valid" not in msg

    def test_it_does_not_shadow_the_architecture_gate(self):
        """A different, permanent, per-model limit with its own remedy."""
        out = "llama_init_from_model: split_mode_tensor not implemented for this arch\n"
        msg = _classify(out, "/models/x.gguf", "local/x", 2)

        assert "architecture" in msg
        assert "b9455" not in msg

    def test_the_marker_is_matched_case_insensitively(self):
        """llama.cpp prints SPLIT_MODE_TENSOR upper-case; the classifier lowers."""
        assert LlamaCppBackend._is_tensor_quant_kv_unsupported(self._OUT)
        assert LlamaCppBackend._is_tensor_quant_kv_unsupported(self._OUT.lower())
        assert not LlamaCppBackend._is_tensor_quant_kv_unsupported("")
        assert not LlamaCppBackend._is_tensor_quant_kv_unsupported(
            "split_mode_tensor not implemented"
        )

    def test_a_hard_crash_carrying_it_is_not_retried_as_a_projector_fault(self):
        """_output_has_nonprojector_diagnostic gates the text-only vision retry.
        Without the marker a doomed tensor load would also pay that retry."""
        assert LlamaCppBackend._output_has_nonprojector_diagnostic(self._OUT)

    def test_it_is_not_the_signal_crash_the_split_axis_latch_requires(self):
        """It is LLAMA_LOG_ERROR + return nullptr, so exit 1 with no signal --
        which is why it needs its own recording path rather than the #6415 one."""
        assert not LlamaCppBackend._should_record_tensor_split_abort(1, self._OUT)
        assert not LlamaCppBackend._is_tensor_split_assert(self._OUT)
