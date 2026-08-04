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
    @pytest.mark.parametrize("arch", sorted(LlamaCppBackend._DIFFUSION_ARCHES))
    def test_every_diffusion_arch_is_recognised(self, arch):
        out = f"error loading model: unknown model architecture: '{arch}'"
        msg = _classify(out, f"/models/{arch}.gguf", f"local/{arch}")
        assert "diffusion" in msg.lower()
        assert "Images page" in msg
        assert arch in msg


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
    _OLLAMA_GGUF = (
        f"/home/u/.ollama{__import__('os').sep}ollama_links" f"{__import__('os').sep}m.gguf"
    )

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

    def test_sigkill_with_no_output_names_oom(self):
        msg = _classify("", "/models/big-bf16.gguf", "local/big", -9)
        assert "signal 9" in msg
        assert "out of memory" in msg.lower()
        assert ".wslconfig" in msg
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

    def test_exit_127_on_a_pinned_binary_does_not_send_it_to_the_updater(
        self, monkeypatch
    ):
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
            "/home/t/.unsloth/llama.cpp/llama-server: 2: exec: "
            "./build/bin/llama-server: not found"
        )
        msg = _classify(out, "/models/x.gguf", "local/x", 127)
        assert "package manager" not in msg
        assert "could not be found or run" in msg

    def test_symbol_lookup_error_is_not_called_a_system_library(self):
        # A mismatched bundled runtime exits 127 with this, not a loader line.
        out = "llama-server: symbol lookup error: llama-server: undefined symbol: ggml_backend_init"
        msg = _classify(out, "/models/x.gguf", "local/x", 127)
        assert "package manager" not in msg

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
