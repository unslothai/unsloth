# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""
Regression tests for studio.backend.loggers.handlers.filter_sensitive_data.

Context: filter_sensitive_data was originally written with a base64-detection
heuristic that truncated any string >100 chars containing ',' or '/' down to
20 chars + '...'. The block was dormant until PR #5246 wired the processor
into the structlog chain to redact native-path leases. Once active, the
heuristic ate normal log lines emitted by llama_cpp_backend (GGUF size
summary, mmproj selection, the full llama-server command line) and any
exception traceback that happened to contain a file path.

These tests pin two properties:

1. Long, comma- or slash-bearing log messages flow through filter_sensitive_data
   unchanged. The exact strings exercised match the call sites at
   studio/backend/core/inference/llama_cpp.py:2117, :2283, and :2312 that
   were truncated in the original bug report.

2. PR #5246's native-path lease redaction still fires for both the inline
   ``native_path_lease=...`` regex form and the ``nativePathLease`` dict-key
   form. This guards against future regressions that strip redaction along
   with the truncation block.
"""

from loggers.handlers import filter_sensitive_data


def _run(event_dict):
    return filter_sensitive_data(logger = None, method_name = "info", event_dict = event_dict)


class TestNoTruncation:
    def test_gguf_size_summary_survives(self):
        # Mirrors the f-string at studio/backend/core/inference/llama_cpp.py:2117
        event = (
            "GGUF size: 232.9 GB, est. KV cache: 87.0 GB, context: 259072, "
            "GPUs free: [(0, 80000), (1, 80000)], selected: [0, 1], fit: False"
        )
        out = _run({"event": event})
        assert out["event"] == event
        assert "..." not in out["event"]

    def test_mmproj_path_survives(self):
        # Mirrors logger.info at studio/backend/core/inference/llama_cpp.py:2283
        event = (
            "Using mmproj for vision: "
            "/home/user/.cache/unsloth/models/some-vision-model-uncensored-r1-distill/mmproj-F16.gguf"
        )
        out = _run({"event": event})
        assert out["event"] == event

    def test_llama_server_command_survives(self):
        # Mirrors logger.info at studio/backend/core/inference/llama_cpp.py:2312
        event = (
            "Starting llama-server: /home/user/.unsloth/studio/llama.cpp/build/bin/llama-server "
            "-m /home/user/.cache/unsloth/models/foo.gguf --port 8090 -c 259072 --parallel 1 "
            "--flash-attn on --mmproj /home/user/.cache/unsloth/models/mmproj-F16.gguf"
        )
        out = _run({"event": event})
        assert out["event"] == event

    def test_traceback_with_paths_survives(self):
        traceback_str = (
            "Traceback (most recent call last):\n"
            '  File "/home/user/.unsloth/studio/unsloth_studio/lib/python3.11/site-packages/'
            'studio/backend/core/inference/llama_cpp.py", line 2312, in start\n'
            '    raise RuntimeError("llama-server crashed: bad alloc, /dev/shm full")\n'
            "RuntimeError: llama-server crashed: bad alloc, /dev/shm full"
        )
        out = _run({"event": "llama-server crashed", "exception": traceback_str})
        assert out["exception"] == traceback_str
        assert "..." not in out["exception"]

    def test_nested_long_string_in_dict_survives(self):
        long_value = (
            "/very/long/path/with,many,commas,and/slashes/that/used/to/get/"
            "chopped/to/twenty/chars/file.gguf"
        )
        out = _run({"event": "load", "details": {"path": long_value}})
        assert out["details"]["path"] == long_value


class TestNativePathLeaseRedactionStillWorks:
    """Guards PR #5246's redaction from being lost alongside the truncation block."""

    def test_inline_native_path_lease_value_redacted(self):
        event = (
            "rejected request: native_path_lease=AAAAAA.BBBBBB extra context "
            "with /some/path,values"
        )
        out = _run({"event": event})
        assert "AAAAAA.BBBBBB" not in out["event"]
        assert "<redacted native path lease>" in out["event"]

    def test_camelcase_native_path_lease_dict_key_redacted(self):
        out = _run({"event": "load", "nativePathLease": "AAAAAA.BBBBBB"})
        assert out["nativePathLease"] == "<redacted native path lease>"

    def test_snakecase_native_path_lease_dict_key_redacted(self):
        out = _run({"event": "load", "native_path_lease": "AAAAAA.BBBBBB"})
        assert out["native_path_lease"] == "<redacted native path lease>"

    def test_nested_native_path_lease_key_redacted(self):
        out = _run({"event": "load", "payload": {"nativePathLease": "AAAAAA.BBBBBB"}})
        assert out["payload"]["nativePathLease"] == "<redacted native path lease>"
