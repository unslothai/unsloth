# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from utils.llama_server_log_lines import llama_server_line_uses_info_level


def test_readiness_lines_are_info():
    assert llama_server_line_uses_info_level("main: server is listening on 127.0.0.1:8080")
    assert llama_server_line_uses_info_level("model loaded")


def test_errors_and_warnings_are_info():
    assert llama_server_line_uses_info_level("ggml: CUDA error")
    assert llama_server_line_uses_info_level("error: failed to load model")
    assert llama_server_line_uses_info_level("warning: something odd")


def test_load_progress_stays_debug():
    assert not llama_server_line_uses_info_level("load_tensors: tensor 42/9000 ( 12.3%)")
    assert not llama_server_line_uses_info_level("offloading 50% of layers to GPU")
