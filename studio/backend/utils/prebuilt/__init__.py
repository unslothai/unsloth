# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Backend-importable prebuilt helpers.

The installers reuse install_llama_prebuilt.py directly; this package holds the
backend-side shapes the studio/ scripts cannot provide (the backend runs with
studio/backend as its sys.path root): runtime_libs (wheel CUDA dirs), child_env
(secret scrubbing + WSL ROCm dirs), freshness_flow and update_flow (the shared
mechanics behind the *_cpp_freshness / *_cpp_update twins).
"""
