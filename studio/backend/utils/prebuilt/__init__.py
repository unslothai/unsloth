# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Backend-importable prebuilt helpers.

The installers reuse install_llama_prebuilt.py directly; only runtime_libs
remains here for the STT sidecar, which cannot import the studio/ scripts.
"""
