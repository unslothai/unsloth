# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Runs pinned vLLM server with API key guarding all routes: vLLM middleware covers only /v1, /v2,
/inference (/cohere in 0.30), not /invocations or /tokenize; key via VLLM_API_KEY, not argv."""

import importlib
import runpy
import sys

# vLLM 0.26 keeps the prefix in serve/utils/server_utils, 0.30 in serve/middleware/authenticate.
for _name in (
    "vllm.entrypoints.serve.middleware.authenticate",
    "vllm.entrypoints.serve.utils.server_utils",
):
    try:
        guard = importlib.import_module(_name)
    except ImportError:
        continue
    if isinstance(getattr(guard, "GUARDED_PREFIX", None), tuple):
        guard.GUARDED_PREFIX = ("/",)
        break
else:
    # A vLLM that moved this again would silently reopen routes; refuse to start instead.
    raise SystemExit("Unsloth: vLLM's API-key middleware changed; refusing to start unguarded.")

if __name__ == "__main__":
    module = sys.argv.pop(1)
    runpy.run_module(module, run_name = "__main__", alter_sys = True)
