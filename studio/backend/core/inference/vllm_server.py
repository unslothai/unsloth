# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Launches the pinned vLLM server with its API key guarding every route.

Executed by the isolated engine Python. vLLM's middleware only guards /v1, /v2, /inference (and /cohere in 0.30),
so /invocations (SageMaker) and /tokenize answer without the key; Studio sends the key on every
request, /health included, so guarding "/" costs it nothing. The key arrives as VLLM_API_KEY,
never on the command line.
"""

import importlib
import runpy
import sys

# Where each locked vLLM keeps the prefix its middleware reads at request time: 0.26 in
# serve/utils/server_utils, 0.30 in serve/middleware/authenticate.
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
