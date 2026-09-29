# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Launches the pinned vLLM server with its API key guarding every route.

Executed by the isolated engine Python. vLLM's middleware only guards /v1, /v2 and /inference,
so /invocations (SageMaker) and /tokenize answer without the key; Studio sends the key on every
request, /health included, so guarding "/" costs it nothing. The key arrives as VLLM_API_KEY,
never on the command line.
"""

import runpy
import sys

from vllm.entrypoints.serve.utils import server_utils

if not isinstance(getattr(server_utils, "GUARDED_PREFIX", None), tuple):
    # A vLLM that moved this would silently reopen routes; refuse to start instead.
    raise SystemExit("Unsloth: vLLM's API-key middleware changed; refusing to start unguarded.")
server_utils.GUARDED_PREFIX = ("/",)

if __name__ == "__main__":
    module = sys.argv.pop(1)
    runpy.run_module(module, run_name = "__main__", alter_sys = True)
