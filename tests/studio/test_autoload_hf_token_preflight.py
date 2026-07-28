# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Background auto-load must prepare the stored HF token before its GGUF
metadata preflight.

The Hub rejects an invalid Authorization header with 401 even for a PUBLIC
repo. ``fetchGgufStagedMetadata`` posts to the same /api/inference/validate
endpoint ``validateModel`` uses, and ``parseJsonOrThrow`` turns a non-OK
response into a throw. In ``loadAutoLoadCandidate`` that preflight runs BEFORE
``validateModel``, and every call site of ``loadAutoLoadCandidate`` is wrapped
in ``catch { hadNonTrustFailure = true; continue; }``. So a stale saved token
made auto-load skip a cached model that would have loaded anonymously, without
ever reaching validateModel's "continue anonymously / replace token" recovery.

The real classification block is sliced verbatim out of chat-adapter.ts and run
under node, so this asserts on the token value that actually reaches the
request rather than on the presence of a symbol.
"""

import json
import os
import shutil
import subprocess
import tempfile
import textwrap
from pathlib import Path

import pytest

WORKDIR = Path(__file__).resolve().parents[2]


def _source_path(relative_path: str) -> Path:
    direct = WORKDIR / relative_path
    if direct.exists():
        return direct
    return WORKDIR / "unsloth_repo" / relative_path


ADAPTER = _source_path("studio/frontend/src/features/chat/api/chat-adapter.ts")
TEMP = WORKDIR / "temp" / "autoload_hf_token_preflight"


def _require_node():
    if shutil.which("node") is None:
        pytest.skip("node not available")
    if not ADAPTER.exists():
        pytest.skip("studio chat sources not present")
    result = subprocess.run(
        ["node", "--experimental-strip-types", "--version"],
        capture_output = True,
        text = True,
        timeout = 5,
    )
    if result.returncode != 0:
        pytest.skip("node --experimental-strip-types not available")


def _classification_slice() -> str:
    """The verbatim `isDiffusion` classification block from loadAutoLoadCandidate.

    Anchored on the declaration and on the `effectiveGpuIds` statement that
    consumes it, so the slice tracks either the prepared-token form or the
    older raw-token ternary.
    """
    src = ADAPTER.read_text(encoding = "utf-8")
    anchor = src.index("async function loadAutoLoadCandidate(")
    starts = [
        pos
        for pos in (
            src.find("let isDiffusion", anchor),
            src.find("const isDiffusion", anchor),
        )
        if pos != -1
    ]
    assert starts, "could not locate the isDiffusion classification block"
    start = min(starts)
    end = src.index("const effectiveGpuIds", start)
    return src[start:end].rstrip()


def _run(script: str, harness: str):
    _require_node()
    TEMP.mkdir(parents = True, exist_ok = True)
    workdir = Path(tempfile.mkdtemp(prefix = "run", dir = TEMP))
    (workdir / "harness.ts").write_text(harness, encoding = "utf-8")
    (workdir / "run.mts").write_text(script, encoding = "utf-8")
    env = dict(os.environ, NODE_NO_WARNINGS = "1")
    result = subprocess.run(
        ["node", "--experimental-strip-types", "--no-warnings", "run.mts"],
        cwd = str(workdir),
        capture_output = True,
        text = True,
        timeout = 30,
        env = env,
    )
    assert result.returncode == 0, f"stderr: {result.stderr}\nstdout: {result.stdout}"
    last = [line for line in result.stdout.strip().splitlines() if line.strip()][-1]
    return json.loads(last)


_HARNESS_TEMPLATE = """\
// Real classification block, sliced verbatim from chat-adapter.ts.
export async function classify(ctx: any) {{
  const {{
    candidate,
    config,
    modelPath,
    hfToken,
    prepareHfTokenForUse,
    fetchGgufStagedMetadata,
  }} = ctx;
{slice}
  return isDiffusion;
}}
"""


_STALE_TOKEN = "hf_staleTokenFromAnEarlierSession"


def _harness() -> str:
    return _HARNESS_TEMPLATE.format(slice = textwrap.indent(_classification_slice(), "  "))


_SCRIPT = textwrap.dedent(
    """
    import { classify } from "./harness.ts";

    const sent: Array<string | null> = [];

    // Mirrors prepareHfTokenForUse: an invalid stored token, with the user's
    // one-shot "continue anonymously" choice, resolves to a null token.
    const prepareHfTokenForUse = async (token: string | null) => {
      if (!token) return { proceed: true, token: null };
      return { proceed: true, token: null };
    };

    // Mirrors the Hub via /api/inference/validate + parseJsonOrThrow: any
    // non-null Authorization value here is the stale token, and the Hub 401s
    // on it even though the repo is public.
    const fetchGgufStagedMetadata = async (payload: any) => {
      sent.push(payload.hf_token ?? null);
      if (payload.hf_token != null) {
        throw new Error("401 Unauthorized: Invalid credentials in Authorization header");
      }
      return { isDiffusion: true };
    };

    let threw: string | null = null;
    let isDiffusion: boolean | null = null;
    try {
      isDiffusion = await classify({
        candidate: { kind: "gguf", ggufVariant: "Q4_K_M" },
        config: { selectedGpuIds: [0] },
        modelPath: "unsloth/some-public-gguf",
        hfToken: %s,
        prepareHfTokenForUse,
        fetchGgufStagedMetadata,
      });
    } catch (e) {
      threw = String((e as Error).message);
    }

    console.log(JSON.stringify({ sent, threw, isDiffusion }));
    """
)


def test_autoload_preflight_sends_the_prepared_token_not_the_stale_one():
    out = _run(_SCRIPT % json.dumps(_STALE_TOKEN), _harness())
    assert out["threw"] is None, (
        "a stale saved token aborted the auto-load metadata preflight; the "
        f"candidate would be skipped: {out['threw']}"
    )
    assert out["sent"] == [None], (
        "the GGUF metadata preflight must send the prepared token, not the raw "
        f"stored one; it sent {out['sent']!r}"
    )
    assert out["isDiffusion"] is True


def test_autoload_preflight_is_skipped_without_a_remembered_gpu_pick():
    """No remembered GPU selection means no preflight and no token use at all."""
    script = _SCRIPT % json.dumps(_STALE_TOKEN)
    script = script.replace("selectedGpuIds: [0]", "selectedGpuIds: null")
    out = _run(script, _harness())
    assert out["sent"] == []
    assert out["threw"] is None
    assert out["isDiffusion"] is False
