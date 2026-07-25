# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Frontend contract for #7066 think-markup neutralization."""

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
PARSE_TS = REPO / "studio/frontend/src/features/chat/utils/parse-assistant-content.ts"
ADAPTER_TS = REPO / "studio/frontend/src/features/chat/api/chat-adapter.ts"


def test_frontend_exports_neutralize_think_markup():
    src = PARSE_TS.read_text(encoding = "utf-8")
    assert "export function neutralizeThinkMarkup" in src
    assert "export function drainThinkMarkupBuffer" in src
    assert "\\u200b" in src or "\u200b" in src
    assert "#7066" in src


def test_chat_adapter_neutralizes_reasoning_before_think_wrap():
    src = ADAPTER_TS.read_text(encoding = "utf-8")
    assert "drainThinkMarkupBuffer" in src
    assert "reasoningMarkupBuffer" in src
    assert "safeReasoning" in src
    # Mixed reasoning/content chunks must not drop delta when reasoning is held.
    assert "if (!safeReasoning) {\n                  continue;" not in src
    assert "`<think>${emit}`" in src


_HARNESS = """
import { parseAssistantContent, hasClosedThinkTag } from "__PARSE_TS__";

const cases = {
  quoted_literal: '<think>user wrote "</think>" here</think>answer',
  closed_fence_literal: "<think>see ```\\n</think>\\n``` example</think>real answer",
  unclosed_fence: "<think>unclosed ```python\\n</think>\\nthe answer",
  literal_only: '<think>only a "</think>" mention, still thinking',
};
const parsed = {};
const closed = {};
for (const [name, raw] of Object.entries(cases)) {
  parsed[name] = parseAssistantContent(raw);
  closed[name] = hasClosedThinkTag(raw);
}

// Perf guard for #7334: literal mentions must not make the parse super-linear.
const LOREM = "reasoning about the training loop in some detail. ";
function words(n) {
  let s = "";
  while (s.length < n) s += LOREM;
  return s.slice(0, n);
}
function span(nLit) {
  if (nLit === 0) return words(8000);
  const chunk = Math.floor(8000 / nLit);
  let s = "";
  for (let i = 0; i < nLit; i++) s += words(Math.max(0, chunk - 10)) + '"</think>"';
  return s;
}
function timeUs(fn) {
  for (let i = 0; i < 50; i++) fn();
  const t0 = process.hrtime.bigint();
  for (let i = 0; i < 200; i++) fn();
  return Number(process.hrtime.bigint() - t0) / 200 / 1000;
}
const clean = `<think>${span(0)}</think>${words(4000)}`;
const many = `<think>${span(200)}</think>${words(4000)}`;
const perf = {
  clean_us: timeUs(() => parseAssistantContent(clean)),
  many_us: timeUs(() => parseAssistantContent(many)),
};
console.log(JSON.stringify({ parsed, closed, perf }));
"""


def _run_parse_harness(tmp_path):
    if shutil.which("node") is None:
        pytest.skip("node not available")
    probe = subprocess.run(
        ["node", "--experimental-strip-types", "--version"],
        capture_output = True,
        text = True,
        timeout = 30,
    )
    if probe.returncode != 0:
        pytest.skip("node --experimental-strip-types not available")
    script = tmp_path / "run.mts"
    script.write_text(_HARNESS.replace("__PARSE_TS__", PARSE_TS.as_posix()), encoding = "utf-8")
    result = subprocess.run(
        ["node", "--experimental-strip-types", "--no-warnings", "run.mts"],
        cwd = str(tmp_path),
        capture_output = True,
        text = True,
        timeout = 300,
        env = dict(os.environ, NODE_NO_WARNINGS = "1"),
    )
    assert result.returncode == 0, f"stderr: {result.stderr}\nstdout: {result.stdout}"
    return json.loads(result.stdout.strip().splitlines()[-1])


def test_parse_assistant_content_literal_close_semantics(tmp_path):
    """Literal vs structural `</think>` classification, end to end (#7066, #7334)."""
    out = _run_parse_harness(tmp_path)
    parsed, closed = out["parsed"], out["closed"]

    # A quoted mention stays inside the thinking block; the bare tag ends it.
    assert parsed["quoted_literal"] == [
        {"type": "reasoning", "text": 'user wrote "</think>" here'},
        {"type": "text", "text": "answer"},
    ]
    assert closed["quoted_literal"] is True

    # A tag inside a CLOSED ``` fence is a fenced example, not the block end.
    assert parsed["closed_fence_literal"][0]["type"] == "reasoning"
    assert "</think>" in parsed["closed_fence_literal"][0]["text"]
    assert parsed["closed_fence_literal"][-1] == {"type": "text", "text": "real answer"}

    # An UNCLOSED fence must not swallow the answer: fall back to structural.
    assert parsed["unclosed_fence"][-1]["type"] == "text"
    assert parsed["unclosed_fence"][-1]["text"].strip() == "the answer"
    assert closed["unclosed_fence"] is True

    # A literal mention alone never closes the block (reasoning timer stays live).
    assert [part["type"] for part in parsed["literal_only"]] == ["reasoning"]
    assert closed["literal_only"] is False


def test_parse_assistant_content_literal_scan_is_single_pass(tmp_path):
    """200 literal mentions in an 8k reasoning span must stay within a small
    multiple of the clean parse; restarting the quote scan per candidate was
    ~6000x and ran on every SSE delta (#7334)."""
    perf = _run_parse_harness(tmp_path)["perf"]
    ratio = perf["many_us"] / perf["clean_us"]
    assert ratio < 500, f"many {perf['many_us']:.1f}us vs clean {perf['clean_us']:.3f}us"
