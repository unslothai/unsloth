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
import {
  parseAssistantContent,
  hasClosedThinkTag,
  structuralThinkCloseIndex,
} from "__PARSE_TS__";

const cases = {
  quoted_literal: '<think>user wrote "</think>" here</think>answer',
  // Mismatched flanks are not a quote span (#7334).
  mismatched_flanks: '<think>I\\'ll answer with `</think>"yes" is the answer',
  // The apostrophe in "It's" is punctuation, not an opening quote (#7334).
  contraction_quoted: "<think>It's discussing '</think>' here</think>answer",
  closed_fence_literal: "<think>see ```\\n</think>\\n``` example</think>real answer",
  unclosed_fence: "<think>unclosed ```python\\n</think>\\nthe answer",
  // Unclosed reasoning fence + a fenced code block in the ANSWER (#7334).
  answer_fence: "<think>draft ```</think>Answer: ```js\\nconst a = 1;\\n```\\ndone",
  literal_only: '<think>only a "</think>" mention, still thinking',
};
const parsed = {};
const closed = {};
for (const [name, raw] of Object.entries(cases)) {
  parsed[name] = parseAssistantContent(raw);
  closed[name] = hasClosedThinkTag(raw);
}

// Mid-stream the enclosing ``` fence may still close in a later delta, so the
// classification of a tag inside it must not flip-flop (#7334).
const fenceDeltas = [
  "<think>marker:\\n",
  "```text\\n",
  "</think>\\n",
  "```\\n",
  "so it is literal.</think>",
  "the answer",
];
const streamClosed = [];
const streamTypes = [];
let cum = "";
for (const delta of fenceDeltas) {
  cum += delta;
  streamClosed.push(hasClosedThinkTag(cum, { streaming: true }));
  streamTypes.push(
    parseAssistantContent(cum, { streaming: true })
      .map((part) => part.type)
      .join("+"),
  );
}
const streamFinal = parseAssistantContent(cum);
const unclosedStreaming = {
  closed: hasClosedThinkTag(cases.unclosed_fence, { streaming: true }),
  types: parseAssistantContent(cases.unclosed_fence, { streaming: true }).map(
    (part) => part.type,
  ),
};
// A delimiter the ADAPTER inserted itself (closing a synthetic
// reasoning_content wrapper) is a known boundary, not an inferred one, so the
// raw-marker deferral must not apply to it (#7334).
const syntheticRaw = "<think>draft ```</think>The answer. See ```js\\ncode\\n```";
const syntheticAt = "<think>draft ```".length;
const isKnownClose = (index) => index === syntheticAt;
const synthetic = {
  known: parseAssistantContent(syntheticRaw, { streaming: true, isKnownClose }),
  knownClosed: hasClosedThinkTag(syntheticRaw, { streaming: true, isKnownClose }),
  rawMarker: parseAssistantContent(syntheticRaw, { streaming: true }).map(
    (part) => part.type,
  ),
};
// A close deferred mid-stream must still be REPORTED, so the adapter can time
// the thought at the instant it arrived instead of at end of stream (#7334).
const deferDeltas = ["<think>draft ```", "</think>", "long answer"];
const deferredSeen = [];
let deferCum = "";
for (const delta of deferDeltas) {
  deferCum += delta;
  hasClosedThinkTag(deferCum, {
    streaming: true,
    onDeferredClose: (index) => deferredSeen.push(index),
  });
}
const deferred = {
  seen: deferredSeen,
  confirmed: structuralThinkCloseIndex(deferCum),
  closedWhileStreaming: hasClosedThinkTag(deferCum, { streaming: true }),
  // A close that is genuinely literal is deferred too, and the final parse
  // then does NOT confirm it, so its timestamp must go unused.
  literalConfirmed: structuralThinkCloseIndex(cases.closed_fence_literal),
  literalFirstDeferred: (() => {
    const seen = [];
    hasClosedThinkTag(cases.closed_fence_literal, {
      streaming: true,
      onDeferredClose: (index) => seen.push(index),
    });
    return seen[0] ?? -1;
  })(),
};

const streaming = {
  streamClosed,
  streamTypes,
  streamFinal,
  unclosedStreaming,
  synthetic,
  deferred,
};

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
function fencedSpan(nLit) {
  // One open fence holding nLit literal close tags, then the fence closes and a
  // long stretch runs before the real close. Every literal takes the odd-fence
  // branch with the SAME "next close tag" answer, so an unmemoized look-ahead
  // rescans that stretch nLit times.
  let s = "```\\n";
  for (let i = 0; i < nLit; i++) s += "</think>\\n" + words(30);
  return s + "```\\n" + words(8000);
}
const clean = `<think>${span(0)}</think>${words(4000)}`;
const many = `<think>${span(200)}</think>${words(4000)}`;
const fenced = `<think>${fencedSpan(200)}</think>${words(4000)}`;
const perf = {
  clean_us: timeUs(() => parseAssistantContent(clean)),
  many_us: timeUs(() => parseAssistantContent(many)),
  fenced_us: timeUs(() => parseAssistantContent(fenced)),
};
console.log(JSON.stringify({ parsed, closed, perf, streaming }));
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

    # Mismatched flanks are not a quoted mention: an odd backtick count before
    # the tag and a double quote after it used to read as a quote span, which
    # hid the entire visible answer in the thinking drawer (#7334).
    assert parsed["mismatched_flanks"] == [
        {"type": "reasoning", "text": "I'll answer with `"},
        {"type": "text", "text": '"yes" is the answer'},
    ]
    assert closed["mismatched_flanks"] is True

    # A contraction before a single-quoted mention must not flip the parity:
    # counting it read the quoted tag as the block end and leaked the rest of
    # the thought into the visible answer (#7334).
    assert parsed["contraction_quoted"] == [
        {"type": "reasoning", "text": "It's discussing '</think>' here"},
        {"type": "text", "text": "answer"},
    ]
    assert closed["contraction_quoted"] is True

    # A literal mention alone never closes the block (reasoning timer stays live).
    assert [part["type"] for part in parsed["literal_only"]] == ["reasoning"]
    assert closed["literal_only"] is False

    # A ``` in the visible ANSWER is not proof that a reasoning-side fence
    # closed: taking it as such made the genuine close look literal and hid the
    # entire answer inside the thinking drawer (#7334).
    assert parsed["answer_fence"] == [
        {"type": "reasoning", "text": "draft ```"},
        {"type": "text", "text": "Answer: ```js\nconst a = 1;\n```\ndone"},
    ]
    assert closed["answer_fence"] is True


def test_mid_stream_unclosed_fence_decision_is_deferred(tmp_path):
    """A tag inside a not-yet-closed ``` fence must not read as the block end.

    Mid-stream ``</think>`` inside a fence that closes a delta later would
    otherwise be called structural, then reclassified as literal once the
    closing backticks arrive: the text bounces out of the thinking drawer and
    back, and `chat-adapter` latches `reasoningDuration` on a tag that was never
    the real close and never corrects it (#7334).
    """
    streaming = _run_parse_harness(tmp_path)["streaming"]

    # The real close is the 5th delta; nothing before it may read as closed.
    assert streaming["streamClosed"] == [False, False, False, False, True, True]
    # ... and no visible text part escapes the drawer before then.
    assert streaming["streamTypes"][:4] == ["reasoning"] * 4
    assert streaming["streamTypes"][-1] == "reasoning+text"

    # The completed stream keeps the fenced sample in reasoning and the answer visible.
    assert streaming["streamFinal"][0]["type"] == "reasoning"
    assert "</think>" in streaming["streamFinal"][0]["text"]
    assert streaming["streamFinal"][-1] == {"type": "text", "text": "the answer"}

    # A genuinely unclosed fence still defers mid-stream; the final parse (asserted
    # in the semantics test above) is what falls back to structural.
    assert streaming["unclosedStreaming"]["closed"] is False
    assert streaming["unclosedStreaming"]["types"] == ["reasoning"]


def test_known_synthetic_close_is_not_re_derived(tmp_path):
    """The adapter's own `</think>` must survive the streaming deferral.

    A provider can end structured reasoning_content inside an unfinished ```
    fence; `closeReasoningContent()` then appends a delimiter whose position is
    already known. Running the raw-marker fence heuristics over it kept every
    answer delta in the thinking drawer until the stream ended (#7334).
    """
    synthetic = _run_parse_harness(tmp_path)["streaming"]["synthetic"]

    assert synthetic["known"] == [
        {"type": "reasoning", "text": "draft ```"},
        {"type": "text", "text": "The answer. See ```js\ncode\n```"},
    ]
    assert synthetic["knownClosed"] is True
    # Without the known boundary the same shape is a RAW model marker, which
    # still defers mid-stream (the ambiguity the heuristics exist for).
    assert synthetic["rawMarker"] == ["reasoning"]


def test_deferred_close_is_reported_for_reasoning_timing(tmp_path):
    """A deferred close must be reported so the thought can be timed at it.

    ``<think>draft ```</think>long answer`` defers the close mid-stream and only
    resolves it as structural at the end, so `reasoningDuration` was measured to
    end of stream and counted the whole visible answer as thought time (#7334).
    """
    deferred = _run_parse_harness(tmp_path)["streaming"]["deferred"]

    # Reported every delta while held, always at the real close offset.
    close_at = len("<think>draft ```")
    assert deferred["seen"], "deferred close was never reported"
    assert set(deferred["seen"]) == {close_at}
    # ... and the final parse confirms exactly that offset, so its timestamp is
    # the one the adapter may use.
    assert deferred["confirmed"] == close_at
    # The deferral itself is unchanged: mid-stream this is still not closed.
    assert deferred["closedWhileStreaming"] is False

    # A genuinely literal close is reported too, but the final parse resolves a
    # LATER offset, so the recorded timestamp is never applied.
    assert deferred["literalFirstDeferred"] != -1
    assert deferred["literalConfirmed"] != deferred["literalFirstDeferred"]


def test_chat_adapter_times_reasoning_from_the_deferred_close(tmp_path):
    """The adapter must record deferred offsets and read them back at finalize."""
    src = ADAPTER_TS.read_text(encoding = "utf-8")
    assert "deferredCloseTimes" in src
    assert "onDeferredClose" in src
    assert "structuralThinkCloseIndex" in src
    # The end-of-stream fallback must prefer the confirmed deferred instant.
    assert "closedAt - reasoningStartAt" in src


def test_chat_adapter_marks_its_own_reasoning_close_as_known(tmp_path):
    """The adapter must record the offsets it inserts and pass them down."""
    src = ADAPTER_TS.read_text(encoding = "utf-8")
    assert "syntheticCloses" in src
    assert "syntheticCloses.add(cumulativeText.length)" in src
    assert "isKnownClose" in src


def test_structured_content_wrapper_closes_are_known(tmp_path):
    """The `<think>` wrapper around a structured thinking part is ours too.

    A provider streaming reasoning as a `delta.content` thinking part that ends
    inside an unfinished ``` fence had its inserted `</think>` re-derived by the
    raw-marker heuristics, keeping every answer delta in the drawer until the
    stream ended (#7334).
    """
    src = ADAPTER_TS.read_text(encoding = "utf-8")
    assert "closeOffsets" in src
    assert "syntheticCloses.add(cumulativeText.length + offset)" in src
    # The wrapper close must be emitted separately so its offset is recorded.
    assert '`<think>${neutralizeThinkMarkup(thinking)}</think>`' not in src


def test_parse_assistant_content_literal_scan_is_single_pass(tmp_path):
    """200 literal mentions in an 8k reasoning span must stay within a small
    multiple of the clean parse; restarting the quote scan per candidate was
    ~6000x and ran on every SSE delta (#7334)."""
    perf = _run_parse_harness(tmp_path)["perf"]
    ratio = perf["many_us"] / perf["clean_us"]
    assert ratio < 500, f"many {perf['many_us']:.1f}us vs clean {perf['clean_us']:.3f}us"
    # 200 FENCED literals sharing one open fence all take the odd-fence branch
    # with the same "is there a later close tag" answer; memoizing it keeps the
    # parse near linear (~7x the clean control, vs ~17x re-scanning and far
    # worse as the trailing span grows).
    fenced_ratio = perf["fenced_us"] / perf["clean_us"]
    assert fenced_ratio < 60, f"fenced {perf['fenced_us']:.1f}us vs clean {perf['clean_us']:.3f}us"
