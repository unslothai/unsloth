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
    # U+2060 WORD JOINER, matching the backend's _THINK_NEUTRAL_ZW: U+200B is
    # line-break class ZW and would let a neutralized tag wrap mid-tag (#7334).
    assert "\\u2060" in src or "\u2060" in src
    assert "\\u200b" not in src and "\u200b" not in src
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
  createScanResumeCache,
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
  // A quote escaped inside a string literal is not a delimiter either (#7334).
  escaped_quoted:
    '<think>He wrote "use \\\\"</think>\\\\" here" and continued</think>Answer',
  closed_fence_literal: "<think>see ```\\n</think>\\n``` example</think>real answer",
  unclosed_fence: "<think>unclosed ```python\\n</think>\\nthe answer",
  // Unclosed reasoning fence + a fenced code block in the ANSWER (#7334).
  answer_fence: "<think>draft ```</think>Answer: ```js\\nconst a = 1;\\n```\\ndone",
  // A quoted mention pairs delimiter RUNS of EQUAL length, so a 1-backtick
  // flank against a 3-backtick one is no span: that ``` opens the ANSWER's
  // fence and the tag was the structural close (#7334).
  unequal_runs: "<think>Use a code fence: `</think>```python\\nprint(1)\\n```",
  // Well-formed markdown reaches an ODD raw backtick count through a
  // nested-backtick code span, so raw parity alone must not decide (#7334).
  nested_backtick_span: "<think>Use ``a ` b``</think>```python\\nprint(1)\\n```",
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

// A resumed scan (`resume`) must answer exactly like a cold one (no `resume`)
// on every delta of every chunking, or it silently reintroduces #7066 by
// skipping a close tag it decided about too early (#7334).
const RESUME_CASES = [
  cases.quoted_literal,
  cases.mismatched_flanks,
  cases.contraction_quoted,
  cases.escaped_quoted,
  cases.closed_fence_literal,
  cases.unclosed_fence,
  cases.answer_fence,
  cases.unequal_runs,
  cases.nested_backtick_span,
  cases.literal_only,
  "<think>a `</think>` b `</think>` c </think>done",
  "<think>```\\n</think>\\n```\\n```\\n</think>\\n```\\ntail</think>answer",
  '<think>mixed `</think>" and "</think>` then </think>visible',
  // A close whose literal verdict needs the char AFTER the trailing quote: the
  // word char here makes the quote an answer opener, not a closing flank, so
  // the tag is structural. A delta ending exactly on that quote leaves the
  // verdict unsettled and must stay re-readable next delta (#7334).
  '<think>reason "</think>"Answer',
  '<think>he wrote \\\\"</think>\\\\" and then ```\\n</think>\\n``` </think>ok',
];
const resumeMismatches = [];
function checkResume(raw, cuts, cache, label) {
  // Stable identity, as the adapter's is: a fresh arrow per delta would miss
  // the slot and hide the very thing under test.
  const known = () => false;
  const warm = { streaming: true, isKnownClose: known, resume: cache };
  // No `resume` means a fresh slot per call, i.e. the full O(buffer) scan.
  const cold = { streaming: true, isKnownClose: known };
  for (const end of cuts) {
    const cum = raw.slice(0, end);
    const got = [
      JSON.stringify(parseAssistantContent(cum, warm)),
      hasClosedThinkTag(cum, warm),
      structuralThinkCloseIndex(cum, warm),
    ];
    const want = [
      JSON.stringify(parseAssistantContent(cum, cold)),
      hasClosedThinkTag(cum, cold),
      structuralThinkCloseIndex(cum, cold),
    ];
    for (let i = 0; i < got.length; i++) {
      if (got[i] !== want[i]) {
        resumeMismatches.push(`${label} end=${end} #${i}: ${got[i]} != ${want[i]}`);
      }
    }
  }
}
for (const raw of RESUME_CASES) {
  for (const step of [1, 3, 8, 9, raw.length]) {
    const cuts = [];
    for (let end = step; end < raw.length; end += step) cuts.push(end);
    cuts.push(raw.length);
    checkResume(raw, cuts, createScanResumeCache(), `step=${step}`);
  }
  // Truncation at the end is the one non-append the cache detects itself, so
  // the same cache must survive it (chat-adapter trims a trailing `${...}`).
  const shared = createScanResumeCache();
  const half = Math.max(1, raw.length >> 1);
  checkResume(raw, [half, raw.length, half - 1, raw.length], shared, "truncate");
}

// The adapter keeps ONE onDeferredClose reference per stream, so the resumed
// scan reports a candidate on the delta it first reaches it and not again.
// Timing the thought off that first report must be unaffected (#7334).
const FIRE_DELTAS = [
  "<think>reasoning ```\\n",
  "</think>\\n",
  "still inside the fence, ",
  "</think>\\n",
  "more reasoning ",
  "and the answer follows",
];
function replayDeferred(deltas, useCache) {
  const resume = useCache ? createScanResumeCache() : undefined;
  const perStep = [];
  const firstAt = {};
  let cum = "";
  let step = 0;
  const record = (index) => {
    perStep[step].push(index);
    if (!(index in firstAt)) firstAt[index] = step;
  };
  const opts = { streaming: true, onDeferredClose: record, resume };
  for (const delta of deltas) {
    cum += delta;
    perStep.push([]);
    hasClosedThinkTag(cum, opts);
    step += 1;
  }
  return { perStep, firstAt, total: perStep.reduce((n, s) => n + s.length, 0) };
}
const firing = {
  warm: replayDeferred(FIRE_DELTAS, true),
  cold: replayDeferred(FIRE_DELTAS, false),
};

// Providers emit `</think>` as one token, so a quoted mention arrives as
// `... "` / `</think>` / `" ...` and the middle delta ends EXACTLY on the tag.
// The absent trailing flank is not an empty one: calling the mention structural
// for that one delta makes chat-adapter latch reasoningDuration off it, and it
// never lowers a nonzero value, so the thought time stops at the mention
// (#7334). Defer instead, and report the candidate for the final parse.
const QUOTE_SPLIT_DELTAS = ['<think>echo "', "</think>", '" here', " still thinking"];
const quoteSplitDeferred = [];
const quoteSplit = { closed: [] };
let qsCum = "";
for (const delta of QUOTE_SPLIT_DELTAS) {
  qsCum += delta;
  quoteSplit.closed.push(
    hasClosedThinkTag(qsCum, {
      streaming: true,
      onDeferredClose: (index) => quoteSplitDeferred.push(index),
    }),
  );
}
quoteSplit.deferred = quoteSplitDeferred;
quoteSplit.finalClosed = hasClosedThinkTag(qsCum);
quoteSplit.finalTypes = parseAssistantContent(qsCum).map((part) => part.type);
// The same deferral must still resolve STRUCTURAL as soon as the flank shows
// the quote opens the ANSWER, or the visible answer never leaves the drawer.
const ANSWER_SPLIT_DELTAS = ['<think>reason "', "</think>", '"Answer'];
const answerSplit = { closed: [] };
let asCum = "";
for (const delta of ANSWER_SPLIT_DELTAS) {
  asCum += delta;
  answerSplit.closed.push(hasClosedThinkTag(asCum, { streaming: true }));
}
answerSplit.finalIndex = structuralThinkCloseIndex(asCum);
answerSplit.finalParts = parseAssistantContent(asCum);
// A reasoning block that simply ENDS on `"</think>` has no more deltas coming,
// so the final parse still falls back to structural.
const quoteAtEof = {
  index: structuralThinkCloseIndex('<think>reason "</think>'),
  streamingClosed: hasClosedThinkTag('<think>reason "</think>', { streaming: true }),
};

const streaming = {
  streamClosed,
  streamTypes,
  streamFinal,
  unclosedStreaming,
  synthetic,
  deferred,
  resumeMismatches,
  firing,
  quoteSplit,
  answerSplit,
  quoteAtEof,
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
  // The space after the closing quote is what makes each of these a prose
  // MENTION, which is what this span is built to hold. A closing quote running
  // straight into the next word is instead the answer's own opening quote, so
  // without the separator the first one is the structural close and the span
  // collapses to nothing (#7334).
  for (let i = 0; i < nLit; i++) s += words(Math.max(0, chunk - 11)) + '"</think>" ';
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
// Replaying a whole stream: without `resume` every delta re-walks the buffer,
// which is the O(n^2) #7334 is about.
function replayStream(raw, useCache) {
  const opts = { streaming: true, resume: useCache ? createScanResumeCache() : undefined };
  let n = 0;
  for (let end = 4; end <= raw.length; end += 4) {
    n += parseAssistantContent(raw.slice(0, end), opts).length;
  }
  return n;
}
function timeMsFew(fn) {
  fn();
  let best = Infinity;
  for (let i = 0; i < 3; i++) {
    const t0 = process.hrtime.bigint();
    fn();
    best = Math.min(best, Number(process.hrtime.bigint() - t0) / 1e6);
  }
  return best;
}
const streamRaw = `<think>${span(200)}${span(200)}`;
const perf = {
  clean_us: timeUs(() => parseAssistantContent(clean)),
  many_us: timeUs(() => parseAssistantContent(many)),
  fenced_us: timeUs(() => parseAssistantContent(fenced)),
  stream_cached_ms: timeMsFew(() => replayStream(streamRaw, true)),
  stream_cold_ms: timeMsFew(() => replayStream(streamRaw, false)),
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

    # Mismatched flanks are no quoted mention: an odd backtick count before the
    # tag and a double quote after it hid the whole answer in the drawer (#7334).
    assert parsed["mismatched_flanks"] == [
        {"type": "reasoning", "text": "I'll answer with `"},
        {"type": "text", "text": '"yes" is the answer'},
    ]
    assert closed["mismatched_flanks"] is True

    # A contraction before a single-quoted mention must not flip the parity, or
    # the quoted tag reads as the block end and leaks the thought (#7334).
    assert parsed["contraction_quoted"] == [
        {"type": "reasoning", "text": "It's discussing '</think>' here"},
        {"type": "text", "text": "answer"},
    ]
    assert closed["contraction_quoted"] is True

    # Escaped quotes belong to the string literal around them, so the mention
    # they wrap stays reasoning and the bare tag after it ends the block (#7334).
    assert parsed["escaped_quoted"] == [
        {
            "type": "reasoning",
            "text": 'He wrote "use \\"</think>\\" here" and continued',
        },
        {"type": "text", "text": "Answer"},
    ]
    assert closed["escaped_quoted"] is True

    # A literal mention alone never closes the block (reasoning timer stays live).
    assert [part["type"] for part in parsed["literal_only"]] == ["reasoning"]
    assert closed["literal_only"] is False

    # A ``` in the visible ANSWER does not prove a reasoning-side fence closed:
    # taking it as such made the genuine close look literal (#7334).
    assert parsed["answer_fence"] == [
        {"type": "reasoning", "text": "draft ```"},
        {"type": "text", "text": "Answer: ```js\nconst a = 1;\n```\ndone"},
    ]
    assert closed["answer_fence"] is True

    # Matching flanks are not enough: a mention pairs delimiter RUNS of EQUAL
    # length (CommonMark), so a 1-backtick flank against the answer's 3-backtick
    # fence is no span, though raw parity called it a mention (#7334).
    assert parsed["unequal_runs"] == [
        {"type": "reasoning", "text": "Use a code fence: `"},
        {"type": "text", "text": "```python\nprint(1)\n```"},
    ]
    assert closed["unequal_runs"] is True

    # Same rule from well-formed markdown: ``a ` b`` is a legal nested span
    # whose 5 backticks make parity odd, which alone swallowed the answer (#7334).
    assert parsed["nested_backtick_span"] == [
        {"type": "reasoning", "text": "Use ``a ` b``"},
        {"type": "text", "text": "```python\nprint(1)\n```"},
    ]
    assert closed["nested_backtick_span"] is True


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


def test_mid_stream_quoted_close_waits_for_its_trailing_flank(tmp_path):
    """A close tag ending the delta must not read as the block end.

    `</think>` is one token for every provider, so a quoted mention arrives as
    `... "` / `</think>` / `" ...` and the middle delta stops exactly on the
    tag. Reading the flank that has not arrived as "not a quote" called the
    mention structural for that one delta; `chat-adapter` latches
    `reasoningDuration` from `hasClosedThinkTag` behind a `!reasoningDuration`
    guard and never lowers a nonzero value, so the reported thinking time
    excluded every second of reasoning after the mention (#7334). The backend
    extractor holds the same buffer (`_should_hold_quoted_think_close`).
    """
    streaming = _run_parse_harness(tmp_path)["streaming"]

    # No delta of a quoted mention ever reads as closed, and the deferred
    # candidate is reported so the adapter can time the thought from it.
    assert streaming["quoteSplit"]["closed"] == [False, False, False, False]
    assert streaming["quoteSplit"]["deferred"] == [len('<think>echo "')]
    assert streaming["quoteSplit"]["finalClosed"] is False
    assert streaming["quoteSplit"]["finalTypes"] == ["reasoning"]

    # Deferring is not swallowing: the delta that reveals the quote opening the
    # ANSWER still reclassifies the tag as structural, so the answer streams.
    assert streaming["answerSplit"]["closed"] == [False, False, True]
    assert streaming["answerSplit"]["finalIndex"] == len('<think>reason "')
    assert streaming["answerSplit"]["finalParts"] == [
        {"type": "reasoning", "text": 'reason "'},
        {"type": "text", "text": '"Answer'},
    ]

    # And a stream that simply ends on the tag falls back to structural.
    assert streaming["quoteAtEof"]["index"] == len('<think>reason "')
    assert streaming["quoteAtEof"]["streamingClosed"] is False


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

    # This replay passes no `resume` cache, so every delta rescans from the top
    # and re-reports; either way the offset is the real close.
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


def test_streaming_resume_matches_a_cold_scan(tmp_path):
    """A resumed scan must answer exactly like a full rescan, every delta.

    The scan carries fence and quote cursors across SSE deltas so a delta costs
    O(new text) instead of O(buffer) (#7334). Resuming past a tag whose verdict
    the inspected prefix does not settle would skip the real close and put the
    visible answer back in the thinking drawer, i.e. #7066 again.
    """
    streaming = _run_parse_harness(tmp_path)["streaming"]
    assert streaming["resumeMismatches"] == []


def test_deferred_close_first_report_is_unchanged_by_resume(tmp_path):
    """Resuming drops repeat reports, never the FIRST one.

    `chat-adapter` records the arrival instant of a deferred close the first
    time it hears about it, so only the first report per index is observable.
    A resumed scan reports each candidate once, on the same delta a cold scan
    first reports it, which is when the tag arrived (#7334).
    """
    firing = _run_parse_harness(tmp_path)["streaming"]["firing"]
    warm, cold = firing["warm"], firing["cold"]

    # The observable part: same offsets, first seen on the same delta.
    assert warm["firstAt"] == cold["firstAt"]
    fence_open = "<think>reasoning ```\n"
    second = fence_open + "</think>\n" + "still inside the fence, "
    assert warm["firstAt"] == {str(len(fence_open)): 1, str(len(second)): 3}

    # ... while the repeats are gone: each candidate is reported exactly once.
    reported = [index for step in warm["perStep"] for index in step]
    assert sorted(reported) == sorted(set(reported))
    assert warm["total"] == 2
    assert cold["total"] > warm["total"]


def test_chat_adapter_times_reasoning_from_the_deferred_close(tmp_path):
    """The adapter must record deferred offsets and read them back at finalize."""
    src = ADAPTER_TS.read_text(encoding = "utf-8")
    assert "deferredCloseTimes" in src
    assert "onDeferredClose" in src
    assert "structuralThinkCloseIndex" in src
    # The end-of-stream fallback must prefer the confirmed deferred instant.
    assert "closedAt - reasoningStartAt" in src
    # The timer starts when raw reasoning arrives, not when the holdback emits:
    # a first delta that is only a marker prefix emits nothing (#7334).
    start_at = src.index("if (reasoning) {")
    assert "reasoningStartAt = Date.now();" in src[start_at : start_at + 600]
    assert src.index("reasoningMarkupBuffer += reasoning;") > src.index(
        "reasoningStartAt = Date.now();", start_at
    )


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
    assert "`<think>${neutralizeThinkMarkup(thinking)}</think>`" not in src


def test_parse_assistant_content_literal_scan_is_single_pass(tmp_path):
    """200 literal mentions in an 8k reasoning span must stay within a small
    multiple of the clean parse; restarting the quote scan per candidate was
    ~6000x and ran on every SSE delta (#7334)."""
    perf = _run_parse_harness(tmp_path)["perf"]
    ratio = perf["many_us"] / perf["clean_us"]
    assert ratio < 500, f"many {perf['many_us']:.1f}us vs clean {perf['clean_us']:.3f}us"
    # 200 FENCED literals share one open fence and one "is there a later close
    # tag" answer; memoizing it keeps the parse near linear (~7x the clean
    # control, vs ~17x re-scanning and worse as the trailing span grows).
    fenced_ratio = perf["fenced_us"] / perf["clean_us"]
    assert fenced_ratio < 60, f"fenced {perf['fenced_us']:.1f}us vs clean {perf['clean_us']:.3f}us"


def test_streaming_replay_is_not_quadratic(tmp_path):
    """Replaying a stream must cost O(text), not O(text) per delta.

    Without the resume cache every SSE delta re-walks the whole cumulative
    buffer, so streaming a 16k reasoning span holding 400 literal mentions cost
    ~50x what resuming does. The bound is loose because CI timing is noisy; the
    real gap is one to two orders of magnitude (#7334).
    """
    perf = _run_parse_harness(tmp_path)["perf"]
    ratio = perf["stream_cold_ms"] / max(perf["stream_cached_ms"], 1e-6)
    assert (
        ratio > 4
    ), f"cached {perf['stream_cached_ms']:.1f}ms vs cold {perf['stream_cold_ms']:.1f}ms"


def test_chat_adapter_resume_caches_are_per_stream(tmp_path):
    """The caches must be minted per stream, and their keys must be stable.

    A cache is only valid while the buffer it scans grows by appending, so it
    belongs to one stream; and the slot is keyed on the callbacks by identity,
    so a fresh arrow per delta would silently disable the resume (#7334).
    """
    src = ADAPTER_TS.read_text(encoding = "utf-8")
    assert "createScanResumeCache" in src
    assert "resume: pollResume" in src
    assert "resume: buildResume" in src
    # Two call sites, both inside the per-stream scope.
    assert src.count("createScanResumeCache()") == 2
    assert src.index("createScanResumeCache()") > src.index('let cumulativeText = "";')
    # The callbacks the slot is keyed on are hoisted, not rebuilt per delta.
    assert "const knownCloseAt = " in src
    assert "isKnownClose: (index) =>" not in src
    assert "onDeferredClose: (index) =>" not in src
