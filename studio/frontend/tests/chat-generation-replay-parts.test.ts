// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// A recovery follower replays a run's stored chunk events. It used to fold them into ONE string and
// re-parse that string on every publish, which had two consequences: the work was quadratic in the
// length of the answer, and a tool-heavy reply had no string form at all -- its calls are PARTS, so
// the replay dropped them and a reopened message lost its pills and their output. These pin the
// accumulator that replaced the string: the same parts the live stream builds, in the order the run
// produced them, extended one event at a time.

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { fileURLToPath } from "node:url";

import { registerBundlerResolver } from "./helpers/kit.ts";

const read = (relative: string) =>
  readFileSync(new URL(relative, import.meta.url), "utf8");

registerBundlerResolver();

const { createRecoveryReplay, seededParkedApprovals } = await import(
  "../src/features/chat/utils/chat-generation-replay.ts"
);

const { parseAssistantContent } = await import(
  "../src/features/chat/utils/parse-assistant-content.ts"
);

const text = (t: string) => ({ type: "text", text: t });
const reasoning = (t: string) => ({ type: "reasoning", text: t });
const tool = (toolCallId: string, toolName: string, extra: object = {}) => ({
  type: "tool-call",
  toolCallId,
  toolName,
  args: {},
  argsText: "{}",
  ...extra,
});

test("a seeded reply keeps the call between its two paragraphs", () => {
  // The flattened string cannot say this. `generationRawContent` could only ever hand back the two
  // runs of prose, and `parseAssistantContent` put them side by side.
  const replay = createRecoveryReplay([
    reasoning("It should sit between the paragraphs."),
    tool("call_0:uuid", "read_file", { result: "the file body" }),
    text("and here is what it said."),
  ]);
  const parts = replay.content() as Array<Record<string, unknown>>;
  assert.deepEqual(
    parts.map((part) => part.type),
    ["reasoning", "tool-call", "text"],
  );
  assert.equal(parts[1].toolCallId, "call_0:uuid");
  assert.equal(parts[1].result, "the file body");
});

test("a replayed tool_start lands where the call happened, not at the end", () => {
  const replay = createRecoveryReplay("");
  replay.applyChunk({ choices: [{ delta: { content: "before " } }] });
  replay.applyChunk({
    _toolEvent: {
      type: "tool_start",
      tool_call_id: "call_0",
      tool_name: "read_file",
      arguments: { path: "a.txt" },
      arguments_text: '{"path":"a.txt"}',
    },
  });
  replay.applyChunk({ choices: [{ delta: { content: "after" } }] });
  const parts = replay.content() as Array<Record<string, unknown>>;
  assert.deepEqual(parts.map((part) => part.type), ["text", "tool-call", "text"]);
  assert.equal(parts[1].toolName, "read_file");
  assert.deepEqual(parts[1].args, { path: "a.txt" });
  assert.equal(
    (parts[0] as { text: string }).text + (parts[2] as { text: string }).text,
    "before after",
    "the prose around a call must survive in order",
  );
});

test("tool_end carries the result onto the card its id names", () => {
  const replay = createRecoveryReplay("");
  replay.applyChunk({ choices: [{ delta: { content: "x" } }] });
  replay.applyChunk({
    _toolEvent: { type: "tool_start", tool_call_id: "call_0", tool_name: "bash", arguments: {} },
  });
  replay.applyChunk({ _toolEvent: { type: "tool_output", tool_call_id: "call_0", text: "line one\n" } });
  replay.applyChunk({
    _toolEvent: { type: "tool_end", tool_call_id: "call_0", result: "exit 0" },
  });
  const parts = replay.content() as Array<Record<string, unknown>>;
  const call = parts.find((part) => part.type === "tool-call")!;
  assert.equal(call.toolName, "bash");
  // The longer captured stream beats the status line, exactly as the live path decides.
  assert.match(String(call.result), /line one/);
  assert.match(String(call.result), /exit 0/);
});

test("index-keyed tool_call deltas accumulate into the ONE card they belong to", () => {
  const replay = createRecoveryReplay("");
  replay.applyChunk({
    choices: [
      {
        delta: {
          tool_calls: [
            { index: 0, function: { name: "read_file", arguments: "{\"a\":" } },
          ],
        },
      },
    ],
  });
  replay.applyChunk({
    choices: [{ delta: { tool_calls: [{ index: 0, function: { arguments: " 1}" } }] } }],
  });
  const parts = replay.content() as Array<Record<string, unknown>>;
  const calls = parts.filter((part) => part.type === "tool-call");
  assert.equal(calls.length, 1, "a continuing call must not open a second card");
  assert.deepEqual(calls[0].args, { a: 1 });
});

test("the backend's own spelling of a minted id reaches the SAME card", () => {
  // Providers stream `tool_calls` with no id; the backend mints `tool_call_<index>` for the slot
  // under the same rule the replay mints it, so its `tool_start` lands on the card the fragments
  // opened instead of opening a second one. A DIFFERENT id is a different call, and gets its own.
  const replay = createRecoveryReplay([]);
  replay.applyChunk({
    choices: [{ delta: { tool_calls: [{ index: 0, function: { name: "grep", arguments: "{}" } }] } }],
  });
  replay.applyChunk({
    _toolEvent: { type: "tool_start", tool_call_id: "tool_call_0", tool_name: "grep", arguments: {} },
  });
  let parts = replay.content() as Array<Record<string, unknown>>;
  assert.equal(parts.filter((part) => part.type === "tool-call").length, 1);
  assert.equal(parts.find((part) => part.type === "tool-call")?.toolCallId, "tool_call_0");

  replay.applyChunk({
    _toolEvent: { type: "tool_start", tool_call_id: "call_1", tool_name: "bash", arguments: {} },
  });
  parts = replay.content() as Array<Record<string, unknown>>;
  const calls = parts.filter((part) => part.type === "tool-call");
  assert.equal(calls.length, 2, "a second call is a second card, not a rename of the first");
  assert.equal(calls[1].toolName, "bash");
});

test("reasoning reopens only when the seed left it open", () => {
  const open = createRecoveryReplay([reasoning("the thought")]);
  open.applyChunk({ choices: [{ delta: { content: "the answer" } }] });
  const parts = open.content() as Array<Record<string, unknown>>;
  assert.deepEqual(parts.map((part) => part.type), ["reasoning", "text"]);
  assert.equal((parts[0] as { text: string }).text, "the thought");
  assert.equal((parts[1] as { text: string }).text, "the answer");

  const closed = createRecoveryReplay([text("plain")]);
  closed.applyChunk({ choices: [{ delta: { reasoning_content: "fresh thought" } }] });
  assert.deepEqual(
    (closed.content() as Array<{ type: string }>).map((part) => part.type),
    ["text", "reasoning"],
  );
});

test("a reasoning_summary frame is not content, and does not extend the reply", () => {
  const replay = createRecoveryReplay("answer");
  assert.equal(replay.applyChunk({ _reasoningDurationMs: 1200 }), false);
  assert.equal(replay.rawText(), "answer");
});

// The live stream closes an open thought before it records a call, so the block's close tag sits at the
// END of the thought run. A replay seeded from parts re-creates those tags from what storage holds, and
// a flag carried across the whole reply keeps closing a block a boundary already closed -- the stray
// `< /think>` then survives as literal text in the part after it. What the reader saw was an answer
// whose first word was a tag, doubled on every reopen.
test("a call landing mid-thought closes the block where the live stream closed it", () => {
  const replay = createRecoveryReplay([reasoning("the thought")]);
  replay.applyChunk({
    choices: [
      { delta: { tool_calls: [{ index: 0, function: { name: "bash", arguments: "{}" } }] } },
    ],
  });
  replay.applyChunk({ choices: [{ delta: { content: "the answer" } }] });
  const parts = replay.content() as Array<Record<string, unknown>>;
  assert.deepEqual(
    parts.map((part) => part.type),
    ["reasoning", "tool-call", "text"],
    "the call sits between the thought and the answer it produced",
  );
  assert.equal(
    (parts[0] as { text: string }).text,
    "the thought",
    "the close belongs to the end of the thought, never to the start of the answer",
  );
  assert.equal((parts[2] as { text: string }).text, "the answer");
});

test("reopening at any point of a run shows what a stream that never left shows", () => {
  // The reader must not be able to tell the tab was shut: replaying from what storage holds has to
  // yield exactly what the same frames would have produced had they arrived in one sitting.
  const frames = [
    { choices: [{ delta: { reasoning_content: "think A" } }] },
    {
      choices: [
        { delta: { tool_calls: [{ index: 0, function: { name: "read_file", arguments: "{}" } }] } },
      ],
    },
    { choices: [{ delta: { content: "the answer" } }] },
    { choices: [{ delta: { reasoning_content: "a second thought" } }] },
    { choices: [{ delta: { content: " and its tail" } }] },
  ];
  const neverLeft = createRecoveryReplay("");
  for (const frame of frames) neverLeft.applyChunk(frame);
  const expected = neverLeft.content() as Array<Record<string, unknown>>;
  assert.deepEqual(
    expected.map((part) => part.type),
    ["reasoning", "tool-call", "text", "reasoning", "text"],
  );
  for (let cut = 1; cut <= frames.length; cut++) {
    const seed = createRecoveryReplay("");
    for (const frame of frames.slice(0, cut)) seed.applyChunk(frame);
    const reopened = createRecoveryReplay(seed.content());
    for (const frame of frames.slice(cut)) reopened.applyChunk(frame);
    assert.deepEqual(
      reopened.content() as Array<Record<string, unknown>>,
      expected,
      `reopening after ${cut} of ${frames.length} frames must show the same reply`,
    );
  }
});

// A model that carries its OWN tags inside delta.content already classifies its own thought: the live
// stream appends such a delta verbatim and lets the tags inside it do the work. The replay closed the
// block it happened to sit in regardless of who opened it, so the first answer delta of a tagged run cut
// the thought mid-sentence and the model's own close tag survived as literal text in the answer part.
test("a run whose own tags carry the thought replays as the live stream read it", () => {
  const frames = [
    { choices: [{ delta: { content: "<think>the thought " } }] },
    { choices: [{ delta: { content: "still thinking</think>" } }] },
    { choices: [{ delta: { content: "the answer" } }] },
  ];
  // What the tab that never left renders: the buffer the live adapter appends to, parsed once.
  const live = parseAssistantContent(
    frames.map((frame) => frame.choices[0].delta.content as string).join(""),
  );
  const replay = createRecoveryReplay("");
  for (const frame of frames) replay.applyChunk(frame, 1000);
  assert.deepEqual(
    replay.content() as Array<Record<string, unknown>>,
    live,
    "the whole thought stays reasoning and only the answer becomes a text part",
  );
});

test("reopening at any point of a tagged run shows what a stream that never left shows", () => {
  const frames = [
    { choices: [{ delta: { content: "<think>the thought " } }] },
    { choices: [{ delta: { content: "still thinking</think>" } }] },
    {
      choices: [
        {
          delta: {
            tool_calls: [{ index: 0, id: "call_0", function: { name: "bash", arguments: "{}" } }],
          },
        },
      ],
    },
    { choices: [{ delta: { content: "the answer" } }] },
  ];
  const neverLeft = createRecoveryReplay("");
  for (const frame of frames) neverLeft.applyChunk(frame);
  const expected = neverLeft.content() as Array<Record<string, unknown>>;
  assert.deepEqual(
    expected.map((part) => part.type),
    ["reasoning", "tool-call", "text"],
    "the model's own tags classify the reply exactly once, and a reopen does not add a second pair",
  );
  for (let cut = 1; cut <= frames.length; cut++) {
    const seed = createRecoveryReplay("");
    for (const frame of frames.slice(0, cut)) seed.applyChunk(frame);
    const reopened = createRecoveryReplay(seed.content());
    for (const frame of frames.slice(cut)) reopened.applyChunk(frame);
    assert.deepEqual(
      reopened.content() as Array<Record<string, unknown>>,
      expected,
      `reopening after ${cut} of ${frames.length} frames must show the same reply`,
    );
  }
});

// A structured `thinking` part arrives inside delta.content already wrapped in a full pair of tags by
// extractDeltaText. When the block was opened by THIS replay (a reasoning_content frame before it), that
// second open tag is not a no-op for the parser: it survives as literal text inside the thought. A chunk
// carrying only a close tag is the opposite case -- that tag ends whichever block is open either way.
test("a thinking part with its own tags lands in the thought the replay opened", () => {
  const frames = [
    { choices: [{ delta: { reasoning_content: "first thought " } }] },
    { choices: [{ delta: { content: [{ type: "thinking", text: "structured thought" }] } }] },
    { choices: [{ delta: { content: [{ type: "text", text: "the answer" }] } }] },
  ];
  // What the live path appends, in order: it wraps reasoning_content itself and appends extractDeltaText's
  // output verbatim, so parsing THAT string is what a reopened tab has to reproduce exactly.
  const live = parseAssistantContent("<think>first thought </think><think>structured thought</think>the answer");
  const replay = createRecoveryReplay("");
  for (const frame of frames) replay.applyChunk(frame, 1000);
  assert.deepEqual(
    replay.content() as Array<Record<string, unknown>>,
    live,
    "a second open tag must not survive as literal text inside the thought",
  );
});

// The follower builds its accumulator when it attaches and only learns WHICH session the run ran in once
// the stored run comes back. A replay that read the option at construction would shape every already-folded
// frame under whatever scope this tab happens to be on, so a python turn that plotted a chart comes back as
// a path from someone else's sandbox instead of an image from the run that made it.
// The laziness above only pays off because BOTH accumulators the follower builds are handed the SAME holder,
// and because the run's own field is what fills it. Pin both call sites the way the clock is pinned: a fresh
// literal at the prefill rebuild silently hands the new accumulator an option it has been folding frames under.
test("the follower hands the run its own sandbox session, at both places it builds an accumulator", () => {
  const provider = read("../src/features/chat/runtime-provider.tsx");
  // Both accumulators get the holder, never a literal: one is built before the stored run exists to be read.
  const builds = provider.split("createRecoveryReplay(").slice(1);
  assert.equal(
    builds.filter((build) => build.slice(0, 240).includes("replayOptions,")).length,
    builds.length,
    "every accumulator the follower builds must be handed the holder it can still be updated through",
  );
  assert.ok(
    // Pinned on shape, not literal: the holder now carries the approval registration too (scoped ids),
    // and it is the carrying that matters -- not the exact spelling of the literal.
    provider.includes("const replayOptions:") &&
      provider.includes("sandboxSessionId?: string;") &&
      provider.includes("toolConfirmations?:"),
    "the session, and a parked call's approval, have to travel in one object both constructions share",
  );
  assert.ok(
    provider.includes("replayOptions.sandboxSessionId ??=") &&
      provider.includes("update.run.requestPayload.session_id;"),
    "a replayed card names the session that ran, which only exists once the follow yields a run",
  );
});

test("a replayed python card names the session that ran, not the tab that reopened it", () => {
  const options: { sandboxSessionId?: string } = {};
  const replay = createRecoveryReplay("", undefined, options);
  // Filled AFTER construction, exactly as the follower fills it once followChatGenerationRun yields a run.
  options.sandboxSessionId = "sess_ran";
  replay.applyChunk({ choices: [{ delta: { content: "here is the plot" } }] });
  replay.applyChunk({
    _toolEvent: {
      type: "tool_start",
      tool_call_id: "call_0",
      tool_name: "python",
      arguments: {},
    },
  });
  replay.applyChunk({
    _toolEvent: {
      type: "tool_end",
      tool_call_id: "call_0",
      result: 'done\n__IMAGES__:["plot_0.png"]',
    },
  });

  const parts = replay.content() as Array<Record<string, unknown>>;
  const card = parts.find((part) => part.type === "tool-call")!;
  assert.deepEqual(
    card.result,
    { text: "done", images: ["plot_0.png"], sessionId: "sess_ran", files: [] },
    "the replayed card must carry the run's session, split out of the wire's marker",
  );
});

test("a reopened tab reads the sources of a run it did not watch", () => {
  // Live derives Sources from the call's result at its final yield and appends them after the reply;
  // a follower that only assembled text and calls showed pills with no panel. Same parts, same order,
  // byte-identical ids: one source is one entry on both sides or the panel disagrees between tabs.
  const replay = createRecoveryReplay("");
  replay.applyChunk({ choices: [{ delta: { content: "as cited." } }] });
  replay.applyChunk({
    _toolEvent: { type: "tool_start", tool_call_id: "call_0", tool_name: "web_search", arguments: {} },
  });
  replay.applyChunk({
    _toolEvent: {
      type: "tool_end",
      tool_call_id: "call_0",
      result: "Title: Alpha\nURL: https://example.com/a\nSnippet: says it.",
    },
  });
  const parts = replay.content() as Array<Record<string, unknown>>;
  assert.deepEqual(parts.map((part) => part.type), ["text", "tool-call", "source"]);
  const src = parts[2] as Record<string, unknown>;
  assert.equal(src.sourceType, "url");
  assert.equal(src.id, "https://example.com/a");
  assert.equal((src.metadata as { description: string }).description, "says it.");

  // And the citations a frame carried land AFTER everything else, exactly where live appends them —
  // under the id live folds from citation type and position, so inline [N] markers match their entry.
  const citedFrame = {
    source: "https://doc.example/x",
    document_title: "Doc X",
    cited_text: "quoted",
    type: "document",
    start_char_index: 10,
    end_char_index: 20,
  };
  replay.applyChunk({ _toolEvent: { type: "document_citations", citations: [citedFrame] } });
  const after = replay.content() as Array<Record<string, unknown>>;
  assert.deepEqual(after.map((part) => part.type), ["text", "tool-call", "source", "source"]);
  assert.equal(after[3].id, "https://doc.example/x#document:10:20");
});

test("a citation frame that arrives twice lands once", () => {
  // Live dedups citation parts by id; a follower folding a re-delivered frame must not double-list [1].
  const replay = createRecoveryReplay("");
  replay.applyChunk({ choices: [{ delta: { content: "per [1]" } }] });
  const cit = { source: "https://doc.example/x", document_title: "Doc X", cited_text: "quoted" };
  replay.applyChunk({ _toolEvent: { type: "document_citations", citations: [cit] } });
  replay.applyChunk({ _toolEvent: { type: "document_citations", citations: [cit] } });
  const parts = replay.content() as Array<Record<string, unknown>>;
  assert.equal(parts.filter((part) => part.type === "source").length, 1);
});

test("a source the closed tab already persisted is not derived a second time", () => {
  // A run that finished after its writer saved leaves BOTH shapes on disk: the call and the source it
  // produced. Deriving again from the call would list the same url twice under one id.
  const seededSource = {
    type: "source",
    sourceType: "url",
    id: "https://example.com/a",
    url: "https://example.com/a",
    title: "Alpha",
  };
  const replay = createRecoveryReplay([
    { type: "text", text: "as cited." },
    {
      type: "tool-call",
      toolCallId: "call_0:uuid",
      toolName: "web_search",
      args: {},
      argsText: "{}",
      result: "Title: Alpha\nURL: https://example.com/a\nSnippet: as cited.",
    },
    seededSource,
  ]);
  const parts = replay.content() as Array<Record<string, unknown>>;
  assert.equal(parts.filter((part) => part.type === "source").length, 1);
});

test("derived web sources land before the citation parts, as live appends them", () => {
  const replay = createRecoveryReplay("");
  replay.applyChunk({ choices: [{ delta: { content: "both kinds" } }] });
  replay.applyChunk({
    _toolEvent: { type: "tool_start", tool_call_id: "call_0", tool_name: "web_fetch", arguments: {} },
  });
  replay.applyChunk({
    _toolEvent: {
      type: "tool_end",
      tool_call_id: "call_0",
      result: "Title: Web\nURL: https://web.example/b\nSnippet: fetched.",
    },
  });
  replay.applyChunk({
    _toolEvent: {
      type: "document_citations",
      citations: [{ source: "https://doc.example/c", document_title: "Doc C", cited_text: "quoted" }],
    },
  });
  const parts = replay.content() as Array<Record<string, unknown>>;
  assert.deepEqual(parts.map((part) => part.type), ["text", "tool-call", "source", "source"]);
  assert.equal(parts[2].id, "https://web.example/b");
  assert.ok(String(parts[3].id).startsWith("https://doc.example/c#"), "citations last, after the web source");
});

test("a parked call reopens on the one card its run keyed it by", () => {
  // Live keys a parked card `${scopeId}:${approvalId}` and registers that id with the store; the seeded
  // part already carries it. Matching it WHOLE is what keeps one parked call on one card -- routing the
  // scoped id through the backend-id map splits at the first colon, reads the session, misses, mints a
  // second card, and the store never hears that the first one is waiting on an approval.
  const calls: Array<[string, string]> = [];
  const replay = createRecoveryReplay(
    [
      { type: "text", text: "runs once" },
      {
        type: "tool-call",
        toolCallId: "sess:thread:appr_1",
        toolName: "bash",
        args: {},
        argsText: "{}",
      },
    ],
    undefined,
    {
      toolConfirmations: {
        scopeId: "sess:thread",
        register: (id, approvalId) => {
          calls.push([id, approvalId]);
        },
        resolve: (id) => {
          calls.push([id, "resolved"]);
        },
      },
    },
  );
  replay.applyChunk({
    _toolEvent: {
      type: "tool_start",
      tool_call_id: "call_0",
      tool_name: "bash",
      arguments: { cmd: "ls" },
      arguments_text: '{"cmd":"ls"}',
      approval_id: "appr_1",
      awaiting_confirmation: true,
    },
  });
  const parts = replay.content() as Array<Record<string, unknown>>;
  const cards = parts.filter((part) => part.type === "tool-call");
  assert.equal(cards.length, 1);
  assert.equal(cards[0].toolCallId, "sess:thread:appr_1");
  assert.deepEqual(calls, [["sess:thread:appr_1", "appr_1"]]);

  // And it stops asking the moment its end arrives -- resolved through the SAME card, not a new one.
  replay.applyChunk({ _toolEvent: { type: "tool_end", tool_call_id: "call_0", result: "done" } });
  const done = replay.content() as Array<Record<string, unknown>>;
  const endCards = done.filter((part) => part.type === "tool-call");
  assert.equal(endCards.length, 1);
  assert.equal(String(endCards[0].result).includes("done"), true);
  assert.deepEqual(calls[1], ["sess:thread:appr_1", "resolved"]);
});

test("a parked call the reader never saw still asks to be approved", () => {
  // No seed at all: the frames alone must produce ONE card, keyed scoped like live keys it, and this
  // tab's store must hear about it exactly as the watching tab did.
  const calls: Array<[string, string]> = [];
  const replay = createRecoveryReplay("", undefined, {
    toolConfirmations: {
      scopeId: "sess:thread",
      register: (id, approvalId) => {
        calls.push([id, approvalId]);
      },
    },
  });
  replay.applyChunk({ choices: [{ delta: { content: "x" } }] });
  replay.applyChunk({
    _toolEvent: {
      type: "tool_start",
      tool_call_id: "call_0",
      tool_name: "bash",
      arguments: {},
      approval_id: "appr_9",
      awaiting_confirmation: true,
    },
  });
  const parts = replay.content() as Array<Record<string, unknown>>;
  const card = parts.find((part) => part.type === "tool-call")!;
  assert.equal(card.toolCallId, "sess:thread:appr_9");
  assert.deepEqual(calls, [["sess:thread:appr_9", "appr_9"]]);
});

test("a call still parked when the tab closed reopens parked even though its tool_start sat below the cursor", () => {
  // The cursor equals the autosave's sequence, so a call ALREADY parked at close time has no frame left to
  // fold and the replay's own registration never fires for it: the SEED itself must re-raise it. Only a
  // scoped-id card carries an approval to restore (a minted-id card carried its approval in the closed tab's
  // separate store, not on the part), and a card already answered must NOT have Approve/Deny raised on it.
  const calls: Array<[string, string]> = [];
  const toolConfirmations = {
    scopeId: "sess:thread",
    register: (id: string, approvalId: string) => {
      calls.push([id, approvalId]);
    },
  };
  const seed = [
    text("do the thing"),
    tool("sess:thread:appr_1", "bash", { argsText: '{"cmd":"ls"}' }),
    tool("call_9:uuid-1", "read_file", { result: "answered while away" }),
    tool("sess:thread:appr_0", "bash", { result: "already answered" }),
  ];
  // What the follower does the moment it learns the run's scope -- before ANY frame has been folded:
  // a string seed holds no cards, and an already-answered scoped card must not re-ask.
  for (const parked of seededParkedApprovals(seed, toolConfirmations.scopeId)) {
    toolConfirmations.register(parked.id, parked.approvalId);
  }
  assert.deepEqual(
    calls,
    [["sess:thread:appr_1", "appr_1"]],
    "the parked card re-raises under the key live keyed it by; the answered and minted-id cards do not",
  );
  assert.deepEqual(seededParkedApprovals("just prose", "sess:thread"), []);
});

test("the follower restores seeded approvals at the moment it learns the scope, not per frame", () => {
  // The restore rides on the SAME holder the accumulator reads lazily, and fires exactly once -- when
  // `scopeId` lands with the first run frame, before any event folds. A fold later re-registering the pair
  // is the same store entry set again (idempotent), and a tool_end resolving it clears it by the SAME key.
  const provider = read("../src/features/chat/runtime-provider.tsx");
  assert.ok(
    provider.includes("for (const parked of seededParkedApprovals(") &&
      provider.includes("storedMessage.content,") &&
      provider.includes("parked.approvalId,"),
    "the seed is where a skipped tool_start left the approval; the follower has to read it back out",
  );
});
