// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  codexLocalToolRoundId,
  localToolExchangeIndexes,
  shouldReplayAssistantReasoning,
  startsNewCodexToolRound,
} from "../src/features/chat/codex-reasoning.ts";

test("reasoning replay follows completion and provider policy", () => {
  const base = {
    reasoningContent: "A structured thought.",
    hasContent: false,
    hasToolCalls: false,
  };

  assert.equal(
    shouldReplayAssistantReasoning({ ...base, enabled: true, incomplete: false }),
    true,
  );
  assert.equal(
    shouldReplayAssistantReasoning({ ...base, enabled: true, incomplete: true }),
    false,
  );
  assert.equal(
    shouldReplayAssistantReasoning({
      ...base,
      enabled: true,
      incomplete: true,
      hasContent: true,
    }),
    true,
  );
  assert.equal(
    shouldReplayAssistantReasoning({ ...base, enabled: false, incomplete: false }),
    false,
  );
  assert.equal(
    shouldReplayAssistantReasoning({
      ...base,
      enabled: true,
      incomplete: false,
      reasoningContent: "",
    }),
    false,
  );
});

test("Codex local tool provenance preserves parallel calls and round boundaries", () => {
  const firstRound = { source: "local", round_id: 1 };
  const secondRound = { source: "local", round_id: 2 };

  assert.equal(codexLocalToolRoundId(firstRound), 1);
  assert.equal(codexLocalToolRoundId({ source: "external", round_id: 1 }), null);
  assert.equal(codexLocalToolRoundId({ source: "local", round_id: "1" }), null);

  assert.equal(startsNewCodexToolRound(null, 1), false);
  assert.equal(startsNewCodexToolRound(1, 1), false);
  assert.equal(startsNewCodexToolRound(1, codexLocalToolRoundId(secondRound)), true);
});

test("a forced retrieval card stays its own exchange before a numbered round", () => {
  const exchanges = (parts: { provenance?: unknown; result?: string }[]) =>
    localToolExchangeIndexes(
      parts,
      (part) => codexLocalToolRoundId(part.provenance),
      (part) =>
        (part.provenance as { source?: unknown } | undefined)?.source === "local" &&
        part.result !== undefined,
    );
  const realRound = { provenance: { source: "local", round_id: 0 }, result: "ok" };
  // What the backend sends for a forced recall: local, no round_id.
  assert.deepEqual(exchanges([{ provenance: { source: "local" }, result: "recall" }, realRound]), [0, 1]);
  // Without provenance the two merge into one assistant message, which Llama 3 templates reject.
  assert.deepEqual(exchanges([{ result: "recall" }, realRound]), [0, 0]);
});

test("the JSONL export also starts a new exchange before a forced recall card", async () => {
  const { readSrc } = await import("./helpers/kit.ts");
  const src = readSrc("features/chat/prompt-storage/prompt-storage-dialog.tsx");
  // Same rule as the chat adapter: flush what is pending before a round-less local pair.
  assert.match(src, /toolCalls\.length > 0 && \(ownPair \|\| startsNewCodexToolRound\(roundId, callRoundId\)\)\) flush\(\)/);
});
