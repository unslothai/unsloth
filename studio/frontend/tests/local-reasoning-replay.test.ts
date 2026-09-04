// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  codexLocalToolRoundId,
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
