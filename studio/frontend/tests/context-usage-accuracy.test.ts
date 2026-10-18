// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { test } from "node:test";

import {
  contextUsageRows,
  deriveContextUsageBar,
  formatTokenCount,
} from "../src/features/chat/lib/context-usage-bar-state.ts";
import {
  branchContextUsage,
  contextUsageMarkers,
  estimateBranch,
  isSummedToolLoopUsage,
  revisionTotal,
  sameContextUsage,
  threadBranchRecords,
} from "../src/features/chat/utils/saved-context-usage.ts";
import { contextUsageReasoningKey } from "../src/features/chat/utils/context-usage-reasoning.ts";

test("counts below a million keep the k suffix", () => {
  assert.equal(formatTokenCount(999), "999");
  assert.equal(formatTokenCount(1000), "1.0k");
  assert.equal(formatTokenCount(322_800), "322.8k");
  assert.equal(formatTokenCount(999_949), "999.9k");
});

test("counts from a million up switch to an M suffix instead of a four-digit k", () => {
  assert.equal(formatTokenCount(999_950), "1.0M");
  assert.equal(formatTokenCount(1_000_000), "1.0M");
  assert.equal(formatTokenCount(2_500_000), "2.5M");
  assert.equal(formatTokenCount(12_345_678), "12.3M");
});

test("a negative or non-finite used count reads as not counted", () => {
  for (const used of [-5, Number.NaN, Number.POSITIVE_INFINITY]) {
    const state = deriveContextUsageBar({ used, total: 32768 });
    assert.ok(state);
    assert.equal(state.percent, null);
    assert.equal(state.face, "— / 32.8k");
  }
});

test("a count past the window still caps the ratio at one hundred percent", () => {
  const state = deriveContextUsageBar({ used: 2_345_678, total: 32768 });
  assert.ok(state);
  assert.equal(state.face, "2.3M / 32.8k");
  assert.equal(state.percent, 100);
});

test("a single request's rows describe the context it reports", () => {
  assert.deepEqual(
    contextUsageRows({ promptTokens: 900, completionTokens: 100, cachedTokens: 800 }, true),
    { promptTokens: 900, completionTokens: 100, cached: 800, cacheWrites: undefined },
  );
  assert.deepEqual(
    contextUsageRows(
      { promptTokens: 900, completionTokens: 100, contextTokens: 1000, cachedTokens: 800, cacheWriteTokens: 50 },
      true,
    ),
    { promptTokens: 900, completionTokens: 100, cached: 800, cacheWrites: 50 },
  );
  assert.deepEqual(contextUsageRows(null, false), {});
});

test("a local tool loop's rows show the last pass's reply, not every pass's", () => {
  assert.deepEqual(
    contextUsageRows({ promptTokens: 625, completionTokens: 150, contextTokens: 654, cachedTokens: 500 }, false),
    { promptTokens: 625, completionTokens: 29, cached: 500, cacheWrites: undefined },
  );
});

test("a managed engine's summed prompt hides the rows unless the turn was one pass", () => {
  assert.deepEqual(
    contextUsageRows({ promptTokens: 3000, completionTokens: 1550, contextTokens: 3500 }, true),
    {},
  );
  assert.deepEqual(
    contextUsageRows({ promptTokens: 2000, completionTokens: 1500, contextTokens: 3500 }, true),
    { promptTokens: 2000, completionTokens: 1500, cached: undefined, cacheWrites: undefined },
  );
});

test("a cloud tool loop's summed billing rows are hidden, even when the last reply is long", () => {
  assert.deepEqual(
    contextUsageRows({ promptTokens: 322000, completionTokens: 760, contextTokens: 52400, cachedTokens: 306600 }, true),
    {},
  );
  assert.deepEqual(
    contextUsageRows({ promptTokens: 6900, completionTokens: 4260, contextTokens: 8100 }, true),
    {},
  );
  assert.deepEqual(
    contextUsageRows({ promptTokens: 203000, completionTokens: 4500, contextTokens: 24000 }, true),
    {},
  );
});

const localToolCall = {
  type: "tool-call",
  toolCallId: "c1",
  toolName: "web_search",
  args: {},
  provenance: { source: "local", round_id: 1 },
};
const hostedToolCall = { type: "tool-call", toolCallId: "c1", toolName: "web_search", args: {} };
const cloudUsage = { modelId: "external::openai::gpt-x" };

test("a cloud turn whose tools Studio ran, saved without its context, is a summed loop figure", () => {
  assert.equal(isSummedToolLoopUsage(cloudUsage, [localToolCall, { type: "text", text: "a" }]), true);
});

test("a saved figure that names its context, a hosted tool, a local model or a plain reply is trusted", () => {
  assert.equal(isSummedToolLoopUsage({ ...cloudUsage, contextTokens: 52400 }, [localToolCall]), false);
  assert.equal(isSummedToolLoopUsage(cloudUsage, [hostedToolCall]), false);
  assert.equal(isSummedToolLoopUsage({ modelId: "unsloth/gguf-model" }, [localToolCall]), false);
  assert.equal(isSummedToolLoopUsage(cloudUsage, [{ type: "text", text: "a" }]), false);
  assert.equal(isSummedToolLoopUsage(cloudUsage, undefined), false);
  assert.equal(isSummedToolLoopUsage(undefined, [localToolCall]), false);
});

const LOCAL = "unsloth/gguf-model";
const CLOUD = "external::openai::gpt-x";

function usage(context: number, modelId?: string) {
  return {
    promptTokens: context - 10,
    completionTokens: 10,
    totalTokens: context,
    cachedTokens: 0,
    ...(modelId ? { modelId } : {}),
  };
}

function row(
  id: string,
  role: "user" | "assistant",
  createdAt: number,
  text: string,
  metadata?: Record<string, unknown>,
) {
  return { id, role, createdAt, content: [{ type: "text" as const, text }], ...(metadata ? { metadata } : {}) };
}

test("a retrieval card added before the first pass leaves a one-pass cloud figure trusted", () => {
  const retrievalCard = {
    type: "tool-call",
    toolCallId: "rag_auto_0123456789ab",
    toolName: "search_knowledge_base",
    args: { query: "q" },
    provenance: { source: "local" },
  };
  assert.equal(isSummedToolLoopUsage(cloudUsage, [retrievalCard, { type: "text", text: "a" }]), false);
  assert.equal(isSummedToolLoopUsage(cloudUsage, [retrievalCard, localToolCall]), true);
  const saved = usage(6330, CLOUD);
  const reply = row("a2", "assistant", 2, "y".repeat(40), { contextUsage: saved });
  const branch = [
    row("u1", "user", 1, "x".repeat(40)),
    { ...reply, content: [retrievalCard, ...reply.content] },
  ] as never;
  assert.equal(branchContextUsage(branch, CLOUD, null, null), saved);
});

const conversation = () => [
  row("u1", "user", 1, "x".repeat(40)),
  row("a1", "assistant", 2, "y".repeat(40), { contextUsage: usage(300, LOCAL) }),
  row("u2", "user", 3, "x".repeat(40)),
  row("a2", "assistant", 4, "y".repeat(40), { contextUsage: usage(500, LOCAL) }),
];

test("a reply's markers record its position, the edits before it and the branch's estimate", () => {
  const branch = conversation();
  branch[2] = { ...branch[2], metadata: { revision: 2 } };
  assert.deepEqual(contextUsageMarkers(branch), { messageCount: 4, revision: 2, estimatedTokens: 40 });
  assert.deepEqual(contextUsageMarkers([]), { messageCount: 0, revision: 0, estimatedTokens: 0 });
});

test("the tip's own count prices its branch", () => {
  const branch = conversation();
  assert.equal(branchContextUsage(branch, LOCAL, 8192, 8192), branch[3].metadata?.contextUsage);
});

test("tagged local counts retain their exactness only under the reasoning mode that produced them", () => {
  const on = contextUsageReasoningKey({ enable_thinking: true, preserve_thinking: true });
  const off = contextUsageReasoningKey({ enable_thinking: false, preserve_thinking: true });
  const branch = conversation();
  branch[3] = {
    ...branch[3],
    metadata: {
      contextUsage: { ...usage(500, LOCAL), reasoningKey: on, estimatedTokens: 40 },
    },
  };
  assert.equal(branchContextUsage(branch, LOCAL, 8192, 8192, on), branch[3].metadata?.contextUsage);
  const changed = branchContextUsage(branch, LOCAL, 8192, 8192, off);
  assert.equal(changed?.estimated, true);
  assert.equal(changed?.totalTokens, 500);
});

test("legacy saved counts remain exact on a local reasoning model", () => {
  const branch = conversation();
  const saved = branch[3].metadata?.contextUsage;
  for (const enabled of [true, false]) {
    const reasoningKey = contextUsageReasoningKey({ enable_thinking: enabled });
    assert.equal(branchContextUsage(branch, LOCAL, 8192, 8192, reasoningKey), saved);
    assert.equal(branchContextUsage(branch, LOCAL, 8192, null, reasoningKey), saved);
    assert.equal(branchContextUsage(branch, LOCAL, null, null, reasoningKey), saved);
  }
  branch[2] = { ...branch[2], metadata: { revision: 1 } };
  assert.equal(branchContextUsage(branch, LOCAL, 8192, 8192, "[true,null,null]")?.estimated, true);
});

test("local reasoning defaults do not invalidate cloud context counts", () => {
  const branch = conversation().map((message) => message.role === "assistant"
    ? { ...message, metadata: { contextUsage: usage(500, CLOUD) } }
    : message);
  assert.equal(
    branchContextUsage(branch, CLOUD, null, null, contextUsageReasoningKey({ enable_thinking: true })),
    branch[3].metadata?.contextUsage,
  );
});

test("a reply edited after it was written is estimated on top of the last count before it", () => {
  const branch = conversation();
  branch[3] = { ...branch[3], content: [{ type: "text", text: "y".repeat(400) }], metadata: { ...branch[3].metadata, revision: 1 } };
  assert.deepEqual(branchContextUsage(branch, LOCAL, 8192, 8192), {
    promptTokens: 410,
    completionTokens: 0,
    totalTokens: 410,
    cachedTokens: 0,
    estimated: true,
  });
});

test("a research report replaces the handoff's count with an estimate from the preceding turn", () => {
  const branch = conversation();
  branch[3] = {
    ...branch[3],
    content: [{ type: "text", text: "r".repeat(400) }],
    metadata: { ...branch[3].metadata, researchRunId: "research-1", researchStatus: "completed" },
  };
  const priced = branchContextUsage(branch, LOCAL, 8192, 8192);
  assert.equal(priced?.estimated, true);
  assert.equal(priced?.totalTokens, 410);
  assert.equal(branchContextUsage(branch.slice(2), LOCAL, 8192, 8192), null);
});

test("a turn after a research report keeps its own exact context count", () => {
  const branch = conversation();
  branch[1] = {
    ...branch[1],
    metadata: { ...branch[1].metadata, researchRunId: "research-1", researchStatus: "completed" },
  };
  assert.equal(branchContextUsage(branch, LOCAL, 8192, 8192), branch[3].metadata?.contextUsage);
});

test("a revision invalidates every count after it and none before it", () => {
  const branch = conversation();
  branch[2] = { ...branch[2], metadata: { revision: 1 } };
  assert.equal(branchContextUsage(branch, LOCAL, 8192, 8192)?.totalTokens, 320);
  branch[0] = { ...branch[0], metadata: { revision: 1 } };
  assert.equal(branchContextUsage(branch, LOCAL, 8192, 8192), null);
});

test("a reply written after the revision is trusted again", () => {
  const branch = conversation();
  branch[2] = { ...branch[2], metadata: { revision: 1 } };
  branch[3] = { ...branch[3], metadata: { contextUsage: { ...usage(500, LOCAL), revision: 1 } } };
  assert.equal(branchContextUsage(branch, LOCAL, 8192, 8192), branch[3].metadata?.contextUsage);
});

test("an unanswered prompt is estimated on top of the reply before it", () => {
  const branch = conversation().slice(0, 3);
  const priced = branchContextUsage(branch, LOCAL, 8192, 8192);
  assert.equal(priced?.totalTokens, 310);
  assert.equal(priced?.estimated, true);
});

test("counts another model, a summed cloud loop or a past-window figure produced are skipped", () => {
  const branch = conversation();
  branch[3] = { ...branch[3], metadata: { contextUsage: usage(500, "unsloth/other-gguf") } };
  assert.equal(branchContextUsage(branch, LOCAL, 8192, 8192)?.totalTokens, 320);
  branch[3] = { ...branch[3], metadata: { contextUsage: usage(9000, LOCAL) } };
  assert.equal(branchContextUsage(branch, LOCAL, 8192, 8192)?.totalTokens, 320);
  assert.equal(branchContextUsage(branch, LOCAL, 16384, null), branch[3].metadata?.contextUsage);
  const cloud = conversation().map((message) =>
    message.role === "assistant"
      ? { ...message, content: [localToolCall, ...message.content], metadata: { contextUsage: usage(322_000, CLOUD) } }
      : message,
  ) as never;
  assert.equal(branchContextUsage(cloud, CLOUD, null, null), null);
});

test("a count saved without a model is trusted only under a local window", () => {
  const branch = conversation().map((message) =>
    message.role === "assistant" ? { ...message, metadata: { contextUsage: usage(500) } } : message,
  );
  assert.equal(branchContextUsage(branch, LOCAL, 8192, 8192)?.totalTokens, 500);
  assert.equal(branchContextUsage(branch, CLOUD, null, null), null);
});

test("the plain estimate follows the branch order, not the timestamps", () => {
  const branch = [row("u1", "user", 50, "x".repeat(40)), row("a1", "assistant", 1, "y".repeat(40))];
  assert.equal(estimateBranch(branch)?.totalTokens, 20);
  assert.equal(estimateBranch([]), null);
});

test("runtime messages read as branch records carrying their own metadata", () => {
  const messages = [
    { id: "u1", role: "user", createdAt: new Date(5), content: [], metadata: { custom: { revision: 2 } } },
    { id: "a1", role: "assistant", createdAt: new Date(6), content: [], metadata: { custom: { revision: 1 } } },
  ] as never;
  assert.deepEqual(threadBranchRecords(messages), [
    { id: "u1", role: "user", content: [], metadata: { revision: 2 } },
    { id: "a1", role: "assistant", content: [], metadata: { revision: 1 } },
  ]);
  assert.equal(revisionTotal(messages), 3);
  assert.equal(revisionTotal([] as never), 0);
});

test("a research report replacing the same message changes the runtime branch revision", () => {
  const message = {
    metadata: { custom: { researchRunId: "research-1", serverRevision: 2, revision: 1 } },
  };
  assert.equal(revisionTotal([message]), 3);
  message.metadata.custom.serverRevision = 20;
  assert.equal(revisionTotal([message]), 21);
  assert.equal(revisionTotal([{
    metadata: { custom: { researchRunId: "research-1", researchRun: { lastEventSeq: 30 } } },
  }]), 30);
});

test("generation stream revisions and malformed research revisions do not invalidate the meter", () => {
  assert.equal(revisionTotal([{
    metadata: { custom: { generationRunId: "generation-1", serverRevision: 100, revision: 2 } },
  }]), 2);
  for (const serverRevision of [undefined, null, "2", -1, Number.NaN, Infinity, 1.5]) {
    assert.equal(revisionTotal([{
      metadata: { custom: { researchRunId: "research-1", serverRevision } },
    }]), 0);
  }
});

test("a shown count matches only the same figure under the same model", () => {
  const saved = usage(500);
  assert.equal(sameContextUsage({ ...saved, modelId: LOCAL }, saved, LOCAL), true);
  assert.equal(sameContextUsage({ ...saved, modelId: "unsloth/other-gguf" }, saved, LOCAL), false);
  assert.equal(sameContextUsage({ ...saved, modelId: LOCAL, estimated: true }, saved, LOCAL), false);
  assert.equal(sameContextUsage(null, saved, LOCAL), false);
});

test("a count stops describing its branch once a turn above it is deleted", () => {
  const withPositions = conversation().map((message, index) =>
    message.role === "assistant"
      ? { ...message, metadata: { contextUsage: { ...usage(index === 1 ? 300 : 500, LOCAL), messageCount: index + 1 } } }
      : message,
  );
  assert.equal(branchContextUsage(withPositions, LOCAL, 8192, 8192)?.totalTokens, 500);
  const afterDelete = [withPositions[0], withPositions[1], withPositions[3]];
  const priced = branchContextUsage(afterDelete, LOCAL, 8192, 8192);
  assert.equal(priced?.totalTokens, 310);
  assert.equal(priced?.estimated, true);
  assert.equal(branchContextUsage(conversation().slice(0, 1).concat(conversation()[3]), LOCAL, 8192, 8192)?.totalTokens, 500);
});

function calibrated() {
  const branch = conversation();
  branch[1] = { ...branch[1], metadata: { contextUsage: { ...usage(300, CLOUD), messageCount: 2, estimatedTokens: 20 } } };
  branch[3] = { ...branch[3], metadata: { contextUsage: { ...usage(500, CLOUD), messageCount: 4, estimatedTokens: 40 } } };
  return branch;
}

test("a count left stale by a delete above it still calibrates the estimate", () => {
  const branch = calibrated();
  const afterDelete = [branch[2], branch[3]];
  assert.deepEqual(branchContextUsage(afterDelete, CLOUD, null, null), {
    promptTokens: 480,
    completionTokens: 0,
    totalTokens: 480,
    cachedTokens: 0,
    estimated: true,
  });
});

test("a reply edited in place is its own count plus the edit", () => {
  const branch = calibrated();
  branch[3] = {
    ...branch[3],
    content: [{ type: "text", text: "y".repeat(400) }],
    metadata: { ...branch[3].metadata, revision: 1 },
  };
  assert.equal(branchContextUsage(branch, CLOUD, null, null)?.totalTokens, 590);
});

test("the newest count wins, whether it is still exact or only calibrates", () => {
  const staleTip = calibrated();
  staleTip[3] = { ...staleTip[3], metadata: { contextUsage: { ...usage(500, CLOUD), messageCount: 9, estimatedTokens: 40 } } };
  assert.deepEqual(branchContextUsage(staleTip, CLOUD, null, null), {
    promptTokens: 500,
    completionTokens: 0,
    totalTokens: 500,
    cachedTokens: 0,
    estimated: true,
  });
  const staleWithoutEstimate = calibrated();
  staleWithoutEstimate[3] = { ...staleWithoutEstimate[3], metadata: { contextUsage: { ...usage(500, CLOUD), messageCount: 9 } } };
  assert.equal(branchContextUsage(staleWithoutEstimate, CLOUD, null, null)?.totalTokens, 320);
  const staleFirst = calibrated();
  staleFirst[1] = { ...staleFirst[1], metadata: { contextUsage: { ...usage(300, CLOUD), messageCount: 9, estimatedTokens: 20 } } };
  assert.equal(branchContextUsage(staleFirst, CLOUD, null, null), staleFirst[3].metadata?.contextUsage);
});

test("a count smaller than its own estimate scales the estimate down instead of going negative", () => {
  const branch = calibrated();
  branch[3] = { ...branch[3], metadata: { contextUsage: { ...usage(19_000, CLOUD), messageCount: 20, estimatedTokens: 51_000 } } };
  const priced = branchContextUsage(branch, CLOUD, null, null);
  assert.equal(priced?.estimated, true);
  assert.equal(priced?.totalTokens, 15);
  const empty = calibrated().map((message) =>
    message.role === "assistant" ? { ...message, content: [] } : { ...message, content: [] },
  );
  empty[3] = { ...empty[3], metadata: { contextUsage: { ...usage(19_000, CLOUD), messageCount: 20, estimatedTokens: 51_000 } } };
  assert.equal(branchContextUsage(empty, CLOUD, null, null)?.totalTokens, 1);
});
