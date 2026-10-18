// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import ts from "@typescript/typescript6";
import { readSrc } from "./helpers/kit.ts";

const source = readSrc("features/chat/api/chat-adapter.ts");
const start = source.indexOf("let yieldedStatus: string | null = null;");
const end = source.indexOf("\n        } catch (error)", start);
assert.ok(start >= 0 && end > start);
const { outputText } = ts.transpileModule(
  `async function* follow() {
    let report = "";
    const createdRun = { id: "research-1" };
    const researchFollowController = new AbortController();
    const resolvedThreadId = "thread-1";
    const unstable_assistantMessageId = "assistant-1";
    ${source.slice(start, end)}
  }`,
  { compilerOptions: { target: ts.ScriptTarget.ES2022 } },
);

type Update = { content: unknown[]; metadata: { custom: { serverRevision: number } } };

// Token-count bookkeeping must not make an already-delivered terminal update wait for storage.
for (const status of ["completed", "failed", "cancelled"]) {
  for (const [historyState, read] of [
    ["stalled", () => new Promise<never>(() => {})],
    ["offline", async () => { throw new Error("offline"); }],
  ] as const) {
    test(`a ${status} research run finishes without reading ${historyState} history`, { timeout: 1000 }, async () => {
      let reads = 0;
      const follow = new Function(
        "watchResearchRun",
        "terminalResearchStatuses",
        "getStoredChatMessage",
        `${outputText}; return follow;`,
      )(
        async function* () {
          yield { id: "research-1", status: "running", report: "draft", lastEventSeq: 1 };
          yield { id: "research-1", status: "running", report: "longer draft", lastEventSeq: 2 };
          yield { id: "research-1", status, report: "report", lastEventSeq: 3 };
        },
        new Set(["completed", "failed", "cancelled"]),
        () => { reads += 1; return read(); },
      ) as () => AsyncGenerator<Update>;

      const updates: Update[] = [];
      for await (const update of follow()) updates.push(update);

      assert.equal(reads, 0);
      assert.equal(updates.length, 2);
      assert.deepEqual(updates[0].content, [{ type: "text", text: "draft" }]);
      assert.deepEqual(updates[1].content, [{ type: "text", text: "report" }]);
      assert.equal(updates[1].metadata.custom.serverRevision, 3);
    });
  }
}
