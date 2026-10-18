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

async function follow(read: () => Promise<unknown>) {
  let reads = 0;
  const run = new Function(
    "watchResearchRun",
    "terminalResearchStatuses",
    "getStoredChatMessage",
    `${outputText}; return follow;`,
  )(
    async function* () {
      yield { id: "research-1", status: "running", report: "draft", lastEventSeq: 1 };
      yield { id: "research-1", status: "running", report: "longer draft", lastEventSeq: 2 };
      yield { id: "research-1", status: "completed", report: "report", lastEventSeq: 3 };
    },
    new Set(["completed", "failed", "cancelled"]),
    async (threadId: string, messageId: string) => {
      assert.equal(threadId, "thread-1");
      assert.equal(messageId, "assistant-1");
      reads += 1;
      return read();
    },
  ) as () => AsyncGenerator<Update>;
  const updates: Update[] = [];
  for await (const update of run()) updates.push(update);
  return { reads, updates };
}

test("the settled research report uses the same reasoning and tools as reopening", async () => {
  const content = [
    { type: "reasoning", text: "research reasoning" },
    { type: "tool-call", toolCallId: "call-1", toolName: "web_search", args: {}, result: "sources" },
    { type: "text", text: "report" },
  ];
  const result = await follow(async () => ({
    role: "assistant", content, metadata: { researchRunId: "research-1" },
  }));
  assert.equal(result.reads, 1);
  assert.equal(result.updates.length, 2);
  assert.deepEqual(result.updates[0].content, [{ type: "text", text: "draft" }]);
  assert.deepEqual(result.updates[1].content, content);
  assert.notEqual(result.updates[1].content, content);
  assert.equal(result.updates[1].metadata.custom.serverRevision, 3);
});

for (const [name, read] of [
  ["missing", async () => undefined],
  ["unavailable", async () => { throw new Error("offline"); }],
  ["another run", async () => ({ role: "assistant", content: [{ type: "text", text: "old" }], metadata: { researchRunId: "old" } })],
] as const) {
  test(`a ${name} stored row preserves the completed report`, async () => {
    const result = await follow(read);
    assert.deepEqual(result.updates.at(-1)?.content, [{ type: "text", text: "report" }]);
  });
}
