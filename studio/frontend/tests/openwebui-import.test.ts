// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Covers branch restoration, old and new tool-call formats, and damaged graphs.

import assert from "node:assert/strict";
import test from "node:test";

import {
  isOpenWebUIRecord,
  openWebUIRecordToConversation,
} from "../src/features/chat/utils/openwebui-import.ts";

type Part = { type: string; [key: string]: unknown };

function parts(conversation: { messages: { content: unknown }[] }, index: number): Part[] {
  return conversation.messages[index].content as Part[];
}

function text(conversation: { messages: { content: unknown }[] }, index: number): string {
  return parts(conversation, index)
    .filter((part) => part.type === "text")
    .map((part) => part.text as string)
    .join("\n");
}

function chatRecord(chat: Record<string, unknown>, outer: Record<string, unknown> = {}) {
  return { id: "rec", user_id: "u", title: "t", chat, created_at: 1_700_000_000, ...outer };
}

function historyOf(messages: Record<string, unknown>[], currentId: string | null) {
  const map: Record<string, unknown> = {};
  for (const message of messages) map[message.id as string] = message;
  return { messages: map, currentId };
}

test("detection separates an Open WebUI record from the OpenAI and ShareGPT lines we already import", () => {
  assert.equal(isOpenWebUIRecord(chatRecord({ history: historyOf([], null) })), true);
  // Legacy bare chat: the record IS the chat blob.
  assert.equal(
    isOpenWebUIRecord({
      id: "c",
      title: "old",
      messages: [{ id: "m", role: "user", content: "hi", timestamp: 1 }],
    }),
    true,
  );
  // Plain OpenAI/ShareGPT JSONL lines must not be mistaken for one.
  assert.equal(
    isOpenWebUIRecord({ messages: [{ role: "user", content: "hi" }] }),
    false,
  );
  assert.equal(
    isOpenWebUIRecord({ conversations: [{ from: "human", value: "hi" }] }),
    false,
  );
  assert.equal(isOpenWebUIRecord(null), false);
});

test("the branch the user had open is imported last, so the thread reopens where they left it", () => {
  // one user turn, three assistant replies; currentId points at the middle one.
  const record = chatRecord({
    title: "regens",
    history: historyOf(
      [
        { id: "u1", parentId: null, childrenIds: ["a1", "a2", "a3"], role: "user", content: "q", timestamp: 100 },
        { id: "a1", parentId: "u1", childrenIds: [], role: "assistant", content: "first", timestamp: 101 },
        { id: "a2", parentId: "u1", childrenIds: [], role: "assistant", content: "kept", timestamp: 102 },
        { id: "a3", parentId: "u1", childrenIds: [], role: "assistant", content: "third", timestamp: 103 },
      ],
      "a2",
    ),
  });

  const conversation = openWebUIRecordToConversation(record, "fallback");
  assert.ok(conversation);
  assert.equal(conversation.messages.length, 4);
  assert.equal(text(conversation, 3), "kept");
  // Every sibling is kept as a branch off the same user turn, not flattened away.
  const userId = conversation.messages[0].id;
  assert.deepEqual(
    conversation.messages.slice(1).map((message) => message.parentId),
    [userId, userId, userId],
  );
  // Strictly increasing timestamps: studio sorts by createdAt, so equal stamps would reshuffle them.
  const stamps = conversation.messages.map((message) => message.createdAt);
  assert.deepEqual([...stamps].sort((a, b) => a - b), stamps);
  assert.equal(new Set(stamps).size, stamps.length);
});

test("a parent cycle and a dangling currentId are imported instead of hanging or dropping the chat", () => {
  const cycle = chatRecord({
    history: historyOf(
      [
        { id: "a", parentId: "b", childrenIds: ["b"], role: "user", content: "one", timestamp: 1 },
        { id: "b", parentId: "a", childrenIds: ["a"], role: "assistant", content: "two", timestamp: 2 },
      ],
      "b",
    ),
  });
  const fromCycle = openWebUIRecordToConversation(cycle, "fallback");
  assert.ok(fromCycle);
  assert.equal(fromCycle.messages.length, 2);

  const dangling = chatRecord({
    history: historyOf(
      [
        { id: "u", parentId: null, childrenIds: ["a"], role: "user", content: "q", timestamp: 1 },
        { id: "a", parentId: "u", childrenIds: [], role: "assistant", content: "a", timestamp: 2 },
      ],
      "does-not-exist",
    ),
  });
  const fromDangling = openWebUIRecordToConversation(dangling, "fallback");
  assert.ok(fromDangling);
  assert.equal(fromDangling.messages.length, 2);
});

test("a message whose parent was deleted becomes a root rather than vanishing", () => {
  const record = chatRecord({
    history: historyOf(
      [{ id: "orphan", parentId: "gone", childrenIds: [], role: "assistant", content: "kept", timestamp: 5 }],
      "orphan",
    ),
  });
  const conversation = openWebUIRecordToConversation(record, "fallback");
  assert.ok(conversation);
  assert.equal(conversation.messages.length, 1);
  assert.equal(conversation.messages[0].parentId, null);
});

test("a turn that renders to nothing is dropped and its children relink to the surviving ancestor", () => {
  const record = chatRecord({
    history: historyOf(
      [
        { id: "u1", parentId: null, childrenIds: ["err"], role: "user", content: "q", timestamp: 1 },
        // A failed turn: Open WebUI keeps the row with empty content and an error blob.
        { id: "err", parentId: "u1", childrenIds: ["u2"], role: "assistant", content: "", error: { content: "boom" }, timestamp: 2 },
        { id: "u2", parentId: "err", childrenIds: [], role: "user", content: "retry", timestamp: 3 },
      ],
      "u2",
    ),
  });
  const conversation = openWebUIRecordToConversation(record, "fallback");
  assert.ok(conversation);
  assert.equal(conversation.messages.length, 2);
  assert.equal(conversation.messages[1].parentId, conversation.messages[0].id);
});

test("older assistant turns keep their inlined tool call and reasoning as real parts", () => {
  const content =
    '<details type="reasoning" done="true" duration="7">\n<summary>Thought for 7 seconds</summary>\n> weighing options\n</details>\n' +
    '<details type="tool_calls" done="true" id="call_1" name="get_weather" arguments="{&quot;city&quot;: &quot;Lisbon&quot;}" result="{&quot;c&quot;: 21}">\n<summary>Tool Executed</summary>\n</details>\n' +
    "It is 21 degrees.";
  const record = chatRecord({
    history: historyOf(
      [{ id: "a", parentId: null, childrenIds: [], role: "assistant", content, timestamp: 1 }],
      "a",
    ),
  });

  const conversation = openWebUIRecordToConversation(record, "fallback");
  assert.ok(conversation);
  const [reasoning, toolCall, body] = parts(conversation, 0);
  assert.deepEqual(reasoning, { type: "reasoning", text: "weighing options" });
  assert.equal(toolCall.type, "tool-call");
  assert.equal(toolCall.toolCallId, "call_1");
  assert.equal(toolCall.toolName, "get_weather");
  assert.deepEqual(toolCall.args, { city: "Lisbon" });
  assert.deepEqual(toolCall.result, { c: 21 });
  assert.deepEqual(body, { type: "text", text: "It is 21 degrees." });
});

test("a tool result that contains a code fence is still lifted into a tool-call part", () => {
  // The newer details format puts the result in the block body, and a result is
  // usually JSON, so the fence sits between the opening and closing tags.
  const content =
    "Here is what the tool said.\n" +
    '<details type="tool_calls" done="true" id="call_7" name="run_python">\n' +
    "<summary>Tool Executed</summary>\n" +
    "```json\n{\"rows\": 3}\n```\n" +
    "</details>\n" +
    "So there are three rows.";
  const record = chatRecord({
    history: historyOf(
      [{ id: "a", parentId: null, childrenIds: [], role: "assistant", content, timestamp: 1 }],
      "a",
    ),
  });

  const conversation = openWebUIRecordToConversation(record, "fallback");
  assert.ok(conversation);
  const [before, toolCall, after] = parts(conversation, 0);
  assert.deepEqual(before, { type: "text", text: "Here is what the tool said." });
  assert.equal(toolCall.type, "tool-call");
  assert.equal(toolCall.toolName, "run_python");
  assert.ok(String(toolCall.result).includes('{"rows": 3}'));
  assert.deepEqual(after, { type: "text", text: "So there are three rows." });
  // The markup must not survive as text alongside the part it became.
  assert.ok(!text(conversation, 0).includes("<details"));
});

test("a details block quoted inside a code fence stays text, because the chat is about the markup", () => {
  const content =
    "Look at this:\n\n```html\n" +
    '<details type="tool_calls" done="true" name="not_a_call">\n<summary>example</summary>\n</details>\n' +
    "```\n\nThat is the markup.";
  const record = chatRecord({
    history: historyOf(
      [{ id: "a", parentId: null, childrenIds: [], role: "assistant", content, timestamp: 1 }],
      "a",
    ),
  });

  const conversation = openWebUIRecordToConversation(record, "fallback");
  assert.ok(conversation);
  assert.equal(parts(conversation, 0).every((part) => part.type === "text"), true);
  assert.ok(text(conversation, 0).includes('<details type="tool_calls"'));
  assert.ok(text(conversation, 0).includes("That is the markup."));
});

test("modern assistant turns rebuild their tool call from the output items, result included", () => {
  const record = chatRecord({
    history: historyOf(
      [
        {
          id: "a",
          parentId: null,
          childrenIds: [],
          role: "assistant",
          content: "Lisbon is sunny.",
          timestamp: 1,
          output: [
            { type: "reasoning", id: "rs_1", summary: [{ type: "output_text", text: "check the tool" }] },
            { type: "function_call", call_id: "call_9", name: "web_search", arguments: '{"query":"lisbon"}' },
            {
              type: "function_call_output",
              call_id: "call_9",
              output: [
                { type: "input_text", text: "sunny, 21C" },
                { type: "input_image", image_url: "data:image/png;base64,AAAA" },
              ],
            },
            { type: "message", id: "msg_1", role: "assistant", content: [{ type: "output_text", text: "Lisbon is sunny." }] },
          ],
        },
      ],
      "a",
    ),
  });

  const conversation = openWebUIRecordToConversation(record, "fallback");
  assert.ok(conversation);
  const [reasoning, toolCall, image, body] = parts(conversation, 0);
  assert.equal(reasoning.type, "reasoning");
  assert.equal(toolCall.type, "tool-call");
  assert.deepEqual(toolCall.args, { query: "lisbon" });
  assert.equal(toolCall.result, "sunny, 21C");
  assert.deepEqual(image, { type: "image", image: "data:image/png;base64,AAAA" });
  assert.deepEqual(body, { type: "text", text: "Lisbon is sunny." });
  // `content` mirrors the final output message, so it must not be appended twice.
  assert.equal(parts(conversation, 0).filter((part) => part.type === "text").length, 1);
});

test("an inline image survives; a document keeps its name but not the text Open WebUI extracted", () => {
  const record = chatRecord({
    history: historyOf(
      [
        {
          id: "u",
          parentId: null,
          childrenIds: [],
          role: "user",
          content: "what is in these?",
          timestamp: 1,
          files: [
            { type: "image", url: "data:image/png;base64,IMG" },
            // A file url is a dead Open WebUI route once exported, so the bytes are gone either way.
            { type: "image", url: "/api/v1/files/abc" },
            {
              type: "file",
              id: "f1",
              name: "spec.pdf",
              url: "/api/v1/files/f1",
              file: { filename: "spec.pdf", data: { content: "SECRET EXTRACTED TEXT" } },
            },
          ],
        },
      ],
      "u",
    ),
  });

  const conversation = openWebUIRecordToConversation(record, "fallback");
  assert.ok(conversation);
  const content = parts(conversation, 0);
  assert.deepEqual(
    content.filter((part) => part.type === "image"),
    [{ type: "image", image: "data:image/png;base64,IMG" }],
  );

  const attachments = conversation.messages[0]
    .attachments as unknown as Array<Record<string, unknown>>;
  assert.equal(attachments.length, 1);
  assert.equal(attachments[0].name, "spec.pdf");
  // Replaying the extracted text into attachment content would resend a whole PDF on the next turn.
  assert.deepEqual(attachments[0].content, []);
  assert.ok(!JSON.stringify(conversation.messages).includes("SECRET EXTRACTED TEXT"));
});

test("a chat with no history falls back to the flat active branch, and archived carries over", () => {
  const record = chatRecord(
    {
      title: "flat",
      messages: [
        { id: "m1", role: "user", content: "q", timestamp: 10 },
        { id: "m2", role: "assistant", content: "a", timestamp: 11 },
      ],
    },
    { archived: true },
  );

  const conversation = openWebUIRecordToConversation(record, "fallback");
  assert.ok(conversation);
  assert.equal(conversation.title, "flat");
  assert.equal(conversation.archived, true);
  assert.equal(conversation.messages.length, 2);
  assert.equal(conversation.messages[1].parentId, conversation.messages[0].id);
});

test("an empty chat imports as nothing rather than as a blank thread", () => {
  assert.equal(openWebUIRecordToConversation(chatRecord({ history: historyOf([], null) }), "x"), null);
  assert.equal(openWebUIRecordToConversation(chatRecord({}), "x"), null);
});

test("epoch seconds on the record and milliseconds on the blob both land in the same era", () => {
  const seconds = openWebUIRecordToConversation(
    chatRecord({ history: historyOf([{ id: "m", parentId: null, role: "user", content: "hi" }], "m") }, { created_at: 1_700_000_000 }),
    "x",
  );
  assert.ok(seconds);
  assert.equal(seconds.createdAt, 1_700_000_000_000);

  const millis = openWebUIRecordToConversation(
    {
      chat: {
        timestamp: 1_700_000_000_000,
        history: historyOf([{ id: "m", parentId: null, role: "user", content: "hi" }], "m"),
      },
    },
    "x",
  );
  assert.ok(millis);
  assert.equal(millis.createdAt, 1_700_000_000_000);
});
