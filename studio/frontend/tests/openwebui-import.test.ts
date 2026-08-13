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

test("a tool result stored as a plain string is kept, not flattened to nothing", () => {
  // The Responses API documents `output` as a string OR a content array, and a
  // string is what a tool returning text produces.
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
            { type: "function_call", call_id: "call_1", name: "web_search", arguments: '{"query":"lisbon"}' },
            { type: "function_call_output", call_id: "call_1", output: '{"temp":"21C"}' },
            { type: "message", id: "msg_1", role: "assistant", content: [{ type: "output_text", text: "Lisbon is sunny." }] },
          ],
        },
      ],
      "a",
    ),
  });

  const conversation = openWebUIRecordToConversation(record, "fallback");
  assert.ok(conversation);
  const toolCall = parts(conversation, 0).find((part) => part.type === "tool-call");
  assert.ok(toolCall);
  assert.equal(toolCall.result, '{"temp":"21C"}');

  // An orphan string output, with no call to attach it to, still shows as text.
  const orphan = openWebUIRecordToConversation(
    chatRecord({
      history: historyOf(
        [
          {
            id: "a",
            parentId: null,
            role: "assistant",
            timestamp: 1,
            output: [{ type: "function_call_output", call_id: "call_gone", output: "21C" }],
          },
        ],
        "a",
      ),
    }),
    "fallback",
  );
  assert.ok(orphan);
  assert.equal(text(orphan, 0), "21C");
});

test("a user turn that uploaded a file without typing survives, and keeps its descendants", () => {
  // Open WebUI sends an empty prompt with a file attached, so real exports
  // contain turns whose only content is the upload.
  const record = chatRecord({
    history: historyOf(
      [
        {
          id: "u",
          parentId: null,
          childrenIds: ["a"],
          role: "user",
          content: "",
          timestamp: 1,
          files: [{ type: "file", id: "f1", name: "spec.pdf", url: "/api/v1/files/f1" }],
        },
        { id: "a", parentId: "u", childrenIds: [], role: "assistant", content: "It is a spec.", timestamp: 2 },
      ],
      "a",
    ),
  });

  const conversation = openWebUIRecordToConversation(record, "fallback");
  assert.ok(conversation);
  assert.equal(conversation.messages.length, 2);
  const attachments = conversation.messages[0]
    .attachments as unknown as Array<Record<string, unknown>>;
  assert.equal(attachments[0].name, "spec.pdf");
  // The reply hangs off the upload, not off the root.
  assert.equal(conversation.messages[1].parentId, conversation.messages[0].id);

  // A turn that renders nothing at all is still dropped.
  const failed = openWebUIRecordToConversation(
    chatRecord({
      history: historyOf([{ id: "u", parentId: null, role: "user", content: "", timestamp: 1 }], "u"),
    }),
    "fallback",
  );
  assert.equal(failed, null);
});

test("a chat thousands of messages deep converts instead of exhausting the call stack", () => {
  // A long-running chat is one long parent chain, and a recursive walk of it
  // throws RangeError, which would abort the whole export mid-stream.
  const messages = Array.from({ length: 20_000 }, (_, index) => ({
    id: `m${index}`,
    parentId: index === 0 ? null : `m${index - 1}`,
    role: index % 2 === 0 ? "user" : "assistant",
    content: `turn ${index}`,
    timestamp: index + 1,
  }));
  const conversation = openWebUIRecordToConversation(
    chatRecord({ history: historyOf(messages, "m19999") }),
    "fallback",
  );
  assert.ok(conversation);
  assert.equal(conversation.messages.length, 20_000);
  assert.equal(text(conversation, 0), "turn 0");
  assert.equal(text(conversation, 19_999), "turn 19999");
});

test("details markup a user typed stays text rather than becoming reasoning", () => {
  const typed =
    'Why does this render oddly?\n<details type="reasoning" done="true">\n<summary>Thought</summary>\n> a plan\n</details>';
  const conversation = openWebUIRecordToConversation(
    chatRecord({
      history: historyOf(
        [
          { id: "u", parentId: null, role: "user", content: typed, timestamp: 1 },
          { id: "a", parentId: "u", role: "assistant", content: typed, timestamp: 2 },
        ],
        "a",
      ),
    }),
    "fallback",
  );
  assert.ok(conversation);
  assert.deepEqual(parts(conversation, 0), [{ type: "text", text: typed }]);
  // The same markup written by Open WebUI's own assistant output still converts.
  assert.equal(parts(conversation, 1).some((part) => part.type === "reasoning"), true);
});

test("a legacy flat chat drops repeated ids without rescanning what it already collected", () => {
  const flat = Array.from({ length: 20_000 }, (_, index) => ({
    id: `f${index % 19_000}`,
    role: index % 2 === 0 ? "user" : "assistant",
    content: `turn ${index}`,
    timestamp: index + 1,
  }));
  const conversation = openWebUIRecordToConversation(chatRecord({ messages: flat }), "fallback");
  assert.ok(conversation);
  assert.equal(conversation.messages.length, 19_000);
  assert.equal(conversation.messages[1].parentId, conversation.messages[0].id);
});

test("a prompt keeps the whitespace it was written with", () => {
  // Leading indentation is a markdown code block, so trimming rewrites the prompt.
  const indented = "    const x = 1;\n    const y = 2;\n";
  const conversation = openWebUIRecordToConversation(
    chatRecord({
      history: historyOf(
        [{ id: "u", parentId: null, role: "user", content: indented, timestamp: 1 }],
        "u",
      ),
    }),
    "fallback",
  );
  assert.ok(conversation);
  assert.deepEqual(parts(conversation, 0), [{ type: "text", text: indented }]);
});

test("a legacy chat with no chat-level date is dated from its earliest message", () => {
  const bare = {
    title: "legacy",
    history: historyOf(
      [
        { id: "m1", parentId: null, role: "user", content: "q", timestamp: 1_600_000_000 },
        { id: "m2", parentId: "m1", role: "assistant", content: "a", timestamp: 1_600_000_060 },
      ],
      "m2",
    ),
  };

  const conversation = openWebUIRecordToConversation(bare, "fallback");
  assert.ok(conversation);
  assert.equal(conversation.createdAt, 1_600_000_000_000);
  assert.equal(conversation.messages[0].createdAt, 1_600_000_000_000);
  assert.equal(conversation.messages[1].createdAt, 1_600_000_060_000);
});

test("the assistant answer survives next to a tool result no call in this turn matched", () => {
  const record = chatRecord({
    history: historyOf(
      [
        {
          id: "a",
          parentId: null,
          role: "assistant",
          timestamp: 1,
          content: "It is sunny in Lisbon.",
          // The call itself was recorded on the previous turn.
          output: [{ type: "function_call_output", call_id: "call_earlier", output: "sunny, 21C" }],
        },
      ],
      "a",
    ),
  });

  const conversation = openWebUIRecordToConversation(record, "fallback");
  assert.ok(conversation);
  assert.ok(text(conversation, 0).includes("sunny, 21C"));
  assert.ok(text(conversation, 0).includes("It is sunny in Lisbon."));
});

test("an updated-only chat is still dated from its first message, not its last edit", () => {
  const bare = {
    updated_at: 1_600_100_000,
    history: historyOf(
      [
        { id: "m1", parentId: null, role: "user", content: "q", timestamp: 1_600_000_000 },
        { id: "m2", parentId: "m1", role: "assistant", content: "a", timestamp: 1_600_000_060 },
      ],
      "m2",
    ),
  };

  const conversation = openWebUIRecordToConversation(bare, "fallback");
  assert.ok(conversation);
  assert.equal(conversation.createdAt, 1_600_000_000_000);
  assert.equal(conversation.messages[1].createdAt, 1_600_000_060_000);
});

test("a timestamped OpenAI conversation is not mistaken for an Open WebUI chat", () => {
  // `timestamp` alone is generic; routing this record here would drop tool_calls.
  const oaiToolConversation = {
    messages: [
      { role: "user", content: "weather?", timestamp: 1 },
      { role: "assistant", content: "", tool_calls: [{ id: "c1", function: { name: "w", arguments: "{}" } }], timestamp: 2 },
      { role: "tool", tool_call_id: "c1", content: "21C", timestamp: 3 },
    ],
  };
  assert.equal(isOpenWebUIRecord(oaiToolConversation), false);
  // Even when the exporter gave every message an id of its own.
  assert.equal(
    isOpenWebUIRecord({
      messages: oaiToolConversation.messages.map((message, index) => ({ ...message, id: `m${index}` })),
    }),
    false,
  );
  // Open WebUI's own fields still identify a flat legacy chat.
  assert.equal(
    isOpenWebUIRecord({
      messages: [{ role: "user", content: "hi", parentId: null, childrenIds: [], timestamp: 1 }],
    }),
    true,
  );
});

test("a generated image comes over as an image, not as a dropped output item", () => {
  const record = chatRecord({
    history: historyOf(
      [
        {
          id: "a",
          parentId: null,
          role: "assistant",
          timestamp: 1,
          content: "Here it is.",
          output: [
            { type: "image_generation_call", id: "img_1", result: "AAAA", output_format: "png" },
            { type: "message", id: "msg_1", role: "assistant", content: [{ type: "output_text", text: "Here it is." }] },
          ],
        },
      ],
      "a",
    ),
  });

  const conversation = openWebUIRecordToConversation(record, "fallback");
  assert.ok(conversation);
  assert.deepEqual(
    parts(conversation, 0).filter((part) => part.type === "image"),
    [{ type: "image", image: "data:image/png;base64,AAAA" }],
  );
});

test("a plain details block the model wrote keeps its markup", () => {
  const answer = "Here:\n<details><summary>Spoiler</summary>secret</details>\nDone.";
  const conversation = openWebUIRecordToConversation(
    chatRecord({
      history: historyOf(
        [{ id: "a", parentId: null, role: "assistant", content: answer, timestamp: 1 }],
        "a",
      ),
    }),
    "fallback",
  );
  assert.ok(conversation);
  assert.ok(text(conversation, 0).includes("<summary>Spoiler</summary>"));
  assert.ok(text(conversation, 0).includes("Done."));
});

test("an empty history map falls back to the flat branch instead of dropping the chat", () => {
  for (const messages of [{}, { m1: "not an object" }]) {
    const conversation = openWebUIRecordToConversation(
      chatRecord({
        history: { messages, currentId: "m1" },
        messages: [
          { id: "m1", role: "user", content: "q", timestamp: 1 },
          { id: "m2", role: "assistant", content: "a", timestamp: 2 },
        ],
      }),
      "fallback",
    );
    assert.ok(conversation);
    assert.equal(conversation.messages.length, 2);
    assert.equal(conversation.messages[1].parentId, conversation.messages[0].id);
  }
});

test("built-in Responses tools import as tool parts rather than disappearing", () => {
  const record = chatRecord({
    history: historyOf(
      [
        {
          id: "a",
          parentId: null,
          role: "assistant",
          timestamp: 1,
          content: "Lisbon is sunny.",
          output: [
            {
              type: "web_search_call",
              id: "ws_1",
              status: "completed",
              action: { type: "search", query: "lisbon weather" },
            },
            { type: "shell_call", call_id: "sh_1", action: { type: "exec", commands: ["ls -la"] } },
            { type: "shell_call_output", call_id: "sh_1", output: "total 0" },
            { type: "message", id: "msg_1", role: "assistant", content: [{ type: "output_text", text: "Lisbon is sunny." }] },
          ],
        },
      ],
      "a",
    ),
  });

  const conversation = openWebUIRecordToConversation(record, "fallback");
  assert.ok(conversation);
  const calls = parts(conversation, 0).filter((part) => part.type === "tool-call");
  assert.deepEqual(
    calls.map((call) => call.toolName),
    ["web_search", "code_execution"],
  );
  assert.deepEqual(calls[0].args, { type: "search", query: "lisbon weather" });
  assert.deepEqual(calls[1].args, { type: "exec", commands: ["ls -la"] });
  assert.equal(calls[1].result, "total 0");
});

test("a multimodal turn keeps its text and image when detection routes it here", () => {
  // Chat Completions records that carry a per-message id and timestamp satisfy
  // isOpenWebUIRecord, and reading only string content dropped the whole turn.
  const record = {
    title: "vision",
    messages: [
      {
        id: "1",
        role: "user",
        timestamp: 1_700_000_000,
        content: [
          { type: "text", text: "what is this" },
          { type: "image_url", image_url: { url: "data:image/png;base64,AAA" } },
        ],
      },
      { id: "2", role: "assistant", timestamp: 1_700_000_001, content: "a cat" },
    ],
  };

  assert.equal(isOpenWebUIRecord(record), true);
  const conversation = openWebUIRecordToConversation(record, "fallback");
  assert.ok(conversation);
  assert.equal(conversation.messages.length, 2);
  assert.equal(text(conversation, 0), "what is this");
  assert.deepEqual(
    parts(conversation, 0).map((part) => part.type),
    ["text", "image"],
  );
  assert.equal(parts(conversation, 0)[1].image, "data:image/png;base64,AAA");
  assert.equal(text(conversation, 1), "a cat");
});

test("hex and decimal character references in details attributes are decoded", () => {
  // Open WebUI decodes these with a full html-entities pass, so an apostrophe
  // arrives as &#x27; as readily as &#39; and both have to survive.
  const record = chatRecord({
    history: historyOf(
      [
        {
          id: "a",
          parentId: null,
          role: "assistant",
          timestamp: 1,
          content:
            '<details type="tool_calls" done="true" id="c1" name="ask" ' +
            'arguments="{&quot;q&quot;: &quot;it&#x27;s ok&quot;}" ' +
            'result="{&quot;a&quot;: &quot;it&#39;s done&quot;}">\n' +
            "<summary>Tool executed</summary>\n</details>\nfinished",
        },
      ],
      "a",
    ),
  });

  const conversation = openWebUIRecordToConversation(record, "fallback");
  assert.ok(conversation);
  const call = parts(conversation, 0).find((part) => part.type === "tool-call");
  assert.ok(call);
  assert.deepEqual(call.args, { q: "it's ok" });
  assert.deepEqual(call.result, { a: "it's done" });
});

test("a doubly escaped ampersand survives one decoding pass as literal text", () => {
  const record = chatRecord({
    history: historyOf(
      [
        {
          id: "a",
          parentId: null,
          role: "assistant",
          timestamp: 1,
          content:
            '<details type="tool_calls" done="true" id="c" name="n" ' +
            'arguments="{&quot;raw&quot;: &quot;&amp;#39;&quot;}">\n</details>',
        },
      ],
      "a",
    ),
  });

  const conversation = openWebUIRecordToConversation(record, "fallback");
  assert.ok(conversation);
  const call = parts(conversation, 0).find((part) => part.type === "tool-call");
  assert.ok(call);
  assert.deepEqual(call.args, { raw: "&#39;" });
});

test("an out of range timestamp is discarded rather than freezing the clock", () => {
  // Past 2^53 `previousTs + 1` stops advancing, so every later message would
  // land on one createdAt and the depth-first order would not survive a reload.
  for (const stamp of [1e16, 1.7e18, Number.MAX_VALUE]) {
    const record = chatRecord({
      history: historyOf(
        [
          { id: "u", parentId: null, role: "user", content: "q", timestamp: stamp },
          { id: "a1", parentId: "u", role: "assistant", content: "one", timestamp: 0 },
          { id: "a2", parentId: "a1", role: "user", content: "two", timestamp: 0 },
          { id: "a3", parentId: "a2", role: "assistant", content: "three", timestamp: 0 },
        ],
        "a3",
      ),
    });

    const conversation = openWebUIRecordToConversation(record, "fallback");
    assert.ok(conversation);
    const stamps = conversation.messages.map((message) => message.createdAt);
    assert.equal(new Set(stamps).size, stamps.length, `duplicate stamps for ${stamp}`);
    for (let i = 1; i < stamps.length; i++) {
      assert.ok(stamps[i] > stamps[i - 1], `not increasing for ${stamp}`);
    }
    const last = stamps[stamps.length - 1];
    assert.ok(
      Number.isSafeInteger(last) && last <= 8.64e15,
      `outside Date's range for ${stamp}`,
    );
  }

  // A stamp inside the range is still honoured.
  const ok = chatRecord({
    history: historyOf(
      [{ id: "u", parentId: null, role: "user", content: "q", timestamp: 1_700_000_000 }],
      "u",
    ),
  });
  const inRange = openWebUIRecordToConversation(ok, "f");
  assert.ok(inRange);
  assert.equal(inRange.messages[0].createdAt, 1_700_000_000_000);
});

test("a generated image given as a url is not wrapped into a broken data url", () => {
  const record = chatRecord({
    history: historyOf(
      [
        {
          id: "a",
          parentId: null,
          role: "assistant",
          content: "",
          timestamp: 1,
          output: [
            { type: "image_generation_call", result: "https://example.com/a.png" },
            { type: "message", content: [{ type: "output_text", text: "there" }] },
          ],
        },
      ],
      "a",
    ),
  });

  const conversation = openWebUIRecordToConversation(record, "fallback");
  assert.ok(conversation);
  const image = parts(conversation, 0).find((part) => part.type === "image");
  assert.ok(image);
  assert.equal(image.image, "https://example.com/a.png");
});

test("malformed tool arguments stay readable instead of posing as an args object", () => {
  const record = chatRecord({
    history: historyOf(
      [
        {
          id: "a",
          parentId: null,
          role: "assistant",
          timestamp: 1,
          content:
            '<details type="tool_calls" done="true" id="c" name="n" arguments="not json">\n</details>',
        },
      ],
      "a",
    ),
  });

  const conversation = openWebUIRecordToConversation(record, "fallback");
  assert.ok(conversation);
  const call = parts(conversation, 0).find((part) => part.type === "tool-call");
  assert.ok(call);
  assert.equal(typeof call.args, "object");
  assert.deepEqual(call.args, { arguments: "not json" });
});

test("a tool that returned only images carries no empty result body", () => {
  const record = chatRecord({
    history: historyOf(
      [
        {
          id: "a",
          parentId: null,
          role: "assistant",
          content: "",
          timestamp: 1,
          output: [
            { type: "function_call", call_id: "c1", name: "plot", arguments: "{}" },
            {
              type: "function_call_output",
              call_id: "c1",
              output: [{ type: "input_image", image_url: "data:image/png;base64,BBB" }],
            },
            { type: "message", content: [{ type: "output_text", text: "done" }] },
          ],
        },
      ],
      "a",
    ),
  });

  const conversation = openWebUIRecordToConversation(record, "fallback");
  assert.ok(conversation);
  const call = parts(conversation, 0).find((part) => part.type === "tool-call");
  assert.ok(call);
  assert.equal(call.result, undefined);
  assert.ok(parts(conversation, 0).some((part) => part.type === "image"));
});

test("a tool result keeps its text under any of the three part names, and portable image urls", () => {
  const record = chatRecord({
    history: historyOf(
      [
        {
          id: "a",
          parentId: null,
          role: "assistant",
          timestamp: 1,
          output: [
            { type: "function_call", call_id: "c1", name: "lookup", arguments: "{}" },
            {
              type: "function_call_output",
              call_id: "c1",
              output: [
                { type: "output_text", text: "first" },
                { type: "text", text: " second" },
                { type: "input_text", text: " third" },
                { type: "input_image", image_url: "https://example.com/chart.png" },
                { type: "input_image", image_url: "/api/v1/files/local-only" },
              ],
            },
          ],
        },
      ],
      "a",
    ),
  });

  const conversation = openWebUIRecordToConversation(record, "fallback");
  assert.ok(conversation);
  const [call] = parts(conversation, 0).filter((part) => part.type === "tool-call");
  assert.equal(call.result, "first second third");
  // The absolute url resolves anywhere; the Open WebUI route does not.
  assert.deepEqual(
    parts(conversation, 0).filter((part) => part.type === "image"),
    [{ type: "image", image: "https://example.com/chart.png" }],
  );
});

test("details markup inside a fence that was never closed stays code", () => {
  // An answer interrupted mid-block still quotes the markup, it does not use it.
  const cut =
    'Like this:\n```markdown\n<details type="reasoning" done="true">\n<summary>Thought</summary>\n> not mine\n</details>\n';
  const conversation = openWebUIRecordToConversation(
    chatRecord({
      history: historyOf(
        [{ id: "a", parentId: null, role: "assistant", content: cut, timestamp: 1 }],
        "a",
      ),
    }),
    "fallback",
  );
  assert.ok(conversation);
  assert.equal(parts(conversation, 0).some((part) => part.type === "reasoning"), false);
  assert.ok(text(conversation, 0).includes('<details type="reasoning"'));
});

test("a disconnected cycle does not displace the branch the user had open", () => {
  const record = chatRecord({
    history: historyOf(
      [
        { id: "u", parentId: null, role: "user", content: "question", timestamp: 1 },
        { id: "a", parentId: "u", role: "assistant", content: "the answer", timestamp: 2 },
        // A component reachable only through its own cycle.
        { id: "c1", parentId: "c2", role: "user", content: "orphaned one", timestamp: 3 },
        { id: "c2", parentId: "c1", role: "assistant", content: "orphaned two", timestamp: 4 },
      ],
      "a",
    ),
  });

  const conversation = openWebUIRecordToConversation(record, "fallback");
  assert.ok(conversation);
  assert.equal(conversation.messages.length, 4);
  // Studio reopens on the last message, so it has to be the selected branch.
  assert.equal(text(conversation, conversation.messages.length - 1), "the answer");
});

test("a stray inline fence does not swallow the tool calls after it", () => {
  // Markdown opens a block fence only at the start of a line. Treating any
  // unclosed ``` as one made a single mid-sentence backtick run quote the rest
  // of the message, so every tool call after it was lost as literal markup.
  const record = chatRecord({
    history: historyOf(
      [
        {
          id: "a",
          parentId: null,
          role: "assistant",
          timestamp: 1,
          content:
            "let me compute ```\n" +
            '<details type="tool_calls" done="true" id="c1" name="calculator" ' +
            'arguments="{}" result="7">\n<summary>Tool Executed</summary>\n</details>\n' +
            "the answer is 7",
        },
      ],
      "a",
    ),
  });

  const conversation = openWebUIRecordToConversation(record, "fallback");
  assert.ok(conversation);
  const call = parts(conversation, 0).find((part) => part.type === "tool-call");
  assert.ok(call, "the tool call was swallowed by the stray fence");
  assert.equal(call.toolName, "calculator");
  assert.equal(call.result, 7);
});

test("a reasoning block quoting one backtick run keeps the next tool call", () => {
  const record = chatRecord({
    history: historyOf(
      [
        {
          id: "a",
          parentId: null,
          role: "assistant",
          timestamp: 1,
          content:
            '<details type="reasoning" done="true">\n<summary>Thought</summary>\n' +
            "> the user asked about ```\n</details>\n" +
            '<details type="tool_calls" done="true" id="c9" name="real_tool" ' +
            'arguments="{}" result="42">\n<summary>Tool Executed</summary>\n</details>\n' +
            "final answer",
        },
      ],
      "a",
    ),
  });

  const conversation = openWebUIRecordToConversation(record, "fallback");
  assert.ok(conversation);
  assert.deepEqual(
    parts(conversation, 0).map((part) => part.type),
    ["reasoning", "tool-call", "text"],
  );
});

test("reasoning survives an empty summary next to populated content", () => {
  const record = chatRecord({
    history: historyOf(
      [
        {
          id: "a",
          parentId: null,
          role: "assistant",
          timestamp: 1,
          output: [
            // Summaries off: the array is present but empty.
            { type: "reasoning", id: "rs_1", summary: [], content: [{ type: "reasoning_text", text: "weighed it up" }] },
            { type: "message", id: "msg_1", role: "assistant", content: [{ type: "output_text", text: "Done." }] },
          ],
        },
      ],
      "a",
    ),
  });

  const conversation = openWebUIRecordToConversation(record, "fallback");
  assert.ok(conversation);
  assert.deepEqual(
    parts(conversation, 0).filter((part) => part.type === "reasoning"),
    [{ type: "reasoning", text: "weighed it up" }],
  );
  // A populated summary still wins over content.
  const both = openWebUIRecordToConversation(
    chatRecord({
      history: historyOf(
        [
          {
            id: "a",
            parentId: null,
            role: "assistant",
            timestamp: 1,
            output: [
              { type: "reasoning", summary: [{ type: "summary_text", text: "the summary" }], content: [{ type: "reasoning_text", text: "the raw trace" }] },
            ],
          },
        ],
        "a",
      ),
    }),
    "fallback",
  );
  assert.ok(both);
  assert.deepEqual(parts(both, 0), [{ type: "reasoning", text: "the summary" }]);
});
