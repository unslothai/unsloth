// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import vm from "node:vm";
import { after, before, test } from "node:test";
import ts from "@typescript/typescript6";
import { type ViteDevServer, createServer } from "vite";
import { toolCallReplayArguments } from "../src/features/chat/tool-call-arguments.ts";
import { codexLocalToolRoundId, startsNewCodexToolRound } from "../src/features/chat/codex-reasoning.ts";
import type { ParsedConversation } from "../src/features/chat/types.ts";

import { readSrc } from "./helpers/kit.ts";

let vite: ViteDevServer;
let parseImportText: (text: string, filename: string) => ParsedConversation[];
let messageToOpenAI: (message: {
  role: unknown;
  content: unknown;
  attachments?: unknown;
}) => unknown[];

function loadMessageToOpenAI(): typeof messageToOpenAI {
  const source = readSrc("features/chat/prompt-storage/prompt-storage-dialog.tsx");
  const start = source.indexOf("type OAIContentPart =");
  const end = source.indexOf("// ShareGPT training JSONL", start);
  assert.notEqual(start, -1, "message serializer start marker must exist");
  assert.notEqual(end, -1, "message serializer end marker must exist");
  const exactSerializer =
    source.slice(start, end) +
    "\nglobalThis.__messageToOpenAI = messageToOpenAI;\n";
  const javascript = ts.transpileModule(exactSerializer, {
    compilerOptions: {
      module: ts.ModuleKind.None,
      target: ts.ScriptTarget.ES2022,
    },
  }).outputText;
  const context = {
    unwrapPastedTextContent: (text: string) => text,
    toolResultModelText: (result: unknown) => result,
    toolCallReplayArguments,
    codexLocalToolRoundId,
    startsNewCodexToolRound,
  } as Record<string, unknown>;
  vm.runInNewContext(javascript, context);
  return context.__messageToOpenAI as typeof messageToOpenAI;
}

before(async () => {
  vite = await createServer({
    appType: "custom",
    server: { middlewareMode: true },
  });
  const loaded = await vite.ssrLoadModule(
    "/src/features/chat/utils/chat-import.ts",
  );
  parseImportText = loaded.parseImportText as typeof parseImportText;
  messageToOpenAI = loadMessageToOpenAI();
});

after(async () => {
  await vite.close();
});

test("markdown transcripts import as conversations", () => {
  const markdown = [
    "## User",
    "",
    "Hello",
    "",
    "## Assistant",
    "",
    "Hi there",
    "",
  ].join("\n");
  const conversations = parseImportText(markdown, "my-chat.md");
  assert.equal(conversations.length, 1);
  assert.equal(conversations[0].title, "my-chat");
  assert.deepEqual(
    conversations[0].messages.map(({ role, content }) => ({
      role,
      content: (content as ReadonlyArray<{ text: string }>)[0]?.text,
    })),
    [
      { role: "user", content: "Hello" },
      { role: "assistant", content: "Hi there" },
    ],
  );
});

test("message JSONL imports as one conversation", () => {
  const conversations = parseImportText(
    '{"role":"user","content":"Hello"}\n' +
      '{"role":"assistant","content":"Hi"}',
    "conversation-messages.jsonl",
  );

  assert.equal(conversations.length, 1);
  assert.equal(conversations[0].title, "conversation-messages");
  assert.deepEqual(
    conversations[0].messages.map(({ role }) => role),
    ["user", "assistant"],
  );
});

test("developer and assistant array content survive message JSONL import", () => {
  const image = "data:image/png;base64,QUFBQQ==";
  const [conversation] = parseImportText(
    '{"role":"developer","content":"Follow policy"}\n' +
      '{"role":"assistant","content":[{"type":"text","text":"Done"},{"type":"image_url","image_url":{"url":"' +
      image +
      '"}}]}',
    "conversation-messages.jsonl",
  );

  assert.deepEqual(
    conversation.messages.map(({ role }) => role),
    ["system", "assistant"],
  );
  assert.deepEqual(conversation.messages[1].content, [
    { type: "text", text: "Done" },
    { type: "image", image },
  ]);
});

for (const layout of ["messages", "training"]) {
  test(`assistant refusals survive ${layout} JSONL import`, () => {
    const refusal = "I cannot help with that request.";
    for (const assistant of [
      { role: "assistant", content: null, refusal },
      { role: "assistant", content: "", refusal },
      { role: "assistant", content: [], refusal },
      { role: "assistant", content: [{ type: "refusal", refusal }] },
    ]) {
      const messages = [
        { role: "user", content: "A question" },
        assistant,
        { role: "user", content: "Another question" },
      ];
      const input = layout === "training"
        ? JSON.stringify({ messages })
        : messages.map((message) => JSON.stringify(message)).join("\n");
      const [conversation] = parseImportText(input, "chat.jsonl");
      assert.deepEqual(
        conversation.messages.map(({ role }) => role),
        ["user", "assistant", "user"],
      );
      assert.deepEqual(conversation.messages[1].content, [{ type: "text", text: refusal }]);
      assert.equal(conversation.messages[1].parentId, conversation.messages[0].id);
      assert.equal(conversation.messages[2].parentId, conversation.messages[1].id);
    }
  });
}

test("assistant text takes precedence over refusal metadata on import", () => {
  const [conversation] = parseImportText(
    JSON.stringify({ messages: [{ role: "assistant", content: "An explanation.", refusal: "Refused" }] }),
    "chat.jsonl",
  );
  assert.deepEqual(conversation.messages[0].content, [{ type: "text", text: "An explanation." }]);
});

test("Responses API part types import instead of dropping every message", () => {
  const image = "data:image/png;base64,QUFBQQ==";
  const [conversation] = parseImportText(
    JSON.stringify({
      messages: [
        { role: "user", content: [{ type: "input_text", text: "Hi" }] },
        {
          role: "assistant",
          content: [
            { type: "output_text", text: "Hello" },
            { type: "input_image", image_url: image },
          ],
        },
      ],
    }),
    "chat.jsonl",
  );

  assert.deepEqual(
    conversation.messages.map(({ role }) => role),
    ["user", "assistant"],
  );
  assert.deepEqual(conversation.messages[0].content, [{ type: "text", text: "Hi" }]);
  assert.deepEqual(conversation.messages[1].content, [
    { type: "text", text: "Hello" },
    { type: "image", image },
  ]);
});

test("a verbatim Responses API object imports its input and output turns", () => {
  const [conversation] = parseImportText(
    JSON.stringify({
      object: "response",
      input: [
        { type: "message", role: "user", content: [{ type: "input_text", text: "Hi" }] },
      ],
      output: [
        { type: "reasoning", summary: [] },
        {
          type: "message",
          role: "assistant",
          content: [{ type: "output_text", text: "Hello" }],
        },
      ],
    }),
    "response.jsonl",
  );

  assert.deepEqual(
    conversation.messages.map(({ role }) => role),
    ["user", "assistant"],
  );
  assert.deepEqual(conversation.messages[1].content, [{ type: "text", text: "Hello" }]);
});

test("a Responses API reasoning item imports as a reasoning part", () => {
  const [conversation] = parseImportText(
    JSON.stringify({
      object: "response",
      output: [
        {
          type: "reasoning",
          summary: [{ type: "summary_text", text: "weighing the options" }],
        },
        {
          type: "message",
          role: "assistant",
          content: [{ type: "output_text", text: "Hello" }],
        },
      ],
    }),
    "response.jsonl",
  );

  assert.deepEqual(conversation.messages.map(({ role }) => role), ["assistant"]);
  assert.deepEqual(conversation.messages[0].content, [
    { type: "reasoning", text: "weighing the options" },
    { type: "text", text: "Hello" },
  ]);
});

test("a headerless csv keeps its first message", () => {
  const [conversation] = parseImportText('"user","hello"\n"assistant","hi"\n', "chat.csv");
  assert.deepEqual(
    conversation.messages.map((message) => [message.role, (message.content[0] as { text: string }).text]),
    [
      ["user", "hello"],
      ["assistant", "hi"],
    ],
  );
});

for (const header of ['"role","content"', "sender,message"]) {
  test(`a csv with a ${header} header row still skips it`, () => {
    const [conversation] = parseImportText(
      `${header}\n"user","hello"\n"assistant","hi"\n`,
      "chat.csv",
    );
    assert.deepEqual(
      conversation.messages.map((message) => [message.role, (message.content[0] as { text: string }).text]),
      [
        ["user", "hello"],
        ["assistant", "hi"],
      ],
    );
  });
}

test("a Responses object keeps tool calls and reads created_at seconds", () => {
  const [conversation] = parseImportText(
    JSON.stringify({
      object: "response",
      created_at: 1740000000,
      output: [
        {
          type: "function_call",
          call_id: "call_1",
          name: "get_time",
          arguments: { zone: "utc" },
        },
        { type: "function_call_output", call_id: "call_1", output: "noon" },
        {
          type: "message",
          role: "assistant",
          content: [{ type: "output_text", text: "It is noon" }],
        },
      ],
    }),
    "response.jsonl",
  );

  assert.deepEqual(
    conversation.messages.map(({ role }) => role),
    ["assistant", "assistant"],
  );
  const toolCall = conversation.messages[0].content[0] as {
    type: string;
    result?: string;
    args?: unknown;
  };
  assert.equal(toolCall.type, "tool-call");
  assert.equal(toolCall.result, "noon");
  assert.deepEqual(toolCall.args, { zone: "utc" });
  assert.ok(
    conversation.messages[0].createdAt > 1e12,
    "created_at seconds must be converted to milliseconds",
  );
});

test("mixed input messages and a string output import the answer too", () => {
  const [conversation] = parseImportText(
    JSON.stringify({
      input: [{ role: "user", content: "Q" }],
      output: "A",
    }),
    "eval.jsonl",
  );

  assert.deepEqual(
    conversation.messages.map(({ role }) => role),
    ["user", "assistant"],
  );
});

test("an empty messages array does not shadow a response output", () => {
  const [conversation] = parseImportText(
    JSON.stringify({
      messages: [],
      output: [
        {
          type: "message",
          role: "assistant",
          content: [{ type: "output_text", text: "Hello" }],
        },
      ],
    }),
    "response.jsonl",
  );

  assert.equal(conversation.messages.length, 1);
  assert.equal(conversation.messages[0].role, "assistant");
  assert.deepEqual(conversation.messages[0].content, [
    { type: "text", text: "Hello" },
  ]);
});

test("alpaca and prompt-completion records import as user/assistant turns", () => {
  for (const [record, expectedUser] of [
    [
      { instruction: "Say hi", input: "context", output: "Hello" },
      "Say hi\n\ncontext",
    ],
    [{ instruction: "Say hi", output: "Hello" }, "Say hi"],
    [{ prompt: "Q", completion: "A" }, "Q"],
  ] as const) {
    const [conversation] = parseImportText(
      JSON.stringify(record),
      "pairs.jsonl",
    );
    assert.deepEqual(
      conversation.messages.map(({ role }) => role),
      ["user", "assistant"],
    );
    assert.deepEqual(conversation.messages[0].content, [
      { type: "text", text: expectedUser },
    ]);
  }
});

test("ShareGPT import maps input/output roles and ChatML-keyed entries", () => {
  const [conversation] = parseImportText(
    JSON.stringify({
      conversations: [
        { from: "input", value: "Hi" },
        { from: "output", value: "Hello" },
        { role: "developer", content: "Be precise" },
        { role: "user", content: "Again" },
        { role: "assistant", content: "Sure" },
      ],
    }),
    "sharegpt.jsonl",
  );

  assert.deepEqual(
    conversation.messages.map(({ role }) => role),
    ["user", "assistant", "system", "user", "assistant"],
  );
});

test("ShareGPT import maps role aliases regardless of case and whitespace", () => {
  const [conversation] = parseImportText(
    JSON.stringify({
      conversations: [
        { from: "system", value: "Be brief" },
        { from: "user", value: "Hi" },
        { from: "assistant", value: "Hello" },
        { from: "Human", value: "Again" },
        { from: " GPT ", value: "Sure" },
        { from: "HUMAN", value: "Bye" },
        { from: "constructor", value: "Unknown" },
      ],
    }),
    "sharegpt.jsonl",
  );

  assert.deepEqual(
    conversation.messages.map(({ role }) => role),
    ["system", "user", "assistant", "user", "assistant", "user", "assistant"],
  );
});

test("ShareGPT import prefers non-empty role/content over empty from/value", () => {
  const [conversation] = parseImportText(
    JSON.stringify({
      conversations: [
        { from: "human", role: "", value: "Hi", content: "" },
        { from: "", role: "gpt", value: "", content: "Hello" },
        { role: "tool", content: "tool result" },
      ],
    }),
    "sharegpt.jsonl",
  );

  assert.deepEqual(
    conversation.messages.map(({ role, content }) => ({
      role,
      text: (content as ReadonlyArray<{ text: string }>)[0]?.text,
    })),
    [
      { role: "user", text: "Hi" },
      { role: "assistant", text: "Hello" },
    ],
  );
});

test("null entries in an import record are skipped, not fatal", () => {
  for (const record of [
    { conversations: [null, { from: "gpt", value: "Hello" }] },
    { messages: [null, { role: "user", content: "Hi" }] },
  ]) {
    const [conversation] = parseImportText(JSON.stringify(record), "chat.jsonl");

    assert.equal(conversation.messages.length, 1);
  }
});

function toolCallResults(conversation: ParsedConversation): string[] {
  return conversation.messages.flatMap((message) =>
    (message.content as ReadonlyArray<{ type: string; result?: unknown }>)
      .filter((part) => part.type === "tool-call")
      .map((part) => String(part.result)),
  );
}

test("each turn keeps its own tool result when a chat file reuses call ids", () => {
  const call =
    '{"role":"assistant","content":null,"tool_calls":[{"id":"tool_call_0","type":"function","function":{"name":"get_time","arguments":"{}"}}]}';
  const [conversation] = parseImportText(
    [
      '{"role":"user","content":"stopped"}',
      call,
      '{"role":"user","content":"first"}',
      call,
      '{"role":"tool","tool_call_id":"tool_call_0","name":"get_time","content":"RESULT_A"}',
      '{"role":"assistant","content":"It is A"}',
      '{"role":"user","content":"again"}',
      call,
      '{"role":"tool","tool_call_id":"tool_call_0","name":"get_time","content":"RESULT_B"}',
      '{"role":"assistant","content":"It is B"}',
    ].join("\n"),
    "conversation-messages.jsonl",
  );

  assert.deepEqual(toolCallResults(conversation), ["undefined", "RESULT_A", "RESULT_B"]);
});

test("parallel calls sharing an id in one turn take their results in call order", () => {
  const [conversation] = parseImportText(
    [
      '{"role":"user","content":"weather"}',
      '{"role":"assistant","content":null,"tool_calls":[' +
        '{"id":"call_0","type":"function","function":{"name":"get_weather","arguments":"{\\"city\\":\\"Paris\\"}"}},' +
        '{"id":"call_0","type":"function","function":{"name":"get_weather","arguments":"{\\"city\\":\\"Tokyo\\"}"}}]}',
      '{"role":"tool","tool_call_id":"call_0","name":"get_weather","content":"PARIS"}',
      '{"role":"tool","tool_call_id":"call_0","name":"get_weather","content":"TOKYO"}',
    ].join("\n"),
    "conversation-messages.jsonl",
  );

  assert.deepEqual(toolCallResults(conversation), ["PARIS", "TOKYO"]);
});

test("out-of-order tool results with unique ids still reach their call", () => {
  const [conversation] = parseImportText(
    '{"role":"tool","tool_call_id":"call_early","name":"lookup","content":"EARLY"}\n' +
      '{"role":"assistant","content":null,"tool_calls":[' +
      '{"id":"call_early","type":"function","function":{"name":"lookup","arguments":"{}"}},' +
      '{"id":"call_late","type":"function","function":{"name":"lookup","arguments":"{}"}}]}\n' +
      '{"role":"assistant","content":"Waiting"}\n' +
      '{"role":"user","content":"Any news?"}\n' +
      '{"role":"tool","tool_call_id":"call_late","name":"lookup","content":"LATE"}',
    "conversation-messages.jsonl",
  );

  assert.deepEqual(toolCallResults(conversation), ["EARLY", "LATE"]);
});

test("assistant images are represented explicitly in JSONL exports", () => {
  const exported = structuredClone(
    messageToOpenAI({
      role: "assistant",
      content: [
        { type: "text", text: "Chart" },
        { type: "image", image: "data:image/png;base64,QUFBQQ==" },
      ],
    }),
  );
  assert.deepEqual(
    exported,
    [{ role: "assistant", content: "Chart\n\n[image attachment]" }],
  );
});

function toolCallPart(id: string, query: string, result: string) {
  return {
    type: "tool-call",
    toolCallId: id,
    toolName: "web_search",
    args: { query },
    argsText: JSON.stringify({ query }),
    result,
  };
}

function webSearchCall(id: string, query: string) {
  return {
    id,
    type: "function",
    function: { name: "web_search", arguments: JSON.stringify({ query }) },
  };
}

test("JSONL exports put the answer after the tool result it used", () => {
  const exported = structuredClone(
    messageToOpenAI({
      role: "assistant",
      content: [
        { type: "reasoning", text: "I should search." },
        { type: "text", text: "Let me look that up." },
        toolCallPart("c1", "capital of Australia", "Canberra is the capital."),
        { type: "text", text: "The capital of Australia is Canberra." },
      ],
    }),
  );
  assert.deepEqual(exported, [
    {
      role: "assistant",
      content: "<thinking>\nI should search.\n</thinking>\n\nLet me look that up.",
      tool_calls: [webSearchCall("c1", "capital of Australia")],
    },
    { role: "tool", tool_call_id: "c1", name: "web_search", content: "Canberra is the capital." },
    { role: "assistant", content: "The capital of Australia is Canberra." },
  ]);
});

test("JSONL exports keep each tool round as its own assistant turn", () => {
  const exported = structuredClone(
    messageToOpenAI({
      role: "assistant",
      content: [
        toolCallPart("a", "first", "A"),
        { type: "text", text: "Let me check more." },
        toolCallPart("b", "second", "B"),
        { type: "text", text: "Final." },
      ],
    }),
  );
  assert.deepEqual(exported, [
    { role: "assistant", content: null, tool_calls: [webSearchCall("a", "first")] },
    { role: "tool", tool_call_id: "a", name: "web_search", content: "A" },
    { role: "assistant", content: "Let me check more.", tool_calls: [webSearchCall("b", "second")] },
    { role: "tool", tool_call_id: "b", name: "web_search", content: "B" },
    { role: "assistant", content: "Final." },
  ]);
});

test("JSONL exports split Studio's back-to-back searches into rounds", () => {
  const local = { provenance: { source: "local" } };
  const exported = structuredClone(
    messageToOpenAI({
      role: "assistant",
      content: [
        { ...toolCallPart("a", "first", "No results found."), ...local },
        { ...toolCallPart("b", "second", "B"), ...local },
        { type: "text", text: "Final." },
      ],
    }),
  );
  assert.deepEqual(exported, [
    { role: "assistant", content: null, tool_calls: [webSearchCall("a", "first")] },
    { role: "tool", tool_call_id: "a", name: "web_search", content: "No results found." },
    { role: "assistant", content: null, tool_calls: [webSearchCall("b", "second")] },
    { role: "tool", tool_call_id: "b", name: "web_search", content: "B" },
    { role: "assistant", content: "Final." },
  ]);
});

test("JSONL exports keep a Studio round's parallel searches in one turn", () => {
  const round = (round_id: number) => ({ provenance: { source: "local", round_id } });
  const exported = structuredClone(
    messageToOpenAI({
      role: "assistant",
      content: [
        { ...toolCallPart("a", "first", "A"), ...round(1) },
        { ...toolCallPart("b", "second", "B"), ...round(1) },
        { ...toolCallPart("c", "third", "C"), ...round(2) },
        { type: "text", text: "Final." },
      ],
    }),
  );
  assert.deepEqual(exported, [
    {
      role: "assistant",
      content: null,
      tool_calls: [webSearchCall("a", "first"), webSearchCall("b", "second")],
    },
    { role: "tool", tool_call_id: "a", name: "web_search", content: "A" },
    { role: "tool", tool_call_id: "b", name: "web_search", content: "B" },
    { role: "assistant", content: null, tool_calls: [webSearchCall("c", "third")] },
    { role: "tool", tool_call_id: "c", name: "web_search", content: "C" },
    { role: "assistant", content: "Final." },
  ]);
});

test("JSONL exports keep parallel tool calls in one assistant turn", () => {
  const exported = structuredClone(
    messageToOpenAI({
      role: "assistant",
      content: [
        toolCallPart("a", "first", "A"),
        toolCallPart("b", "second", "B"),
        { type: "text", text: "Both done." },
      ],
    }),
  );
  assert.deepEqual(exported, [
    {
      role: "assistant",
      content: null,
      tool_calls: [webSearchCall("a", "first"), webSearchCall("b", "second")],
    },
    { role: "tool", tool_call_id: "a", name: "web_search", content: "A" },
    { role: "tool", tool_call_id: "b", name: "web_search", content: "B" },
    { role: "assistant", content: "Both done." },
  ]);
});

test("JSONL exports keep a call without a result in the turn of the text after it", () => {
  const { result: _omitted, ...unanswered } = toolCallPart("a", "first", "");
  const exported = structuredClone(
    messageToOpenAI({
      role: "assistant",
      content: [unanswered, { type: "text", text: "Answer." }],
    }),
  );
  assert.deepEqual(exported, [
    { role: "assistant", content: "Answer.", tool_calls: [webSearchCall("a", "first")] },
  ]);
});

test("imports without message send times mark their ordering timestamps as estimated", () => {
  for (const [filename, source] of [
    ["messages.jsonl", '{"messages":[{"role":"user","content":"Hello"}]}'],
    ["sharegpt.jsonl", '{"created_at":1700000000000,"conversations":[{"from":"human","value":"Hello"}]}'],
    ["messages.csv", "role,content\nuser,Hello"],
  ]) {
    const conversation = parseImportText(source, filename)[0];
    assert.ok(conversation, filename);
    assert.equal(conversation.messages[0].metadata?.createdAtEstimated, true, filename);
  }
});

test("a Responses object keeps its instructions and custom tool call input", () => {
  const [conversation] = parseImportText(
    JSON.stringify({
      object: "response",
      instructions: "Answer tersely.",
      input: [
        { role: "user", content: [{ type: "input_text", text: "Patch it" }] },
        { type: "custom_tool_call", call_id: "c1", name: "apply_patch", input: "*** Begin Patch" },
        { type: "custom_tool_call_output", call_id: "c1", output: "Done" },
      ],
      output: [
        { type: "message", role: "assistant", content: [{ type: "output_text", text: "Patched." }] },
      ],
    }),
    "response.json",
  );

  assert.equal(conversation.messages[0].role, "system");
  assert.deepEqual(conversation.messages[0].content, [{ type: "text", text: "Answer tersely." }]);
  const call = conversation.messages
    .flatMap((m) => m.content as ReadonlyArray<{ type?: string; toolName?: string; args?: unknown; result?: unknown }>)
    .find((p) => p.type === "tool-call");
  assert.equal(call?.toolName, "apply_patch");
  assert.deepEqual(call?.args, { input: "*** Begin Patch" });
  assert.equal(call?.result, "Done");
});

test("a Responses object keeps parallel calls in one turn, namespaces and generated images", () => {
  const [conversation] = parseImportText(
    JSON.stringify({
      object: "response",
      input: [
        { role: "user", content: "Weather and a picture?" },
        { type: "function_call", call_id: "a", namespace: "mcp__weather", name: "get_forecast", arguments: "{}" },
        { type: "function_call", call_id: "b", name: "get_time", arguments: "{}" },
        { type: "function_call_output", call_id: "a", output: "Sunny" },
        { type: "function_call_output", call_id: "b", output: "Noon" },
      ],
      output: [{ type: "image_generation_call", id: "ig", result: "QUJD", output_format: "webp" }],
    }),
    "response.json",
  );

  const roles = conversation.messages.map((m) => m.role);
  assert.deepEqual(roles, ["user", "assistant", "assistant"]);
  const calls = conversation.messages[1].content as ReadonlyArray<{ toolName?: string; result?: unknown }>;
  assert.deepEqual(
    calls.map((p) => [p.toolName, p.result]),
    [["mcp__weather__get_forecast", "Sunny"], ["get_time", "Noon"]],
  );
  assert.deepEqual(conversation.messages[2].content, [
    { type: "image", image: "data:image/webp;base64,QUJD" },
  ]);
});

test("alpaca system, text-part tool outputs and bundled call outputs survive import", () => {
  const [alpaca] = parseImportText(
    JSON.stringify({ system: "Be brief.", instruction: "Name a prime", input: "", output: "7" }),
    "alpaca.jsonl",
  );
  assert.deepEqual(alpaca.messages.map((m) => m.role), ["system", "user", "assistant"]);

  const [response] = parseImportText(
    JSON.stringify({
      object: "response",
      input: [
        { role: "user", content: "Run it" },
        { type: "function_call", call_id: "f1", name: "read", arguments: "{}" },
        { type: "function_call_output", call_id: "f1", output: [{ type: "input_text", text: "file body" }] },
        { type: "shell_call", call_id: "s1", action: { commands: ["ls"] }, output: "a.txt" },
      ],
      output: [{ type: "message", role: "assistant", content: [{ type: "output_text", text: "Done." }] }],
    }),
    "response.json",
  );
  const calls = response.messages.flatMap(
    (m) => m.content as ReadonlyArray<{ type?: string; result?: unknown }>,
  ).filter((p) => p.type === "tool-call");
  assert.deepEqual(calls.map((p) => p.result), ["file body", "a.txt"]);
  assert.equal(response.messages.at(-1)?.role, "assistant");
  assert.deepEqual((response.messages.at(-1)?.content as ReadonlyArray<unknown>).at(-1), { type: "text", text: "Done." });
});

test("a finished web_search_call imports as an answered call", () => {
  const [conversation] = parseImportText(
    JSON.stringify({
      object: "response",
      input: [{ role: "user", content: "News?" }],
      output: [
        { type: "web_search_call", id: "ws1", status: "completed", action: { type: "search", query: "news" } },
        { type: "message", role: "assistant", content: [{ type: "output_text", text: "Here is the news." }] },
      ],
    }),
    "response.json",
  );

  const call = conversation.messages
    .flatMap((m) => m.content as ReadonlyArray<{ type?: string; toolName?: string; result?: unknown }>)
    .find((p) => p.type === "tool-call");
  assert.equal(call?.toolName, "web_search");
  assert.equal(call?.result, "completed");
  assert.deepEqual((conversation.messages.at(-1)?.content as ReadonlyArray<unknown>).at(-1), { type: "text", text: "Here is the news." });
});

test("a builtin call keeps its explicit output item over a status placeholder", () => {
  const [conversation] = parseImportText(
    JSON.stringify({
      object: "response",
      input: [
        { role: "user", content: "List files" },
        { type: "shell_call", call_id: "s1", status: "completed", action: { commands: ["ls"] } },
        { type: "shell_call_output", call_id: "s1", output: "a.txt" },
      ],
      output: [{ type: "message", role: "assistant", content: [{ type: "output_text", text: "One file." }] }],
    }),
    "response.json",
  );

  const call = conversation.messages
    .flatMap((m) => m.content as ReadonlyArray<{ type?: string; result?: unknown }>)
    .find((p) => p.type === "tool-call");
  assert.equal(call?.result, "a.txt");
});

test("hosted Responses calls import as server-side tool cards with their code and logs", () => {
  const [conversation] = parseImportText(
    JSON.stringify({
      object: "response",
      input: [{ role: "user", content: "Compute" }],
      output: [
        { type: "web_search_call", id: "ws1", status: "completed", action: { type: "search", query: "q" } },
        { type: "code_interpreter_call", id: "ci1", status: "completed", code: "print(1)", outputs: [{ type: "logs", logs: "1" }] },
        { type: "message", role: "assistant", content: [{ type: "output_text", text: "1" }] },
      ],
    }),
    "response.json",
  );

  const calls = conversation.messages
    .flatMap((m) => m.content as ReadonlyArray<{ type?: string; toolName?: string; args?: Record<string, unknown>; result?: unknown }>)
    .filter((p) => p.type === "tool-call");
  assert.deepEqual(
    calls.map((p) => [p.toolName, p.args?._server_tool, p.result]),
    [["web_search", true, "completed"], ["code_execution", true, "1"]],
  );
  assert.equal(calls[1].args?.command, "print(1)");
});

test("a Responses mcp_call keeps its server and tool name", () => {
  const [conversation] = parseImportText(
    JSON.stringify({
      object: "response",
      input: [{ role: "user", content: "Find docs" }],
      output: [
        { type: "mcp_call", id: "m1", server_label: "docs", name: "search_docs", arguments: "{\"q\":\"x\"}", output: "hit" },
        { type: "message", role: "assistant", content: [{ type: "output_text", text: "Found." }] },
      ],
    }),
    "response.json",
  );

  const call = conversation.messages
    .flatMap((m) => m.content as ReadonlyArray<{ type?: string; toolName?: string; result?: unknown }>)
    .find((p) => p.type === "tool-call");
  assert.equal(call?.toolName, "docs__search_docs");
  assert.equal(call?.result, "hit");
});

test("Responses shell calls: hosted ones render their command, client ones and MCP errors stay honest", () => {
  const [conversation] = parseImportText(
    JSON.stringify({
      object: "response",
      input: [
        { role: "user", content: "Go" },
        { type: "shell_call", call_id: "s1", status: "completed", action: { commands: ["ls", "pwd"] } },
        { type: "local_shell_call", call_id: "l1", action: { type: "exec", command: ["ls"] } },
        { type: "local_shell_call_output", call_id: "l1", output: "local.txt" },
        { type: "mcp_call", id: "m1", server_label: "docs", name: "search", arguments: "{}", output: null, error: "Authentication failed" },
      ],
    }),
    "response.json",
  );

  const calls = conversation.messages
    .flatMap((m) => m.content as ReadonlyArray<{ type?: string; toolName?: string; args?: Record<string, unknown>; result?: unknown }>)
    .filter((p) => p.type === "tool-call");
  assert.deepEqual(calls[0].args, { kind: "bash", command: "ls\npwd", _server_tool: true });
  assert.equal(calls[1].toolName, "local_shell");
  assert.equal(calls[1].args?._server_tool, undefined);
  assert.equal(calls[1].result, "local.txt");
  assert.equal(calls[2].result, "Authentication failed");
});

test("images in Responses tool outputs and code interpreter outputs stay images", () => {
  const [conversation] = parseImportText(
    JSON.stringify({
      object: "response",
      input: [
        { role: "user", content: "Plot" },
        { type: "function_call", call_id: "f1", name: "snap", arguments: "{}" },
        {
          type: "function_call_output",
          call_id: "f1",
          output: [
            { type: "input_text", text: "shot" },
            { type: "input_image", image_url: "data:image/png;base64,QUJD" },
          ],
        },
      ],
      output: [
        {
          type: "code_interpreter_call",
          id: "ci1",
          code: "plot()",
          outputs: [{ type: "logs", logs: "ok" }, { type: "image", url: "https://example.com/p.png" }],
        },
      ],
    }),
    "response.json",
  );

  const results = conversation.messages
    .flatMap((m) => m.content as ReadonlyArray<{ type?: string; result?: unknown }>)
    .filter((p) => p.type === "tool-call")
    .map((p) => p.result);
  assert.deepEqual(results, [
    { text: "shot", images: [{ mimeType: "image/png", data: "QUJD" }] },
    "ok\nhttps://example.com/p.png",
  ]);
});

test("file search results and whitespace-sensitive prompts survive import", () => {
  const [search] = parseImportText(
    JSON.stringify({
      object: "response",
      input: [{ role: "user", content: "Find it" }],
      output: [
        { type: "file_search_call", id: "fs1", status: "completed", queries: ["refund policy"], results: [{ file_id: "f", filename: "p.md", text: "30 days" }] },
      ],
    }),
    "response.json",
  );
  const call = search.messages
    .flatMap((m) => m.content as ReadonlyArray<{ type?: string; args?: unknown; result?: unknown }>)
    .find((p) => p.type === "tool-call");
  assert.deepEqual(call?.args, { queries: ["refund policy"], _server_tool: true });
  assert.equal(call?.result, "30 days");

  const [pair] = parseImportText(
    JSON.stringify({ prompt: "def f():\n    ", completion: "return 1" }),
    "pairs.jsonl",
  );
  assert.deepEqual(pair.messages[0].content, [{ type: "text", text: "def f():\n    " }]);
});

test("hosted results do not split the assistant run they belong to", () => {
  const [conversation] = parseImportText(
    JSON.stringify({
      object: "response",
      input: [{ role: "user", content: "News?" }],
      output: [
        { type: "web_search_call", id: "ws1", status: "completed", action: { type: "search", query: "news" } },
        { type: "shell_call", call_id: "s1", action: { commands: ["date"] } },
        { type: "shell_call_output", call_id: "s1", output: "Monday" },
        { type: "message", role: "assistant", content: [{ type: "output_text", text: "Here is the news." }] },
      ],
    }),
    "response.json",
  );

  assert.deepEqual(conversation.messages.map((m) => m.role), ["user", "assistant"]);
  const parts = conversation.messages[1].content as ReadonlyArray<{ type?: string; text?: string; result?: unknown }>;
  assert.deepEqual(
    parts.map((p) => (p.type === "tool-call" ? p.result : p.text)),
    ["completed", "Monday", "Here is the news."],
  );
});

test("string output, id-keyed local shell output and lone screenshots import intact", () => {
  const [conversation] = parseImportText(
    JSON.stringify({
      object: "response",
      input: [
        { role: "user", content: "Check" },
        { type: "local_shell_call", id: "lsc1", call_id: "l1", action: { type: "exec", command: ["date"] } },
        { type: "local_shell_call_output", id: "l1", output: "Monday" },
        { type: "computer_call", call_id: "c1", action: { type: "screenshot" } },
        { type: "computer_call_output", call_id: "c1", output: { type: "computer_screenshot", image_url: "data:image/png;base64,QUJD" } },
        { type: "web_search_call", id: "ws1", status: "completed", action: { type: "search", query: "q" } },
      ],
      output: "All good.",
    }),
    "response.json",
  );

  const results = conversation.messages
    .flatMap((m) => m.content as ReadonlyArray<{ type?: string; result?: unknown }>)
    .filter((p) => p.type === "tool-call")
    .map((p) => p.result);
  assert.deepEqual(results, ["Monday", { text: "", images: [{ mimeType: "image/png", data: "QUJD" }] }, "completed"]);
  const last = conversation.messages.at(-1);
  assert.equal(last?.role, "assistant");
  assert.ok((last?.content as ReadonlyArray<{ type?: string }>).some((p) => p.type === "tool-call"));
});

test("raw Responses item lines do not import as standalone answers", () => {
  const conversations = parseImportText(
    [
      { type: "function_call", call_id: "c", name: "add", arguments: "{}" },
      { type: "function_call_output", call_id: "c", output: "8" },
    ].map((item) => JSON.stringify(item)).join("\n"),
    "items.jsonl",
  );

  assert.deepEqual(conversations, []);
});

test("a hosted call before its answer exports in that order", () => {
  const [conversation] = parseImportText(
    JSON.stringify({
      object: "response",
      input: [{ role: "user", content: "Compute" }],
      output: [
        { type: "code_interpreter_call", id: "ci1", code: "print(2)", outputs: [{ type: "logs", logs: "2" }] },
        { type: "message", role: "assistant", content: [{ type: "output_text", text: "It printed 2." }] },
      ],
    }),
    "response.json",
  );

  const exported = messageToOpenAI({
    role: "assistant",
    content: conversation.messages[1].content,
  }) as Array<{ role: string; content?: unknown }>;
  const answerAt = exported.findIndex(
    (m) => m.role === "assistant" && JSON.stringify(m.content ?? "").includes("It printed 2."),
  );
  assert.equal(answerAt, exported.length - 1);
});

test("legacy function_call turns survive message JSONL import", () => {
  const [conversation] = parseImportText(
    [
      '{"role":"user","content":"Weather in Paris?"}',
      '{"role":"assistant","content":null,"function_call":{"name":"get_weather","arguments":"{\\"city\\":\\"Paris\\"}"}}',
      '{"role":"function","name":"get_weather","content":"72F sunny"}',
      '{"role":"assistant","content":"It is 72 and sunny."}',
    ].join("\n"),
    "conversation-messages.jsonl",
  );

  assert.deepEqual(
    conversation.messages.map(({ role }) => role),
    ["user", "assistant", "assistant"],
  );
  assert.deepEqual(toolCallResults(conversation), ["72F sunny"]);
  const call = (
    conversation.messages[1].content as ReadonlyArray<{
      type: string;
      toolName?: string;
      args?: unknown;
    }>
  )[0];
  assert.equal(call.type, "tool-call");
  assert.equal(call.toolName, "get_weather");
  assert.deepEqual(call.args, { city: "Paris" });
});

test("legacy function results reach their call inside a messages record", () => {
  const [conversation] = parseImportText(
    JSON.stringify({
      messages: [
        { role: "user", content: "Add 2+2" },
        {
          role: "assistant",
          content: null,
          function_call: { name: "calc", arguments: '{"e":"2+2"}' },
        },
        { role: "function", name: "calc", content: "4" },
        { role: "assistant", content: "4" },
      ],
    }),
    "chat.jsonl",
  );

  assert.deepEqual(
    conversation.messages.map(({ role }) => role),
    ["user", "assistant", "assistant"],
  );
  assert.deepEqual(toolCallResults(conversation), ["4"]);
});

test("each legacy function result reaches the call that named it", () => {
  const [conversation] = parseImportText(
    [
      '{"role":"user","content":"Time and weather?"}',
      '{"role":"assistant","content":null,"function_call":{"name":"get_time","arguments":"{}"}}',
      '{"role":"assistant","content":null,"function_call":{"name":"get_weather","arguments":"{}"}}',
      '{"role":"function","name":"get_time","content":"NOON"}',
      '{"role":"function","name":"get_weather","content":"RAIN"}',
    ].join("\n"),
    "conversation-messages.jsonl",
  );

  assert.deepEqual(toolCallResults(conversation), ["NOON", "RAIN"]);
});

test("a legacy function result ahead of its call still lands on it", () => {
  const [conversation] = parseImportText(
    '{"role":"function","name":"lookup","content":"EARLY"}\n' +
      '{"role":"assistant","content":null,"function_call":{"name":"lookup","arguments":"{}"}}\n' +
      '{"role":"assistant","content":"Done"}',
    "conversation-messages.jsonl",
  );

  assert.deepEqual(toolCallResults(conversation), ["EARLY"]);
});

test("assistant text stays beside its legacy function call", () => {
  const [conversation] = parseImportText(
    '{"role":"assistant","content":"Checking.","function_call":{"name":"lookup","arguments":"not json"}}\n' +
      '{"role":"function","name":"lookup","content":"FOUND"}',
    "conversation-messages.jsonl",
  );

  const call = (
    conversation.messages[0].content as ReadonlyArray<{
      type: string;
      toolCallId: string;
    }>
  )[1];
  assert.equal(typeof call.toolCallId, "string");
  assert.notEqual(call.toolCallId.length, 0);
  assert.deepEqual(conversation.messages[0].content, [
    { type: "text", text: "Checking." },
    {
      type: "tool-call",
      toolCallId: call.toolCallId,
      toolName: "lookup",
      args: { _raw: "not json" },
      argsText: "not json",
      result: "FOUND",
    },
  ]);
});

test("already-parsed legacy function arguments survive import", () => {
  const [conversation] = parseImportText(
    '{"role":"assistant","content":null,"function_call":{"name":"f","arguments":{"a":1}}}',
    "conversation-messages.jsonl",
  );

  const call = (
    conversation.messages[0].content as ReadonlyArray<{
      args?: unknown;
      argsText?: string;
    }>
  )[0];
  assert.deepEqual(call.args, { a: 1 });
  assert.equal(call.argsText, '{"a":1}');
});

test("a structured legacy function result imports as json text", () => {
  const [conversation] = parseImportText(
    '{"role":"assistant","content":null,"function_call":{"name":"f","arguments":"{}"}}\n' +
      '{"role":"function","name":"f","content":{"rows":3}}',
    "conversation-messages.jsonl",
  );

  assert.deepEqual(toolCallResults(conversation), ['{"rows":3}']);
});

test("an assistant line carrying only function_call still counts as a message", () => {
  const [conversation] = parseImportText(
    '{"role":"assistant","function_call":{"name":"f","arguments":"{}"}}',
    "conversation-messages.jsonl",
  );

  assert.equal(conversation.messages.length, 1);
  assert.deepEqual(toolCallResults(conversation), ["undefined"]);
});

test("an early function result reaches only one same-name call", () => {
  const [conversation] = parseImportText(
    '{"role":"function","name":"f","content":"EARLY"}\n' +
      '{"role":"assistant","content":null,"function_call":{"name":"f","arguments":"{}"}}\n' +
      '{"role":"assistant","content":null,"function_call":{"name":"f","arguments":"{}"}}',
    "conversation-messages.jsonl",
  );

  assert.deepEqual(toolCallResults(conversation), ["EARLY", "undefined"]);
});

test("early function results land on same-name calls in order", () => {
  const [conversation] = parseImportText(
    '{"role":"function","name":"f","content":"FIRST"}\n' +
      '{"role":"function","name":"f","content":"SECOND"}\n' +
      '{"role":"assistant","content":null,"function_call":{"name":"f","arguments":"{}"}}\n' +
      '{"role":"assistant","content":null,"function_call":{"name":"f","arguments":"{}"}}',
    "conversation-messages.jsonl",
  );

  assert.deepEqual(toolCallResults(conversation), ["FIRST", "SECOND"]);
});

test("batched same-name results bind the newest call still waiting", () => {
  const [conversation] = parseImportText(
    '{"role":"assistant","content":null,"function_call":{"name":"f","arguments":"{}"}}\n' +
      '{"role":"assistant","content":null,"function_call":{"name":"f","arguments":"{}"}}\n' +
      '{"role":"function","name":"f","content":"R1"}\n' +
      '{"role":"function","name":"f","content":"R2"}',
    "conversation-messages.jsonl",
  );

  assert.deepEqual(toolCallResults(conversation), ["R2", "R1"]);
});

test("a legacy function result can name a modern tool call", () => {
  const [conversation] = parseImportText(
    '{"role":"assistant","content":null,"tool_calls":[{"id":"call_1","type":"function","function":{"name":"search","arguments":"{}"}}]}\n' +
      '{"role":"function","name":"search","content":"RESULT"}',
    "conversation-messages.jsonl",
  );

  assert.deepEqual(toolCallResults(conversation), ["RESULT"]);
});

test("a message-shaped record keeps its nested messages conversation", () => {
  const [conversation] = parseImportText(
    '{"role":"assistant","function_call":{"name":"f","arguments":"{}"},' +
      '"messages":[{"role":"user","content":"hi"}]}',
    "chat.jsonl",
  );

  assert.deepEqual(
    conversation.messages.map(({ role }) => role),
    ["user"],
  );
});

test("flat tool_calls entries keep their name and arguments", () => {
  const [conversation] = parseImportText(
    '{"role":"assistant","content":null,"tool_calls":[{"id":"t1","name":"search","arguments":{"q":"x"}}]}\n' +
      '{"role":"tool","tool_call_id":"t1","content":"HIT"}',
    "conversation-messages.jsonl",
  );

  const call = (
    conversation.messages[0].content as ReadonlyArray<{
      toolCallId?: string;
      toolName?: string;
      args?: unknown;
    }>
  )[0];
  assert.equal(call.toolCallId, "t1");
  assert.equal(call.toolName, "search");
  assert.deepEqual(call.args, { q: "x" });
  assert.deepEqual(toolCallResults(conversation), ["HIT"]);
});

test("already-parsed arguments on a modern tool call survive import", () => {
  const [conversation] = parseImportText(
    '{"role":"assistant","content":null,"tool_calls":[{"id":"t1","type":"function","function":{"name":"f","arguments":{"a":1}}}]}',
    "conversation-messages.jsonl",
  );

  const call = (
    conversation.messages[0].content as ReadonlyArray<{
      args?: unknown;
      argsText?: string;
    }>
  )[0];
  assert.deepEqual(call.args, { a: 1 });
  assert.equal(call.argsText, '{"a":1}');
});

test("call_id identifies a flat tool call", () => {
  const [conversation] = parseImportText(
    '{"role":"assistant","content":null,"tool_calls":[{"call_id":"c9","name":"f","arguments":"{}"}]}\n' +
      '{"role":"tool","tool_call_id":"c9","content":"OK"}',
    "conversation-messages.jsonl",
  );

  const call = (
    conversation.messages[0].content as ReadonlyArray<{ toolCallId?: string }>
  )[0];
  assert.equal(call.toolCallId, "c9");
  assert.deepEqual(toolCallResults(conversation), ["OK"]);
});

test("a tool result that only names its call still reaches it", () => {
  const [conversation] = parseImportText(
    '{"role":"assistant","content":null,"tool_calls":[{"id":"t1","type":"function","function":{"name":"f","arguments":"{}"}}]}\n' +
      '{"role":"tool","name":"f","content":"BYNAME"}',
    "conversation-messages.jsonl",
  );

  assert.deepEqual(toolCallResults(conversation), ["BYNAME"]);
});

test("a tool result falls back to its name when its id matches nothing", () => {
  const [conversation] = parseImportText(
    '{"role":"assistant","content":null,"function_call":{"name":"f","arguments":"{}"}}\n' +
      '{"role":"tool","tool_call_id":"unrelated","name":"f","content":"FELLBACK"}',
    "conversation-messages.jsonl",
  );

  assert.deepEqual(toolCallResults(conversation), ["FELLBACK"]);
});

test("an unnamed function result lands on the newest unanswered call", () => {
  const [conversation] = parseImportText(
    '{"role":"assistant","content":null,"function_call":{"name":"f","arguments":"{}"}}\n' +
      '{"role":"assistant","content":null,"function_call":{"name":"g","arguments":"{}"}}\n' +
      '{"role":"function","content":"NONAME"}',
    "conversation-messages.jsonl",
  );

  assert.deepEqual(toolCallResults(conversation), ["undefined", "NONAME"]);
});

test("unnamed results for parallel calls bind in call order", () => {
  const [conversation] = parseImportText(
    '{"role":"assistant","content":null,"tool_calls":[{"name":"f","arguments":{}},{"name":"g","arguments":{}}]}\n' +
      '{"role":"tool","content":"FIRST"}\n' +
      '{"role":"tool","content":"SECOND"}',
    "conversation-messages.jsonl",
  );

  assert.deepEqual(toolCallResults(conversation), ["FIRST", "SECOND"]);
});

test("an unnamed early function result waits for a call", () => {
  const [conversation] = parseImportText(
    '{"role":"function","content":"ORPHAN"}\n' +
      '{"role":"assistant","content":null,"function_call":{"name":"f","arguments":"{}"}}',
    "conversation-messages.jsonl",
  );

  assert.deepEqual(toolCallResults(conversation), ["ORPHAN"]);
});

test("a function_call nested under additional_kwargs survives", () => {
  const [conversation] = parseImportText(
    '{"role":"assistant","content":"","additional_kwargs":{"function_call":{"name":"f","arguments":"{\\"a\\":1}"}}}\n' +
      '{"role":"function","name":"f","content":"KW"}',
    "conversation-messages.jsonl",
  );

  const call = (
    conversation.messages[0].content as ReadonlyArray<{
      toolName?: string;
      args?: unknown;
    }>
  )[0];
  assert.equal(call.toolName, "f");
  assert.deepEqual(call.args, { a: 1 });
  assert.deepEqual(toolCallResults(conversation), ["KW"]);
});

test("tool_calls nested under additional_kwargs survive", () => {
  const [conversation] = parseImportText(
    '{"role":"assistant","content":"","additional_kwargs":{"tool_calls":[{"id":"k1","type":"function","function":{"name":"f","arguments":"{}"}}]}}\n' +
      '{"role":"tool","tool_call_id":"k1","content":"KWTC"}',
    "conversation-messages.jsonl",
  );

  assert.deepEqual(toolCallResults(conversation), ["KWTC"]);
});

test("a function_call duplicated inside tool_calls produces one call", () => {
  const [conversation] = parseImportText(
    '{"role":"assistant","content":null,"tool_calls":[{"id":"t1","type":"function","function":{"name":"f","arguments":"{\\"a\\":1}"}}],"function_call":{"name":"f","arguments":"{\\"a\\":1}"}}',
    "conversation-messages.jsonl",
  );

  const parts = conversation.messages[0].content as ReadonlyArray<{
    type: string;
  }>;
  assert.deepEqual(
    parts.map(({ type }) => type),
    ["tool-call"],
  );
});

test("a keyed early result is not stolen by an earlier different-name call", () => {
  const [conversation] = parseImportText(
    '{"role":"function","name":"g","content":"Rg"}\n' +
      '{"role":"assistant","content":null,"function_call":{"name":"f","arguments":"{}"}}\n' +
      '{"role":"assistant","content":null,"function_call":{"name":"g","arguments":"{}"}}',
    "conversation-messages.jsonl",
  );

  assert.deepEqual(toolCallResults(conversation), ["undefined", "Rg"]);
});

test("a tool result waits for the call its id names", () => {
  const [conversation] = parseImportText(
    '{"role":"assistant","content":null,"function_call":{"name":"f","arguments":"{}"}}\n' +
      '{"role":"tool","tool_call_id":"x","name":"f","content":"R"}\n' +
      '{"role":"assistant","content":null,"tool_calls":[{"id":"x","type":"function","function":{"name":"f","arguments":"{}"}}]}',
    "conversation-messages.jsonl",
  );

  assert.deepEqual(toolCallResults(conversation), ["undefined", "R"]);
});

test("a function_call id lets a modern tool result bind it", () => {
  const [conversation] = parseImportText(
    '{"role":"assistant","content":null,"function_call":{"id":"fc1","name":"f","arguments":"{}"}}\n' +
      '{"role":"tool","tool_call_id":"fc1","content":"BYID"}',
    "conversation-messages.jsonl",
  );

  assert.deepEqual(toolCallResults(conversation), ["BYID"]);
});

test("same-name calls in one turn take results in call order", () => {
  const [conversation] = parseImportText(
    '{"role":"assistant","content":null,"tool_calls":[' +
      '{"id":"a","type":"function","function":{"name":"f","arguments":"{}"}},' +
      '{"id":"b","type":"function","function":{"name":"f","arguments":"{}"}}]}\n' +
      '{"role":"tool","name":"f","content":"R1"}\n' +
      '{"role":"tool","name":"f","content":"R2"}',
    "conversation-messages.jsonl",
  );

  assert.deepEqual(toolCallResults(conversation), ["R1", "R2"]);
});

test("a function_call with different arguments is not deduplicated", () => {
  const [conversation] = parseImportText(
    '{"role":"assistant","content":null,"tool_calls":[' +
      '{"id":"t1","type":"function","function":{"name":"f","arguments":"{\\"a\\":1}"}}],' +
      '"function_call":{"name":"f","arguments":"{\\"a\\":2}"}}',
    "conversation-messages.jsonl",
  );

  assert.equal(conversation.messages[0].content.length, 2);
});

test("a keyed result for a call that never appears drops instead of lying", () => {
  const [conversation] = parseImportText(
    '{"role":"tool","tool_call_id":"ghost","name":"stock","content":"AAPL"}\n' +
      '{"role":"assistant","content":null,"function_call":{"name":"f","arguments":"{}"}}',
    "conversation-messages.jsonl",
  );

  assert.deepEqual(toolCallResults(conversation), ["undefined"]);
});

for (const role of ["tool", "function"]) {
  for (const resultFirst of [false, true]) {
    test(`a ${role} result with an unmatched id cannot bind by name (${resultFirst ? "early" : "late"})`, () => {
      const call = {
        role: "assistant",
        content: null,
        tool_calls: [{ id: "real-call", type: "function", function: { name: "lookup", arguments: "{}" } }],
      };
      const result = { role, tool_call_id: "missing-call", name: "lookup", content: "WRONG RESULT" };
      const messages = resultFirst ? [result, call] : [call, result];
      const [conversation] = parseImportText(
        messages.map((message) => JSON.stringify(message)).join("\n"),
        "conversation-messages.jsonl",
      );

      assert.deepEqual(toolCallResults(conversation), ["undefined"]);
    });
  }
}

test("a null tool_calls entry does not lose the other calls", () => {
  const [conversation] = parseImportText(
    '{"role":"assistant","content":null,"tool_calls":[null,{"id":"t1","type":"function","function":{"name":"f","arguments":"{}"}}]}\n' +
      '{"role":"tool","tool_call_id":"t1","content":"OK"}',
    "conversation-messages.jsonl",
  );

  assert.deepEqual(toolCallResults(conversation), ["OK"]);
});

test("a null entry inside a messages record does not lose the conversation", () => {
  const [conversation] = parseImportText(
    '{"messages":[null,{"role":"user","content":"hi"},7,{"role":"assistant","content":"yo"}]}',
    "chat.jsonl",
  );

  assert.deepEqual(
    conversation.messages.map(({ role }) => role),
    ["user", "assistant"],
  );
});

test("an early keyed result is not displaced by a later anonymous one", () => {
  const [conversation] = parseImportText(
    '{"role":"tool","tool_call_id":"a","content":"A"}\n' +
      '{"role":"assistant","content":null,"tool_calls":[{"id":"a","type":"function","function":{"name":"f","arguments":"{}"}},{"id":"b","type":"function","function":{"name":"g","arguments":"{}"}}]}\n' +
      '{"role":"tool","content":"B"}',
    "conversation-messages.jsonl",
  );

  assert.deepEqual(toolCallResults(conversation), ["A", "B"]);
});

test("a reserialized function_call copy of a tool call is not duplicated", () => {
  const [conversation] = parseImportText(
    '{"role":"assistant","content":null,"tool_calls":[{"id":"c1","type":"function","function":{"name":"f","arguments":"{\\"a\\":1}"}}],"function_call":{"name":"f","arguments":"{\\"a\\": 1}"}}\n' +
      '{"role":"tool","tool_call_id":"c1","content":"R"}',
    "conversation-messages.jsonl",
  );

  assert.deepEqual(toolCallResults(conversation), ["R"]);
});

test("a legacy copy with a different explicit id stays its own call", () => {
  const [conversation] = parseImportText(
    '{"role":"assistant","content":null,"tool_calls":[{"id":"a","type":"function","function":{"name":"f","arguments":"{}"}}],"function_call":{"id":"b","name":"f","arguments":"{}"}}\n' +
      '{"role":"tool","tool_call_id":"a","content":"A"}\n' +
      '{"role":"tool","tool_call_id":"b","content":"B"}',
    "conversation-messages.jsonl",
  );

  assert.deepEqual(toolCallResults(conversation), ["A", "B"]);
});

test("a name-only result ahead of its calls keeps arrival order", () => {
  const [conversation] = parseImportText(
    '{"role":"tool","name":"f","content":"A"}\n' +
      '{"role":"assistant","content":null,"tool_calls":[{"id":"a","type":"function","function":{"name":"f","arguments":"{}"}},{"id":"b","type":"function","function":{"name":"f","arguments":"{}"}}]}\n' +
      '{"role":"tool","name":"f","content":"B"}',
    "conversation-messages.jsonl",
  );

  assert.deepEqual(toolCallResults(conversation), ["A", "B"]);
});

test("an anonymous result ahead of its calls keeps arrival order", () => {
  const [conversation] = parseImportText(
    '{"role":"tool","content":"A"}\n' +
      '{"role":"assistant","content":null,"tool_calls":[{"id":"a","type":"function","function":{"name":"f","arguments":"{}"}},{"id":"b","type":"function","function":{"name":"g","arguments":"{}"}}]}\n' +
      '{"role":"tool","content":"B"}',
    "conversation-messages.jsonl",
  );

  assert.deepEqual(toolCallResults(conversation), ["A", "B"]);
});
