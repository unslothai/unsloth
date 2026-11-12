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
  assert.deepEqual(response.messages.at(-1)?.content, [{ type: "text", text: "Done." }]);
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
  assert.deepEqual(conversation.messages.at(-1)?.content, [{ type: "text", text: "Here is the news." }]);
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
  assert.equal(calls[1].args?.code, "print(1)");
});
