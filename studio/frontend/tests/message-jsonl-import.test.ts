// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import vm from "node:vm";
import { after, before, test } from "node:test";
import ts from "typescript";
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
