// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import {
  installLocalStorageFake,
  registerBundlerResolver,
} from "./helpers/kit.ts";

registerBundlerResolver();
installLocalStorageFake();
Object.assign((globalThis.window as { location: object }).location, {
  href: "http://localhost/",
});

const {
  buildTitleRefreshRequest,
  buildTitleRequest,
  heuristicChatTitle,
  titleCheckpoint,
  titleRefreshExcerpt,
} = await import("../src/features/chat/utils/chat-title.ts");
const { buildExternalModelId, parseExternalModelId } = await import("../src/features/chat/external-providers.ts");
type MessageRecord = import("../src/features/chat/types.ts").MessageRecord;

const LOCAL = "unsloth/gemma-4-E2B-it-GGUF";

function message(
  index: number,
  role: MessageRecord["role"],
  content: MessageRecord["content"],
): MessageRecord {
  return {
    id: `m${index}`,
    threadId: "t",
    role,
    content,
    createdAt: index,
  } as MessageRecord;
}

const text = (value: string) => [{ type: "text", text: value }] as MessageRecord["content"];

test("a refresh reads the latest turns, oldest first, not the opening exchange", () => {
  const messages: MessageRecord[] = [
    message(0, "user", text("help me write a follow-up email")),
    message(1, "assistant", text("Sure, here is a draft ".repeat(40))),
  ];
  for (let i = 2; i < 12; i += 2) {
    messages.push(message(i, "user", text(`what about clause ${i} of the contract`)));
    messages.push(message(i + 1, "assistant", text(`Clause ${i} limits liability. `.repeat(20))));
  }
  const excerpt = titleRefreshExcerpt(messages);
  assert.ok(excerpt.length <= 900, String(excerpt.length));
  assert.doesNotMatch(excerpt, /follow-up email/);
  const lines = excerpt.split("\n");
  assert.match(lines.at(-1) ?? "", /^Assistant: Clause 10 limits liability/);
  assert.match(lines.at(-2) ?? "", /^User: what about clause 10 of the contract$/);
  for (const line of lines) {
    assert.ok(line.length <= 300 + "Assistant: ".length, line);
    // A message is sent whole, or cut to at least 40 characters, never as a stub.
    const body = line.replace(/^(User|Assistant): /, "");
    assert.ok(body.length >= 40 || /^what about clause \d+ of the contract$/.test(body), line);
  }
});

test("a message that would fit only as a stub is left out", () => {
  // Newest first: 312 + 307 + 239 = 858 used, so the opening prompt has 36 characters of room.
  const excerpt = titleRefreshExcerpt([
    message(0, "user", text("opening question about the email we started with")),
    message(1, "assistant", text("a".repeat(227))),
    message(2, "user", text("d".repeat(400))),
    message(3, "assistant", text("c".repeat(400))),
  ]);
  assert.doesNotMatch(excerpt, /opening/);
  assert.match(excerpt, /^Assistant: a{227}$/m);
});

test("only visible text is sent: no reasoning, tool calls, images or system turns", () => {
  const excerpt = titleRefreshExcerpt([
    message(0, "system", text("You are terse")),
    message(1, "user", "a legacy string message" as unknown as MessageRecord["content"]),
    message(2, "assistant", [
      { type: "reasoning", text: "SECRET THOUGHTS" },
      { type: "tool-call", toolName: "web_search", args: {}, result: "SEARCH DUMP" },
      { type: "text", text: "Here are   the\nresults" },
    ] as unknown as MessageRecord["content"]),
    message(3, "user", [{ type: "image", image: "data:image/png;base64,AAAA" }] as unknown as MessageRecord["content"]),
  ]);
  assert.equal(excerpt, "User: a legacy string message\nAssistant: Here are the results");
});

test("an empty chat has nothing to title", () => {
  assert.equal(titleRefreshExcerpt([]), "");
  assert.equal(titleRefreshExcerpt([message(0, "assistant", text("   "))]), "");
});

test("a refresh is as cheap as the automatic title, with its own prompt", async () => {
  const automatic = await buildTitleRequest(LOCAL, "User: hi");
  const refresh = await buildTitleRefreshRequest(LOCAL, "User: hi");
  // The automatic prompt is unchanged byte for byte.
  assert.equal(
    automatic?.messages?.[0]?.content,
    "Write 1 concise chat title summarizing the conversation topic, not the user's exact wording. Use the assistant reply as context when provided. Rules: 2-6 words, no quotes, no punctuation, ASCII only, do not echo input. Output title only.",
  );
  assert.match(String(refresh?.messages?.[0]?.content), /latest messages/);
  assert.deepEqual(
    { ...refresh, messages: undefined },
    { ...automatic, messages: undefined },
  );
  assert.deepEqual(
    [refresh?.max_tokens, refresh?.enable_thinking, refresh?.enable_tools],
    [24, false, false],
  );
});

test("a refresh follows the connection that answered, and asks a local model only while it is serving", () => {
  const source = readFileSync(
    new URL("../src/features/chat/components/chat-row-menu.ts", import.meta.url),
    "utf8",
  ).replace(/\s+/g, " ");
  const serving = source.slice(
    source.indexOf("async function titleModelServing"),
    source.indexOf("export async function regenerateChatTitle"),
  );
  // An idle-unloaded local model is reloaded by any completion naming it, so residency is read first,
  // bounded: a stalled status check must still reach the heuristic fallback.
  // With the WebKitGTK-safe helper (no AbortSignal.timeout before 2.38), disposed once settled.
  assert.match(serving, /const wait = disposableTimeoutSignal\(TITLE_STATUS_WAIT_MS\);/);
  assert.match(serving, /const status = await Promise\.race\(\[getInferenceStatus\(wait\.signal, checkpoint\), timedOut\]\);/);
  assert.match(serving, /finally \{ wait\.dispose\(\); \}/);
  assert.doesNotMatch(source, /AbortSignal\.timeout\(/);
  assert.match(serving, /if \(!status\.active_model \|\| status\.is_audio \|\| status\.is_diffusion\) return false;/);
  // The model's wait covers building the request (a stored key is encrypted through the backend) as well as the stream.
  const fromModel = source.slice(
    source.indexOf("async function titleFromModel"),
    source.indexOf("async function titleModelServing"),
  );
  assert.match(fromModel, /await Promise\.race\(\[buildTitleRefreshRequest\(checkpoint, excerpt\), timedOut\]\)/);
  assert.match(fromModel, /await Promise\.race\(\[titleFromStream\(streamChatCompletions\(request, controller\.signal\)\), timedOut\]\)/);
  const body = source.slice(
    source.indexOf("export async function regenerateChatTitle"),
    source.indexOf("/** The sandbox"),
  );
  assert.match(body, /const selected = !modelLoading \? params\.checkpoint : "";/);
  // The latest answer on the branch names the connection; never the live selection alone.
  assert.match(body, /const lastMessage = branch\.at\(-1\);/);
  assert.match(
    body,
    /lastMessage\?\.role === "assistant" \? answeringCheckpoint\(lastMessage\.metadata\?\.custom\) : "";/,
  );
  assert.match(body, /const checkpoint = titleCheckpoint\(answeredWith, selected\);/);
  assert.match(
    body,
    /\(serving \? await titleFromModel\(checkpoint, excerpt\) : null\) \?\? heuristicChatTitle\(branch\)/,
  );
  // Each pane is guarded with its own starting title: a comparison's panes can differ.
  assert.match(body, /item\.type === "single" \? Promise\.resolve\(null\) : listStoredChatThreads\(\{ pairId: item\.id, includeArchived: true \}\)/);
  assert.match(body, /updateChatThread\(id, \{ title \}, before === undefined \? \{\} : \{ expectedTitle: before \}\)/);
});

test("a locally answered chat is never sent to a selected external provider", () => {
  const external = buildExternalModelId("prov-1", "gpt-x");
  assert.notEqual(parseExternalModelId(external), null);
  // Local answer, external selected: no model asked, the messages pick the title.
  assert.equal(titleCheckpoint(LOCAL, external), "");
  assert.equal(titleCheckpoint("", external), "");
  // Local answer, local selected: unchanged.
  assert.equal(titleCheckpoint(LOCAL, LOCAL), LOCAL);
  // External answer: the connection that already has these messages.
  assert.equal(titleCheckpoint(external, LOCAL), external);
});

const DRIFTED: MessageRecord[] = [
  message(0, "user", text("Help me write a short follow-up email to a client after our kickoff meeting.")),
  message(1, "assistant", text("Sure! **Subject:** Follow-Up on Our Kickoff Meeting. Dear client, thanks for the meeting.")),
  message(2, "user", text("Actually the client now wants to renegotiate the contract. They want to lower the liability cap. What should I push back on?")),
  message(3, "assistant", text("## Liability cap negotiation\nIf the client asks to **lower the liability cap**, push back on carve-outs for indemnification.")),
  message(4, "user", text("How should indemnification and limitation of liability clauses interact in a software license agreement?")),
  message(5, "assistant", text("## 1. Indemnification vs limitation of liability\nIn a **software license agreement**, indemnification is often carved out of the liability cap.")),
];

test("without a model the title follows where the chat went, not where it started", () => {
  assert.equal(heuristicChatTitle(DRIFTED), "Indemnification and limitation of liability clauses");
});

test("without a model a closing thanks or nudge does not become the title", () => {
  assert.equal(
    heuristicChatTitle([
      message(0, "user", text("How do I fine-tune Llama 3 with LoRA on a single GPU?")),
      message(1, "assistant", text("## LoRA fine-tuning on one GPU\nUse **QLoRA** with 4-bit quantization.")),
      message(2, "user", text("thanks!")),
    ]),
    "LoRA on a single GPU",
  );
  assert.equal(
    heuristicChatTitle([
      message(0, "user", text("Write a story about a dragon who loves baking bread")),
      message(1, "assistant", text("Once upon a time, a dragon named Ember baked bread.")),
      message(2, "user", text("ok continue")),
    ]),
    "Dragon who loves baking bread",
  );
});

test("without a model text with no spaced words keeps its first line", () => {
  assert.equal(heuristicChatTitle([message(0, "user", text("如何用Python读取CSV文件？"))]), "如何用Python读取CSV文件");
  assert.equal(heuristicChatTitle([]), null);
  assert.equal(heuristicChatTitle([message(0, "assistant", text("Hello there"))]), null);
});

test("a paste-only turn titles from its pasted text", () => {
  const paste = {
    ...message(2, "user", text("")),
    attachments: [{ type: "document", name: "notes.txt", content: [{ type: "text", text: "Quarterly revenue forecast for the Berlin retail stores" }] }],
  } as unknown as MessageRecord;
  const turns = [
    message(0, "user", text("Help me write a short follow-up email to a client")),
    message(1, "assistant", text("Sure, here is a draft.")),
    paste,
  ];
  assert.match(titleRefreshExcerpt(turns), /^User: .*Quarterly revenue forecast/m);
  assert.match(heuristicChatTitle(turns) ?? "", /revenue forecast/i);
});

test("without a model words with combining marks stay whole", () => {
  const hindi = "मुझे पायथन डेकोरेटर समझाओ, खासकर तर्क वाले डेकोरेटर";
  const title = heuristicChatTitle([message(0, "user", text(hindi))]) ?? "";
  assert.ok(title.length > 0);
  for (const word of title.split(" ")) assert.ok(hindi.split(/[\s,]+/).includes(word), word);
});

test("without a model reasoning and tool output never pick the title", () => {
  const title = heuristicChatTitle([
      message(0, "user", text("Plan a three day trip to Kyoto in autumn")),
      message(1, "assistant", [
        { type: "reasoning", text: "Database migration database migration database migration" },
        { type: "tool-call", toolName: "web_search", args: {}, result: "database migration" },
        { type: "text", text: "Day one: **Kyoto temples** in autumn." },
      ] as unknown as MessageRecord["content"]),
    ]);
  assert.equal(title, "Day trip to Kyoto in autumn");
  assert.doesNotMatch(title ?? "", /database|migration/i);
});

test("the Library's Regenerate title is off while any of the chat's panes is generating", () => {
  const source = readFileSync(
    new URL("../src/features/library/chats/chats-items.tsx", import.meta.url),
    "utf8",
  ).replace(/\s+/g, " ");
  const item = source.slice(
    source.indexOf("function RegenerateTitleItem"),
    source.indexOf("function ForkItem"),
  );
  // A comparison's id is its pair's; runningByThreadId is keyed by each pane's thread.
  assert.match(
    item,
    /\(chat\.threadIds\?\.length \? chat\.threadIds : \[chat\.id\]\)\.some\(\(id\) => Boolean\(s\.runningByThreadId\[id\]\)\)/,
  );
  assert.match(item, /disabled=\{generating\}/);
});

test("a second Regenerate title while one runs leaves the running request's toast alone", () => {
  const source = readFileSync(
    new URL("../src/features/library/chats/chats-library.tsx", import.meta.url),
    "utf8",
  ).replace(/\s+/g, " ");
  const fn = source.slice(
    source.indexOf("async function regenerateChatTitleWithToast"),
    source.indexOf("function isDateField"),
  );
  assert.match(fn, /if \(outcome === "busy"\) return; if \(outcome === "empty"\)/);
});
