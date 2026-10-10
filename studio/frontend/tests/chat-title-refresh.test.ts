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

const { buildTitleRefreshRequest, buildTitleRequest, titleRefreshExcerpt } =
  await import("../src/features/chat/utils/chat-title.ts");
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
  assert.ok(excerpt.length <= 1200, String(excerpt.length));
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
  // Newest first: 307 + 312 + 307 + 232 = 1158 used, so the opening prompt has 36 characters of room.
  const excerpt = titleRefreshExcerpt([
    message(0, "user", text("opening question about the email we started with")),
    message(1, "assistant", text("a".repeat(220))),
    message(2, "user", text("b".repeat(400))),
    message(3, "assistant", text("c".repeat(400))),
    message(4, "user", text("d".repeat(400))),
  ]);
  assert.doesNotMatch(excerpt, /opening/);
  assert.match(excerpt, /^Assistant: a{220}$/m);
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

test("a refresh asks the selected model, never the one that answered", () => {
  const source = readFileSync(
    new URL("../src/features/chat/components/chat-row-menu.ts", import.meta.url),
    "utf8",
  ).replace(/\s+/g, " ");
  const body = source.slice(source.indexOf("export async function regenerateChatTitle"));
  assert.match(body, /const \{ params, modelLoading \} = useChatRuntimeStore\.getState\(\);/);
  assert.match(body, /buildTitleRefreshRequest\(params\.checkpoint, excerpt\)/);
  assert.doesNotMatch(body.slice(0, body.indexOf("/** The sandbox")), /answeringCheckpoint|titleCheckpoint/);
});
