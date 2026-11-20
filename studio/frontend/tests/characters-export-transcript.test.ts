// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import { characterChatTranscript } from "../src/features/characters/export-transcript.ts";

const text = (value: string) => [{ type: "text", text: value }];

test("an ordinary chat is not a character transcript", () => {
  assert.equal(characterChatTranscript([{ role: "user", content: text("hi") }]), null);
});

test("a character chat exports without turn markers, each reply named once", () => {
  const records = [
    { role: "user", content: text("[Start a new chat]"), metadata: { roleplay: { kind: "opening" } } },
    { role: "assistant", content: text("Mara Quill: Welcome."), metadata: { roleplay: { kind: "reply", speakerId: "m", speakerName: "Mara Quill" } } },
    { role: "user", content: text("[Pip's turn]"), metadata: { roleplay: { kind: "turn", speakerId: "p" } } },
    { role: "assistant", content: text("Pip: Hello!"), metadata: { roleplay: { kind: "reply", speakerId: "p", speakerName: "Pip" } } },
    { role: "user", content: text("Mara: are you there?") },
  ];
  const out = characterChatTranscript(records);
  assert.deepEqual(
    out?.map((r) => [r.role, (r.content as { text: string }[])[0].text]),
    [
      ["assistant", "Mara Quill: Welcome."],
      ["assistant", "Pip: Hello!"],
      // The user's own words are never rewritten.
      ["user", "Mara: are you there?"],
    ],
  );
});

test("a reply the model wrote without its name still exports under that name", () => {
  const reply = { role: "assistant", content: text("I will take the roof."), metadata: { roleplay: { kind: "reply", speakerId: "r", speakerName: "Rin" } } };
  const out = characterChatTranscript([reply]);
  assert.equal((out?.[0].content as { text: string }[])[0].text, "Rin: I will take the roof.");
});
