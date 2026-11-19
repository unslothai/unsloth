// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { splitSpoken, stripForSpeech } from "../src/features/chat/voice/speech-text.ts";
import { planTtsChunk, spokenChunk } from "../src/features/chat/voice/tts-chunk-plan.ts";

const REPLY = "Here are the top 10 emergency numbers from different countries:\n\n1. **United States**: 911  \n2. **United Kingdom**: 999 or 112  \n3. **Canada**: 911  \n4. **Australia**: 000  \n5. **Germany**: 112  \n6. **France**: 112  \n7. **Japan**: 110 (police), 119 (fire/ambulance)  \n8. **China**: 110 (police), 120 (ambulance), 119 (fire)  \n9. **India**: 112 (universal emergency number)  \n10. **Russia**: 112 or 102 (police), 103 (ambulance), 101 (fire)  \n\nNote: In many countries, dialing **112** works as a universal emergency number, even if you run out of airtime on your mobile phone. Always use the local emergency number for the fastest response!";

function chunks(text: string): string[] {
  const pending = splitSpoken(text).map((s) => stripForSpeech(s.text));
  const out: string[] = [];
  while (pending.length) {
    const take = Math.max(1, planTtsChunk({ pending, first: out.length === 0, final: true, playing: false, waitedMs: 0 }));
    out.push(spokenChunk(pending.splice(0, take)));
  }
  return out;
}

test("list items merged into one request each end in a stop", () => {
  const spoken = chunks(REPLY).join(" ");
  assert.doesNotMatch(spoken, /911 United Kingdom/);
  assert.match(spoken, /United States: 911\. United Kingdom: 999 or 112\. Canada: 911\./);
  assert.match(spoken, /101 \(fire\)\. Note:/);
});

test("punctuated sentences and clause marks are left as they are", () => {
  assert.equal(spokenChunk(["Hello there.", "How are you?", "Wow!", "Wait…", "Here they are:", "One, two;"]), "Hello there. How are you? Wow! Wait… Here they are: One, two;");
  assert.equal(spokenChunk(['He said "go."', "(see above)"]), 'He said "go." (see above).');
  assert.equal(spokenChunk(["Canada: 911", "", "Japan: 110"]), "Canada: 911. Japan: 110.");
});
