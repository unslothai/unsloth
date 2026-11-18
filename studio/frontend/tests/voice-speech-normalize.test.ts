// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { normalizeForSpeech } from "../src/features/chat/voice/speech-normalize.ts";

// One case per form the Kokoro listening sample got wrong, confirmed by ear.
const FAILING_FORMS: { input: string; locale: string; spoken: string }[] = [
  { input: "The total is $1,250.", locale: "en", spoken: "The total is one thousand two hundred fifty dollars." },
  { input: "A coffee is $3.50.", locale: "en-US", spoken: "A coffee is three dollars fifty." },
  { input: "The app costs $0.99.", locale: "en", spoken: "The app costs ninety nine cents." },
  { input: "It costs $1.", locale: "en", spoken: "It costs one dollar." },
  { input: "They raised $1.5M.", locale: "en", spoken: "They raised one point five million dollars." },
  { input: "The book is £15.99.", locale: "en-GB", spoken: "The book is fifteen pounds ninety nine." },
  { input: "We sold 1,000 units.", locale: "en", spoken: "We sold one thousand units." },
  { input: "Pi is about 3.14.", locale: "en", spoken: "Pi is about three point one four." },
  { input: "The rate is 3.5%.", locale: "en", spoken: "The rate is three point five percent." },
  { input: "Set the alarm for 6:00 a.m.", locale: "en", spoken: "Set the alarm for six a m." },
  { input: "Call me at 7:05 tomorrow.", locale: "en", spoken: "Call me at seven oh five tomorrow." },
  { input: "Lunch is at 12:00.", locale: "en", spoken: "Lunch is at twelve o'clock." },
  { input: "Add 1/2 a cup.", locale: "en", spoken: "Add one half a cup." },
  { input: "About 3/4 of them.", locale: "en", spoken: "About three quarters of them." },
  { input: "Released on 2024-03-15.", locale: "en", spoken: "Released on March fifteenth, twenty twenty four." },
  { input: "See No. 5 on the list.", locale: "en", spoken: "See Number 5 on the list." },
  { input: "Louis XIV built Versailles.", locale: "en", spoken: "Louis the fourteenth built Versailles." },
  { input: "World War II ended.", locale: "en", spoken: "World War two ended." },
  { input: "Call (555) 123-4567.", locale: "en", spoken: "Call five five five, one two three, four five six seven." },
  { input: "Dial +1-800-555-0199.", locale: "en", spoken: "Dial plus one, eight zero zero, five five five, zero one nine nine." },
  { input: "Save the GGUF file.", locale: "en", spoken: "Save the G G U F file." },
  // The 13-voice sweep: every voice read these wrong ("K M", "four H thirty M", "Tue" as "two").
  { input: "A marathon is 42.195 km.", locale: "en", spoken: "A marathon is forty two point one nine five kilometers." },
  { input: "I ran 5km.", locale: "en", spoken: "I ran 5 kilometers." },
  { input: "It weighs 1 kg.", locale: "en", spoken: "It weighs 1 kilogram." },
  { input: "He is 6 ft tall.", locale: "en", spoken: "He is 6 feet tall." },
  { input: "It is 21°C today.", locale: "en", spoken: "It is 21 degrees Celsius today." },
  { input: "Most finish in about 4h 30m.", locale: "en", spoken: "Most finish in about 4 hours and 30 minutes." },
  { input: "Wait 1 hr.", locale: "en", spoken: "Wait 1 hour." },
  { input: "It takes 90 min.", locale: "en", spoken: "It takes 90 minutes." },
  { input: "The meeting moved to Tue, Mar 4 at 3:30 PM.", locale: "en", spoken: "The meeting moved to Tuesday, March fourth at three thirty p m." },
  { input: "Released March 15, 2024.", locale: "en", spoken: "Released March fifteenth, 2024." },
];

test("forms the voice reads well, and look-alikes, pass through", () => {
  for (const text of ["Dr. Smith", "Update to v2.1.3.", "The score was 3-2.", "Mix in a 2:1 ratio.",
    "Room 101.", "It finished in 00:45.", "The deadline is 5/6/2024.", "Size XL fits.", "I am here.", "5/3 is more than one.",
    "May 5 people come?", "Sun, sea and sand.", "Set 30m.", "Use ffmpeg -h", "The h264 codec.", "Mar the finish."]) {
    assert.equal(normalizeForSpeech(text, "en"), text);
  }
});

test("other languages keep their text", () => {
  assert.equal(normalizeForSpeech("Il coûte $5 à 6:00.", "fr-FR"), "Il coûte $5 à 6:00.");
});

test("a locale named like an object property has no rules", () => {
  for (const locale of ["constructor", "__proto__-x", "toString"]) {
    assert.equal(normalizeForSpeech("$5", locale), "$5");
  }
});

for (const form of FAILING_FORMS) {
  test(`reads ${form.input} as ${form.spoken}`, () => {
    assert.equal(normalizeForSpeech(form.input, form.locale), form.spoken);
  });
}

test("a few cents read as cents, and the pronoun I after Mark stays a word", () => {
  assert.equal(normalizeForSpeech("It costs $1.01.", "en"), "It costs one dollar and one cent.");
  assert.equal(normalizeForSpeech("It costs $3.05.", "en"), "It costs three dollars and five cents.");
  assert.equal(normalizeForSpeech("It costs $5.25.", "en"), "It costs five dollars twenty five.");
  assert.equal(normalizeForSpeech("Mark I think so.", "en"), "Mark I think so.");
  assert.equal(normalizeForSpeech("Read Book I'd say first.", "en"), "Read Book I'd say first.");
  assert.equal(normalizeForSpeech("Read Chapter I of the report.", "en"), "Read Chapter one of the report.");
  // A capital after the amount is a scale only on its own: "T-shirt" is a word.
  assert.equal(normalizeForSpeech("A $5 T-shirt.", "en"), "A five dollars T-shirt.");
  assert.equal(normalizeForSpeech("$5K now", "en"), "five thousand dollars now");
  assert.equal(normalizeForSpeech("1,000, 2,000 and 3,000.", "en"), "one thousand, two thousand and three thousand.");
  // Before a lowercase word it is the pronoun far more often; on its own it is still the numeral.
  assert.equal(normalizeForSpeech("It was the Mark I.", "en"), "It was the Mark one.");
});
