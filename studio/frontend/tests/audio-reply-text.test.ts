// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  splitAudioReply,
  spokenReplyText,
} from "../src/components/assistant-ui/audio-reply-text.ts";
import { readSrc } from "./helpers/kit.ts";

const PLAYER = '<audio-player src="data:audio/wav;base64,QUJD" />';

test("a clip-only reply is the player and no text", () => {
  assert.deepEqual(splitAudioReply(PLAYER), {
    audioSrc: "data:audio/wav;base64,QUJD",
    text: "",
  });
});

test("the text an audio model spoke survives beside its player", () => {
  assert.deepEqual(splitAudioReply(`${PLAYER}\n\nHello there. How can I help?`), {
    audioSrc: "data:audio/wav;base64,QUJD",
    text: "Hello there. How can I help?",
  });
});

test("text without a player is passed through untouched", () => {
  const text = "No audio here, just **markdown**.";
  assert.deepEqual(splitAudioReply(text), { audioSrc: null, text });
});

test("the renderer no longer returns the bare player for a reply that has text", () => {
  // The old branch returned <AudioPlayer/> for ANY match, which discarded the
  // transcript the adapter had just appended.
  const src = readSrc("components/assistant-ui/markdown-text.tsx");
  assert.doesNotMatch(src, /displayText\.match\(AUDIO_PLAYER_RE\)/);
  assert.match(src, /splitAudioReply\(displayText\)/);
  assert.match(src, /audioSrc !== null && markdownText === ""/);
});

test("the spoken text is rendered whole, and an older backend's status label is not", () => {
  assert.equal(spokenReplyText("  Hello there.  "), "Hello there.");
  assert.equal(spokenReplyText(null), "");
  assert.equal(spokenReplyText('[Generated audio from: "Hello there. How can I"]'), "");
});
