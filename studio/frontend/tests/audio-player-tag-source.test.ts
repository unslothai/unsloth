// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import { readSrc } from "./helpers/kit.ts";

const MARKDOWN_TEXT = readSrc("components/assistant-ui/markdown-text.tsx");

function audioPlayerRe(): RegExp {
  const literal = MARKDOWN_TEXT.match(/const AUDIO_PLAYER_RE = \/(.+)\/;/);
  assert.ok(literal, "AUDIO_PLAYER_RE not found");
  return new RegExp(literal[1]);
}

test("the inline wav the chat adapter writes still becomes a player", () => {
  const match = '<audio-player src="data:audio/wav;base64,UklGRg==" />'.match(audioPlayerRe());
  assert.equal(match?.[1], "data:audio/wav;base64,UklGRg==");
});

test("a reply cannot make the player fetch a remote or same-origin url", () => {
  for (const src of ["https://attacker.example/x.wav", "//attacker.example/x.wav", "/api/health", "blob:x"]) {
    assert.equal(`<audio-player src="${src}" />`.match(audioPlayerRe()), null, src);
  }
});
