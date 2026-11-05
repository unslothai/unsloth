// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc, registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const globals = globalThis as unknown as Record<string, unknown>;
globals.requestAnimationFrame = (cb: () => void) => cb();
globals.document = { execCommand: () => false };

function composer(value: string, caret = value.length) {
  const input = {
    value,
    selectionStart: caret,
    selectionEnd: caret,
    focus: () => {},
    setSelectionRange: () => {},
  };
  const writes: string[] = [];
  const setText = (next: string) => {
    writes.push(next);
    input.value = next;
  };
  return { input: input as unknown as HTMLTextAreaElement, writes, setText };
}

async function pick(value: string, prompt: string, caret?: number) {
  const { insertSavedPrompt } = await import(
    "../src/features/chat/prompt-storage/insert-prompt.ts"
  );
  const c = composer(value, caret);
  insertSavedPrompt(c.input, prompt, c.setText);
  return c.writes;
}

test("a saved prompt fills an empty composer as before", async () => {
  assert.deepEqual(await pick("", "Summarize:"), ["Summarize:"]);
});

test("a saved prompt keeps the typed draft", async () => {
  assert.deepEqual(await pick("my text", "Summarize:"), ["my text\n\nSummarize:"]);
  assert.deepEqual(await pick("my text", "Summarize:", 0), ["Summarize:\n\nmy text"]);
  assert.deepEqual(await pick("a\nb", "P", 2), ["a\n\nP\n\nb"]);
  assert.deepEqual(await pick("a\nb", "P", 1), ["a\n\nP\n\nb"]);
  assert.deepEqual(await pick("a\n\n\nb", "P", 2), ["a\n\nP\n\nb"]);
});

for (const [file, writes] of [
  ["components/assistant-ui/thread.tsx", ["composer().setText(p.text)", "composer().setText(text)"]],
  ["features/chat/shared-composer.tsx", ["setCurrentText(p.text)", "setCurrentText(t)"]],
] as const) {
  test(`${file}: saved prompts go through the caret insert`, () => {
    const source = readSrc(file);
    for (const write of writes) assert.ok(!source.includes(write), write);
    assert.equal(source.match(/insertSavedPrompt\(/g)?.length, 2);
  });
}
