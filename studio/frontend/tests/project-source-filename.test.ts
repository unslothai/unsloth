// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import { installLocalStorageFake, registerStoreStubResolver } from "./helpers/kit.ts";

// save-markdown-source reaches lib/toast and the auth barrel through rag-api,
// both .tsx node's type stripping cannot load.
installLocalStorageFake();
registerStoreStubResolver();

const { projectSourceFileName } = await import(
  "../src/features/rag/api/save-markdown-source.ts"
);

test("keeps an ordinary chat title readable", () => {
  assert.equal(projectSourceFileName("Fix my regex"), "Fix my regex.md");
  assert.equal(projectSourceFileName("v1.2 notes"), "v1.2 notes.md");
  assert.equal(projectSourceFileName(".hidden"), ".hidden.md");
});

test("replaces every character Windows reserves in a name", () => {
  assert.equal(
    projectSourceFileName('a\\b/c:d*e?f"g<h>i|j'),
    "a_b_c_d_e_f_g_h_i_j.md",
  );
  assert.equal(projectSourceFileName("50/50 split"), "50_50 split.md");
});

test("drops control characters instead of sending them in a filename", () => {
  for (let code = 0; code <= 0x1f; code++) {
    const name = projectSourceFileName(`a${String.fromCharCode(code)}b`);
    assert.equal(name, "a b.md", `U+${code.toString(16)} survived as ${name}`);
  }
  assert.equal(projectSourceFileName("a\u007fb"), "a b.md");
});

test("breaks every Windows device name, extension or not", () => {
  const devices = [
    "CON",
    "PRN",
    "AUX",
    "NUL",
    ...Array.from({ length: 9 }, (_, i) => `COM${i + 1}`),
    ...Array.from({ length: 9 }, (_, i) => `LPT${i + 1}`),
    "COM¹",
    "COM²",
    "COM³",
    "LPT¹",
    "LPT²",
    "LPT³",
  ];
  for (const device of devices) {
    for (const title of [device, device.toLowerCase(), `${device}.txt`]) {
      const name = projectSourceFileName(title);
      const head = name.slice(0, name.indexOf("."));
      assert.equal(
        devices.some((d) => d.toLowerCase() === head.toLowerCase()),
        false,
        `${title} still resolves to the ${device} device as ${name}`,
      );
    }
  }
});

test("a name that only looks like a device is left alone", () => {
  assert.equal(projectSourceFileName("CONSOLE"), "CONSOLE.md");
  assert.equal(projectSourceFileName("COM10"), "COM10.md");
  assert.equal(projectSourceFileName("COM"), "COM.md");
});

test("never ends the name in a period or a space", () => {
  assert.equal(projectSourceFileName("trailing dot."), "trailing dot.md");
  assert.equal(projectSourceFileName("trailing dots..."), "trailing dots.md");
  assert.equal(projectSourceFileName("trailing space "), "trailing space.md");
  assert.equal(projectSourceFileName("mixed . . "), "mixed.md");
});

test("falls back for a title that leaves nothing usable", () => {
  for (const title of ["", "   ", "\t\n", "...", "..", ".", "///", "***"]) {
    assert.equal(projectSourceFileName(title), "chat.md", `for ${title}`);
  }
});

test("falls back where the backend would list the source as _.md", () => {
  // The backend collapses anything outside [A-Za-z0-9._-] to "_", so a title
  // with no ASCII word character would otherwise arrive as "_.md".
  assert.equal(projectSourceFileName("你好世界"), "chat.md");
  assert.equal(projectSourceFileName("разговор"), "chat.md");
  assert.equal(projectSourceFileName("\u{1f600}\u{1f680}"), "chat.md");
  // One ASCII character is enough to keep the title.
  assert.equal(projectSourceFileName("Qwen 你好"), "Qwen 你好.md");
});

/** Any surrogate left over once the well-formed pairs are removed is a half. */
function loneSurrogates(text: string): number {
  return (
    text.replace(/[\ud800-\udbff][\udc00-\udfff]/g, "").match(/[\ud800-\udfff]/g)
      ?? []
  ).length;
}

test("truncates on a code point boundary, within a byte budget", () => {
  const emoji = projectSourceFileName("\u{1f600}a".repeat(200));
  const bytes = new TextEncoder().encode(emoji).length;
  assert.ok(bytes <= 183, `${bytes} bytes`);
  assert.equal(loneSurrogates(emoji), 0);
  const cjk = projectSourceFileName(`${"好".repeat(300)}x`);
  assert.ok(new TextEncoder().encode(cjk).length <= 183);
  const ascii = projectSourceFileName("a".repeat(300));
  assert.equal(ascii, `${"a".repeat(180)}.md`);
});

test("a lone surrogate in the title never reaches the filename", () => {
  assert.equal(projectSourceFileName("chat \ud83d end"), "chat end.md");
  assert.equal(loneSurrogates(projectSourceFileName("\udfff a \ud800")), 0);
});
