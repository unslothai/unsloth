// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Two modules decide which local sources reach a picker, and they are documented as being the
// same set: `PICKER_LOCAL_SOURCES` in the model picker's inventory hook, and
// `CHAT_LOCAL_SOURCES` in chat's option builder. Nothing enforced it, so adding a source to
// one and not the other left the model listed in the hub and absent from the picker. This
// pins the two together by reading the source of both.

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { fileURLToPath } from "node:url";

function declaredSources(relPath: string, constName: string): string[] {
  const source = readFileSync(
    fileURLToPath(new URL(`../src/${relPath}`, import.meta.url)),
    "utf8",
  );
  const start = source.indexOf(constName);
  assert.notEqual(start, -1, `${constName} not found in ${relPath}`);
  const open = source.indexOf("new Set([", start);
  assert.notEqual(open, -1, `${constName} is no longer a Set literal`);
  const close = source.indexOf("])", open);
  const body = source.slice(open + "new Set([".length, close);
  return [...body.matchAll(/"([a-z_]+)"/g)].map((m) => m[1]).sort();
}

test("the chat picker and chat's option builder allow the same local sources", () => {
  const picker = declaredSources(
    "features/model-picker/inventory/use-chat-picker-inventory.ts",
    "PICKER_LOCAL_SOURCES",
  );
  const chat = declaredSources(
    "features/chat/local-model-options.ts",
    "CHAT_LOCAL_SOURCES",
  );

  assert.ok(picker.length > 0, "picker allowlist parsed as empty");
  assert.deepEqual(picker, chat);
});

test("oMLX rows reach both", () => {
  for (const [relPath, constName] of [
    [
      "features/model-picker/inventory/use-chat-picker-inventory.ts",
      "PICKER_LOCAL_SOURCES",
    ],
    ["features/chat/local-model-options.ts", "CHAT_LOCAL_SOURCES"],
  ] as const) {
    assert.ok(
      declaredSources(relPath, constName).includes("omlx"),
      `${constName} does not allow omlx`,
    );
  }
});
