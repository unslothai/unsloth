// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { fileURLToPath } from "node:url";

import ts from "typescript";

// The shipped helper, lifted out of chat-adapter.ts rather than copied, so a
// change to the real source is what these assert against. Importing the module
// would drag in the stores and the toast layer for one pure string function.
const adapterPath = fileURLToPath(
  new URL("../src/features/chat/api/chat-adapter.ts", import.meta.url),
);
const source = readFileSync(adapterPath, "utf8");
const start = source.indexOf("function normalizeTarget(");
assert.ok(start >= 0, "normalizeTarget is no longer defined in chat-adapter.ts");
const declaration = source.slice(start, source.indexOf("\n}", start) + 2);
const normalizeTarget = new Function(
  `${ts.transpileModule(declaration, {
    compilerOptions: { target: ts.ScriptTarget.ES2020 },
  }).outputText}; return normalizeTarget;`,
)() as (value: string) => string;

const sameKey = (a: string, b: string) => normalizeTarget(a) === normalizeTarget(b);

test("one Windows file spelled with either separator is one candidate", () => {
  // Two keys meant the same file twice: one spelling burned an auto-load
  // attempt, and a remembered record written as C:\ never matched C:/.
  assert.ok(sameKey("C:\\Users\\a\\models\\M.gguf", "C:/Users/a/models/M.gguf"));
});

test("Windows and UNC paths still fold case", () => {
  assert.ok(sameKey("C:\\Users\\a\\M.gguf", "c:\\users\\a\\m.gguf"));
  assert.ok(sameKey("\\\\srv\\share\\M.gguf", "\\\\SRV\\share\\m.gguf"));
});

test("WSL UNC paths keep their case, because they address ext4", () => {
  // Folding merged two real files onto one key, so the second never loaded.
  assert.ok(
    !sameKey("\\\\wsl$\\Ubuntu\\home\\a\\M.gguf", "\\\\wsl$\\Ubuntu\\home\\a\\m.gguf"),
  );
  assert.ok(sameKey("\\\\wsl$\\Ubuntu\\home\\a\\M.gguf", "//wsl$/Ubuntu/home/a/M.gguf"));
});

test("POSIX paths keep their case", () => {
  assert.ok(!sameKey("/home/a/M.gguf", "/home/a/m.gguf"));
});

test("a decomposed filename is the same candidate as its composed form", () => {
  // macOS hands back NFD, so a remembered model was never re-attempted.
  assert.ok(sameKey("/home/a/caf\u00e9.gguf", "/home/a/cafe\u0301.gguf"));
});

test("repo ids still fold case", () => {
  assert.ok(sameKey("unsloth/Qwen3-0.6B-GGUF", "UNSLOTH/qwen3-0.6b-gguf"));
});
