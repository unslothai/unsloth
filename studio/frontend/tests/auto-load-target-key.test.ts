// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { fileURLToPath } from "node:url";

import ts from "typescript";

// The shipped helper is lifted out of chat-adapter.ts rather than copied, so
// these assert against the real source. Importing the module would drag in the
// stores and the toast layer for one pure string function.
const adapterPath = fileURLToPath(
  new URL("../src/features/chat/api/chat-adapter.ts", import.meta.url),
);
const source = readFileSync(adapterPath, "utf8");
const start = source.indexOf("function normalizeTarget(");
assert.ok(
  start >= 0,
  "normalizeTarget is no longer defined in chat-adapter.ts",
);
const declaration = source.slice(start, source.indexOf("\n}", start) + 2);
const normalizeTarget = new Function(
  `${
    ts.transpileModule(declaration, {
      compilerOptions: { target: ts.ScriptTarget.ES2020 },
    }).outputText
  }; return normalizeTarget;`,
)() as (value: string) => string;

const sameKey = (a: string, b: string) =>
  normalizeTarget(a) === normalizeTarget(b);

type TestSource = {
  id: string;
  loadId: string;
  kind: string;
  sizeBytes: number;
  listVariants: () => Promise<
    { quant: string; downloaded: boolean; partial?: boolean }[]
  >;
};
const declarations = ["isRememberedSource", "prioritizeRememberedQuantSources"]
  .map((name) => {
    const pos = source.indexOf(
      `${name === "prioritizeRememberedQuantSources" ? "async " : ""}function ${name}(`,
    );
    assert.ok(pos >= 0);
    return source.slice(pos, source.indexOf("\n}", pos) + 2);
  })
  .join("\n");
const prioritize = new Function(
  "normalizeTarget",
  "isAutoLoadableGgufVariant",
  `${ts.transpileModule(declarations, { compilerOptions: { target: ts.ScriptTarget.ES2020 } }).outputText}; return prioritizeRememberedQuantSources;`,
)(normalizeTarget, () => true) as (
  sources: TestSource[],
  remembered: {
    id: string;
    kind: string;
    ggufVariant: string;
    loadId?: string;
  },
) => Promise<TestSource[]>;

test("same-quant copies prefer the remembered physical load target", async () => {
  const sources: TestSource[] = ["Org/Model", "/custom/rev"].map(
    (loadId, index) => ({
      id: "Org/Model",
      loadId,
      kind: "gguf",
      sizeBytes: index ? 9000 : 8000,
      listVariants: async () => [{ quant: "Q8_0", downloaded: true }],
    }),
  );
  const ordered = await prioritize(sources, {
    id: "Org/Model",
    kind: "gguf",
    ggufVariant: "Q8_0",
    loadId: "/custom/rev",
  });
  assert.equal(ordered[0].loadId, "/custom/rev");
});

test("remembered Q8 in a previous cache precedes smaller Q6 in the active cache", async () => {
  const reads = [0, 0];
  const sources: TestSource[] = ["Q6_K", "Q8_0"].map((quant, index) => ({
    id: "Org/Model",
    loadId: index ? "/custom/rev" : "Org/Model",
    kind: "gguf",
    sizeBytes: index ? 8000 : 6000,
    listVariants: async () => {
      reads[index]++;
      return [{ quant, downloaded: true }];
    },
  }));
  const ordered = await prioritize(sources, {
    id: "Org/Model",
    kind: "gguf",
    ggufVariant: "Q8_0",
  });
  assert.equal(ordered[0].loadId, "/custom/rev");
  assert.equal((await ordered[0].listVariants())[0].quant, "Q8_0");
  assert.deepEqual(reads, [1, 1], "reuse the location probe in the load loop");
});

test("missing or incomplete remembered quant preserves fallback ordering", async () => {
  const sources: TestSource[] = ["Q6_K", "Q8_0"].map((quant, index) => ({
    id: "Org/Model",
    loadId: index ? "/custom/rev" : "Org/Model",
    kind: "gguf",
    sizeBytes: index ? 8000 : 6000,
    listVariants: async () => [
      { quant, downloaded: true, partial: index === 1 },
    ],
  }));
  const ordered = await prioritize(sources, {
    id: "Org/Model",
    kind: "gguf",
    ggufVariant: "Q8_0",
  });
  assert.equal(ordered[0], sources[0]);
});

test("one Windows file spelled with either separator is one candidate", () => {
  // Two keys meant one spelling burned an attempt on the same file, and a
  // remembered record written as C:\ never matched C:/.
  assert.ok(
    sameKey("C:\\Users\\a\\models\\M.gguf", "C:/Users/a/models/M.gguf"),
  );
});

test("Windows and UNC paths still fold case", () => {
  assert.ok(sameKey("C:\\Users\\a\\M.gguf", "c:\\users\\a\\m.gguf"));
  assert.ok(sameKey("\\\\srv\\share\\M.gguf", "\\\\SRV\\share\\m.gguf"));
});

test("WSL UNC paths keep their case, because they address ext4", () => {
  // Folding merged two real files onto one key, so the second never loaded.
  assert.ok(
    !sameKey(
      "\\\\wsl$\\Ubuntu\\home\\a\\M.gguf",
      "\\\\wsl$\\Ubuntu\\home\\a\\m.gguf",
    ),
  );
  assert.ok(
    sameKey("\\\\wsl$\\Ubuntu\\home\\a\\M.gguf", "//wsl$/Ubuntu/home/a/M.gguf"),
  );
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
