// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc, registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const { hubRepoIdFor, modelInfoFacts } = await import(
  "../src/features/model-picker/components/model-info-facts.ts"
);

const base = {
  id: "unsloth/Qwen3.5-0.8B-GGUF",
  downloads: 0,
  likes: 0,
  isGguf: true,
};

test("Hub facts render as label/value rows, unknown fields left out", () => {
  const rows = modelInfoFacts({
    ...base,
    tags: ["gguf", "license:apache-2.0"],
    totalParams: 752_393_024,
    libraryName: "transformers",
    pipelineTag: "image-text-to-text",
    createdAt: "2026-03-01T00:00:00Z",
    gated: "manual",
  });
  const byLabel = Object.fromEntries(rows);
  assert.equal(byLabel.License, "apache-2.0");
  assert.equal(byLabel.Parameters, "752.4M");
  assert.equal(byLabel.Library, "transformers");
  assert.equal(byLabel.Task, "image text to text");
  assert.equal(byLabel.Access, "Gated");
  assert.ok(byLabel.Created);
  for (const absent of ["Size (estimated)", "Updated"])
    assert.equal(byLabel[absent], undefined);
});

test("a repo with no license tag says so rather than dropping the row", () => {
  assert.deepEqual(
    modelInfoFacts({ ...base, tags: [], updatedAt: "not a date" }),
    [["License", "Not specified"]],
  );
});

test("only Hub repos are looked up: snapshot paths map back, other local paths do not", () => {
  assert.equal(hubRepoIdFor("unsloth/Qwen3-0.6B"), "unsloth/Qwen3-0.6B");
  assert.equal(hubRepoIdFor("lex-au/Orpheus-3b-FT-Q8_0.gguf"), "lex-au/Orpheus-3b-FT-Q8_0.gguf");
  assert.equal(
    hubRepoIdFor(
      "/home/u/.cache/huggingface/hub/models--unsloth--Qwen3-0.6B/snapshots/abc123",
    ),
    "unsloth/Qwen3-0.6B",
  );
  assert.equal(hubRepoIdFor("/home/u/models/my-finetune"), null);
  assert.equal(hubRepoIdFor("C:\\models\\my-finetune.gguf"), null);
});

test("Information is the last settings section, collapsed by default, for the loaded Hub model", () => {
  const sheet = readSrc("features/chat/chat-settings-sheet.tsx");
  const info = sheet.indexOf(
    '<CollapsibleSection label="Information">{modelInfo}</CollapsibleSection>',
  );
  assert.ok(info > 0, "Information section rendered");
  assert.equal(
    sheet.indexOf("<CollapsibleSection", info + 1),
    -1,
    "nothing after it",
  );
  const page = readSrc("features/chat/chat-page.tsx");
  assert.match(page, /\? hubRepoIdFor\(inferenceCheckpoint\)\n\s*: null;/);
  assert.match(
    page,
    /modelInfoRepoId \? <ModelInfo repoId=\{modelInfoRepoId\} \/> : null/,
  );
});
