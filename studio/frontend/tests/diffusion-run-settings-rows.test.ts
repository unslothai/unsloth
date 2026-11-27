// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The config page cannot render under Node, so these checks are source-level.

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";

const LINES = readSrc(
  "features/model-picker/components/model-config-page.tsx",
).split("\n");

function ggufAdvancedLines(): string[] {
  const start = LINES.findIndex((line) =>
    line.startsWith("function GgufAdvancedSettings("),
  );
  assert.notEqual(start, -1, "no GgufAdvancedSettings");
  const length = LINES.slice(start + 1).findIndex((line) =>
    /^(export )?function \w+\(/.test(line),
  );
  return LINES.slice(start, start + 1 + length);
}

function groupLines(title: string): { gate: string; body: string } {
  const lines = ggufAdvancedLines();
  const open = lines.findIndex(
    (line) => line.trim() === `<AdvancedGroup title="${title}">`,
  );
  assert.notEqual(open, -1, `no ${title} group`);
  const close = lines.findIndex(
    (line, i) => i > open && line.trim() === "</AdvancedGroup>",
  );
  return {
    gate: lines[open - 1].trim(),
    body: lines.slice(open, close).join("\n"),
  };
}

test("the diffusion runner has no slots or drafter argument, so the Speed group is not offered", () => {
  const speed = groupLines("Speed");
  assert.equal(speed.gate, "{!isDiffusion && (");
  assert.ok(speed.body.includes(">Speculative Decoding</span>"));
  assert.ok(speed.body.includes("<ParallelSlotsRow"));
});

test("KV Cache Dtype is not offered for a diffusion model", () => {
  const lines = ggufAdvancedLines();
  const label = lines.findIndex((line) =>
    line.includes(">KV Cache Dtype</span>"),
  );
  assert.notEqual(label, -1, "no KV Cache Dtype row");
  // A `)}` first means the gate closed before the row.
  const above = lines
    .slice(0, label)
    .reverse()
    .map((line) => line.trim())
    .find((line) => line === "{!isDiffusion && (" || line === ")}");
  assert.equal(above, "{!isDiffusion && (");
});

test("the Memory group renders for diffusion only when it has the GPU picker to show", () => {
  assert.equal(
    groupLines("Memory").gate,
    "{(!isDiffusion || showGpuPicker) && (",
  );
});

test("a diffusion context the runner did not apply says so, with the value it runs", () => {
  const page = LINES.join("\n");
  const start = page.indexOf("function DiffusionContextNote(");
  assert.notEqual(start, -1, "no DiffusionContextNote");
  const note = page.slice(start, page.indexOf("\nfunction ", start + 1));
  assert.match(
    note,
    /if \(loadedContext == null \|\| value !== requested \|\| value === loadedContext\) \{\s+return null;/,
  );
  assert.match(note, /running with \{loadedContext\.toLocaleString\(\)\} tokens/);
  // It replaces the VRAM-capacity warning, which describes a context the runner never uses.
  assert.match(
    page.replace(/\s+/g, " "),
    /resolvedIsDiffusion \? \( <DiffusionContextNote value=\{v\} requested=\{loadedConfig\?\.customContextLength \?\? null\} loadedContext=\{activeLoadedContext\} \/> \) : isActiveModel &&/,
  );
});
