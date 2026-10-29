// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Flows type: Studio's text-ui steps, sans numbers, mono only for code.

import assert from "node:assert/strict";
import { readdirSync, readFileSync, statSync } from "node:fs";
import { join } from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

const root = fileURLToPath(new URL("../src/features/flows", import.meta.url));

function files(dir: string): string[] {
  return readdirSync(dir).flatMap((name) => {
    const path = join(dir, name);
    if (statSync(path).isDirectory()) return path.endsWith(join("player", "world")) ? [] : files(path);
    return /\.tsx?$/.test(name) ? [path] : [];
  });
}

// Pinned per line, not by scanning the folder: on the stack, files from the parallel branches
// still carry their own sizes and mono, and those are fixed on their branches.
const read = (rel: string) => readFileSync(join(root, rel), "utf8");

test("numbers and ids use the sans font with tabular-nums", () => {
  assert.match(read("editor/run-drawer.tsx"), /w-14 text-right text-ui-11 text-muted-foreground tabular-nums/);
  assert.match(read("editor/flow-node.tsx"), /<Badge variant="secondary" className="text-ui-11 tabular-nums">\s*\{String\(value\)\}/);
  const viewers = read("editor/value-viewers-full.tsx");
  assert.match(viewers, /column\.type === "number" \? "text-right tabular-nums" : ""/);
  assert.match(viewers, /text-ui-11 text-muted-foreground tabular-nums">\s*shape/);
  assert.match(read("editor/dialogs.tsx"), /className="min-w-0 flex-1 truncate text-xs"\s*title=\{model\.repo\}/);
});

test("mono stays on code: logs, JSON output and editors, pasted flow JSON, the webhook command", () => {
  assert.match(read("editor/run-drawer.tsx"), /<pre className="px-4 py-2 font-mono/);
  assert.match(read("editor/output-value.tsx"), /<pre className="[^"]*font-mono/);
  assert.match(read("editor/param-fields.tsx"), /\$\{FIELD\} font-mono text-ui-12/);
  assert.match(read("editor/trigger-panel.tsx"), /<code className="[^"]*font-mono/);
});

test("this branch's Flows files use text-ui steps, not pixel sizes", () => {
  for (const rel of ["editor/param-fields.tsx", "gallery/flows-gallery-page.tsx"])
    assert.doesNotMatch(read(rel), /text-\[\d+(\.\d+)?px\]/, rel);
});

test("Flows icons never draw with a translucent currentColor", () => {
  for (const path of files(root)) {
    const src = readFileSync(path, "utf8");
    for (const icon of src.match(/<HugeiconsIcon\b[^>]*?\/>/gs) ?? [])
      assert.doesNotMatch(icon, /text-[a-z-]+(-\d+)?\/\d+/, `${path}: ${icon}`);
  }
});
