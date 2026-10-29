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

test("no arbitrary pixel font sizes outside the game player", () => {
  for (const path of files(root))
    assert.doesNotMatch(readFileSync(path, "utf8"), /text-\[\d+(\.\d+)?px\]/, path);
});

test("font-mono is left on code, JSON and logs only", () => {
  const mono = files(root).flatMap((path) =>
    readFileSync(path, "utf8")
      .split("\n")
      .filter((line) => line.includes("font-mono"))
      .map((line) => `${path.slice(root.length + 1)}: ${line.trim()}`),
  );
  const allowed = [
    /gallery\/flows-gallery-page\.tsx: className="rounded-md border bg-transparent px-2 py-1\.5 font-mono text-xs"/, // pasted flow JSON
    /editor\/run-drawer\.tsx: <pre className=/, // logs
    /editor\/output-value\.tsx: <pre className=/, // JSON output
    /editor\/param-fields\.tsx: className=\{`\$\{FIELD\} font-mono/, // JSON param editor
    /editor\/flow-node\.tsx: <Badge variant="secondary" className="max-w-full font-mono/, // object keys
    /editor\/trigger-panel\.tsx: <code /, // webhook command
  ];
  for (const line of mono)
    assert.ok(allowed.some((rule) => rule.test(line)), `font-mono on a non-code value: ${line}`);
});

test("Flows icons never draw with a translucent currentColor", () => {
  for (const path of files(root)) {
    const src = readFileSync(path, "utf8");
    for (const icon of src.match(/<HugeiconsIcon\b[^>]*?\/>/gs) ?? [])
      assert.doesNotMatch(icon, /text-[a-z-]+(-\d+)?\/\d+/, `${path}: ${icon}`);
  }
});
