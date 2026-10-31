// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// A static import of these from the chat graph puts them back in the eager startup bundle.
import assert from "node:assert/strict";
import { readdirSync, readFileSync, statSync } from "node:fs";
import { join } from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

import { readSrc } from "./helpers/kit.ts";

const SRC = fileURLToPath(new URL("../src/", import.meta.url));

function sources(dir: string): string[] {
  return readdirSync(dir).flatMap((name) => {
    const path = join(dir, name);
    if (statSync(path).isDirectory()) return sources(path);
    return /\.(ts|tsx)$/.test(name) && !/\.(test|check)\.tsx?$/.test(name) ? [path] : [];
  });
}

function staticValueImporters(specifier: string): string[] {
  const escaped = specifier.replace(/[.*+?^${}()|[\]\\/]/g, "\\$&");
  const pattern = new RegExp(`^\\s*(import|export)\\s+(?!type\\b)[^;]*?\\bfrom\\s+["']${escaped}["']`, "m");
  return sources(SRC)
    .filter((path) => pattern.test(readFileSync(path, "utf8")))
    .map((path) => path.slice(SRC.length));
}

test("node-forge is only imported dynamically", () => {
  assert.deepEqual(staticValueImporters("node-forge"), []);
});

test("mermaid is only imported dynamically", () => {
  assert.deepEqual(staticValueImporters("@streamdown/mermaid"), []);
  assert.deepEqual(staticValueImporters("mermaid"), []);
});

test("the training barrel does not re-export the YAML helpers", () => {
  assert.doesNotMatch(readSrc("features/training/index.ts"), /from\s+["']\.\/lib\/yaml-config["']/);
});
