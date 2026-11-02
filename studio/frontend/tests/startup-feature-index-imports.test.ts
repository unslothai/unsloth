// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Importing a feature index from startup code makes every module it re-exports part of the
// startup bundle, so these indexes must stay out of main.tsx's static import graph.
import assert from "node:assert/strict";
import { existsSync, readFileSync } from "node:fs";
import { dirname, join, relative } from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

const SRC = fileURLToPath(new URL("../src/", import.meta.url));

const LAZY_FEATURE_INDEXES = [
  "chat",
  "hub",
  "library",
  "model-picker",
  "rag",
  "settings",
  "training",
].map((name) => `features/${name}/index.ts`);

// Static value imports and re-exports, plus bare side-effect imports. `import type` is erased.
const STATIC_IMPORT =
  /^\s*(?:import|export)\s+(?!type\s)(?:[^;'"]*?\bfrom\s+)?["']([^"']+)["']/gm;

function resolveSource(fromFile: string, specifier: string): string | null {
  let base: string;
  if (specifier.startsWith("@/")) base = join(SRC, specifier.slice(2));
  else if (specifier.startsWith(".")) base = join(dirname(fromFile), specifier);
  else return null;
  for (const suffix of ["", ".ts", ".tsx", "/index.ts", "/index.tsx"]) {
    const candidate = base + suffix;
    if (/\.(ts|tsx)$/.test(candidate) && existsSync(candidate)) return candidate;
  }
  return null;
}

function startupGraph(): Map<string, string> {
  const entry = join(SRC, "main.tsx");
  const parent = new Map<string, string>([[entry, ""]]);
  const stack = [entry];
  while (stack.length > 0) {
    const file = stack.pop() as string;
    for (const match of readFileSync(file, "utf8").matchAll(STATIC_IMPORT)) {
      const target = resolveSource(file, match[1] as string);
      if (target && !parent.has(target)) {
        parent.set(target, file);
        stack.push(target);
      }
    }
  }
  return parent;
}

function chain(parent: Map<string, string>, file: string): string {
  const steps: string[] = [];
  for (let at = file; at; at = parent.get(at) ?? "") {
    steps.push(relative(SRC, at));
  }
  return steps.join(" <- ");
}

test("startup code imports lazy features' modules, not their indexes", () => {
  const parent = startupGraph();
  const reached = LAZY_FEATURE_INDEXES.flatMap((index) => {
    const file = join(SRC, index);
    return parent.has(file) ? [chain(parent, file)] : [];
  });
  assert.deepEqual(reached, []);
});

test("the markdown renderer (Streamdown, KaTeX, Shiki) is not in the startup graph", () => {
  const parent = startupGraph();
  const preview = join(SRC, "components/markdown/markdown-preview.tsx");
  const exporters = join(SRC, "features/chat/prompt-storage/prompt-storage-dialog.tsx");
  const reached = [preview, exporters].flatMap((file) =>
    parent.has(file) ? [chain(parent, file)] : [],
  );
  assert.deepEqual(reached, []);
});

test("the startup graph walk sees the shell", () => {
  const parent = startupGraph();
  for (const shell of ["app/routes/__root.tsx", "components/app-sidebar.tsx", "app/provider.tsx"]) {
    assert.ok(parent.has(join(SRC, shell)), `${shell} not reached from main.tsx`);
  }
  assert.ok(parent.size > 300, `only ${parent.size} startup modules found`);
});
