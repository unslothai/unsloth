// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync, readdirSync, statSync } from "node:fs";
import path from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

// A value-import cycle through tool-fallback.tsx once left ToolFallback in its TDZ while build and unit tests stayed green.

const SRC = fileURLToPath(new URL("../src", import.meta.url));
const ENTRY = "components/assistant-ui/tool-fallback.tsx";

function sourceFiles(dir: string, out: string[] = []): string[] {
  for (const name of readdirSync(dir)) {
    const full = path.join(dir, name);
    if (statSync(full).isDirectory()) sourceFiles(full, out);
    else if (/\.tsx?$/.test(name)) out.push(full);
  }
  return out;
}

function resolveSpec(spec: string, from: string): string | null {
  let base: string;
  if (spec.startsWith("@/")) base = path.join(SRC, spec.slice(2));
  else if (spec.startsWith(".")) base = path.resolve(path.dirname(from), spec);
  else return null;
  for (const ext of [".ts", ".tsx", "/index.ts", "/index.tsx"]) {
    try {
      const candidate = `${base}${ext}`;
      if (statSync(candidate).isFile()) return candidate;
    } catch {
    }
  }
  return null;
}

const VALUE_IMPORT =
  /^[ \t]*(?:import|export)\s+(?!type[\s{])(?:[^'"]*?from\s+)?['"]([^'"]+)['"]/gm;

function importGraph(): Map<string, string[]> {
  const graph = new Map<string, string[]>();
  for (const file of sourceFiles(SRC)) {
    const text = readFileSync(file, "utf8");
    const deps = new Set<string>();
    for (const match of text.matchAll(VALUE_IMPORT)) {
      const resolved = resolveSpec(match[1], file);
      if (resolved) deps.add(resolved);
    }
    graph.set(file, [...deps]);
  }
  return graph;
}

test("tool-fallback is not on a value-import cycle", () => {
  const graph = importGraph();
  const entry = path.join(SRC, ENTRY);
  assert.ok(graph.has(entry), `${ENTRY} is no longer where this test looks`);

  const queue: string[][] = [[entry]];
  const seen = new Set([entry]);
  let cycle: string[] | null = null;
  while (queue.length && !cycle) {
    const trail = queue.shift() as string[];
    for (const dep of graph.get(trail[trail.length - 1]) ?? []) {
      if (dep === entry) {
        cycle = [...trail, dep];
        break;
      }
      if (!seen.has(dep)) {
        seen.add(dep);
        queue.push([...trail, dep]);
      }
    }
  }

  assert.equal(
    cycle,
    null,
    cycle
      ? `tool-fallback.tsx is back on an import cycle:\n  ${cycle
          .map((f) => path.relative(SRC, f))
          .join("\n  -> ")}\nImport from the defining module rather than a barrel to cut it.`
      : "",
  );
});
