// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync, readdirSync } from "node:fs";
import { join, relative } from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

const SRC = fileURLToPath(new URL("../src", import.meta.url));

function sources(dir: string): string[] {
  return readdirSync(dir, { withFileTypes: true }).flatMap((entry) => {
    const path = join(dir, entry.name);
    if (entry.isDirectory()) return sources(path);
    return /\.tsx?$/.test(entry.name) ? [path] : [];
  });
}

// @streamdown/code memoises tokenisations in a never-evicting Map; import its types only.
test("no source module loads @streamdown/code at runtime", () => {
  const offenders = sources(SRC).filter((path) => {
    const text = readFileSync(path, "utf8");
    return (
      /^import\s+(?!type\b)[^;]*?from\s+["']@streamdown\/code["']/m.test(
        text,
      ) || /import\(\s*["']@streamdown\/code["']\s*\)/.test(text)
    );
  });
  assert.deepEqual(
    offenders.map((path) => relative(SRC, path)),
    [],
  );
});

test("finished-code surfaces share the bounded plugin", () => {
  for (const path of [
    "components/assistant-ui/tool-code-cell.tsx",
    "components/markdown/markdown-preview.tsx",
    "features/hub/catalog/model-readme.tsx",
  ]) {
    assert.match(
      readFileSync(join(SRC, path), "utf8"),
      /import \{ codePlugin \} from "(?:\.|@\/components\/assistant-ui)\/shared-code-plugin";/,
      path,
    );
  }
});
