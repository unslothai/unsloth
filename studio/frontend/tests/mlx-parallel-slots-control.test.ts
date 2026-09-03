// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0


import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import path from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const read = (relative: string) =>
  readFileSync(path.join(HERE, "..", relative), "utf8");

const CONFIG_PAGE = read(
  "src/features/model-picker/components/model-config-page.tsx",
);
const RUNTIME = read("src/features/chat/hooks/use-chat-model-runtime.ts");
const COMPOSER = read("src/features/chat/shared-composer.tsx");

function bodyOf(source: string, name: string): string {
  const lines = source.split("\n");
  const start = lines.findIndex((line) =>
    line.startsWith(`function ${name}(`),
  );
  assert.notEqual(start, -1, `no top-level function ${name}`);
  let end = lines.length;
  for (let i = start + 1; i < lines.length; i++) {
    if (/^(export )?function \w+\(/.test(lines[i])) {
      end = i;
      break;
    }
  }
  return lines.slice(start, end).join("\n");
}

test("each panel's note tells the truth about what it costs", () => {
  const hint = (name: string) => {
    const at = CONFIG_PAGE.indexOf(`const ${name} =`);
    assert.notEqual(at, -1, `${name} is gone`);
    return CONFIG_PAGE.slice(at, CONFIG_PAGE.indexOf("\n\n", at)).replace(
      /"\s*\+\s*\n?\s*"/g,
      "",
    );
  };
  assert.match(hint("GGUF_PARALLEL_HINT"), /fewer slots are launched/);
  assert.match(hint("MLX_PARALLEL_HINT"), /nothing reduces the number to fit/);
  assert.doesNotMatch(hint("MLX_PARALLEL_HINT"), /llama-server|fewer are run|fewer slots/);
});

test("both load paths commit a width for exactly the runtimes that decode one", () => {
  const paths: Array<[string, string, string]> = [
    ["primary", RUNTIME, "loadResponse"],
    ["compare", COMPOSER, "resp"],
  ];
  for (const [name, source, response] of paths) {
    const at = source.indexOf("const committedSlots");
    assert.notEqual(at, -1, `${name} load commits no width at all`);
    const committed = source.slice(at, source.indexOf("committedNBatch", at));
    for (const required of [
      `${response}.is_gguf`,
      `${response}.is_diffusion`,
      `${response}.is_mlx`,
    ]) {
      assert.ok(
        committed.includes(required),
        `the ${name} load decides its width without ${required}`,
      );
    }
    assert.match(
      committed,
      /\?\s*\(?[\w.]*[Nn]Parallel[^:]*:\s*null/,
      `the ${name} load commits its width on the wrong branch`,
    );
  }
});
