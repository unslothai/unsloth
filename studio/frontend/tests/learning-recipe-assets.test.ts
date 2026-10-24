// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import { fileURLToPath } from "node:url";
import { build } from "vite";

test("production learning recipes emit JSON assets fetchable under the same-origin CSP", async () => {
  const result = await build({
    configFile: false,
    root: fileURLToPath(new URL("../", import.meta.url)),
    logLevel: "silent",
    build: {
      write: false,
      minify: false,
      rolldownOptions: {
        input: fileURLToPath(
          new URL(
            "../src/features/data-recipes/learning-recipes/index.ts",
            import.meta.url,
          ),
        ),
        preserveEntrySignatures: "strict",
      },
    },
  });
  assert(!Array.isArray(result) && "output" in result);
  const books = result.output.find(
    (file) =>
      file.type === "asset" &&
      /books-and-documents-.*\.json$/.test(file.fileName),
  );
  assert(
    books?.type === "asset",
    "the small books template must be emitted as a JSON file",
  );
  const source =
    typeof books.source === "string"
      ? books.source
      : new TextDecoder().decode(books.source);
  assert(Array.isArray(JSON.parse(source).recipe.columns));
  for (const file of result.output) {
    if (file.type === "chunk") {
      assert.doesNotMatch(
        file.code,
        /data:application\/json/,
        "fetching an inline template is blocked by Unsloth's connect-src policy",
      );
    }
  }
});
