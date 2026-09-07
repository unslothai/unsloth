// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import ts from "typescript";
import {
  pinKey,
  pinnedQuantEntries,
} from "../src/features/model-picker/components/model-selector/pinned-models.ts";

test("GGUF pins survive deleting a duplicate and disappear with the last copy", async () => {
  const source = readFileSync(
    new URL(
      "../src/features/model-picker/components/model-selector/reconcile-gguf-pins.ts",
      import.meta.url,
    ),
    "utf8",
  );
  const declaration = source
    .slice(source.indexOf("export async function"))
    .replace("export ", "");
  const compile = new Function(
    "listCachedGguf",
    "listGgufVariants",
    "pinKey",
    "pinnedQuantEntries",
    "usePinnedModelsStore",
    ts.transpileModule(declaration, {
      compilerOptions: { target: ts.ScriptTarget.ES2020 },
    }).outputText + "; return reconcileGgufPinsAfterDelete;",
  );
  for (const remaining of [true, false, "unavailable"]) {
    let pinned = ["Org/Model", pinKey("Org/Model", "Q8_0")];
    const run = compile(
      async () => {
        if (remaining === "unavailable") throw new Error("scan unavailable");
        return remaining ? [{ repo_id: "Org/Model" }] : [];
      },
      async () => ({ variants: [{ quant: "Q8_0", downloaded: true }] }),
      pinKey,
      pinnedQuantEntries,
      {
        getState: () => ({
          pinned,
          togglePinned: (repo: string, quant?: string) => {
            pinned = pinned.filter((p) => p !== pinKey(repo, quant));
          },
        }),
      },
    );
    await run("Org/Model");
    assert.equal(pinned.length, remaining ? 2 : 0);
  }
});
