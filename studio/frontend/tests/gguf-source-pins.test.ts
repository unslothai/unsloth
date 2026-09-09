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
import {
  ggufVariantsMatchForPicker,
  modelIdsMatchForPicker,
} from "../src/features/model-picker/components/model-selector/row-identity.ts";

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
  "fetchCachedGgufInventory",
  "listGgufVariants",
  "pinnedQuantEntries",
  "usePinnedModelsStore",
  "modelIdsMatchForPicker",
  "ggufVariantsMatchForPicker",
  ts.transpileModule(declaration, {
    compilerOptions: { target: ts.ScriptTarget.ES2020 },
  }).outputText + "; return reconcileGgufPinsAfterDelete;",
);

for (const requested of ["Org/Model", "org/model"]) {
  for (const remaining of [
    "anonymous",
    "anonymous-partial-inventory",
    "duplicate",
    "absent",
    "partial",
    "unavailable",
    "unconfirmed-absent",
    "unconfirmed-partial",
    "lowercase-quant",
    "other-quant",
    "legacy-confirmation",
  ]) {
    test(`GGUF pins after deleting ${requested}: ${remaining}`, async () => {
      const original = ["Org/Model", pinKey("Org/Model", "Q8_0")];
      let pinned = [...original];
      let variantCalls = 0;
      const inventory = async () => {
        if (remaining === "unavailable") throw new Error("scan unavailable");
        return {
          cached: remaining.endsWith("absent")
            ? []
            : [
                {
                  repo_id: "org/model",
                  partial:
                    remaining.endsWith("partial") ||
                    remaining === "anonymous-partial-inventory",
                },
              ],
          scan_confirmed:
            remaining === "legacy-confirmation"
              ? undefined
              : !remaining.startsWith("unconfirmed"),
        };
      };
      const run = compile(
        async () => (await inventory()).cached,
        inventory,
        async () => {
          variantCalls += 1;
          return {
            variants: [
              {
                quant:
                  remaining === "lowercase-quant"
                    ? "q8_0"
                    : remaining === "other-quant"
                      ? "Q6_K"
                      : "Q8_0",
                downloaded:
                  !remaining.startsWith("anonymous") &&
                  !remaining.endsWith("partial"),
                partial: remaining.endsWith("partial"),
              },
            ],
          };
        },
        pinnedQuantEntries,
        {
          getState: () => ({
            pinned,
            togglePinned: (repo: string, quant?: string) => {
              pinned = pinned.filter((p) => p !== pinKey(repo, quant));
            },
          }),
        },
        modelIdsMatchForPicker,
        ggufVariantsMatchForPicker,
      );
      await run(requested);
      const expected =
        remaining === "absent" ||
        remaining === "partial" ||
        remaining === "anonymous-partial-inventory"
          ? []
          : remaining === "other-quant"
            ? ["Org/Model"]
            : original;
      assert.deepEqual(pinned, expected);
      if (remaining.startsWith("unconfirmed")) assert.equal(variantCalls, 0);
    });
  }
}
