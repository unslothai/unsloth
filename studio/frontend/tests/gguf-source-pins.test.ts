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
  // Both functions travel together: the reconciler calls the non-GGUF survivor probe.
  .slice(source.indexOf("async function hasRunnableNonGgufCopy"))
  .replace("export ", "");
const compile = new Function(
  "listCachedGguf",
  "fetchCachedGgufInventory",
  "fetchCachedModelsInventory",
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
    "non-gguf-survivor",
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
      const modelsInventory = async () => ({
        cached:
          remaining === "non-gguf-survivor"
            ? [
                {
                  repo_id: "org/model",
                  model_format: "safetensors",
                  partial: false,
                  capabilities: { can_chat: true },
                },
              ]
            : [],
        scan_confirmed: true,
      });
      const run = compile(
        // The bare pin must survive the last quant when a non-GGUF copy of the same
        // repo is still cached and runnable, and fall only when none is left.
        async () => (await inventory()).cached,
        inventory,
        modelsInventory,
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
                  !remaining.endsWith("partial") &&
                  // The last quant is what was just deleted; only a non-GGUF copy is left.
                  remaining !== "non-gguf-survivor",
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
      // The survivor case carries a token: an anonymous listing stops at the guard that
      // trusts the complete copy on disk, before any pin is reconsidered.
      await run(
        requested,
        remaining === "non-gguf-survivor" ? "hf_fixture_token" : undefined,
      );
      const expected =
        remaining === "absent" ||
        remaining === "partial" ||
        remaining === "anonymous-partial-inventory"
          ? []
          : remaining === "other-quant" || remaining === "non-gguf-survivor"
            ? ["Org/Model"]
            : original;
      assert.deepEqual(pinned, expected);
      if (remaining.startsWith("unconfirmed")) assert.equal(variantCalls, 0);
    });
  }
}
