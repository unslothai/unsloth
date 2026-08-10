// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * A quant row has to advertise the whole pick, and only the checkpoint has to fit in memory.
 *
 * Sized from the GGUF repo alone, a `unsloth/Qwen-Image-Edit-2511-GGUF` BF16 row read 40.87 GB and
 * then fetched 57.73 GB: the base repo's text encoder, VAE, tokenizer and configs were never named.
 * Adding the companions to the row fixes that number, but they must not reach the OOM/TIGHT badge,
 * which judges what the denoiser needs resident and would invent OOM rows out of disk bytes.
 */

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import test from "node:test";

const pickers = readFileSync(
  fileURLToPath(
    new URL(
      "../src/features/model-picker/components/model-selector/pickers.tsx",
      import.meta.url,
    ),
  ),
  "utf8",
);

test("the quant row advertises the companions it also downloads", () => {
  assert.ok(
    pickers.includes("formatBytes(v.size_bytes + companionBytes)"),
    "the row size must include the companion bytes",
  );
  assert.ok(
    !/SizeText value={formatBytes\(v\.size_bytes\)}/.test(pickers),
    "no row may still show the GGUF size alone",
  );
});

test("the memory verdict stays on the checkpoint alone", () => {
  // The text encoder is a disk cost the badge has no say over: an offloaded one never competes
  // with the denoiser for VRAM, so folding it in would mark quants OOM that run fine.
  assert.ok(
    pickers.includes("const fit = getGgufFit(v.size_bytes);"),
    "the fit classification must read the checkpoint size",
  );
  assert.ok(
    !/getGgufFit\([^)]*companionBytes/.test(pickers),
    "companion bytes must never reach the fit classification",
  );
});

test("the chat picker asks for no companion sizes", () => {
  // The resolver is supplied by the Images and Video pages, which own the load settings the plan
  // depends on. Without one the hook answers 0 and the rows read exactly as they always did.
  const selector = readFileSync(
    fileURLToPath(
      new URL(
        "../src/features/model-picker/components/model-selector.tsx",
        import.meta.url,
      ),
    ),
    "utf8",
  );
  assert.ok(selector.includes("resolveCompanionBytes ?? null"));

  const companionBytes = readFileSync(
    fileURLToPath(
      new URL(
        "../src/features/model-picker/components/model-selector/companion-bytes.ts",
        import.meta.url,
      ),
    ),
    "utf8",
  );
  assert.ok(
    companionBytes.includes("if (!resolve || !key || !sampleGgufFilename) return;"),
    "no resolver means no request",
  );
});
