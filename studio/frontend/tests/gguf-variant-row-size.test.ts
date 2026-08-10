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
    pickers.includes("v.size_bytes + (companionBytes.get(v.filename) ?? 0)"),
    "the row size must include that row's own companion bytes",
  );
  assert.ok(
    !/SizeText value={formatBytes\(v\.size_bytes\)}/.test(pickers),
    "no row may still show the GGUF size alone",
  );
});

test("companion bytes are asked for per row, not per repo", () => {
  // The companion set follows the checkpoint: MiniMax-H3 pairs a -Q2_ transformer with the Q2
  // Qwen3-VL encoder and every other quant with the Q4 one, so one sample answer applied to every
  // row misreports whichever tier it did not come from by several GB.
  assert.ok(
    pickers.includes("(displayVariants ?? []).map((variant) => variant.filename)"),
    "every displayed row must be in the query",
  );
  // Local paths included: only the checkpoint is on disk, and the base it resolves is not.
  assert.ok(
    !/isLocalPath \? \[\]/.test(pickers),
    "a local checkpoint still needs its remote companions sized",
  );
  assert.ok(
    pickers.includes("companionBytes.get(v.filename)"),
    "each row must read its own answer",
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
    companionBytes.includes("if (!resolve || !key) return;"),
    "no resolver means no request",
  );
});

test("an answer is scoped to the settings it was planned under", () => {
  // A precision change hands the picker a new resolver, but the rows are the same filenames. Keyed
  // on those alone, the old totals kept satisfying the check while the replacement was in flight,
  // so a click could stage tens of GB the row never advertised.
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
    companionBytes.includes("`${resolverId(resolve)}\\n${ggufFilenames.join(\"\\n\")}`"),
    "the answer key must carry the resolver identity, not just the filenames",
  );
});

test("a row cannot be picked before its size is known", () => {
  // Until the batch lands there is no companion figure, so the row would show the GGUF-only size
  // and still be clickable: picking there stages the full plan, which is the original bug.
  assert.ok(
    pickers.includes("const awaitingSize = companionBytesBlocked.has(v.filename);"),
    "a row waits on its OWN answer, not on the batch and not on v.downloaded",
  );
  assert.ok(
    !pickers.includes("v.downloaded !== true"),
    "downloaded covers the checkpoint only, so it cannot excuse a row from waiting",
  );
  assert.ok(
    pickers.includes("disabled={unusableLocal || awaitingSize}"),
    "a row awaiting its size must not be selectable",
  );
  assert.ok(
    /awaitingSize\s*\?\s*"…"/.test(pickers),
    "and must not present the checkpoint-only number as the total",
  );
});

test("a repo larger than the request cap is still sizable", () => {
  // The route rejects a payload over MAX_COMPANION_SIZE_QUERIES outright, and browse lists show
  // every quant, so one oversized request would 422 and leave every row of the repo disabled.
  const companionBytes = readFileSync(
    fileURLToPath(
      new URL(
        "../src/features/model-picker/components/model-selector/companion-bytes.ts",
        import.meta.url,
      ),
    ),
    "utf8",
  );
  assert.ok(companionBytes.includes("const MAX_FILENAMES_PER_REQUEST = 96;"));
  assert.ok(
    companionBytes.includes("i += MAX_FILENAMES_PER_REQUEST"),
    "filenames must be chunked to the server's limit",
  );
  // Each request is its own batch on the server, with its own concurrency bound and its own
  // listings, so chunks in parallel would multiply the planning and re-fetch the same metadata.
  assert.ok(
    companionBytes.includes("for (const chunk of chunks) {"),
    "chunks must go one at a time",
  );
  assert.ok(
    !companionBytes.includes("Promise.all("),
    "and never all at once",
  );
});

test("a request that failed outright leaves its rows usable", () => {
  // Disabling a row is for "asked and got no size". A failed request answered nothing, so the row
  // falls back to what it showed before any of this rather than becoming unpickable.
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
    /catch \{\s*for \(const name of chunk\) blocked\.delete\(name\);/.test(companionBytes),
    "a failed chunk must release its own rows",
  );
  assert.ok(
    pickers.includes("const awaitingSize = companionBytesBlocked.has(v.filename);"),
    "and the row must gate on that set",
  );
});

test("an abandoned expander stops its batch", () => {
  // Collapsing must cancel the request, not just the setState: one abandoned 63-quant batch keeps
  // the backend planning every candidate against the Hub.
  const companionBytes = readFileSync(
    fileURLToPath(
      new URL(
        "../src/features/model-picker/components/model-selector/companion-bytes.ts",
        import.meta.url,
      ),
    ),
    "utf8",
  );
  assert.ok(companionBytes.includes("new AbortController()"));
  assert.ok(
    companionBytes.includes("controller.abort();"),
    "the effect cleanup must abort the in-flight batch",
  );
  assert.ok(
    /resolve\(repoId,[\s\S]*?controller\.signal\)/.test(companionBytes),
    "the signal must reach the resolver",
  );
});
