// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const { isRetryableDownload, retryRequestFor } =
  await import("../src/features/hub/download-manager/retry-download.ts");

function job(over: Record<string, unknown> = {}) {
  const variant = (over.variant as string | null | undefined) ?? null;
  return {
    kind: "model",
    repoId: "Org/Repo",
    variant,
    state: "running",
    downloadedBytes: 1024,
    completedBytes: 0,
    completeOnDisk: false,
    expectedBytes: 4096,
    fraction: 0.25,
    bytesPerSec: 0,
    etaSeconds: 0,
    error: null,
    startedAt: 1,
    ...over,
  } as never;
}

test("a scoped row retries with scope_id and its files, a plain variant without", () => {
  const scoped = retryRequestFor(
    job({
      variant: "@hub-assets",
      scopedFiles: ["vae.safetensors"],
      checkpoint: false,
    }),
  );
  assert.equal(scoped.scopeId, "hub-assets");
  assert.deepEqual(scoped.files, ["vae.safetensors"]);
  assert.equal(scoped.checkpoint, false);

  const gguf = retryRequestFor(
    job({ variant: "Q4_K_M", state: "cancelled", downloadDir: "/models" }),
  );
  assert.equal(gguf.scopeId, undefined);
  assert.equal(gguf.variant, "Q4_K_M");
  assert.equal(gguf.downloadDir, "/models");
  // A failed start re-resolves the current folder, which may be the fix.
  const failed = retryRequestFor(
    job({ variant: "Q4_K_M", state: "error", downloadDir: "/missing" }),
  );
  assert.equal("downloadDir" in failed, false);
});

test("rows owned by other flows are not retryable from the panel", () => {
  assert.equal(isRetryableDownload(job()), true);
  assert.equal(
    isRetryableDownload(job({ external: true, repoId: "vLLM" })),
    false,
  );
  assert.equal(
    isRetryableDownload(job({ variant: `@training-${"a".repeat(32)}` })),
    false,
  );
  assert.equal(isRetryableDownload(job({ variant: "@hub-assets" })), false);
});
