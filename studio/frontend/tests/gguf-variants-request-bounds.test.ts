// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The picker's quant listing was issued with no abort signal and no offline handling, so
// an unanswered request left the expander on "Loading variants…" and auto-load behind it,
// while the Hub page, bounded and offline-aware, kept working.

import assert from "node:assert/strict";
import test from "node:test";

import { registerBundlerResolver } from "./helpers/kit.ts";

// The module reaches the abort-signal ponyfills through "@/".
registerBundlerResolver();
const { GGUF_VARIANTS_TIMEOUT_MS, ggufVariantsAbort, ggufVariantsQuery } =
  await import("../src/features/chat/api/gguf-variants-request.ts");

const REPO = "unsloth/Qwen3-8B-GGUF";

test("every listing carries a signal, so an unanswered one can still be given up on", () => {
  const abort = ggufVariantsAbort();
  try {
    assert.ok(abort.signal instanceof AbortSignal);
    assert.equal(abort.signal.aborted, false);
  } finally {
    abort.dispose();
  }
  assert.ok(GGUF_VARIANTS_TIMEOUT_MS > 0 && Number.isFinite(GGUF_VARIANTS_TIMEOUT_MS));
});

test("collapsing the row aborts the listing it started", () => {
  const controller = new AbortController();
  const abort = ggufVariantsAbort(controller.signal);
  try {
    assert.equal(abort.signal.aborted, false);
    controller.abort();
    assert.equal(abort.signal.aborted, true);
  } finally {
    abort.dispose();
  }
});

test("a row already collapsed never opens a request", () => {
  const controller = new AbortController();
  controller.abort();
  const abort = ggufVariantsAbort(controller.signal);
  try {
    assert.equal(abort.signal.aborted, true);
  } finally {
    abort.dispose();
  }
});

test("an unreachable Hub asks for the cached listing", () => {
  const params = ggufVariantsQuery(REPO, undefined, true);
  assert.equal(params.get("repo_id"), REPO);
  assert.equal(params.get("offline"), "true");
  // Without this the route takes the remote path anyway and spends the whole timeout.
  assert.equal(params.get("prefer_local_cache"), "true");
});

test("a reachable Hub is still asked for the remote listing", () => {
  const params = ggufVariantsQuery(REPO, undefined, false);
  assert.equal(params.get("offline"), null);
  assert.equal(params.get("prefer_local_cache"), null);
});

test("the row's own directory is scoped to, and blank paths are dropped", () => {
  const scoped = ggufVariantsQuery(
    REPO,
    { preferLocalCache: true, localPath: "  /models/qwen  " },
    false,
  );
  assert.equal(scoped.get("local_path"), "/models/qwen");
  assert.equal(scoped.get("prefer_local_cache"), "true");

  const blank = ggufVariantsQuery(REPO, { localPath: "   " }, false);
  assert.equal(blank.get("local_path"), null);
});
