// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { register } from "node:module";
import test from "node:test";

register("./store-stub-resolver.mjs", import.meta.url);

const { BenchmarkRequestError, isRecoverableTransportError } = await import(
  "../src/features/benchmark/api/benchmark-api.ts"
);

test("a dropped evaluation request recovers via status, a real rejection does not", () => {
  // Shape authFetch throws when the connection is lost (features/auth/api.ts).
  const dropped = Object.assign(new Error("Studio is not answering."), {
    unslothTransportFailure: true,
  });
  assert.equal(isRecoverableTransportError(dropped), true);
  assert.equal(
    isRecoverableTransportError(new BenchmarkRequestError("gw", 504)),
    true,
  );
  assert.equal(
    isRecoverableTransportError(new BenchmarkRequestError("bad", 400)),
    false,
  );
  assert.equal(
    isRecoverableTransportError(new Error("Benchmark failed")),
    false,
  );
  assert.equal(isRecoverableTransportError(null), false);
});
