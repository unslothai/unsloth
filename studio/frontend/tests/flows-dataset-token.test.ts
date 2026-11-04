// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { registerStoreStubResolver } from "./helpers/kit.ts";

registerStoreStubResolver();

test("the dataset viewer reads with this account's Hugging Face token", async () => {
  const { datasetRows, setFlowsTransport } = await import(
    "../src/features/flows/api.ts"
  );
  const { useHfTokenStore } = await import(
    "../src/features/hub/stores/hf-token-store.ts"
  );
  const seen: Record<string, string>[] = [];
  setFlowsTransport(async (_url, init) => {
    seen.push(init?.headers as Record<string, string>);
    return new Response(
      JSON.stringify({ rows: [], total: 0, offset: 0, view_rows: 1000 }),
    );
  });
  useHfTokenStore.setState({ token: "hf_private" });
  await datasetRows({ kind: "dataset", source: "hf", ref: "org/gated" }, 0, 20);
  setFlowsTransport(null);
  assert.equal(seen[0]["X-Unsloth-HF-Token"], "hf_private");
  assert.equal(seen[0]["Content-Type"], "application/json");
});
