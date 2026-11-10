// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const { fetchModelSize } = await import(
  "../src/features/hub/lib/dataset-size.ts"
);

function stubSiblings(sizes: Record<string, number>): void {
  globalThis.fetch = (async () =>
    new Response(
      JSON.stringify({
        siblings: Object.entries(sizes).map(([rfilename, size]) => ({
          rfilename,
          size,
        })),
      }),
      { status: 200, headers: { "content-type": "application/json" } },
    )) as typeof fetch;
}

test("a diffusers pipeline skips the pickles its safetensors make redundant", async () => {
  stubSiblings({
    "model_index.json": 1,
    "unet/diffusion_pytorch_model.safetensors": 100,
    "unet/diffusion_pytorch_model.bin": 100,
    "v1-5-pruned.safetensors": 50,
    "v1-5-pruned.ckpt": 50,
    "Model-FP8.pt": 30,
  });
  // The root .pt with no safetensors twin is the only copy and stays.
  assert.deepEqual(await fetchModelSize("acme/sd-shaped"), {
    totalBytes: 1 + 100 + 50 + 30,
    weightsBytes: 100 + 50 + 30,
  });
});

test("an adapter safetensors never drops a component's only .bin", async () => {
  stubSiblings({
    "model_index.json": 1,
    "transformer/adapter_model.safetensors": 5,
    "transformer/diffusion_pytorch_model.bin": 200,
    "vae/diffusion_pytorch_model.fp16.safetensors": 10,
    "vae/diffusion_pytorch_model.bin": 20,
  });
  assert.deepEqual(await fetchModelSize("acme/adapter-shaped"), {
    totalBytes: 1 + 5 + 200 + 10 + 20,
    weightsBytes: 5 + 200 + 10 + 20,
  });
});
