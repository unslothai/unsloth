// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const { fetchModelSize } = await import(
  "../src/features/hub/lib/dataset-size.ts"
);

function stubSiblings(sizes: Record<string, number>, pipelineClass?: string): void {
  globalThis.fetch = (async () =>
    new Response(
      JSON.stringify({
        config: { diffusers: { _class_name: pipelineClass } },
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

test("SDXL checkpoint candidates include both Animagine variants and exclude components and shards", async () => {
  stubSiblings({
    "model_index.json": 671,
    "unet/diffusion_pytorch_model.safetensors": 512,
    "model-00001-of-00002.safetensors": 256,
    "animagine-xl-4.0.safetensors": 6938434056,
    "animagine-xl-4.0-opt.safetensors": 6938350040,
  }, "StableDiffusionXLPipeline");
  const info = await fetchModelSize("cagliostrolab/animagine-xl-4.0");
  assert.deepEqual(info?.imageCheckpointCandidates, [
    { filename: "animagine-xl-4.0.safetensors", bytes: 6938434056 },
    { filename: "animagine-xl-4.0-opt.safetensors", bytes: 6938350040 },
  ]);
});

function deferredResponse() {
  let resolve!: (response: Response) => void;
  const promise = new Promise<Response>((done) => { resolve = done; });
  return { promise, resolve };
}

test("reopening a model while its canceled metadata lookup settles starts one fresh lookup", async (t) => {
  const repo = "acme/reopened-sdxl";
  const first = new AbortController();
  const canceled = deferredResponse();
  const fresh = deferredResponse();
  const requests = [canceled, fresh];
  const fetch = t.mock.method(globalThis, "fetch", () => {
    const request = requests.shift();
    assert.ok(request, "concurrent readers must share the replacement lookup");
    return request.promise;
  });
  const initial = fetchModelSize(repo, undefined, first.signal);
  first.abort();
  const reopened = fetchModelSize(repo);
  assert.equal(await initial, null);
  // Let the old fetch finish after the replacement has taken ownership of the key.
  canceled.resolve(Response.json({ siblings: [] }));
  await new Promise<void>((resolve) => setImmediate(resolve));
  const secondReader = fetchModelSize(repo);
  fresh.resolve(Response.json({ siblings: [{ rfilename: "weights.safetensors", size: 42 }] }));
  const expected = { totalBytes: 42, weightsBytes: 42 };
  assert.deepEqual(await reopened, expected);
  assert.deepEqual(await secondReader, expected);
  assert.equal(fetch.mock.callCount(), 2);
});

test("the picker offers only confirmed checkpoints from a mixed SDXL repo", async (t) => {
  const { confirmedSdxlCheckpoints } = await import("../src/features/hub/lib/sdxl-checkpoints.ts");
  const { installLocalStorageFake } = await import("./helpers/kit.ts");
  installLocalStorageFake();
  const names = ["edition.safetensors", "unet.safetensors", "model.safetensors"];
  const info = {
    revision: "1".repeat(40), totalBytes: null, weightsBytes: null,
    imageCheckpointCandidates: names.map((filename) => ({ filename, bytes: null })),
  };
  const calls: string[] = [];
  const fetch = t.mock.method(globalThis, "fetch", async (input: RequestInfo | URL) => {
    const params = new URL(String(input), "http://localhost").searchParams;
    assert.equal(params.get("revision"), info.revision);
    assert.equal(params.get("repo_id"), "acme/mixed");
    const filename = params.get("filename")!;
    calls.push(filename);
    return Response.json(filename === "model.safetensors");
  });
  const read = () => confirmedSdxlCheckpoints("acme/mixed", info, undefined, new AbortController().signal);
  const expected = [info.imageCheckpointCandidates[2]];
  assert.deepEqual(await read(), expected);
  assert.deepEqual(await read(), expected);
  assert.deepEqual(calls.sort(), [...names].sort());
  assert.equal(fetch.mock.callCount(), 3);
});

test("confirmed SDXL checkpoints survive a reload for offline reopening", async () => {
  const { rememberSdxlCheckpoints, rememberedSdxlCheckpoints } = await import(
    "../src/features/hub/lib/sdxl-checkpoints.ts"
  );
  const { installLocalStorageFake } = await import("./helpers/kit.ts");
  installLocalStorageFake();
  assert.equal(rememberedSdxlCheckpoints("acme/sdxl"), undefined);
  const files = [{ filename: "edition.safetensors", bytes: 7 }];
  rememberSdxlCheckpoints("Acme/SDXL", files);
  assert.deepEqual(rememberedSdxlCheckpoints("acme/sdxl"), files);
  for (let i = 0; i < 60; i++) rememberSdxlCheckpoints(`acme/r${i}`, files);
  assert.equal(rememberedSdxlCheckpoints("acme/sdxl"), undefined);
  assert.deepEqual(rememberedSdxlCheckpoints("acme/r59"), files);
  window.localStorage.setItem("unsloth.hub.sdxlCheckpoints", JSON.stringify({ "acme/x": [{ filename: "../a.safetensors" }] }));
  assert.deepEqual(rememberedSdxlCheckpoints("acme/x"), []);
});
