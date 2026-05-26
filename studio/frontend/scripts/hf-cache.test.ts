// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { registerHooks } from "node:module";
import { resolve } from "node:path";
import test from "node:test";
import { pathToFileURL } from "node:url";

const srcRoot = resolve(import.meta.dirname, "../src");

registerHooks({
  resolve(specifier, context, nextResolve) {
    if (specifier.startsWith("@/")) {
      return {
        shortCircuit: true,
        url: pathToFileURL(resolve(srcRoot, `${specifier.slice(2)}.ts`)).href,
      };
    }
    return nextResolve(specifier, context);
  },
});

const { cachedModelInfo } = await import("../src/lib/hf-cache.ts");

test("cachedModelInfo requests and preserves Hub gguf and config fields", async () => {
  const requests: URL[] = [];
  const fetchModelInfo: typeof fetch = async (input) => {
    const rawUrl =
      typeof input === "string"
        ? input
        : input instanceof URL
          ? input.href
          : input.url;
    requests.push(new URL(rawUrl));

    return new Response(
      JSON.stringify({
        _id: "model-id",
        id: "org/model",
        private: false,
        pipeline_tag: "text-generation",
        downloads: 10,
        gated: false,
        likes: 5,
        lastModified: "2026-01-01T00:00:00.000Z",
        safetensors: { total: 123, parameters: { F32: 123 } },
        tags: ["text-generation"],
        gguf: { total: 456, architecture: "llama" },
        config: {
          architectures: ["LlamaForCausalLM"],
          model_type: "llama",
        },
      }),
      {
        status: 200,
        headers: { "content-type": "application/json" },
      },
    );
  };

  const result = await cachedModelInfo({
    name: "org/model",
    fetch: fetchModelInfo,
  });

  assert.equal(requests.length, 1);
  const expand = requests[0].searchParams.getAll("expand");
  assert.equal(expand.includes("safetensors"), true);
  assert.equal(expand.includes("tags"), true);
  assert.equal(expand.includes("gguf"), true);
  assert.equal(expand.includes("config"), true);
  assert.deepEqual(result.safetensors, {
    total: 123,
    parameters: { F32: 123 },
  });
  assert.deepEqual(result.tags, ["text-generation"]);
  assert.deepEqual(result.gguf, { total: 456, architecture: "llama" });
  assert.deepEqual(result.config, {
    architectures: ["LlamaForCausalLM"],
    model_type: "llama",
  });
});
