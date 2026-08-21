import assert from "node:assert/strict";
import test from "node:test";

import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const { inferenceBackendFromSystem } = await import(
  "../src/hooks/gpu-backend.ts",
);

test("inference GPU info uses the llama.cpp backend when it differs from torch", () => {
  assert.equal(
    inferenceBackendFromSystem({
      device_backend: "cuda",
      inference_gpu: { backend: "vulkan", available: true, devices: [] },
    } as never),
    "vulkan",
  );
  assert.equal(
    inferenceBackendFromSystem({ device_backend: "rocm" } as never),
    "rocm",
  );
});
