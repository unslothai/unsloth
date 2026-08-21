import assert from "node:assert/strict";
import test from "node:test";

import { resolveLlamaBackendForWarning } from "../src/hooks/llama-backend-warning.ts";

test("environment backend override is authoritative for the warning", () => {
  assert.equal(
    resolveLlamaBackendForWarning({ backend: "cuda", envBackend: "vulkan" }),
    "vulkan",
  );
  assert.equal(
    resolveLlamaBackendForWarning({ backend: "cuda", envBackend: "auto" }),
    "cuda",
  );
  assert.equal(
    resolveLlamaBackendForWarning({ backend: "rocm", envBackend: null }),
    "rocm",
  );
});
