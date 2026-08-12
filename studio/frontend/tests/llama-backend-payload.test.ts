// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import {
  parseLlamaBackendStatus,
  visibleLlamaBackendOptions,
} from "../src/features/settings/api/llama-backend-payload.ts";

const FULL_PAYLOAD = {
  supported: true,
  reason: null,
  env_backend: null,
  backend: "cuda",
  backend_request: "auto",
  installed_tag: "b9596-mix-abc",
  options: [
    {
      backend: "auto",
      available: true,
      resolved_backend: "cuda",
      release_tag: "b9596-mix-abc",
      download_size_bytes: 183239972,
    },
    { backend: "rocm", available: false, unavailable_reason: "unavailable" },
  ],
  job: { state: "idle", message: "", progress: null },
};

test("a status payload is read into the picker's shape", () => {
  const status = parseLlamaBackendStatus(FULL_PAYLOAD);

  assert.equal(status.supported, true);
  assert.equal(status.backend, "cuda");
  assert.equal(status.backendRequest, "auto");
  assert.equal(status.options[0]?.resolvedBackend, "cuda");
  assert.equal(status.options[0]?.downloadSizeBytes, 183239972);
  assert.equal(status.options[1]?.available, false);
});

test("a backend this build does not know about is dropped, not offered", () => {
  // Older clients cannot label or submit unknown backends.
  const status = parseLlamaBackendStatus({
    ...FULL_PAYLOAD,
    options: [...FULL_PAYLOAD.options, { backend: "sycl", available: true }],
  });

  assert.deepEqual(
    status.options.map((option) => option.backend),
    ["auto", "rocm"],
  );
});

test("a backend older than this client's list still reads as automatic", () => {
  // An unknown recorded choice must not become automatic.
  const status = parseLlamaBackendStatus({
    ...FULL_PAYLOAD,
    backend_request: "sycl",
  });

  assert.equal(status.backendRequest, "auto");
});

test("a missing or malformed payload degrades instead of throwing", () => {
  const status = parseLlamaBackendStatus(null);

  assert.equal(status.supported, false);
  assert.equal(status.backendRequest, "auto");
  assert.deepEqual(status.options, []);
  assert.equal(status.job.state, "idle");
});

test("a running job is read with its progress so the bar can move", () => {
  const status = parseLlamaBackendStatus({
    ...FULL_PAYLOAD,
    job: {
      state: "running",
      operation: "switch",
      requested_backend: "vulkan",
      message: "Installing the vulkan llama.cpp build...",
      progress: 0.42,
      reload_required: null,
      started_at: "2026-08-11T12:00:00Z",
      finished_at: null,
    },
  });

  assert.equal(status.job.state, "running");
  assert.equal(status.job.operation, "switch");
  assert.equal(status.job.requestedBackend, "vulkan");
  assert.equal(status.job.startedAt, "2026-08-11T12:00:00Z");
  assert.equal(status.job.progress, 0.42);
  assert.equal(status.job.message, "Installing the vulkan llama.cpp build...");
});

test("only installable backends are offered", () => {
  const status = parseLlamaBackendStatus(FULL_PAYLOAD);

  assert.deepEqual(
    visibleLlamaBackendOptions(status, "auto").map((option) => option.backend),
    ["auto"],
  );
});

test("the selected backend stays listed even when it stops being installable", () => {
  // Keep the selected value in the control even after it becomes unavailable.
  const status = parseLlamaBackendStatus(FULL_PAYLOAD);

  assert.deepEqual(
    visibleLlamaBackendOptions(status, "rocm").map((option) => option.backend),
    ["auto", "rocm"],
  );
});
