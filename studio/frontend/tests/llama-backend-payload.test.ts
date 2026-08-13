// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import {
  llamaBackendSelectionNeedsApply,
  parseLlamaBackendStatus,
  visibleLlamaBackendOptions,
} from "../src/features/settings/api/llama-backend-payload.ts";

const FULL_PAYLOAD = {
  supported: true,
  reason: null,
  env_backend: null,
  backend: "cuda",
  backend_request: "auto",
  selection_applied: true,
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

test("a backend older than this client's list stays unknown", () => {
  // An unknown recorded choice must not become automatic: the picker would then
  // show detection for an install that is pinned, and overwrite it on the next
  // apply. Untouched it is not dirty, but picking over it deliberately is.
  const status = parseLlamaBackendStatus({
    ...FULL_PAYLOAD,
    backend_request: "sycl",
  });

  assert.equal(status.backendRequest, null);
  assert.equal(llamaBackendSelectionNeedsApply(status, null), false);
  assert.equal(llamaBackendSelectionNeedsApply(status, "cuda"), true);
});

test("a missing or malformed payload degrades instead of throwing", () => {
  const status = parseLlamaBackendStatus(null);

  assert.equal(status.supported, false);
  assert.equal(status.backendRequest, null);
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

test("automatic can be applied again when it now resolves differently", () => {
  const status = parseLlamaBackendStatus({
    ...FULL_PAYLOAD,
    backend: "cpu",
    selection_applied: false,
  });

  assert.equal(llamaBackendSelectionNeedsApply(status, null), true);
  assert.equal(llamaBackendSelectionNeedsApply(status, "auto"), true);
});

test("older status payloads do not become dirty without server evidence", () => {
  const { selection_applied: _selectionApplied, ...olderPayload } =
    FULL_PAYLOAD;
  assert.equal(_selectionApplied, true);
  const status = parseLlamaBackendStatus(olderPayload);

  assert.equal(status.selectionApplied, true);
  assert.equal(llamaBackendSelectionNeedsApply(status, null), false);
});

test("an environment pin is never left with Apply as the only live control", () => {
  // The Select is disabled whenever the environment pins the backend. Dirtiness is
  // computed independently, so an automatic install whose detection has since drifted
  // (a GPU appeared under an env-pinned CPU install) makes the row dirty while the
  // Select is disabled. The server refuses that POST with environment_override, so the
  // button must be disabled by the same condition rather than offering the round trip.
  const status = parseLlamaBackendStatus({
    ...FULL_PAYLOAD,
    env_backend: "cpu",
    backend: "cpu",
    backend_request: "auto",
    selection_applied: false,
  });

  assert.equal(status.envBackend, "cpu");
  assert.equal(llamaBackendSelectionNeedsApply(status, status.backendRequest), true);
  const envLocked = status.envBackend !== null;
  const dirty = llamaBackendSelectionNeedsApply(status, status.backendRequest);
  // What the component computes for each control.
  assert.equal(!status.supported || envLocked, true, "Select is disabled");
  assert.equal(!dirty || !status.supported || envLocked, true, "Apply is disabled too");
});

test("every unsupported reason survives the parser verbatim", () => {
  // The section maps these to distinct explanations, including the no_install_dir
  // alias, and a reason that stops round-tripping silently degrades to the generic
  // "could not be checked" copy.
  for (const reason of [
    "not_installed",
    "local_link",
    "source_build",
    "no_install_dir",
    "unresolved",
  ]) {
    const status = parseLlamaBackendStatus({
      ...FULL_PAYLOAD,
      supported: false,
      reason,
      options: [],
    });
    assert.equal(status.reason, reason);
    assert.equal(status.supported, false);
    assert.deepEqual(visibleLlamaBackendOptions(status, null), []);
    // Nothing to apply on an install that cannot be switched.
    assert.equal(llamaBackendSelectionNeedsApply(status, status.backendRequest), false);
  }
});

test("macOS reports Metal as the running backend and offers only automatic", () => {
  // resolve_backends_payload enumerates ("auto",) on macOS, and metal is deliberately
  // absent from LLAMA_BACKENDS: it is a backend an install can RUN, never one a user
  // can request. The parser must keep it as the effective backend all the same.
  const status = parseLlamaBackendStatus({
    ...FULL_PAYLOAD,
    backend: "metal",
    backend_request: "auto",
    options: [
      {
        backend: "auto",
        available: true,
        resolved_backend: "metal",
        release_tag: "b9596-mix-abc",
        download_size_bytes: null,
      },
    ],
  });

  assert.equal(status.backend, "metal");
  assert.equal(status.options.length, 1);
  assert.equal(status.options[0].resolvedBackend, "metal");
  assert.equal(status.options[0].downloadSizeBytes, null);
  assert.equal(llamaBackendSelectionNeedsApply(status, "auto"), false);
});

test("terminal job states round-trip, including a job that failed elsewhere", () => {
  for (const [state, error] of [
    ["success", null],
    ["error", "no cuda prebuilt bundle attempts were available"],
    ["nonsense-from-a-newer-server", null],
  ] as const) {
    const status = parseLlamaBackendStatus({
      ...FULL_PAYLOAD,
      job: {
        state,
        operation: "switch",
        requested_backend: "vulkan",
        message: "done",
        error,
        finished_at: "2026-08-13T00:00:00Z",
      },
    });
    assert.equal(
      status.job.state,
      state === "nonsense-from-a-newer-server" ? "idle" : state,
    );
    assert.equal(status.job.operation, "switch");
    assert.equal(status.job.requestedBackend, "vulkan");
    assert.equal(status.job.error, error);
  }
});
