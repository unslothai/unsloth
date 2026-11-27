// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import {
  llamaBackendSelectionNeedsApply,
  llamaBackendUnavailableKey,
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
  // An unknown recorded choice must not become automatic, or the next apply overwrites a pin.
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

test("an unavailable backend this host's GPUs could run is greyed with its reason, not hidden", () => {
  // Mixed NVIDIA + AMD host on Linux where ROCm was never installed.
  const status = parseLlamaBackendStatus({
    ...FULL_PAYLOAD,
    options: [
      FULL_PAYLOAD.options[0],
      { backend: "cuda", available: true, resolved_backend: "cuda", applies_to_host: true },
      {
        backend: "rocm",
        available: false,
        unavailable_reason: "rocm_not_installed",
        unavailable_hint: "ROCm is not installed, so llama.cpp cannot use the AMD card through it.",
        applies_to_host: true,
      },
      { backend: "vulkan", available: false, unavailable_reason: "no_prebuilt", applies_to_host: true },
    ],
  });

  const shown = visibleLlamaBackendOptions(status, "auto");
  assert.deepEqual(
    shown.map((option) => [option.backend, llamaBackendUnavailableKey(option)]),
    [
      ["auto", null],
      ["cuda", null],
      ["rocm", "rocmNotInstalled"],
      ["vulkan", "noBuild"],
    ],
  );
  assert.match(shown[2].unavailableHint ?? "", /ROCm is not installed/);
});

test("a closed /dev/kfd or render node reads as no access", () => {
  const status = parseLlamaBackendStatus({
    ...FULL_PAYLOAD,
    options: [
      {
        backend: "rocm",
        available: false,
        unavailable_reason: "rocm_no_access",
        unavailable_hint: "sudo usermod -a -G render,video someone",
        applies_to_host: true,
      },
    ],
  });

  const [rocm] = visibleLlamaBackendOptions(status, "auto").filter(
    (option) => option.backend === "rocm",
  );
  assert.equal(llamaBackendUnavailableKey(rocm), "rocmNoAccess");
  assert.equal(rocm.unavailableHint, "sudo usermod -a -G render,video someone");
});

test("a backend only a newer release ships asks for an update, not a build", () => {
  const status = parseLlamaBackendStatus({
    ...FULL_PAYLOAD,
    options: [
      { backend: "vulkan", available: false, unavailable_reason: "other_release", applies_to_host: true },
    ],
  });

  const [vulkan] = visibleLlamaBackendOptions(status, "auto").filter(
    (option) => option.backend === "vulkan",
  );
  assert.equal(llamaBackendUnavailableKey(vulkan), "otherRelease");
});

test("the recorded selection with no options is not called a missing build", () => {
  const status = parseLlamaBackendStatus({ ...FULL_PAYLOAD, supported: false, reason: "source_build", options: [] });

  const [cuda] = visibleLlamaBackendOptions(status, "cuda");
  assert.equal(cuda.backend, "cuda");
  assert.equal(llamaBackendUnavailableKey(cuda), null);
});

test("a backend for a GPU this host does not have stays hidden", () => {
  const status = parseLlamaBackendStatus({
    ...FULL_PAYLOAD,
    options: [
      FULL_PAYLOAD.options[0],
      { backend: "rocm", available: false, unavailable_reason: "unavailable", applies_to_host: false },
    ],
  });

  assert.deepEqual(
    visibleLlamaBackendOptions(status, "auto").map((option) => option.backend),
    ["auto"],
  );
});

test("the picker renders greyed options disabled with the reason in the label", async () => {
  // Pinned by source: node --test has no DOM to render the Radix Select into.
  const { readFile } = await import("node:fs/promises");
  const source = await readFile(
    new URL("../src/features/settings/components/llama-backend-section.tsx", import.meta.url),
    "utf8",
  );
  assert.match(source, /disabled=\{\s*unavailableKey !== null/);
  assert.match(source, /settings\.resources\.llamaBackend\.unavailableOption/);
  assert.match(source, /llama-backend-hint-\$\{option\.backend\}/);
  const en = await readFile(new URL("../src/i18n/locales/en.ts", import.meta.url), "utf8");
  assert.doesNotMatch(en, /Only backends with a build for this machine are listed/);
  for (const key of ["rocmNotInstalled", "rocmNoAccess", "otherRelease", "checkFailed", "noBuild"]) {
    assert.match(en, new RegExp(`${key}: "`));
  }
});

test("the selected backend stays listed even when it stops being installable", () => {
  const status = parseLlamaBackendStatus(FULL_PAYLOAD);

  assert.deepEqual(
    visibleLlamaBackendOptions(status, "rocm").map((option) => option.backend),
    ["auto", "rocm"],
  );
});

test("a selection with no options left is still listed, not rendered blank", () => {
  // No options while resolution is pending or failed, or a job runs: the recorded choice
  // still needs an item, and automatic names what the install runs on.
  const status = parseLlamaBackendStatus({
    ...FULL_PAYLOAD,
    backend: "vulkan",
    options: [],
  });

  const [auto] = visibleLlamaBackendOptions(status, "auto");
  assert.equal(auto.backend, "auto");
  assert.equal(auto.available, false);
  assert.equal(auto.resolvedBackend, "vulkan");
  assert.deepEqual(
    visibleLlamaBackendOptions(status, "cpu").map((option) => [
      option.backend,
      option.resolvedBackend,
    ]),
    [["cpu", null]],
  );
  assert.equal(visibleLlamaBackendOptions(null, "auto").length, 0);
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
  // The server refuses an env-pinned POST with environment_override, so the button shares the gate.
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
  assert.equal(!status.supported || envLocked, true, "Select is disabled");
  assert.equal(!dirty || !status.supported || envLocked, true, "Apply is disabled too");
});

test("every unsupported reason survives the parser verbatim", () => {
  for (const reason of [
    "not_installed",
    "local_link",
    "source_build",
    "no_install_dir",
    "custom_path",
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
    assert.equal(llamaBackendSelectionNeedsApply(status, status.backendRequest), false);
  }
});

test("macOS reports Metal as the running backend and offers only automatic", () => {
  // metal is absent from LLAMA_BACKENDS: an install can run it but a user cannot request it.
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
