// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// start_install emits install-failed { message, diskFull } and then rejects with the same
// message. The disk-full dialog has to survive whichever of the two the webview sees last.
// The hook cannot be rendered, so its functions are lifted by regex, as in
// managed-environment-wait.test.ts.

import assert from "node:assert/strict";
import test from "node:test";

import { readSrcAsync } from "./helpers/kit.ts";

const src = await readSrcAsync("hooks/use-tauri-backend.ts");

function lift(pattern: RegExp, what: string): string {
  const found = pattern.exec(src);
  assert.ok(found, `could not find ${what} in use-tauri-backend.ts`);
  return found[0];
}

const statusBody = lift(
  /  function setBackendStatus\([^)]*\) \{[\s\S]*?\n  \}\n/,
  "setBackendStatus",
);
const errorBody = lift(
  /  function setBackendError\([\s\S]*?\n  \}\n/,
  "setBackendError",
);
const installBody = lift(
  /  async function startInstall\(\) \{[\s\S]*?\n  \}\n/,
  "startInstall",
);
const listenerBody = lift(
  /register<\{ message: string; diskFull: boolean \}>\("install-failed", \(e\) => \{[\s\S]*?\}\);/,
  "the install-failed listener",
)
  .replace(/^[\s\S]*?=> \{/, "")
  .replace(/\}\);$/, "");

const MESSAGE =
  "Installation failed: install unsloth failed (exit code 1): No space left on device (os error 28)";

async function failInstall(order: "event-first" | "reject-first") {
  const state = { status: "", error: null as string | null, diskFull: false };
  let reject: (reason: string) => void = () => {};
  const noop = () => {};
  const scope: Record<string, unknown> = {
    authFailureRef: { current: null },
    statusRef: { current: "not-installed" },
    elevationResumeRef: { current: null },
    seenStepsRef: { current: new Set() },
    stopManagedEnvironmentWait: noop,
    syncTrayStatus: noop,
    setCurrentStepIndex: noop,
    setProgressDetail: noop,
    setLogs: noop,
    setStatus: (status: string) => {
      state.status = status;
    },
    setError: (error: string | null) => {
      state.error = error;
    },
    setInstallDiskFull: (diskFull: boolean) => {
      state.diskFull = diskFull;
    },
    clearBackendError: () => {
      state.error = null;
    },
    startServer: () => Promise.resolve(),
    // The bare specifier does not resolve inside `new Function`.
    importTauriCore: () =>
      Promise.resolve({
        invoke: () =>
          new Promise((_resolve, rejectInvoke) => {
            reject = rejectInvoke;
          }),
      }),
  };
  const source = `
${statusBody.replace(": BackendStatus", "")}
${errorBody.replace(/: (string|BackendStatus)/g, "")}
${installBody.replace('await import("@tauri-apps/api/core")', "await importTauriCore()")}
    return { startInstall, onInstallFailed: (e) => {${listenerBody}} };
  `;
  const keys = Object.keys(scope);
  const hook = new Function(...keys, source)(
    ...keys.map((key) => scope[key]),
  ) as {
    startInstall: () => Promise<void>;
    onInstallFailed: (e: {
      payload: { message: string; diskFull: boolean };
    }) => void;
  };

  const install = hook.startInstall();
  await new Promise((resolve) => setImmediate(resolve));
  const event = { payload: { message: MESSAGE, diskFull: true } };
  if (order === "event-first") {
    hook.onInstallFailed(event);
    reject(MESSAGE);
    await install;
  } else {
    reject(MESSAGE);
    await install;
    hook.onInstallFailed(event);
  }
  return state;
}

for (const order of ["event-first", "reject-first"] as const) {
  test(`a disk-full install keeps its dialog (${order})`, async () => {
    const state = await failInstall(order);
    assert.equal(state.status, "install-error");
    assert.equal(state.error, MESSAGE);
    assert.equal(state.diskFull, true);
  });
}
