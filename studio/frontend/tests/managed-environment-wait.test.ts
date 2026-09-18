// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// A reload during an install or update must wait it out: repairing instead is what
// produced "Update failed: Repair is already running.". The hook cannot be rendered,
// so its functions are lifted by regex, as in forced-repair-retry.test.ts.

import assert from "node:assert/strict";
import test from "node:test";

import {
  MANAGED_ENVIRONMENT_BUSY,
  MANAGED_ENVIRONMENT_UPDATING,
} from "../src/hooks/backend-preflight-message.ts";
import { UPDATE_STARTUP_MESSAGE } from "../src/components/tauri/startup-messages.ts";
import { readSrcAsync } from "./helpers/kit.ts";

const src = await readSrcAsync("hooks/use-tauri-backend.ts");

function lift(pattern: RegExp, what: string): string {
  const found = pattern.exec(src);
  assert.ok(found, `could not find ${what} in use-tauri-backend.ts`);
  return found[0];
}

const checkBody = lift(
  /  async function checkInstallAndStart\(\) \{[\s\S]*?\n  \}\n/,
  "checkInstallAndStart",
);
const waitBody = lift(
  /  function waitForManagedEnvironment\([^)]*\) \{[\s\S]*?\n  \}\n/,
  "waitForManagedEnvironment",
);
const statusBody = lift(
  /  function setBackendStatus\([^)]*\) \{[\s\S]*?\n  \}\n/,
  "setBackendStatus",
);
const errorBody = lift(
  /  function setBackendError\([\s\S]*?\n  \}\n/,
  "setBackendError",
);
const stopBody = lift(
  /  function stopManagedEnvironmentWait\(\) \{[\s\S]*?\n  \}\n/,
  "stopManagedEnvironmentWait",
);
const POLL_LIMIT = Number(
  lift(/const MANAGED_ENVIRONMENT_WAIT_POLLS = \d+;/, "the poll limit").replace(/\D/g, ""),
);

interface Preflight {
  disposition: string;
  reason: string | null;
  can_auto_repair: boolean;
  port: number | null;
}

const BUSY: Preflight = {
  disposition: "managed_stale",
  // The busy branch must win over it: the busy environment refuses that repair.
  can_auto_repair: true,
  reason: MANAGED_ENVIRONMENT_BUSY,
  port: null,
};

function harness(preflight: Preflight, authFailure: string | null = null) {
  const errors: string[] = [];
  const statuses: string[] = [];
  const messages: string[] = [];
  let repairs = 0;
  let starts = 0;
  let armed: (() => void) | null = null;
  let armedCount = 0;
  const environmentWaitPollsRef = { current: 0 };
  const noop = () => {};

  const scope: Record<string, unknown> = {
    MANAGED_ENVIRONMENT_BUSY,
    MANAGED_ENVIRONMENT_UPDATING,
    UPDATE_STARTUP_MESSAGE,
    MANAGED_ENVIRONMENT_POLL_MS: 5_000,
    MANAGED_ENVIRONMENT_WAIT_POLLS: POLL_LIMIT,
    SERVER_STARTUP_MESSAGE: "Nearly done...",
    authFailureRef: { current: authFailure },
    // Fresh per harness: the hook's single-flight guard reads it before the probe and releases it
    // after, so one left true here would make every later case return without doing anything.
    preflightInFlightRef: { current: false },
    environmentWaitRef: { current: null as unknown },
    environmentWaitPollsRef,
    statusRef: { current: "checking" },
    portRef: { current: null as number | null },
    // The bare specifier does not resolve inside `new Function`.
    importTauriCore: () =>
      Promise.resolve({
        invoke: (command: string) => {
          if (command === "desktop_preflight") return Promise.resolve(preflight);
          if (command === "start_managed_repair") {
            repairs += 1;
            return Promise.reject("An update or repair is already running.");
          }
          return Promise.resolve(null);
        },
      }),
    setTimeout: (fn: () => void) => {
      armed = fn;
      armedCount += 1;
      return armedCount;
    },
    clearTimeout: () => {
      armed = null;
    },
    hasServerStopIntent: () => false,
    setStartupMessage: (message: string) => messages.push(message),
    setStatus: (status: string) => statuses.push(status),
    syncTrayStatus: noop,
    setError: (error: string) => errors.push(error),
    setApiBase: noop,
    setIsExternalServer: noop,
    stopExternalServerPoll: noop,
    startExternalServerPoll: noop,
    setRunningStatus: noop,
    startManagedServer: () => {
      starts += 1;
      return Promise.resolve();
    },
    startRepair: () => {
      repairs += 1;
      return Promise.resolve();
    },
    preflightStaleMessage: (_d: string, reason: string | null) => `stale:${reason}`,
    externalConflictMessage: () => "conflict",
  };

  const source = `
${stopBody}
${statusBody.replace(": BackendStatus", "")}
${errorBody.replace(/: (string|BackendStatus)/g, "")}
${waitBody.replace("bounded: boolean", "bounded")}
${checkBody
  .replace('await import("@tauri-apps/api/core")', "await importTauriCore()")
  .replace("invoke<DesktopPreflightResult>(", "invoke(")}
    return checkInstallAndStart;
  `;
  const keys = Object.keys(scope);
  const check = new Function(...keys, source)(
    ...keys.map((key) => scope[key]),
  ) as () => Promise<void>;

  return {
    check,
    errors,
    statuses,
    messages,
    get starts() {
      return starts;
    },
    get repairs() {
      return repairs;
    },
    get armedCount() {
      return armedCount;
    },
    get waiting() {
      return armed !== null;
    },
    polls: () => environmentWaitPollsRef.current,
    async fireWait() {
      const next = armed;
      assert.ok(next, "no wait was armed");
      armed = null;
      next();
      // The timer dispatches checkInstallAndStart as a floating promise.
      for (let i = 0; i < 8; i += 1) await Promise.resolve();
    },
  };
}

test("a busy environment waits instead of repairing", async () => {
  const run = harness(BUSY);
  await run.check();

  assert.equal(run.repairs, 0, "a busy environment refuses the repair it would start");
  assert.deepEqual(run.errors, [], "the user must not see a failure for a normal update");
  assert.deepEqual(run.statuses, ["starting"]);
  assert.deepEqual(run.messages, [UPDATE_STARTUP_MESSAGE]);
  assert.ok(run.waiting, "the wait must re-poll rather than give up on one answer");
});

test("the wait re-polls and starts by itself once the environment frees up", async () => {
  const preflight: Preflight = { ...BUSY };
  const run = harness(preflight);
  await run.check();
  assert.equal(run.polls(), 1);

  await run.fireWait();
  assert.equal(run.polls(), 2, "still busy, so the wait re-arms");
  assert.equal(run.repairs, 0);

  preflight.disposition = "managed_ready";
  preflight.reason = null;
  await run.fireWait();
  assert.equal(run.waiting, false, "a ready install ends the wait");
  assert.equal(run.polls(), 0, "and retires the count with it");
  assert.equal(run.starts, 1);
  assert.equal(run.repairs, 0);
});

test("a stale install that is not busy still repairs", async () => {
  const run = harness({ ...BUSY, reason: "cli_unusable" });
  await run.check();

  assert.equal(run.repairs, 1, "only the busy reason may skip the repair");
  assert.equal(run.waiting, false);
});

test("the wait is bounded, so a gate nobody releases still reaches Retry", async () => {
  const run = harness(BUSY);
  await run.check();
  for (let i = 1; i < POLL_LIMIT; i += 1) await run.fireWait();
  assert.equal(run.polls(), POLL_LIMIT, "the count tracks every consecutive busy answer");
  assert.equal(run.repairs, 0);

  await run.fireWait();
  assert.equal(run.waiting, false, "the wait gives up rather than spinning for ever");
  assert.equal(run.statuses.at(-1), "error", "which is the screen that carries Retry");
  assert.match(run.errors.at(-1) ?? "", /install or update/);
});

test("our own install or update is waited out without a bound", async () => {
  const run = harness({ ...BUSY, reason: MANAGED_ENVIRONMENT_UPDATING });
  await run.check();
  for (let i = 0; i < POLL_LIMIT + 1; i += 1) await run.fireWait();

  assert.ok(run.waiting, "the app's own mutation always ends, so it is never cut short");
  assert.equal(run.polls(), 0);
  assert.deepEqual(run.errors, []);
  assert.equal(run.repairs, 0);
});

test("a persisted auth failure is not buried under the wait", async () => {
  const run = harness(BUSY, "Desktop auth failed");
  await run.check();

  assert.equal(run.waiting, false);
  assert.equal(run.armedCount, 0);
  assert.equal(run.repairs, 0);
});
