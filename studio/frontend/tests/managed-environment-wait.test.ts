// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// A reload during an install or update finds an environment that is being rewritten.
// The preflight reports it busy, and the app has to wait it out: repairing instead is
// what produced "Update failed: An update or repair is already running.".
//
// The hook cannot be rendered here, so checkInstallAndStart and the wait it arms are
// lifted by regex and run under an injected scope, the way forced-repair-retry.test.ts
// does beside it.

import assert from "node:assert/strict";
import test from "node:test";

import { MANAGED_ENVIRONMENT_BUSY } from "../src/hooks/backend-preflight-message.ts";
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
  /  function waitForManagedEnvironment\(\) \{[\s\S]*?\n  \}\n/,
  "waitForManagedEnvironment",
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
  // can_auto_repair true on purpose: the busy branch must win over it, because the
  // repair it would start is the thing the busy environment goes on to refuse.
  can_auto_repair: true,
  reason: MANAGED_ENVIRONMENT_BUSY,
  port: null,
};

function harness(preflight: Preflight, authFailure: string | null = null) {
  const errors: string[] = [];
  const statuses: string[] = [];
  const messages: string[] = [];
  let repairs = 0;
  let armed: (() => void) | null = null;
  let armedCount = 0;
  const environmentWaitPollsRef = { current: 0 };
  const noop = () => {};

  const scope: Record<string, unknown> = {
    MANAGED_ENVIRONMENT_BUSY,
    UPDATE_STARTUP_MESSAGE,
    MANAGED_ENVIRONMENT_POLL_MS: 5_000,
    MANAGED_ENVIRONMENT_WAIT_POLLS: POLL_LIMIT,
    SERVER_STARTUP_MESSAGE: "Nearly done...",
    authFailureRef: { current: authFailure },
    environmentWaitRef: { current: null as unknown },
    environmentWaitPollsRef,
    statusRef: { current: "checking" },
    portRef: { current: null as number | null },
    // The lifted body's `await import("@tauri-apps/api/core")` is rewritten to this:
    // the bare specifier does not resolve inside `new Function`.
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
    startManagedServer: () => Promise.resolve(),
    startRepair: () => {
      repairs += 1;
      return Promise.resolve();
    },
    preflightStaleMessage: (_d: string, reason: string | null) => `stale:${reason}`,
    externalConflictMessage: () => "conflict",
  };

  const source = `
${stopBody}
    function setBackendStatus(nextStatus) {
      if (authFailureRef.current) return;
      stopManagedEnvironmentWait();
      statusRef.current = nextStatus;
      setStatus(nextStatus);
    }
    function setBackendError(nextError, nextStatus = "error") {
      if (authFailureRef.current) return;
      stopManagedEnvironmentWait();
      statusRef.current = nextStatus;
      setStatus(nextStatus);
      setError(nextError);
    }
${waitBody}
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
  assert.match(run.errors.at(-1) ?? "", /still finishing an install or update/);
});

test("a persisted auth failure is not buried under the wait", async () => {
  // setBackendStatus is a no-op behind an auth failure, so an unguarded wait would
  // poll on for ever behind the error screen the user is already reading.
  const run = harness(BUSY, "Desktop auth failed");
  await run.check();

  assert.equal(run.waiting, false);
  assert.equal(run.armedCount, 0);
  assert.equal(run.repairs, 0);
});
