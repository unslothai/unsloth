// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// A forced repair runs the transactional installer; a generic retry would find the restored
// install ready and just restart the backend.

import assert from "node:assert/strict";
import test from "node:test";

import { readSrcAsync } from "./helpers/kit.ts";

const src = await readSrcAsync("hooks/use-tauri-backend.ts");

function lift(pattern: RegExp, what: string): string {
  const found = pattern.exec(src);
  assert.ok(found, `could not find ${what} in use-tauri-backend.ts`);
  return found[0];
}

const body = lift(
  /const retry = useCallback\(\(\) => \{[\s\S]*?\n  \}, \[\]\);/,
  "the retry callback",
);

type Run = {
  repairs: boolean[];
  preflights: number;
  forcedAfter: boolean;
  heldRepairAllowed: boolean;
};

function runRetry(status: string, forced: boolean): Run {
  const repairs: boolean[] = [];
  const forcedRepairRef = { current: forced };
  const allowHeldRuntimeRepairRef = { current: false };
  let preflights = 0;
  const noop = () => {};
  const scope = {
    statusRef: { current: status },
    forcedRepairRef,
    allowHeldRuntimeRepairRef,
    startRepair: (options?: { forceInstaller?: boolean }) => {
      forcedRepairRef.current = options?.forceInstaller === true;
      repairs.push(options?.forceInstaller === true);
      return Promise.resolve();
    },
    checkInstallAndStart: () => {
      preflights += 1;
    },
    elevationResumeRef: { current: null as string | null },
    startingRef: { current: true },
    portRef: { current: 1 as number | null },
    startTimedOutRef: { current: true },
    seenStepsRef: { current: new Set<string>() },
    useCallback: (fn: unknown) => fn,
    clearAuthFailure: noop,
    clearServerStopIntent: noop,
    setError: noop,
    setLogs: noop,
    setCurrentStepIndex: noop,
    setProgressDetail: noop,
    setElevationPackages: noop,
    setIsExternalServer: noop,
    stopExternalServerPoll: noop,
    stopManagedEnvironmentWait: noop,
  };
  const keys = Object.keys(scope);
  new Function(
    ...keys,
    `${body.replace(/^const retry = /, "return ")}`.replace(/;\s*$/, ";"),
  )(...keys.map((key) => (scope as Record<string, unknown>)[key]))();
  return {
    repairs,
    preflights,
    forcedAfter: forcedRepairRef.current,
    heldRepairAllowed: allowHeldRuntimeRepairRef.current,
  };
}

test("retry after a forced repair re-runs the forced repair", () => {
  const run = runRetry("repair-error", true);
  assert.deepEqual(run.repairs, [true], "the retry must force the installer again");
  assert.equal(run.preflights, 0, "the preflight would restart the same broken backend");
  assert.equal(run.forcedAfter, true, "startRepair re-arms it for the elevation resume");
});

test("retry after an automatic repair still runs the preflight", () => {
  const run = runRetry("repair-error", false);
  assert.deepEqual(run.repairs, []);
  assert.equal(run.preflights, 1);
});

test("retry from any other failure is untouched", () => {
  for (const status of ["error", "install-error", "not-installed", "stopped"]) {
    const run = runRetry(status, true);
    assert.deepEqual(run.repairs, [], `${status} must not start a repair`);
    assert.equal(run.preflights, 1, `${status} must still run the preflight`);
    assert.equal(
      run.forcedAfter,
      false,
      `${status} leaves the generic path, so the forced flag must not survive it`,
    );
  }
});

test("retry lets its preflight repair a recently repaired runtime; a forced resume does not need to", () => {
  for (const status of ["error", "repair-error"]) {
    const run = runRetry(status, false);
    assert.equal(run.preflights, 1, `${status}: the retry still runs the preflight`);
    assert.deepEqual(run.repairs, [], `${status}: no repair without the preflight`);
    assert.equal(run.heldRepairAllowed, true, `${status}: the hold is lifted for it`);
  }
  assert.equal(runRetry("repair-error", true).heldRepairAllowed, false);
});
