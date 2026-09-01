// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Retry, on a repair the user forced from Settings, has to re-run that repair.
//
// startRepair({ forceInstaller: true }) skips `studio update` and runs the bundled
// installer, which is the only thing that replaces a CPU-only wheel. The installer is
// transactional: a failed attempt over an existing install restores the desktop-ready
// environment it found. So the generic retry path -- clear state, run the preflight --
// finds a ready install and restarts the very backend the user pressed Repair about.
// The elevation resume already carried the flag; the error path did not.
//
// The hook cannot be rendered here, so the callback is lifted by regex and evaluated,
// the way gpu-torch-mismatch.test.ts and system-status-verdict.test.ts do beside it.

import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const src = await readFile(
  new URL("../src/hooks/use-tauri-backend.ts", import.meta.url),
  "utf8",
);

function lift(pattern: RegExp, what: string): string {
  const found = pattern.exec(src);
  assert.ok(found, `could not find ${what} in use-tauri-backend.ts`);
  return found[0];
}

const body = lift(
  /const retry = useCallback\(\(\) => \{[\s\S]*?\n  \}, \[\]\);/,
  "the retry callback",
);

type Run = { repairs: boolean[]; preflights: number; forcedAfter: boolean };

function runRetry(status: string, forced: boolean): Run {
  const repairs: boolean[] = [];
  const forcedRepairRef = { current: forced };
  let preflights = 0;
  const noop = () => {};
  const scope = {
    statusRef: { current: status },
    forcedRepairRef,
    startRepair: (options?: { forceInstaller?: boolean }) => {
      // The real one records the flag first; the fake mirrors that so the test can see
      // whether it survived the state reset that now runs ahead of the call.
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
  };
  const keys = Object.keys(scope);
  new Function(
    ...keys,
    `${body.replace(/^const retry = /, "return ")}`.replace(/;\s*$/, ";"),
  )(...keys.map((key) => (scope as Record<string, unknown>)[key]))();
  return { repairs, preflights, forcedAfter: forcedRepairRef.current };
}

test("retry after a forced repair re-runs the forced repair", () => {
  const run = runRetry("repair-error", true);
  assert.deepEqual(run.repairs, [true], "the retry must force the installer again");
  assert.equal(run.preflights, 0, "the preflight would restart the same broken backend");
  // The ref is read into a local and cleared with the rest of the state, so the flag
  // reaches startRepair (which sets it again) without surviving as a latch.
  assert.equal(run.forcedAfter, true, "startRepair re-arms it for the elevation resume");
});

test("retry after an automatic repair still runs the preflight", () => {
  // The automatic callers leave forceInstaller off, and an out-of-date venv really is
  // the common case there: nothing about that path changed.
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
