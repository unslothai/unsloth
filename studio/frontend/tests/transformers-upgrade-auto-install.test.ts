// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// A PyPI release installs behind a notice with no dialog; transformers main, a model with its own
// code, and any scan verdict short of clean still open it.

import assert from "node:assert/strict";
import test from "node:test";
import * as zustand from "zustand";

import { calls as toastCalls, toast } from "./helpers/toast-stub.mjs";
import { loadWithStubs } from "./helpers/module-stubs.ts";
import type { TransformersUpgradeInfo } from "../src/features/transformers-upgrade/types.ts";

type InstallResponse = {
  success: boolean;
  version: string;
  message: string;
  model_unloaded?: boolean;
  requires_consent?: boolean;
  scan_status?: string | null;
};

const installCalls: { version: string; force: boolean; auto: boolean }[] = [];
let nextResponse: InstallResponse | Error = {
  success: true,
  version: "5.30.0",
  message: "ok",
  // biome-ignore lint/style/useNamingConvention: API schema
  scan_status: "clean",
};

const { useTransformersUpgradeDialogStore } = loadWithStubs<
  typeof import("../src/features/transformers-upgrade/stores/transformers-upgrade-dialog-store.ts")
>(
  new URL(
    "../src/features/transformers-upgrade/stores/transformers-upgrade-dialog-store.ts",
    import.meta.url,
  ),
  {
    zustand,
    "@/lib/toast": { toast },
    "../api/transformers-upgrade-api": {
      installLatestTransformers: async (
        version: string,
        force = false,
        auto = false,
      ) => {
        installCalls.push({ version, force, auto });
        if (nextResponse instanceof Error) throw nextResponse;
        return nextResponse;
      },
    },
  },
  { relativePassthrough: true },
);

const PYPI: TransformersUpgradeInfo = {
  // biome-ignore lint/style/useNamingConvention: API schema
  model_type: "muse_glimmer",
  // biome-ignore lint/style/useNamingConvention: API schema
  pypi_version: "5.30.0",
  // biome-ignore lint/style/useNamingConvention: API schema
  supported_in_pypi: true,
  // biome-ignore lint/style/useNamingConvention: API schema
  supported_in_main: true,
  // biome-ignore lint/style/useNamingConvention: API schema
  main_version: "5.31.0.dev0",
};
const MAIN_ONLY: TransformersUpgradeInfo = {
  ...PYPI,
  // biome-ignore lint/style/useNamingConvention: API schema
  supported_in_pypi: false,
};

const store = () => useTransformersUpgradeDialogStore.getState();
const tick = () => new Promise((r) => setTimeout(r, 0));

function reset(response: InstallResponse | Error) {
  installCalls.length = 0;
  toastCalls.length = 0;
  nextResponse = response;
}

test("a clean PyPI release installs with no dialog and a progress notice", async () => {
  reset({
    success: true,
    version: "5.30.0",
    message: "ok",
    // biome-ignore lint/style/useNamingConvention: API schema
    scan_status: "clean",
  });
  const pending = store().requestConsent("org/muse", PYPI);
  assert.equal(store().open, false, "the dialog must not open for a clean PyPI release");
  assert.equal(await pending, true);
  assert.deepEqual(installCalls, [{ version: "5.30.0", force: false, auto: true }]);
  assert.equal(store().open, false);
  assert.equal(store().installRan, true);
  const kinds = toastCalls.map((c) => c.kind);
  assert.deepEqual(kinds, ["loading", "success"]);
});

test("transformers main opens the dialog and installs nothing until confirmed", async () => {
  reset({ success: true, version: "5.31.0.dev0", message: "ok" });
  const pending = store().requestConsent("org/muse", MAIN_ONLY);
  await tick();
  assert.equal(store().open, true);
  assert.equal(store().phase, "consent");
  assert.deepEqual(installCalls, []);
  await store().install();
  assert.equal(await pending, true);
  assert.deepEqual(installCalls, [{ version: "5.31.0.dev0", force: false, auto: false }]);
});

test("a scan that is not clean falls back to the dialog with the reason", async () => {
  reset({
    success: false,
    version: "5.30.0",
    message: "Automatic install skipped: the OSV advisory database could not be reached.",
    // biome-ignore lint/style/useNamingConvention: API schema
    requires_consent: true,
    // biome-ignore lint/style/useNamingConvention: API schema
    scan_status: "review",
  });
  const pending = store().requestConsent("org/muse", PYPI);
  await tick();
  assert.equal(store().open, true);
  assert.equal(store().phase, "consent");
  assert.match(store().autoInstallNotice ?? "", /OSV/);
  assert.ok(toastCalls.some((c) => c.kind === "dismiss"));
  // Confirming installs as before, without the auto flag.
  reset({ success: true, version: "5.30.0", message: "ok" });
  await store().install();
  assert.equal(await pending, true);
  assert.deepEqual(installCalls, [{ version: "5.30.0", force: false, auto: false }]);
});

test("a flagged release opens the dialog with the error and no install action", async () => {
  reset({
    success: false,
    version: "5.30.0",
    message: "transformers 5.30.0 was not installed: the package scan flagged it.",
    // biome-ignore lint/style/useNamingConvention: API schema
    scan_status: "flagged",
  });
  const pending = store().requestConsent("org/muse", PYPI);
  await tick();
  assert.equal(store().open, true);
  assert.equal(store().phase, "error");
  assert.equal(store().installBlocked, true);
  await store().install();
  assert.equal(installCalls.length, 1, "a flagged release is never retried");
  store().resolve(false);
  assert.equal(await pending, false);
});

test("a failed automatic install opens the dialog in its error state", async () => {
  reset(new Error("Another inference request is in progress."));
  const pending = store().requestConsent("org/muse", PYPI, {
    forceCancelActive: true,
  });
  await tick();
  assert.equal(store().open, true);
  assert.equal(store().phase, "error");
  assert.deepEqual(installCalls, [{ version: "5.30.0", force: true, auto: true }]);
  store().resolve(false);
  assert.equal(await pending, false);
});

test("a model with its own code keeps the dialog: it is the only 4-bit path", async () => {
  reset({ success: true, version: "5.30.0", message: "ok" });
  const pending = store().requestConsent("org/muse", PYPI, {
    trustRemoteCodeFallback: true,
  });
  await tick();
  assert.equal(store().open, true);
  assert.deepEqual(installCalls, []);
  store().resolve(true);
  assert.equal(await pending, true);
});
