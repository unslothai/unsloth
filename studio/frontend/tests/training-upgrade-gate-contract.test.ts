// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The gap this closes: `confirmTransformersUpgradeIfNeeded` had exactly two callers,
// both in chat, so a Train-tab run on a model whose architecture no installed
// transformers ships was accepted and then died at model load with
//   "... is not supported yet in transformers==5.3.0"
// and no prompt. Reading the start paths (rather than driving them) keeps this a cheap
// guard against the gate being dropped from either one.

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

const START_PATHS = [
  "../src/features/training/lib/start-fresh-training-run.ts",
  "../src/features/training/lib/resume-training-run.ts",
] as const;

function read(relative: string): string {
  return readFileSync(new URL(relative, import.meta.url), "utf8");
}

test("both training start paths consult the transformers-upgrade gate", () => {
  for (const file of START_PATHS) {
    assert.ok(
      read(file).includes("confirmTrainingTransformersUpgrade"),
      `${file} must consult the transformers-upgrade gate: without it a model whose ` +
        "architecture no installed transformers ships is accepted and then dies at " +
        "model load, with no prompt and no way for the user to act on it",
    );
  }
});

test("the upgrade dialog is raised before the custom-code dialog", () => {
  // Chat's order, and for the same reason: installing a newer transformers changes
  // what the load would even run, so consenting to the install has to come first.
  for (const file of START_PATHS) {
    const source = read(file);
    const upgradeAt = source.indexOf("confirmTrainingTransformersUpgrade(");
    const remoteCodeAt = source.indexOf("confirmRemoteCodeIfNeeded(");
    // Both must be present: a missing call indexes to -1, which would otherwise
    // satisfy the ordering assertion without either gate existing.
    assert.ok(
      upgradeAt >= 0 && remoteCodeAt >= 0,
      `${file} must run both gates`,
    );
    assert.ok(
      upgradeAt < remoteCodeAt,
      `${file} must raise the upgrade dialog before the custom-code dialog`,
    );
  }
});

test("both gates on a start path inspect the same copy of the model", () => {
  // The upgrade check used to be handed the Hub identifier while the custom-code gate
  // resolved the pinned snapshot, so a cached model could be judged on the repo's
  // current config.json and the snapshot's -- two different architectures. One resolver
  // per start path is what keeps them from drifting apart again.
  for (const [file, resolver] of [
    [
      "../src/features/training/lib/start-fresh-training-run.ts",
      "freshModelCachePin(",
    ],
    [
      "../src/features/training/lib/resume-training-run.ts",
      "resumeModelCachePin(",
    ],
  ] as const) {
    const source = read(file);
    assert.equal(
      source.split(resolver).length - 1,
      3,
      `${file} must resolve the cache pin once and pass it to both gates`,
    );
  }
});

test("the resume gate names the run it precedes", () => {
  // Without the run id the check cannot tell that installing would permanently strand
  // a checkpoint attested against a 4-bit model load the latest sidecar refuses.
  const source = read("../src/features/training/lib/resume-training-run.ts");
  assert.ok(source.includes("resumeRunId"));
});

test("the gate reaches the install through the shared consent dialog", () => {
  // Not a second implementation of the flow chat already owns.
  const gate = read(
    "../src/features/training/lib/training-transformers-upgrade.ts",
  );
  assert.ok(gate.includes("confirmTransformersUpgradeIfNeeded"));
  assert.ok(gate.includes("checkTransformersUpgrade"));
});

test("the Configure preview re-asks the check after an install", () => {
  // The hook itself is React, so this is the cheap guard on the wiring the notice cache
  // depends on: the store counts completed installs, and the hook keys its cached
  // answers on that count. Break either end and Configure keeps showing the pre-install
  // answer -- an install that already ran, and 4-bit for a run the new sidecar loads in
  // 16-bit.
  const store = read(
    "../src/features/transformers-upgrade/stores/transformers-upgrade-dialog-store.ts",
  );
  assert.ok(
    /sidecarGeneration:\s*get\(\)\.sidecarGeneration \+ 1/.test(store),
    "a successful install must advance sidecarGeneration",
  );

  const hook = read(
    "../src/features/training/hooks/use-training-transformers-upgrade-notice.ts",
  );
  assert.ok(hook.includes("s.sidecarGeneration"));
  assert.ok(
    /upgradeNoticeCacheKey\(\s*sidecarGeneration/.test(hook),
    "the preview cache key must carry the generation, or an install cannot retire it",
  );
});

test("the consent dialog offers the custom-code way out before an install fails", () => {
  // Training raises this dialog before a run starts, so what it offers first decides
  // what a run can be. For a model that ships its own modeling code, the install is not
  // the only way forward and is the more expensive one: it activates the latest sidecar,
  // which trains 16-bit. Gating the fallback on the error phase left a QLoRA run with
  // Install or Cancel, neither of which starts the 4-bit run the user asked for.
  const dialog = read(
    "../src/features/transformers-upgrade/components/transformers-upgrade-dialog.tsx",
  );
  assert.ok(
    dialog.includes("upgradeDialogActions"),
    "the dialog must take its actions from the shared decision, not re-derive them",
  );
  assert.doesNotMatch(
    dialog,
    /phase === "error" && trustRemoteCodeFallback/,
    "the custom-code fallback must not wait for an install to fail first",
  );
});

test("both start paths carry the upgrade gate's custom-code verdict forward", () => {
  // confirmRemoteCodeIfNeeded falls back to the caller's requiresTrustRemoteCode when the
  // scan request fails, and the stored flag is false on a fresh run. The upgrade check
  // has already answered the question, so it has to be the one that travels.
  for (const file of START_PATHS) {
    const source = read(file);
    assert.ok(
      source.includes(
        "verdict.requiresTrustRemoteCode = outcome.requiresTrustRemoteCode",
      ),
      `${file} must record the upgrade gate's custom-code verdict`,
    );
    assert.ok(
      source.includes("upgradeRequiresTrustRemoteCode"),
      `${file} must pass that verdict into the custom-code gate`,
    );
  }
});
