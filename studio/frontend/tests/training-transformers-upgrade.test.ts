// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The Train start path must consult the transformers-upgrade check, pause on the consent
// dialog, and abandon the start when declined.

import assert from "node:assert/strict";
import { register } from "node:module";
import test from "node:test";

register("./helpers/transformers-upgrade-resolver.mjs", import.meta.url);

const stub = await import("./helpers/transformers-upgrade-stub.mjs");
const { confirmTrainingTransformersUpgrade } = await import(
  "../src/features/training/lib/training-transformers-upgrade.ts"
);

const MODEL = "unsloth/Muse-Glimmer-30B-unsloth-bnb-4bit";
const UPGRADE = {
  // biome-ignore lint/style/useNamingConvention: API schema
  model_type: "muse_glimmer",
  // biome-ignore lint/style/useNamingConvention: API schema
  pypi_version: "5.15.0",
  // biome-ignore lint/style/useNamingConvention: API schema
  supported_in_pypi: true,
  // biome-ignore lint/style/useNamingConvention: API schema
  supported_in_main: true,
};

test("a model no installed transformers ships pauses the start on the dialog", async () => {
  stub.resetStub();
  stub.state.checkResult = {
    upgrade: UPGRADE,
    requiresTrustRemoteCode: false,
    latestTierActive: false,
    forces16Bit: true,
  };
  stub.state.consentResult = true;
  stub.state.installRan = true;

  const outcome = await confirmTrainingTransformersUpgrade({
    modelName: MODEL,
    hfToken: "hf_token",
  });

  assert.deepEqual(outcome, {
    proceed: true,
    error: null,
    forces16Bit: true,
    requiresTrustRemoteCode: false,
  });
  assert.equal(stub.calls[0]?.name, "checkTransformersUpgrade");
  assert.deepEqual(stub.calls[0]?.args.slice(0, 2), [MODEL, "hf_token"]);
  assert.equal(stub.calls[1]?.name, "confirmTransformersUpgradeIfNeeded");
  assert.equal(stub.calls[1]?.args[0].modelName, MODEL);
  assert.equal(stub.calls[1]?.args[0].upgrade, UPGRADE);
});

test("declining the install abandons the start instead of spawning a doomed run", async () => {
  stub.resetStub();
  stub.state.checkResult = {
    upgrade: UPGRADE,
    requiresTrustRemoteCode: false,
    latestTierActive: false,
    forces16Bit: true,
  };
  stub.state.consentResult = false;

  const outcome = await confirmTrainingTransformersUpgrade({
    modelName: MODEL,
    hfToken: null,
  });

  assert.equal(outcome.proceed, false);
  assert.equal(outcome.forces16Bit, false);
  assert.match(String(outcome.error), /is not supported yet/);
  assert.ok(String(outcome.error).includes(MODEL));
});

test("a model shipping its own code keeps the custom-code way out", async () => {
  stub.resetStub();
  stub.state.checkResult = {
    upgrade: { ...UPGRADE, supported_in_pypi: false },
    requiresTrustRemoteCode: true,
    latestTierActive: false,
    forces16Bit: false,
  };

  await confirmTrainingTransformersUpgrade({ modelName: MODEL });

  assert.equal(
    stub.calls[1]?.args[0].trustRemoteCodeFallback,
    true,
    "the dialog must offer the trust_remote_code fallback, like chat does",
  );
  // Training raises no "stop N chats" prompt, so it must never cancel another tab's stream.
  assert.equal(stub.calls[1]?.args[0].forceCancelActive, undefined);
});

test("a resolved custom-code fallback still loads 4-bit", async () => {
  stub.resetStub();
  stub.state.checkResult = {
    upgrade: { ...UPGRADE, supported_in_pypi: false },
    requiresTrustRemoteCode: true,
    latestTierActive: false,
    forces16Bit: false,
  };
  // The fallback resolves true WITHOUT installing, so the sidecar never activates.
  stub.state.consentResult = true;
  stub.state.installRan = false;

  const outcome = await confirmTrainingTransformersUpgrade({
    modelName: MODEL,
  });

  assert.deepEqual(outcome, {
    proceed: true,
    error: null,
    forces16Bit: false,
    requiresTrustRemoteCode: true,
  });
});

test("an already-routed model reports 16-bit without a dialog", async () => {
  stub.resetStub();
  stub.state.checkResult = {
    upgrade: null,
    requiresTrustRemoteCode: false,
    latestTierActive: true,
    forces16Bit: true,
  };

  const outcome = await confirmTrainingTransformersUpgrade({
    modelName: MODEL,
  });

  assert.deepEqual(outcome, {
    proceed: true,
    error: null,
    forces16Bit: true,
    requiresTrustRemoteCode: false,
  });
  assert.equal(
    stub.calls.length,
    1,
    "nothing to install, so nothing to consent to",
  );
});

test("an exact 4-bit resume is never offered an install that strands it", async () => {
  // The sidecar is persistent and refuses this checkpoint's attested 4-bit load, while the
  // model's own code resumes today, so consenting would strand it.
  stub.resetStub();
  stub.state.checkResult = {
    upgrade: UPGRADE,
    requiresTrustRemoteCode: true,
    latestTierActive: false,
    forces16Bit: false,
    installBreaksExactResume: true,
  };

  const outcome = await confirmTrainingTransformersUpgrade({
    modelName: MODEL,
    resumeRunId: "run-42",
  });

  assert.deepEqual(outcome, {
    proceed: true,
    error: null,
    forces16Bit: false,
    requiresTrustRemoteCode: true,
  });
  assert.equal(
    stub.calls.length,
    1,
    "no install may be offered when accepting it would strand the checkpoint",
  );
  assert.equal(stub.calls[0]?.args[2]?.resumeRunId, "run-42");
});

test("a resume with no custom-code way out is not offered a doomed install", async () => {
  // Installing would make effective_training_load_in_4bit raise for this config, so the start is
  // refused with a reason instead of offering the install.
  stub.resetStub();
  stub.state.checkResult = {
    upgrade: UPGRADE,
    requiresTrustRemoteCode: false,
    latestTierActive: false,
    forces16Bit: true,
    installBreaksExactResume: true,
  };
  stub.state.consentResult = true;
  stub.state.installRan = true;

  const outcome = await confirmTrainingTransformersUpgrade({
    modelName: MODEL,
    resumeRunId: "run-42",
  });

  assert.equal(outcome.proceed, false);
  assert.equal(outcome.forces16Bit, false);
  assert.match(String(outcome.error), /4-bit model load/);
  assert.match(String(outcome.error), /Start a new run/);
  assert.equal(
    stub.calls.length,
    1,
    "an install that cannot rescue the resume must never be offered",
  );
});

test("a resume with nothing to install is told the truth about why", async () => {
  // installBreaksExactResume ignores whether a release exists; for a dev-only architecture the
  // dev-only explanation is the accurate one.
  stub.resetStub();
  stub.state.checkResult = {
    upgrade: { ...UPGRADE, supported_in_pypi: false },
    requiresTrustRemoteCode: false,
    latestTierActive: false,
    forces16Bit: false,
    installBreaksExactResume: true,
  };
  stub.state.consentResult = false;

  const outcome = await confirmTrainingTransformersUpgrade({
    modelName: MODEL,
    resumeRunId: "run-42",
  });

  assert.equal(outcome.proceed, false);
  assert.doesNotMatch(String(outcome.error), /Start a new run/);
  assert.match(String(outcome.error), /development branch/);
  assert.match(String(outcome.error), /next transformers release/);
});

test("declining a dev-only upgrade is not told to start again and install it", async () => {
  // Dev-only architecture: no Install action, so the installable wording would loop forever.
  stub.resetStub();
  stub.state.checkResult = {
    upgrade: { ...UPGRADE, supported_in_pypi: false },
    requiresTrustRemoteCode: false,
    latestTierActive: false,
    forces16Bit: false,
  };
  stub.state.consentResult = false;

  const outcome = await confirmTrainingTransformersUpgrade({
    modelName: MODEL,
  });

  assert.equal(outcome.proceed, false);
  assert.doesNotMatch(String(outcome.error), /Start the run again/);
  assert.match(String(outcome.error), /development branch/);
  assert.match(String(outcome.error), /next transformers release/);
});

test("declining an installable upgrade is still told how to get it", async () => {
  stub.resetStub();
  stub.state.checkResult = {
    upgrade: UPGRADE,
    requiresTrustRemoteCode: false,
    latestTierActive: false,
    forces16Bit: true,
  };
  stub.state.consentResult = false;

  const outcome = await confirmTrainingTransformersUpgrade({
    modelName: MODEL,
  });

  assert.match(String(outcome.error), /Start the run again to install it/);
});

test("the check is asked about the copy the run will load", async () => {
  // A cached model loads from its pinned snapshot, whose architecture can differ from the repo's.
  stub.resetStub();

  await confirmTrainingTransformersUpgrade({
    modelName: MODEL,
    modelCachePin: {
      preferLocalCache: true,
      modelLocalPath: "/cache/models--org--model",
      modelSnapshotPath: "/cache/models--org--model/snapshots/abc",
      modelSnapshotRepoId: "org/model",
    },
  });

  assert.deepEqual(stub.calls[0]?.args[2], {
    preferLocalCache: true,
    modelLocalPath: "/cache/models--org--model",
    modelSnapshotPath: "/cache/models--org--model/snapshots/abc",
    modelSnapshotRepoId: "org/model",
    resumeRunId: undefined,
  });
});

test("a backend without the check leaves the start exactly as it was", async () => {
  stub.resetStub();
  stub.state.checkResult = new Error("404 Not Found");

  const outcome = await confirmTrainingTransformersUpgrade({
    modelName: MODEL,
  });

  assert.deepEqual(outcome, {
    proceed: true,
    error: null,
    forces16Bit: false,
    requiresTrustRemoteCode: false,
  });
  assert.equal(stub.calls.length, 1);
});

test("the custom-code verdict travels to the next gate", async () => {
  // confirmRemoteCodeIfNeeded falls back to the caller's flag (false on a fresh run) when the
  // scan fails, so the gate's knowledge of custom code must be carried out.
  stub.resetStub();
  stub.state.checkResult = {
    upgrade: { ...UPGRADE, supported_in_pypi: false },
    requiresTrustRemoteCode: true,
    latestTierActive: false,
    forces16Bit: false,
  };
  stub.state.consentResult = true;
  stub.state.installRan = false;

  const outcome = await confirmTrainingTransformersUpgrade({
    modelName: MODEL,
  });

  assert.equal(outcome.proceed, true);
  assert.equal(outcome.requiresTrustRemoteCode, true);
});

test("a model without custom code reports no verdict to carry", async () => {
  stub.resetStub();
  stub.state.checkResult = {
    upgrade: null,
    requiresTrustRemoteCode: false,
    latestTierActive: false,
    forces16Bit: false,
  };

  const outcome = await confirmTrainingTransformersUpgrade({
    modelName: MODEL,
  });

  assert.equal(outcome.requiresTrustRemoteCode, false);
});

// A bundle newer than its backend (mid-upgrade) gets a 404 here; the start must still work.
for (const [label, failure] of [
  ["a 404 from a backend without the route", new Error("404: API endpoint not found")],
  ["a 405 from a backend without the route", new Error("405: Method Not Allowed")],
  ["a 500 from a broken backend", new Error("500: Internal Server Error")],
  ["a network failure", new TypeError("Failed to fetch")],
] as const) {
  test(`${label} leaves the start exactly as it was`, async () => {
    stub.resetStub();
    stub.state.checkResult = failure;
    const outcome = await confirmTrainingTransformersUpgrade({ modelName: MODEL });
    assert.equal(outcome.proceed, true);
    assert.equal(outcome.error, null);
    assert.equal(outcome.forces16Bit, false);
    // The start path ORs this with the stored flag, so a failed check never removes consent.
    assert.equal(outcome.requiresTrustRemoteCode, false);
    assert.equal(
      stub.calls.filter((c) => c.name === "confirmTransformersUpgradeIfNeeded").length,
      0,
      "a preflight that could not run must raise no dialog",
    );
  });
}

test("an upgrade with no version is never offered as an install", async () => {
  stub.resetStub();
  stub.state.checkResult = {
    upgrade: {
      // biome-ignore lint/style/useNamingConvention: API schema
      model_type: "muse_glimmer",
      // biome-ignore lint/style/useNamingConvention: API schema
      pypi_version: null,
      // biome-ignore lint/style/useNamingConvention: API schema
      supported_in_pypi: true,
    },
    requiresTrustRemoteCode: false,
    latestTierActive: false,
    forces16Bit: false,
  };
  stub.state.consentResult = false;
  const outcome = await confirmTrainingTransformersUpgrade({ modelName: MODEL });
  assert.equal(outcome.proceed, false);
  assert.match(String(outcome.error), /no released transformers version supports it/);
});

test("a field from a newer backend is ignored rather than fatal", async () => {
  stub.resetStub();
  stub.state.checkResult = {
    upgrade: null,
    requiresTrustRemoteCode: false,
    latestTierActive: false,
    forces16Bit: false,
  };
  // Object.assign because the stub's types are deliberately exact, and a newer backend sends unknown keys.
  Object.assign(stub.state.checkResult, {
    someVerdictThisBuildHasNeverHeardOf: { nested: true },
  });
  const outcome = await confirmTrainingTransformersUpgrade({ modelName: MODEL });
  assert.equal(outcome.proceed, true);
});
