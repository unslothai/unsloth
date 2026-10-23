// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import {
  isRetiredCopySnapshot,
  matchesRememberedModel,
  readImageModel,
  redirectRetiredImagePick,
  rememberImageModel,
  retiredSnapshotAsLoaded,
} from "../src/features/images/image-model-recall.ts";
import { installLocalStorageFake, readSrc } from "./helpers/kit.ts";

installLocalStorageFake();

test("a pipeline pick of a retired FLUX.3 full copy loads the base it copies", () => {
  assert.deepEqual(redirectRetiredImagePick({ repoId: "unsloth/flux-3-image-klein", kind: "pipeline" }), {
    repoId: "black-forest-labs/flux-3-image-klein",
    kind: "pipeline",
  });
  assert.equal(
    redirectRetiredImagePick({ repoId: "Unsloth/FLUX-3-Image-Klein-Base", kind: "pipeline" as const }).repoId,
    "black-forest-labs/flux-3-image-klein-base",
  );
});

test("its root bf16 transformer is the pipeline load; other picks stay as named", () => {
  assert.deepEqual(
    redirectRetiredImagePick({
      repoId: "unsloth/flux-3-image-klein",
      kind: "single_file" as const,
      filename: "model.safetensors",
    }),
    { repoId: "black-forest-labs/flux-3-image-klein", kind: "pipeline" },
  );
  // Supplied encoder / VAE files make it another build, which stays on the copy (the backend does the same).
  const withVae = {
    repoId: "unsloth/flux-3-image-klein",
    kind: "single_file" as const,
    filename: "model.safetensors",
    vaeFile: "x/vae.safetensors",
  };
  assert.equal(redirectRetiredImagePick(withVae), withVae);
  const withEncoder = { ...withVae, vaeFile: undefined, textEncoderFiles: ["x/te.safetensors"] };
  assert.equal(redirectRetiredImagePick(withEncoder), withEncoder);
  const other = { repoId: "unsloth/flux-3-image-klein-FP8", kind: "pipeline" as const };
  assert.equal(redirectRetiredImagePick(other), other);
});

test("a downloaded snapshot of a retired copy loads from disk, not through its base", () => {
  // The copy holds the base's own files; redirecting it would need the base's mirror, which may not be cached
  // (offline it cannot load at all). The backend keeps a complete snapshot the same way.
  const downloaded = {
    repoId: "/hf/hub/models--unsloth--flux-3-image-klein-base/snapshots/abc",
    kind: "pipeline" as const,
    displayRepoId: "unsloth/flux-3-image-klein-base",
  };
  assert.equal(redirectRetiredImagePick(downloaded), downloaded);
  const rootTransformer = {
    repoId: "/hf/hub/models--unsloth--flux-3-image-klein/snapshots/abc",
    kind: "single_file" as const,
    filename: "model.safetensors",
    displayRepoId: "unsloth/flux-3-image-klein",
  };
  assert.equal(redirectRetiredImagePick(rootTransformer), rootTransformer);
});

test("a remembered retired copy is recalled as its base, so it matches the status the load reports", () => {
  rememberImageModel({ repoId: "unsloth/flux-3-image-klein", kind: "pipeline" });
  const recalled = readImageModel();
  assert.deepEqual(recalled, { repoId: "black-forest-labs/flux-3-image-klein", kind: "pipeline" });
  assert.equal(
    matchesRememberedModel(recalled!, {
      loaded: true,
      repo_id: "black-forest-labs/flux-3-image-klein",
      model_kind: "pipeline",
    }),
    true,
  );
});

test("a model remembered as the copy's snapshot path is recalled as that snapshot", () => {
  const snapshot = "/home/me/.cache/huggingface/hub/models--unsloth--flux-3-image-klein/snapshots/0123abc";
  rememberImageModel({ repoId: snapshot, kind: "pipeline" });
  assert.deepEqual(readImageModel(), { repoId: snapshot, kind: "pipeline" });
  const mirror = {
    repoId: "/home/me/.cache/huggingface/hub/models--unsloth--flux-3-image-klein-FP8/snapshots/0123abc",
    kind: "pipeline" as const,
  };
  assert.equal(redirectRetiredImagePick(mirror), mirror);
});

test("a retired copy's snapshot is planned before it loads, since only the backend knows it is complete", () => {
  // An incomplete snapshot loads through the base: a direct load would pull the base's mirror inline, outside the
  // download plan and its staging. So the snapshot is planned by its path, with the copy as the display id.
  const snapshot = "/hf/hub/models--unsloth--flux-3-image-klein/snapshots/abc";
  assert.equal(isRetiredCopySnapshot({ repoId: snapshot, displayRepoId: "unsloth/flux-3-image-klein" }), true);
  assert.equal(isRetiredCopySnapshot({ repoId: snapshot }), true);
  assert.equal(isRetiredCopySnapshot({ repoId: "/models/klein", displayRepoId: "unsloth/flux-3-image-klein-base" }), true);
  // The copy's id is redirected to the base outright; other repos and their snapshots are not this case.
  assert.equal(isRetiredCopySnapshot({ repoId: "unsloth/flux-3-image-klein" }), false);
  assert.equal(
    isRetiredCopySnapshot({ repoId: "/hf/hub/models--unsloth--flux-3-image-klein-FP8/snapshots/abc" }),
    false,
  );
  assert.equal(isRetiredCopySnapshot({ repoId: "/models/other", displayRepoId: "unsloth/flux-3-image-klein-FP8" }), false);

  const page = readSrc("features/images/images-page.tsx");
  const body = page.slice(page.indexOf("const loadOrStage = useCallback("), page.indexOf("const loadOrStageRef"));
  assert.match(body, /const retiredSnapshot = isRetiredCopySnapshot\(\{ repoId, displayRepoId: opts\.displayRepoId \}\);\s*if \(retiredSnapshot\) source = "hub";/);
  // Planned as the path (the display id rides along for the backend's redirect), and nothing it returns is dropped.
  assert.match(body, /requestDownloadPlan\(repoId, opts, advanced, opts\.displayRepoId\)/);
  assert.match(body, /retiredSnapshot\s*\? diffusionStagingEntries\(plan\.entries, repoId, \{ filename: opts\.filename \}\)/);
  // The source check that sends a local pick straight to the load comes after the retired-snapshot override.
  assert.ok(body.indexOf("if (retiredSnapshot) source") < body.indexOf('if (source !== "hub" && !downloadOnly)'));
});

test("a retired copy's snapshot that loaded through its base is remembered as that base", () => {
  // Incomplete on disk, the backend loads the base it copies, and the status names that base: the snapshot pick never
  // matches it, so the selection would not replace the remembered model and Reapply would still name the snapshot.
  const snapshot = "/hf/hub/models--unsloth--flux-3-image-klein/snapshots/abc";
  const base = { loaded: true, repo_id: "black-forest-labs/flux-3-image-klein", model_kind: "pipeline" };
  const pipeline = { repoId: snapshot, kind: "pipeline" as const, displayRepoId: "unsloth/flux-3-image-klein" };
  const resolved = retiredSnapshotAsLoaded(pipeline, base);
  assert.deepEqual(resolved, { repoId: "black-forest-labs/flux-3-image-klein", kind: "pipeline" });
  assert.equal(matchesRememberedModel(resolved, base), true);
  // Its root transformer pick, and a snapshot path remembered alone, resolve the same way.
  const rootTransformer = { repoId: snapshot, kind: "single_file" as const, filename: "model.safetensors", baseRepo: snapshot };
  assert.deepEqual(retiredSnapshotAsLoaded(rootTransformer, base), {
    repoId: "black-forest-labs/flux-3-image-klein",
    kind: "pipeline",
  });
  // A complete snapshot loaded as itself, and a status naming something else, keep the pick as it was.
  const asItself = { loaded: true, repo_id: snapshot, model_kind: "pipeline" };
  assert.equal(retiredSnapshotAsLoaded(pipeline, asItself), pipeline);
  const other = { loaded: true, repo_id: "black-forest-labs/flux-3-image-klein-base", model_kind: "pipeline" };
  assert.equal(retiredSnapshotAsLoaded(pipeline, other), pipeline);
  const unrelated = { repoId: "/models/sdxl", kind: "pipeline" as const };
  assert.equal(retiredSnapshotAsLoaded(unrelated, base), unrelated);

  // The ready poll reconciles the Reapply target before it decides what to remember.
  const page = readSrc("features/images/images-page.tsx");
  const ready = page.slice(page.indexOf('toast.success("Model loaded"'));
  const reconcile = ready.indexOf("lastLoad.current = retiredSnapshotAsLoaded(lastLoad.current, loaded)");
  assert.ok(reconcile > 0 && reconcile < ready.indexOf("matchesRememberedModel(lastLoad.current, loaded)"));
});
