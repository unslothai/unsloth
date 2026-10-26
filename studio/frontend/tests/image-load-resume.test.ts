// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0


import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import {
  clearPendingImageLoad,
  finishedLoadTarget,
  forgetStaleResidentComponents,
  readImageComponents,
  readImageModel,
  rememberImageModel,
  rememberLoadedImageComponents,
  rememberPendingImageLoad,
} from "../src/features/images/image-model-recall.ts";
import { isLoadInFlightPhase } from "../src/lib/swap-save-phase.ts";
import { installLocalStorageFake, readSrc } from "./helpers/kit.ts";

const { store } = installLocalStorageFake();

const zImage = { repoId: "/comfy/models/diffusion_models", kind: "gguf", filename: "z.gguf" } as const;
const zFiles = { textEncoderFiles: ["../text_encoders/qwen_3_4b.safetensors"], vaeFile: "../vae/ae.safetensors" };
const ready = {
  loaded: true,
  repo_id: zImage.repoId,
  model_kind: "gguf",
  gguf_filename: zImage.filename,
  engine: "diffusers",
  component_files: { text_encoder: "qwen_3_4b.safetensors", vae: "ae.safetensors" },
};

test("every nonterminal backend load phase counts as in flight on page open", () => {
  const models = readFileSync(new URL("../../backend/models/inference.py", import.meta.url), "utf8");
  const response = models.slice(models.indexOf("class DiffusionLoadProgressResponse"));
  const phases = [.../Literal\[([^\]]+)\]/.exec(response)![1].matchAll(/"([^"]+)"/g)].map((m) => m[1]);
  const nonterminal = phases.filter((p) => p !== "ready" && p !== "error");
  assert.ok(nonterminal.includes("saving_chat_context"));
  for (const phase of nonterminal) assert.equal(isLoadInFlightPhase(phase), true, phase);
  for (const phase of ["ready", "error", null, undefined]) assert.equal(isLoadInFlightPhase(phase), false, String(phase));
  const source = readSrc("features/images/images-page.tsx");
  assert.match(source, /inFlight = isLoadInFlightPhase\(p\.phase\);/);
});

test("a load resumed after a page reload records the paths it was started with", () => {
  store.clear();
  rememberPendingImageLoad({ ...zImage, ...zFiles, fieldFiles: zFiles });
  const resident = { repoId: "org/old", kind: "pipeline" } as const;
  const target = finishedLoadTarget<typeof resident>(resident, ready);
  assert.deepEqual(target, { ...zImage, ...zFiles, fieldFiles: zFiles });
  assert.deepEqual(rememberLoadedImageComponents(target!, ready), zFiles);
  assert.deepEqual(readImageComponents(zImage), zFiles);
  assert.equal(finishedLoadTarget(null, ready), null);

  rememberPendingImageLoad({ ...zImage, fieldFiles: zFiles });
  assert.deepEqual(finishedLoadTarget(null, ready)?.fieldFiles, zFiles);

  rememberPendingImageLoad({ ...zImage, fieldFiles: zFiles });
  assert.equal(finishedLoadTarget(null, { ...ready, gguf_filename: "other.gguf" }), null);
  const own = { ...zImage, fieldFiles: { vaeFile: "mine.safetensors" } };
  rememberPendingImageLoad({ ...zImage, fieldFiles: zFiles });
  assert.equal(finishedLoadTarget(own, ready), own);
  rememberPendingImageLoad({ ...zImage, fieldFiles: zFiles });
  clearPendingImageLoad();
  assert.equal(finishedLoadTarget(null, ready), null);
});

test("the page stores the pending load at start and clears it on every way a load ends", () => {
  const source = readSrc("features/images/images-page.tsx");
  assert.match(source, /lastLoad\.current = \{[\s\S]*?fieldFiles: \{[^}]*\},\s*\};[\s\S]{0,200}rememberPendingImageLoad\(lastLoad\.current\);/);
  assert.match(source, /const started = lastLoad\.current;\s*const finished = finishedLoadTarget\(started, loaded\);[\s\S]{0,200}lastLoad\.current = finished \? retiredSnapshotAsLoaded\(finished, loaded\) : finished;\s*if \(lastLoad\.current\) rememberFinishedLoad\(lastLoad\.current, loaded, finished !== started\);/);
  assert.match(source, /if \(p\.phase === "error"\) \{\s*clearPendingImageLoad\(\);/);
  assert.match(source, /if \(p\.phase === null\) \{\s*clearPendingImageLoad\(\);/);
  assert.match(source, /\} catch \(err\) \{\s*pendingEvicted\.current = \[\];\s*clearPendingImageLoad\(\);/);
  const drop = source.slice(source.indexOf("const dropResidentState = useCallback"));
  assert.ok(drop.slice(0, drop.indexOf("}, [dismissLoadToast")).includes("clearPendingImageLoad();"));
  assert.match(source, /if \(!inFlight\) clearPendingImageLoad\(\);/);
});

test("a load that finished while the page was closed records its paths before the descriptor is cleared", () => {
  const source = readSrc("features/images/images-page.tsx");
  const mount = source.slice(source.indexOf("inFlight = isLoadInFlightPhase(p.phase);"));
  const probe = mount.slice(0, mount.indexOf("if (!inFlight) clearPendingImageLoad();"));
  assert.match(
    probe,
    /\} else if \(p\.phase === "ready"\) \{[\s\S]*?const loaded = await getDiffusionStatus\(\);\s*setStatusIfNewest\(ticket, loaded\);\s*const resumed = finishedLoadTarget<LastLoad>\(null, loaded\);\s*if \(resumed\) \{\s*lastLoad\.current = resumed;\s*rememberFinishedLoad\(resumed, loaded, true\);/,
  );
  const recorder = source.slice(source.indexOf("const rememberFinishedLoad = useCallback"));
  const body = recorder.slice(0, recorder.indexOf("}, []);"));
  assert.match(body, /if \(!matchesRememberedModel\(target, loaded\)\) return;/);
  assert.match(body, /rememberImageModel\(remembered\);\s*setRememberedModel\(remembered\);/);
  assert.match(body, /rememberLoadedImageComponents\(target, loaded\)/);

  store.clear();
  rememberPendingImageLoad({ ...zImage, fieldFiles: zFiles });
  const resumed = finishedLoadTarget(null, ready);
  assert.deepEqual(resumed?.fieldFiles, zFiles);
  assert.deepEqual(rememberLoadedImageComponents(resumed!, ready), zFiles);
  assert.deepEqual(readImageComponents(zImage), zFiles);
});

test("the mount probe publishes the finished build's status before the reconcilers run", () => {
  const source = readSrc("features/images/images-page.tsx");
  const mount = source.slice(source.indexOf("inFlight = isLoadInFlightPhase(p.phase);"));
  const probe = mount.slice(0, mount.indexOf('setInitialLoadProbe(inFlight ? "loading" : "idle");'));
  assert.match(
    probe,
    /\} else if \(p\.phase === "ready"\) \{[\s\S]*?const ticket = \+\+statusTicket\.current;\s*const loaded = await getDiffusionStatus\(\);\s*setStatusIfNewest\(ticket, loaded\);/,
  );

  const newFiles = { vaeFile: "/models/new-vae.safetensors" };
  const oldBuild = { ...ready, component_files: { vae: "old-vae.safetensors" } };
  const newBuild = { ...ready, component_files: { vae: "new-vae.safetensors" } };
  store.clear();
  rememberPendingImageLoad({ ...zImage, fieldFiles: newFiles });
  const resumed = finishedLoadTarget(null, newBuild)!;
  rememberLoadedImageComponents(resumed, newBuild);
  rememberImageModel({ ...zImage, ...newFiles });
  forgetStaleResidentComponents(newBuild);
  assert.deepEqual(readImageComponents(zImage), newFiles);
  assert.equal(readImageModel()?.vaeFile, newFiles.vaeFile);
  forgetStaleResidentComponents(oldBuild);
  assert.equal(readImageComponents(zImage), null);
});

test("a pending load whose files the resident build does not use keeps the saved entry", () => {
  const working = { vaeFile: "/models/working-ae.safetensors" };
  const other = { vaeFile: "/models/other-ae.safetensors" };
  const resident = { ...ready, component_files: { vae: "working-ae.safetensors" } };
  store.clear();
  rememberLoadedImageComponents({ ...zImage, fieldFiles: working }, resident);
  rememberPendingImageLoad({ ...zImage, ...other, fieldFiles: other });
  const resumed = finishedLoadTarget(null, resident);
  assert.equal(resumed, null);
  assert.equal(finishedLoadTarget(null, resident), null, "consumed");
  forgetStaleResidentComponents(resident);
  assert.deepEqual(readImageComponents(zImage), working);

  const own = { ...zImage };
  rememberPendingImageLoad({ ...zImage, fieldFiles: other });
  assert.equal(finishedLoadTarget(own, resident), own);
  assert.equal(rememberLoadedImageComponents(own, resident), null);
  assert.deepEqual(readImageComponents(zImage), working);

  rememberPendingImageLoad({ ...zImage, fieldFiles: working });
  assert.deepEqual(finishedLoadTarget(null, resident)?.fieldFiles, working);
  const native = { ...ready, engine: "sd_cpp", component_files: {} };
  rememberPendingImageLoad({ ...zImage, fieldFiles: other });
  assert.deepEqual(finishedLoadTarget(null, native)?.fieldFiles, other);
});

test("a retired snapshot's load resumed after a page reload is recovered as what it loaded as", () => {
  store.clear();
  const snapshot = "/hf/hub/models--unsloth--flux-3-image-klein/snapshots/abc";
  const asPipeline = { loaded: true, repo_id: snapshot, model_kind: "pipeline" };
  rememberPendingImageLoad({ repoId: snapshot, kind: "single_file", filename: "model.safetensors", baseRepo: snapshot });
  const root = finishedLoadTarget(null, asPipeline);
  assert.deepEqual([root?.repoId, root?.kind, root?.filename], [snapshot, "pipeline", undefined]);
  const asBase = { loaded: true, repo_id: "black-forest-labs/flux-3-image-klein", model_kind: "pipeline" };
  rememberPendingImageLoad({ repoId: snapshot, kind: "pipeline", displayRepoId: "unsloth/flux-3-image-klein" });
  const redirected = finishedLoadTarget(null, asBase);
  assert.deepEqual([redirected?.repoId, redirected?.kind], ["black-forest-labs/flux-3-image-klein", "pipeline"]);
});
