// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// A diffusion load survives a page reload. The reopened page must treat every nonterminal load phase as in flight, and
// the requested component paths (status names only basenames) must survive the reload so the finished load records them.

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import {
  clearPendingImageLoad,
  finishedLoadTarget,
  readImageComponents,
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
  // The page started the load and stored its request, then reloaded: lastLoad is gone (or the replaced build's).
  rememberPendingImageLoad({ ...zImage, ...zFiles, fieldFiles: zFiles });
  const resident = { repoId: "org/old", kind: "pipeline" } as const;
  const target = finishedLoadTarget<typeof resident>(resident, ready);
  assert.deepEqual(target, { ...zImage, ...zFiles, fieldFiles: zFiles });
  assert.deepEqual(rememberLoadedImageComponents(target!, ready), zFiles);
  assert.deepEqual(readImageComponents(zImage), zFiles);
  // Consumed: a later finish does not reuse it.
  assert.equal(finishedLoadTarget(null, ready), null);

  // First load on a fresh page: no resident descriptor at all.
  rememberPendingImageLoad({ ...zImage, fieldFiles: zFiles });
  assert.deepEqual(finishedLoadTarget(null, ready)?.fieldFiles, zFiles);

  // A build that is another model ignores it, and the page's own load keeps its own request.
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
  // Success consumes it before the descriptor is checked against the status.
  assert.match(source, /lastLoad\.current = finishedLoadTarget\(lastLoad\.current, loaded\);\s*if \(lastLoad\.current && matchesRememberedModel\(lastLoad\.current, loaded\)\)/);
  // Failed after starting, cancelled or evicted, refused at start, cancelled or ejected by the page, nothing to resume.
  assert.match(source, /if \(p\.phase === "error"\) \{\s*clearPendingImageLoad\(\);/);
  assert.match(source, /if \(p\.phase === null\) \{\s*clearPendingImageLoad\(\);/);
  assert.match(source, /\} catch \(err\) \{\s*pendingEvicted\.current = \[\];\s*clearPendingImageLoad\(\);/);
  const drop = source.slice(source.indexOf("const dropResidentState = useCallback"));
  assert.ok(drop.slice(0, drop.indexOf("}, [dismissLoadToast")).includes("clearPendingImageLoad();"));
  assert.match(source, /if \(!inFlight\) clearPendingImageLoad\(\);/);
});
