// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import {
  ENGINE_OPTIONS,
  engineModelKey,
  engineRequestField,
  engineSelectVisible,
  loadedEngineChoice,
  nativeLoadAdvanced,
  nativeReloadTarget,
  offersNativeLoraReload,
  readEngineChoices,
  rememberLoadedEngine,
  sendsEngine,
  withEngineChoice,
} from "../src/features/images/lib/engine-choice.ts";
import { formatResolvedValue, resolvedBadge } from "../src/lib/resolved-precision.ts";
import { installLocalStorageFake, readSrc } from "./helpers/kit.ts";

const { storage, store } = installLocalStorageFake();

const ggufOnDiffusers = {
  loaded: true,
  engine: "diffusers",
  engine_forced: null,
  native_engine_eligible: true,
  lora_via_native: true,
  supports_lora: false,
};

test("the select offers Auto, Diffusers and Native, and shows only for a GGUF native can run", () => {
  assert.deepEqual(
    ENGINE_OPTIONS.map(([v]) => v),
    ["auto", "diffusers", "sd_cpp"],
  );
  assert.equal(engineSelectVisible(ggufOnDiffusers), true);
  assert.equal(engineSelectVisible({ ...ggufOnDiffusers, native_engine_eligible: false }), false);
  assert.equal(engineSelectVisible({ loaded: false }), false);
  // An older backend sends no eligibility.
  assert.equal(engineSelectVisible({ loaded: true, engine: "diffusers" }), false);
  assert.equal(engineSelectVisible(null), false);
});

test("only a GGUF load carries the pick, and Auto omits it", () => {
  assert.equal(sendsEngine("gguf", "x.gguf"), true);
  assert.equal(sendsEngine("pipeline", undefined), false);
  assert.equal(sendsEngine("single_file", "x.safetensors"), false);
  assert.equal(sendsEngine(undefined, "Model-Q4_K_M.GGUF"), true);
  assert.equal(engineRequestField("auto"), undefined);
  assert.equal(engineRequestField(undefined), undefined);
  assert.equal(engineRequestField("sd_cpp"), "sd_cpp");
  assert.equal(engineRequestField("diffusers"), "diffusers");
});

test("the choice is remembered per checkpoint and Auto forgets it", () => {
  storage.removeItem("unsloth:images:engine-by-model");
  const key = engineModelKey("unsloth/FLUX.2-klein-4B-GGUF", "flux-2-klein-4b-Q4_K_M.gguf");
  assert.equal(key, "unsloth/FLUX.2-klein-4B-GGUF|flux-2-klein-4b-Q4_K_M.gguf");
  assert.equal(engineModelKey("repo", null), "");
  let choices = withEngineChoice({}, key, "sd_cpp");
  assert.deepEqual(readEngineChoices(), { [key]: "sd_cpp" });
  choices = withEngineChoice(choices, "other|b.gguf", "diffusers");
  choices = withEngineChoice(choices, key, "auto");
  assert.deepEqual(choices, { "other|b.gguf": "diffusers" });
  assert.deepEqual(readEngineChoices(), { "other|b.gguf": "diffusers" });
  // No key, nothing stored.
  assert.deepEqual(withEngineChoice(choices, "", "sd_cpp"), choices);
  // Junk in storage reads as no choice.
  storage.setItem("unsloth:images:engine-by-model", JSON.stringify({ a: "vulkan", b: 3, c: "sd_cpp" }));
  assert.deepEqual(readEngineChoices(), { c: "sd_cpp" });
  storage.setItem("unsloth:images:engine-by-model", "not json");
  assert.deepEqual(readEngineChoices(), {});
});

test("the loaded build's select value follows the backend's engine provenance", () => {
  const resolved = (entry: { source: string; requested?: string | null; value?: string }) => ({
    ...ggufOnDiffusers,
    resolved: { engine: entry },
  });
  assert.equal(loadedEngineChoice(resolved({ source: "explicit", requested: "sd_cpp", value: "sd_cpp" })), "sd_cpp");
  assert.equal(loadedEngineChoice(resolved({ source: "explicit", requested: "diffusers" })), "diffusers");
  assert.equal(loadedEngineChoice(resolved({ source: "auto", requested: null, value: "diffusers" })), "auto");
  assert.equal(loadedEngineChoice(ggufOnDiffusers), "auto");
  assert.equal(loadedEngineChoice({ loaded: false, resolved: { engine: { source: "explicit", requested: "sd_cpp" } } }), "auto");
});

test("the LoRA note offers a native reload only where it would turn LoRA on", () => {
  assert.equal(offersNativeLoraReload(ggufOnDiffusers), true);
  assert.equal(offersNativeLoraReload({ ...ggufOnDiffusers, lora_via_native: false }), false);
  assert.equal(offersNativeLoraReload({ ...ggufOnDiffusers, supports_lora: true }), false);
  // The server pin would ignore the reload's pick.
  assert.equal(offersNativeLoraReload({ ...ggufOnDiffusers, engine_forced: "diffusers" }), false);
  assert.equal(offersNativeLoraReload({ loaded: false }), false);
});

test("the engine badge names the engine", () => {
  assert.equal(formatResolvedValue("engine", "sd_cpp"), "Native");
  assert.equal(formatResolvedValue("engine", "diffusers"), "Diffusers");
  assert.equal(formatResolvedValue("engine", "mlx"), "MLX");
  assert.deepEqual(resolvedBadge("engine", { value: "diffusers", requested: null, source: "auto", status: "applied", reason: "GPU backend 'cuda' uses diffusers" }), {
    label: "Auto: Diffusers",
    tone: "auto",
    tooltip: "GPU backend 'cuda' uses diffusers",
  });
  assert.equal(resolvedBadge("engine", { value: "sd_cpp", requested: "sd_cpp", source: "explicit", status: "applied", reason: "selected for this load" }), null);
});

test("the page sends the pick on the load and the plan, and wires the select and the note", () => {
  const source = readSrc("features/images/images-page.tsx");
  assert.equal(source.match(/engine: engineFor\(repoId, opts, advanced\),/g)?.length, 2);
  assert.match(source, /label="Engine"[\s\S]{0,400}controlKey="engine"/);
  assert.match(source, /offersNativeLoraReload\(status\)[\s\S]{0,600}onClick=\{handleNativeReload\}[\s\S]{0,80}Reload with native engine/);
  assert.match(source, /\{ \.\.\.currentLoadAdvanced\(l\.repoId\), engine: "sd_cpp" \}/);
  // Recorded from the load that landed, never from the select alone.
  assert.match(source, /setEngineChoices\(\(all\) => withEngineChoice\(all, key, loadedEngineChoice\(loaded\)\)\)/);
});

test("the native reload targets the resident GGUF, not a stale page load another client replaced", () => {
  const resident = { loaded: true, model_kind: "gguf", repo_id: "org/new-GGUF", gguf_filename: "new-Q4.gguf", display_repo_id: null };
  const stale = { repoId: "org/old-GGUF", kind: "gguf" as const, filename: "old-Q4.gguf" };
  assert.deepEqual(nativeReloadTarget(stale, resident), {
    repoId: "org/new-GGUF",
    kind: "gguf",
    filename: "new-Q4.gguf",
    displayRepoId: undefined,
  });
  const own = { repoId: "org/new-GGUF", kind: "gguf" as const, filename: "new-Q4.gguf", displayRepoId: "Shown" };
  assert.equal(nativeReloadTarget(own, resident), own);
  assert.equal(nativeReloadTarget(own, { loaded: false }), null);
  assert.equal(nativeReloadTarget(own, { ...resident, model_kind: "pipeline" }), null);
  const source = readSrc("features/images/images-page.tsx");
  assert.match(source, /const l = nativeReloadTarget\(lastLoad\.current, status\);/);
});

test("a refused load reseeds the Engine select from the resident build", () => {
  const source = readSrc("features/images/images-page.tsx");
  assert.match(source, /setEngineSeed\(""\);[\s\S]{0,80}dismissLoadToast\(\);\s*reportLoadFailure\(err, "Failed to start load"\)/);
});

test("a staged pick snapshots its resolved engine for both the plan and the load", () => {
  const source = readSrc("features/images/images-page.tsx");
  assert.match(
    source,
    /if \(!advanced\.engine && sendsEngine\(opts\.kind, opts\.filename\)\) \{\s*advanced = \{ \.\.\.advanced, engine: engineFor\(repoId, opts, advanced\) \?\? "auto" \};\s*\}\s*if \(source !== "hub" && !downloadOnly\)/,
  );
  // "auto" in the snapshot omits the field, so an untouched load stays byte-identical, and a later select edit
  // cannot leak into it.
  assert.equal(engineRequestField("auto"), undefined);
});

test("a native load drops the precision and component files the native routes refuse", () => {
  const advanced = {
    cpu_offload: false,
    speed_mode: "eager",
    transformer_quant: "fp8",
    text_encoder_quant: "fp8",
    text_encoder_file: ["/m/te.safetensors"],
    vae_file: "/m/vae.safetensors",
    loras: [{ id: "a", weight: 1 }],
  };
  const native = nativeLoadAdvanced(advanced, "sd_cpp");
  assert.equal(native.transformer_quant, undefined);
  assert.equal(native.text_encoder_quant, undefined);
  assert.equal(native.text_encoder_file, undefined);
  assert.equal(native.vae_file, undefined);
  // Everything else, LoRAs included (the point of the reload), is carried.
  assert.deepEqual(native.loras, advanced.loras);
  assert.equal(native.speed_mode, "eager");
  // Auto, diffusers, or a server pin to diffusers (which ignores the pick) keep the values.
  assert.equal(nativeLoadAdvanced(advanced, undefined), advanced);
  assert.equal(nativeLoadAdvanced(advanced, "diffusers"), advanced);
  assert.equal(nativeLoadAdvanced(advanced, "sd_cpp", "diffusers"), advanced);
  assert.equal(nativeLoadAdvanced(advanced, "sd_cpp", "sd_cpp").text_encoder_quant, undefined);
  // Applied to both the load and its plan.
  const source = readSrc("features/images/images-page.tsx");
  assert.equal(
    source.match(/const advanced = nativeLoadAdvanced\(picked, engineFor\(repoId, opts, picked\), engineSelection\.current\.forced\);/g)?.length,
    2,
  );
});

test("an explicit choice is persisted from any accepted loaded status, not only the poll's ready branch", () => {
  store.clear();
  const explicit = {
    loaded: true,
    model_kind: "gguf",
    repo_id: "org/klein-GGUF",
    gguf_filename: "klein-Q4.gguf",
    resolved: { engine: { source: "explicit", requested: "sd_cpp" } },
  };
  const stored = rememberLoadedEngine({}, explicit);
  assert.deepEqual(stored, { "org/klein-GGUF|klein-Q4.gguf": "sd_cpp" });
  assert.deepEqual(readEngineChoices(), stored);
  // Unchanged: the same object, so nothing re-renders.
  assert.equal(rememberLoadedEngine(stored, explicit), stored);
  // Auto, an unloaded status and a non-GGUF never write or clear.
  const auto = { ...explicit, resolved: { engine: { source: "auto", requested: null } } };
  assert.equal(rememberLoadedEngine(stored, auto), stored);
  assert.equal(rememberLoadedEngine(stored, { loaded: false }), stored);
  assert.equal(rememberLoadedEngine(stored, { ...explicit, model_kind: "pipeline" }), stored);
  const source = readSrc("features/images/images-page.tsx");
  assert.match(source, /setStatus\(next\);[\s\S]{0,160}setEngineChoices\(\(all\) => rememberLoadedEngine\(all, next\)\)/);
});
