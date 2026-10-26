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
  offersCompiledLoraReload,
  offersNativeLoraReload,
  reapplyChangesEngine,
  reapplyTarget,
  readEngineChoices,
  reloadSource,
  residentReloadTarget,
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

test("the LoRA note offers a plain reload once Speed=Auto compiled LoRA off", () => {
  const compiled = { loaded: true, engine: "diffusers", supports_lora: false, lora_after_reload: true };
  assert.equal(offersCompiledLoraReload(compiled), true);
  assert.equal(offersCompiledLoraReload({ ...compiled, lora_after_reload: false }), false);
  assert.equal(offersCompiledLoraReload({ ...compiled, supports_lora: true }), false);
  // An older backend sends no field.
  assert.equal(offersCompiledLoraReload({ loaded: true, supports_lora: false }), false);
  assert.equal(offersCompiledLoraReload({ loaded: false, lora_after_reload: true }), false);
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
  assert.match(source, /setEngineChoice\("sd_cpp"\);[\s\S]{0,120}engineSelection\.current = \{ \.\.\.engineSelection\.current, choice: "sd_cpp" \};/);
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
  assert.match(source, /setEngineSeed\(""\);[\s\S]{0,80}dismissLoadToast\(\);\s*(?:if \(!offerLoadAnywayFor\(err\)\) )?reportLoadFailure\(err, "Failed to start load"\)/);
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
  assert.equal(nativeLoadAdvanced(advanced, "sd_cpp", "sd_cpp"), advanced);
  // Applied to both the load and its plan.
  const source = readSrc("features/images/images-page.tsx");
  assert.equal(
    source.match(/const advanced = nativeLoadAdvanced\(\s*picked,\s*engineFor\(repoId, opts, picked\),\s*engineSelection\.current\.forced,/g)?.length,
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
  // An unloaded status and a non-GGUF never write or clear.
  assert.equal(rememberLoadedEngine(stored, { loaded: false }), stored);
  assert.equal(rememberLoadedEngine(stored, { ...explicit, model_kind: "pipeline" }), stored);
  const source = readSrc("features/images/images-page.tsx");
  assert.match(source, /setStatus\(next\);[\s\S]{0,160}setEngineChoices\(\(all\) => rememberLoadedEngine\(all, next\)\)/);
});

test("a resident load that reports Auto clears the stored explicit choice: the latest load wins", () => {
  store.clear();
  const key = "org/klein-GGUF|klein-Q4.gguf";
  const auto = {
    loaded: true,
    model_kind: "gguf",
    repo_id: "org/klein-GGUF",
    gguf_filename: "klein-Q4.gguf",
    resolved: { engine: { source: "auto", requested: null } },
  };
  const stored = { [key]: "sd_cpp" as const, "other|x.gguf": "diffusers" as const };
  const cleared = rememberLoadedEngine(stored, auto);
  assert.deepEqual(cleared, { "other|x.gguf": "diffusers" });
  assert.deepEqual(readEngineChoices(), cleared);
  // Nothing stored for it: unchanged, no write.
  assert.equal(rememberLoadedEngine(cleared, auto), cleared);
});

test("a resident GGUF the page did not load gets a Reapply target from status", () => {
  const gguf = { loaded: true, model_kind: "gguf", repo_id: "org/klein-GGUF", gguf_filename: "klein-Q4.gguf", display_repo_id: null };
  assert.deepEqual(residentReloadTarget(gguf), { repoId: "org/klein-GGUF", kind: "gguf", filename: "klein-Q4.gguf", displayRepoId: undefined });
  assert.deepEqual(residentReloadTarget({ loaded: true, model_kind: "pipeline", repo_id: "org/p", display_repo_id: "Shown" }), {
    repoId: "org/p",
    kind: "pipeline",
    displayRepoId: "Shown",
  });
  // Status names component files by basename only: reloading from it would drop them.
  assert.equal(residentReloadTarget({ ...gguf, component_files: { vae: "ae.safetensors" } }), null);
  assert.equal(residentReloadTarget({ ...gguf, gguf_filename: null }), null);
  assert.equal(residentReloadTarget({ loaded: false }), null);
  const source = readSrc("features/images/images-page.tsx");
  assert.match(source, /const resident = residentReloadTarget\(\{[\s\S]{0,300}\}\);\s*if \(resident\) lastLoad\.current = resident;/);
  // The Reapply button shows for that target too, not only for a pipeline.
  assert.match(source, /status\?\.loaded && \(canReapply \|\| residentReloadTarget\(status\) !== null\) &&/);
});

test("an engine-changing reload is planned and staged, a same-engine Reapply loads directly", () => {
  const onDiffusers = { loaded: true, model_kind: "gguf", engine: "diffusers", resolved: { engine: { source: "auto", requested: null } } };
  assert.equal(reapplyChangesEngine("sd_cpp", onDiffusers), true);
  assert.equal(reapplyChangesEngine("diffusers", onDiffusers), true);
  assert.equal(reapplyChangesEngine("auto", onDiffusers), false);
  const onNative = { ...onDiffusers, engine: "sd_cpp", resolved: { engine: { source: "explicit", requested: "sd_cpp" } } };
  assert.equal(reapplyChangesEngine("sd_cpp", onNative), false);
  assert.equal(reapplyChangesEngine("auto", onNative), true);
  // A server pin ignores the pick; a pipeline has no engine choice.
  assert.equal(reapplyChangesEngine("sd_cpp", { ...onDiffusers, engine_forced: "diffusers" }), false);
  assert.equal(reapplyChangesEngine("sd_cpp", { ...onDiffusers, model_kind: "pipeline" }), false);
  assert.equal(reloadSource("unsloth/FLUX.2-klein-GGUF"), "hub");
  assert.equal(reloadSource("/models/klein"), "local");
  assert.equal(reloadSource("C:\\models\\klein"), "local");
  const source = readSrc("features/images/images-page.tsx");
  // Both the note's native reload and an engine-changing Reapply go through loadOrStage, loading even under Download only.
  assert.equal(source.match(/reloadSource\(l\.repoId\),?\s*undefined,\s*true,\s*undefined,\s*true,?\s*\)/g)?.length, 2);
  assert.match(source, /if \(reapplyChangesEngine\(engineChoice, status\)\) \{\s*void loadOrStage\(/);
  assert.match(source, /downloadSnapshot !== undefined \|\| \(!forceLoad && modelSelectionAction === "download"\)/);
});

test("Reapply resolves its target against the checkpoint status reports", () => {
  const own = { repoId: "org/A-GGUF", kind: "gguf" as const, filename: "a.gguf" };
  const onA = { loaded: true, model_kind: "gguf", repo_id: "org/A-GGUF", gguf_filename: "a.gguf" };
  assert.equal(reapplyTarget(own, onA), own);
  // Another client replaced A with B: Reapply reloads B, never swaps back to A.
  const onB = { loaded: true, model_kind: "gguf", repo_id: "org/B-GGUF", gguf_filename: "b.gguf", base_repo: "org/B-base" };
  assert.deepEqual(reapplyTarget(own, onB), { repoId: "org/B-GGUF", kind: "gguf", filename: "b.gguf", displayRepoId: undefined, baseRepo: "org/B-base" });
  // B is not reconstructable from status: nothing, rather than A.
  assert.equal(reapplyTarget(own, { ...onB, component_files: { vae: "ae.safetensors" } }), null);
  // A pipeline matches by repo id alone; nothing resident keeps the page's own target.
  const pipe = { repoId: "org/P", kind: "pipeline" as const };
  assert.equal(reapplyTarget(pipe, { loaded: true, model_kind: "pipeline", repo_id: "org/P" }), pipe);
  assert.equal(reapplyTarget(own, { loaded: false }), own);
  const source = readSrc("features/images/images-page.tsx");
  assert.match(source, /const l = reapplyTarget\(lastLoad\.current, status\);/);
});

test("a reload of the resident GGUF keeps the base repo it was loaded with", () => {
  const status = {
    loaded: true,
    model_kind: "gguf",
    repo_id: "org/flux-gguf",
    gguf_filename: "flux-Q4_K_M.gguf",
    display_repo_id: null,
    base_repo: "org/custom-base",
    component_files: {},
  };
  const seeded = residentReloadTarget(status);
  assert.equal(seeded?.baseRepo, "org/custom-base");
  assert.equal(nativeReloadTarget(seeded, status)?.baseRepo, "org/custom-base");
  assert.equal(reapplyTarget(seeded, status)?.baseRepo, "org/custom-base");
  // A target seeded without it (an older page, or status that named none then) picks it up from status.
  const bare = { repoId: "org/flux-gguf", kind: "gguf" as const, filename: "flux-Q4_K_M.gguf" };
  assert.deepEqual(nativeReloadTarget(bare, status), { ...bare, baseRepo: "org/custom-base" });
  assert.deepEqual(reapplyTarget(bare, status), { ...bare, baseRepo: "org/custom-base" });
  // A base the page chose itself is kept, and a pipeline carries none.
  const own = { ...bare, baseRepo: "org/other-base" };
  assert.equal(reapplyTarget(own, status), own);
  const pipe = { repoId: "org/P", kind: "pipeline" as const };
  assert.equal(reapplyTarget(pipe, { loaded: true, model_kind: "pipeline", repo_id: "org/P", base_repo: "org/x" }), pipe);
  const source = readSrc("features/images/images-page.tsx");
  assert.match(source, /const resident = residentReloadTarget\(\{[\s\S]{0,300}base_repo: status\?\.base_repo,[\s\S]{0,120}\}\);/);
});

test("a server pin never strips options: its native route falls back to diffusers when native cannot run", () => {
  const advanced = { transformer_quant: "fp8", text_encoder_quant: "fp8", text_encoder_file: ["/m/te.safetensors"], vae_file: "/m/vae.safetensors" };
  // The pin ignores the pick, and its fallback must build with the requested options.
  for (const engine of [undefined, "auto", "diffusers", "sd_cpp"] as const) {
    assert.equal(nativeLoadAdvanced(advanced, engine, "sd_cpp", true), advanced, String(engine));
  }
  // An explicit pick (refused, never swapped, when native cannot run) still drops them; a non-GGUF never does.
  assert.equal(nativeLoadAdvanced(advanced, "sd_cpp", null, true).text_encoder_quant, undefined);
  assert.equal(nativeLoadAdvanced(advanced, "sd_cpp", null, false), advanced);
  const source = readSrc("features/images/images-page.tsx");
  assert.equal(source.match(/engineSelection\.current\.forced,\s*sendsEngine\(opts\.kind, opts\.filename\),/g)?.length, 2);
});

test("a background load that fails reseeds the Engine select, like a refused start", () => {
  const source = readSrc("features/images/images-page.tsx");
  assert.match(source, /if \(p\.phase === "error"\) \{[\s\S]{0,1200}setEngineSeed\(""\);[\s\S]{0,200}void refreshStatus\(\);\s*return;/);
});

test("a reload from status keeps the resident GGUF's base repo, on the load and the plan", () => {
  const gguf = { loaded: true, model_kind: "gguf", repo_id: "org/klein-GGUF", gguf_filename: "klein-Q4.gguf", base_repo: "org/custom-base" };
  assert.equal(residentReloadTarget(gguf)?.baseRepo, "org/custom-base");
  // With no last load T has no baseRepo, so name it: the status-built target is the branch that carries one.
  assert.equal(nativeReloadTarget<{ repoId: string; baseRepo?: string }>(null, gguf)?.baseRepo, "org/custom-base");
  assert.equal("baseRepo" in (residentReloadTarget({ ...gguf, base_repo: null }) ?? {}), false);
  const source = readSrc("features/images/images-page.tsx");
  assert.equal(source.match(/base_repo: opts\.baseRepo,/g)?.length, 2);
  assert.equal(source.match(/displayRepoId: l\.displayRepoId, baseRepo: l\.baseRepo \}/g)?.length, 2);
  // Carried into the next Reapply target.
  assert.match(source, /opts = \{ kind: opts\.kind, filename: opts\.filename, displayRepoId: opts\.displayRepoId, baseRepo: opts\.baseRepo \};/);
  assert.match(source, /displayRepoId: opts\.displayRepoId,\s*baseRepo: opts\.baseRepo,/);
});
