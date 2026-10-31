// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import ts from "@typescript/typescript6";
import {
  componentFieldsOwnerAfterLoad,
  componentFilesForModel,
  forgetStaleImageComponents,
  forgetStaleResidentComponents,
  readImageComponents,
  readImageModel,
  recalledComponentFiles,
  recoveredComponentFields,
  reconcileRememberedComponents,
  rememberImageComponents,
  rememberLoadedImageComponents,
  rememberImageModel,
  sameComponentFiles,
  seedImageComponents,
} from "../src/features/images/image-model-recall.ts";
import { splitComponentFileList } from "../src/features/images/component-files.ts";
import { reapplyTarget } from "../src/features/images/lib/engine-choice.ts";
import { installLocalStorageFake, readSrc } from "./helpers/kit.ts";

const { store } = installLocalStorageFake();

const zImage = { repoId: "/comfy/models/diffusion_models", kind: "gguf", filename: "z.gguf" } as const;
const flux = { repoId: "/comfy/models/diffusion_models", kind: "single_file", filename: "flux.safetensors" } as const;
const zFiles = { textEncoderFiles: ["../text_encoders/qwen_3_4b.safetensors"], vaeFile: "../vae/ae.safetensors" };

test("component files are stored per model and a load with none forgets them", () => {
  store.clear();
  rememberImageComponents(zImage, zFiles);
  rememberImageComponents(flux, { textEncoderFiles: ["clip_l.safetensors", "t5xxl.safetensors"] });
  assert.deepEqual(readImageComponents(zImage), zFiles);
  assert.deepEqual(readImageComponents(flux), { textEncoderFiles: ["clip_l.safetensors", "t5xxl.safetensors"] });
  assert.equal(readImageComponents({ ...zImage, filename: "z-Q8_0.gguf" }), null);
  rememberImageComponents(zImage, {});
  assert.equal(readImageComponents(zImage), null);
  assert.ok(readImageComponents(flux));
});

test("the component map keeps the 50 most recent models and survives bad storage", () => {
  store.clear();
  for (let i = 0; i < 60; i++)
    rememberImageComponents({ repoId: `org/m${i}`, kind: "gguf", filename: "m.gguf" }, { vaeFile: `vae${i}` });
  assert.equal(JSON.parse(store.get("unsloth:images:component-files") ?? "[]").length, 50);
  assert.equal(readImageComponents({ repoId: "org/m9", kind: "gguf", filename: "m.gguf" }), null);
  rememberImageComponents({ repoId: "org/m10", kind: "gguf", filename: "m.gguf" }, { vaeFile: "vae10" });
  rememberImageComponents({ repoId: "org/new", kind: "gguf", filename: "m.gguf" }, { vaeFile: "new" });
  assert.deepEqual(readImageComponents({ repoId: "org/m10", kind: "gguf", filename: "m.gguf" }), { vaeFile: "vae10" });
  assert.equal(readImageComponents({ repoId: "org/m11", kind: "gguf", filename: "m.gguf" }), null);

  store.set("unsloth:images:component-files", "{not json");
  assert.equal(readImageComponents(zImage), null);
  rememberImageComponents(zImage, zFiles);
  assert.deepEqual(readImageComponents(zImage), zFiles);
});

test("the map leaves the last-model slot alone", () => {
  store.clear();
  rememberImageModel({ ...zImage, ...zFiles });
  rememberImageComponents(flux, { vaeFile: "ae.safetensors" });
  assert.deepEqual(readImageModel(), { ...zImage, ...zFiles });
});

test("sameComponentFiles ignores formatting and empty fields", () => {
  assert.ok(sameComponentFiles({}, { textEncoderFiles: [], vaeFile: "" }));
  assert.ok(sameComponentFiles({ textEncoderFiles: ["a"], vaeFile: " v " }, { textEncoderFiles: ["a"], vaeFile: "v" }));
  assert.ok(!sameComponentFiles({ textEncoderFiles: ["a", "b"] }, { textEncoderFiles: ["b", "a"] }));
  assert.ok(!sameComponentFiles(zFiles, {}));
});

test("a model pick fills the Advanced fields from that model, clears another model's, and keeps the user's edits", () => {
  const source = readSrc("features/images/images-page.tsx");
  const tree = ts.createSourceFile("images-page.tsx", source, ts.ScriptTarget.Latest, true, ts.ScriptKind.TSX);
  let declaration = "";
  const visit = (node: ts.Node) => {
    if (ts.isVariableDeclaration(node) && node.name.getText(tree) === "pickComponentFiles")
      declaration = `const ${node.getText(tree)};`;
    ts.forEachChild(node, visit);
  };
  visit(tree);
  assert.ok(declaration);
  const { outputText } = ts.transpileModule(`${declaration}\nreturn pickComponentFiles;`, {
    compilerOptions: { target: ts.ScriptTarget.ES2022 },
  });
  const page = (fields: { te: string; vae: string }, owner: object) => {
    const state = { ...fields };
    const scope = {
      useCallback: (fn: unknown) => fn,
      textEncoderFiles: fields.te,
      vaeFile: fields.vae,
      componentFieldsOwner: { current: owner },
      splitComponentFileList,
      sameComponentFiles,
      readImageComponents,
      componentFilesForModel,
      setTextEncoderFiles: (v: string) => (state.te = v),
      setVaeFile: (v: string) => (state.vae = v),
    };
    const pick = new Function(...Object.keys(scope), outputText)(...Object.values(scope));
    return { state, scope, pick };
  };

  store.clear();
  rememberImageComponents(zImage, zFiles);
  const fluxFields = { te: "clip_l.safetensors", vae: "flux_vae.safetensors" };
  const fluxOwner = { textEncoderFiles: ["clip_l.safetensors"], vaeFile: "flux_vae.safetensors" };
  const toZ = page(fluxFields, fluxOwner);
  assert.deepEqual(toZ.pick(zImage), zFiles);
  assert.deepEqual(toZ.state, { te: zFiles.textEncoderFiles.join("\n"), vae: zFiles.vaeFile });
  assert.deepEqual(toZ.scope.componentFieldsOwner.current, zFiles);

  const toFlux = page({ te: zFiles.textEncoderFiles[0], vae: zFiles.vaeFile }, zFiles);
  assert.deepEqual(toFlux.pick(flux), {});
  assert.deepEqual(toFlux.state, { te: "", vae: "" });

  const typed = page({ te: "a.safetensors, b.safetensors", vae: "" }, zFiles);
  assert.deepEqual(typed.pick(zImage), { textEncoderFiles: ["a.safetensors", "b.safetensors"], vaeFile: "" });
  assert.deepEqual(typed.state, { te: "a.safetensors, b.safetensors", vae: "" });
});

test("a successful load stores the requested files per model and the page starts on them", () => {
  const source = readSrc("features/images/images-page.tsx");
  assert.match(
    source,
    /rememberImageModel\(remembered\);\s*setRememberedModel\(remembered\);\s*const files = rememberLoadedImageComponents\(target, loaded\);/,
  );
  assert.match(source, /const fieldFiles = componentFileFields\(opts\.kind, picked\.text_encoder_file, picked\.vae_file\);/);
  assert.match(source, /recalledComponentFiles\(rememberedModel\)/);
  assert.match(source, /downloadSnapshot \|\| downloadOnly \? null : pickComponentFiles\(\{ repoId, \.\.\.opts \}\)/);
});

test("Reapply of a resident model sends that model's files, never the fields another model filled in", () => {
  const source = readSrc("features/images/images-page.tsx");
  const tree = ts.createSourceFile("images-page.tsx", source, ts.ScriptTarget.Latest, true, ts.ScriptKind.TSX);
  const declarations: Record<string, string> = {};
  const visit = (node: ts.Node) => {
    if (ts.isVariableDeclaration(node) && ["pickComponentFiles", "handleReapply"].includes(node.name.getText(tree)))
      declarations[node.name.getText(tree)] = `const ${node.getText(tree)};`;
    ts.forEachChild(node, visit);
  };
  visit(tree);
  assert.ok(declarations.pickComponentFiles && declarations.handleReapply);
  const { outputText } = ts.transpileModule(
    `${declarations.pickComponentFiles}\n${declarations.handleReapply}\nreturn handleReapply;`,
    { compilerOptions: { target: ts.ScriptTarget.ES2022 } },
  );
  const reapply = (fields: { te: string; vae: string }, owner: object, status: object) => {
    const loads: { repoId: string; advanced: Record<string, unknown> }[] = [];
    const state = { ...fields };
    const currentLoadAdvanced = () => ({
      cpu_offload: false,
      text_encoder_file: splitComponentFileList(state.te).length ? splitComponentFileList(state.te) : undefined,
      vae_file: state.vae.trim() || undefined,
    });
    const scope = {
      useCallback: (fn: unknown) => fn,
      textEncoderFiles: fields.te,
      vaeFile: fields.vae,
      componentFieldsOwner: { current: owner },
      splitComponentFileList,
      sameComponentFiles,
      readImageComponents,
      componentFilesForModel,
      setTextEncoderFiles: (v: string) => (state.te = v),
      setVaeFile: (v: string) => (state.vae = v),
      lastLoad: { current: null },
      status,
      engineChoice: "auto",
      reapplyTarget,
      reapplyChangesEngine: () => false,
      reloadSource: () => "hub",
      loadOrStage: () => assert.fail("no engine change here"),
      currentLoadAdvanced,
      handleLoad: (repoId: string, _opts: unknown, pinned?: Record<string, unknown>) => {
        loads.push({ repoId, advanced: pinned ?? currentLoadAdvanced() });
        return Promise.resolve(true);
      },
    };
    new Function(...Object.keys(scope), outputText)(...Object.values(scope))();
    return { loads, state };
  };

  store.clear();
  rememberImageComponents(zImage, zFiles);
  const other = { loaded: true, repo_id: "org/other-GGUF", model_kind: "gguf", gguf_filename: "o.gguf", component_files: {} };
  const toOther = reapply({ te: zFiles.textEncoderFiles.join("\n"), vae: zFiles.vaeFile }, zFiles, other);
  assert.equal(toOther.loads.length, 1);
  assert.equal(toOther.loads[0].repoId, "org/other-GGUF");
  assert.equal(toOther.loads[0].advanced.text_encoder_file, undefined);
  assert.equal(toOther.loads[0].advanced.vae_file, undefined);
  assert.deepEqual(toOther.state, { te: "", vae: "" });

  const resident = { loaded: true, repo_id: zImage.repoId, model_kind: "gguf", gguf_filename: zImage.filename, component_files: {} };
  const toZ = reapply({ te: zFiles.textEncoderFiles.join("\n"), vae: zFiles.vaeFile }, zFiles, resident);
  assert.deepEqual(toZ.loads[0].advanced.text_encoder_file, zFiles.textEncoderFiles);
  assert.equal(toZ.loads[0].advanced.vae_file, zFiles.vaeFile);
  const typed = reapply({ te: "mine.safetensors", vae: "" }, zFiles, other);
  assert.deepEqual(typed.loads[0].advanced.text_encoder_file, ["mine.safetensors"]);
});

test("a Load-mode footprint plans the shown model with its own files, a Download-mode one with the fields", async () => {
  const source = readSrc("features/images/images-page.tsx");
  const tree = ts.createSourceFile("images-page.tsx", source, ts.ScriptTarget.Latest, true, ts.ScriptKind.TSX);
  let declaration = "";
  const visit = (node: ts.Node) => {
    if (ts.isVariableDeclaration(node) && node.name.getText(tree) === "resolveDownloadFootprint")
      declaration = `const ${node.getText(tree)};`;
    ts.forEachChild(node, visit);
  };
  visit(tree);
  assert.ok(declaration);
  const { outputText } = ts.transpileModule(`${declaration}\nreturn resolveDownloadFootprint;`, {
    compilerOptions: { target: ts.ScriptTarget.ES2022 },
  });
  const footprint = async (action: "load" | "download") => {
    const fields = { te: zFiles.textEncoderFiles.join("\n"), vae: zFiles.vaeFile };
    const planned: Record<string, unknown>[] = [];
    const scope = {
      useCallback: (fn: unknown) => fn,
      textEncoderFiles: fields.te,
      vaeFile: fields.vae,
      componentFieldsOwner: { current: zFiles },
      modelSelectionAction: action,
      splitComponentFileList,
      readImageComponents,
      componentFilesForModel,
      currentLoadAdvanced: () => ({
        cpu_offload: false,
        text_encoder_file: splitComponentFileList(fields.te),
        vae_file: fields.vae,
      }),
      requestDownloadPlan: (_repo: string, _opts: unknown, picked: Record<string, unknown>) => {
        planned.push(picked);
        return Promise.resolve({ required_bytes: 1, checkpoint_bytes: 1 });
      },
      pickGuard: null,
      stage: null,
    };
    const resolve = new Function(...Object.keys(scope), outputText)(...Object.values(scope));
    await resolve("org/other-GGUF", { ggufFilename: "o.gguf" });
    return planned[0];
  };

  store.clear();
  rememberImageComponents(zImage, zFiles);
  const load = await footprint("load");
  assert.equal(load.text_encoder_file, undefined);
  assert.equal(load.vae_file, undefined);
  assert.equal(load.cpu_offload, false);
  const download = await footprint("download");
  assert.deepEqual(download.text_encoder_file, zFiles.textEncoderFiles);
});

test("a recalled model's files come from the per-model map, the last-model slot only as a fallback", () => {
  store.clear();
  rememberImageComponents(zImage, zFiles);
  assert.deepEqual(recalledComponentFiles({ ...zImage }), zFiles);
  assert.deepEqual(recalledComponentFiles({ ...flux, vaeFile: "ae.safetensors" }), { vaeFile: "ae.safetensors" });
  assert.deepEqual(recalledComponentFiles({ ...flux }), {});

  const source = readSrc("features/images/images-page.tsx");
  const recall = source.slice(source.indexOf("const handleGenerateWithRecall = useCallback"));
  const pinnedFiles = recall.slice(0, recall.indexOf("if (!started) pendingRecalledGeneration.current = null;"));
  assert.match(pinnedFiles, /const recalled = recalledComponentFiles\(rememberedModel\);/);
  assert.match(pinnedFiles, /text_encoder_file: recalled\.textEncoderFiles,\s*vae_file: recalled\.vaeFile,/);
});

test("a finished load owns the fields only while they show its files", () => {
  assert.deepEqual(componentFieldsOwnerAfterLoad(zFiles, zFiles, {}, "gguf"), zFiles);
  assert.deepEqual(componentFieldsOwnerAfterLoad(zFiles, {}, zFiles, "pipeline"), zFiles);
  assert.deepEqual(componentFieldsOwnerAfterLoad(zFiles, {}, zFiles, "single_file"), zFiles);
  const source = readSrc("features/images/images-page.tsx");
  assert.match(
    source,
    /componentFieldsOwner\.current = componentFieldsOwnerAfterLoad\(componentFieldsShown\.current, files, componentFieldsOwner\.current, target\.kind\);/,
  );
});

test("a load that ignores the fields keeps a cleared field the user's edit", () => {
  const shown = { vaeFile: zFiles.vaeFile };
  assert.deepEqual(componentFieldsOwnerAfterLoad(shown, {}, zFiles, "pipeline"), zFiles);
  rememberImageComponents(flux, { textEncoderFiles: ["../text_encoders/t5.safetensors"], vaeFile: "../vae/flux_ae.safetensors" });
  const owner = componentFieldsOwnerAfterLoad(shown, {}, zFiles, "pipeline");
  assert.deepEqual(componentFilesForModel(shown, owner, flux).files, { vaeFile: "../vae/flux_ae.safetensors" });
  assert.deepEqual(componentFieldsOwnerAfterLoad(shown, {}, zFiles, "single_file"), { vaeFile: zFiles.vaeFile });
});

test("a Load anyway retry keeps the requested files, which a native load strips from what it sends", () => {
  const source = readSrc("features/images/images-page.tsx");
  const start = source.indexOf("const retryOversized = retryWhileCurrent(");
  const retry = source.slice(start, source.indexOf("lastLoadAttempt.current = {", start));
  assert.match(retry, /retryLoadRef\.current\?\.\(repoId, opts, \{ \.\.\.picked, allow_oversized_load: true \}\)/);
});

test("an upgrade seeds the map from the last-model slot, so picking that model keeps its paths", () => {
  store.clear();
  rememberImageModel({ ...zImage, ...zFiles });
  const remembered = readImageModel()!;
  const initial = seedImageComponents(remembered);
  assert.deepEqual(initial, zFiles);
  assert.deepEqual(readImageComponents(zImage), zFiles);
  assert.deepEqual(componentFilesForModel(zFiles, initial, zImage).files, zFiles);
  assert.deepEqual(componentFilesForModel(zFiles, initial, flux).files, {});
  rememberImageComponents(zImage, { vaeFile: "newer.safetensors" });
  assert.deepEqual(seedImageComponents(remembered), { vaeFile: "newer.safetensors" });
  assert.deepEqual(readImageComponents(zImage), { vaeFile: "newer.safetensors" });
  store.clear();
  assert.deepEqual(seedImageComponents({ ...flux }), {});
  assert.equal(store.get("unsloth:images:component-files"), undefined);

  const source = readSrc("features/images/images-page.tsx");
  assert.match(source, /useState<ImageComponentFiles>\(\(\) =>\s*rememberedModel \? seedImageComponents\(rememberedModel\) : \{\},/);
});

test("a resident build with other components than the map's forgets the entry; a matching one keeps it", () => {
  store.clear();
  rememberImageComponents(zImage, zFiles);
  rememberImageComponents(flux, { vaeFile: "../vae/flux_ae.safetensors" });
  forgetStaleImageComponents(zImage, { engine: "sd_cpp", component_files: null });
  assert.deepEqual(readImageComponents(zImage), zFiles);
  forgetStaleImageComponents(zImage, { engine: "diffusers", component_files: {} });
  assert.equal(readImageComponents(zImage), null);
  forgetStaleImageComponents(flux, { component_files: { vae: "flux_ae.safetensors" } });
  assert.deepEqual(readImageComponents(flux), { vaeFile: "../vae/flux_ae.safetensors" });
  forgetStaleImageComponents(flux, { component_files: { vae: "other.safetensors" } });
  assert.equal(readImageComponents(flux), null);

  const source = readSrc("features/images/images-page.tsx");
  const recall = source.slice(source.indexOf("const handleGenerateWithRecall = useCallback"));
  assert.ok(recall.slice(0, recall.indexOf("if (!rememberedModel || !prompt.trim())")).includes("forgetStaleImageComponents(model, status);"));
});

test("a resumed load the page did not start keeps the saved files it engages; a page load stores its request", () => {
  store.clear();
  rememberImageComponents(zImage, zFiles);
  const resumed = { ...zImage, kind: "gguf" as const };
  const engaged = { engine: "diffusers", component_files: { text_encoder: "qwen_3_4b.safetensors", vae: "ae.safetensors" } };
  assert.equal(rememberLoadedImageComponents(resumed, engaged), null);
  assert.deepEqual(readImageComponents(zImage), zFiles);
  assert.equal(rememberLoadedImageComponents(resumed, { engine: "sd_cpp", component_files: null }), null);
  assert.deepEqual(readImageComponents(zImage), zFiles);
  assert.equal(rememberLoadedImageComponents(resumed, { engine: "diffusers", component_files: {} }), null);
  assert.equal(readImageComponents(zImage), null);
  assert.deepEqual(rememberLoadedImageComponents({ ...zImage, fieldFiles: zFiles }, { component_files: {} }), zFiles);
  assert.deepEqual(readImageComponents(zImage), zFiles);
  assert.deepEqual(rememberLoadedImageComponents({ ...zImage, fieldFiles: {} }, { component_files: {} }), {});
  assert.equal(readImageComponents(zImage), null);

  const source = readSrc("features/images/images-page.tsx");
  assert.match(
    source,
    /const files = rememberLoadedImageComponents\(target, loaded\);\s*if \(files && recovered\) \{[\s\S]*?\}\s*if \(files\) componentFieldsOwner\.current = componentFieldsOwnerAfterLoad\(/,
  );
});

test("the encoder and VAE fields are owned separately, so editing one does not carry the other to another model", () => {
  store.clear();
  const fluxFiles = { textEncoderFiles: ["clip_l.safetensors", "t5xxl.safetensors"], vaeFile: "flux_ae.safetensors" };
  rememberImageComponents(flux, fluxFiles);
  rememberImageComponents(zImage, zFiles);
  const typed = { textEncoderFiles: zFiles.textEncoderFiles, vaeFile: "mine.safetensors" };
  const pick = componentFilesForModel(typed, zFiles, flux);
  assert.deepEqual(pick.files, { textEncoderFiles: fluxFiles.textEncoderFiles, vaeFile: "mine.safetensors" });
  assert.deepEqual(pick.owner, { textEncoderFiles: fluxFiles.textEncoderFiles, vaeFile: zFiles.vaeFile });
  assert.deepEqual(componentFilesForModel(pick.files, pick.owner, zImage).files, {
    textEncoderFiles: zFiles.textEncoderFiles,
    vaeFile: "mine.safetensors",
  });
  assert.deepEqual(componentFilesForModel(zFiles, zFiles, flux).files, fluxFiles);
  assert.deepEqual(componentFilesForModel({ textEncoderFiles: ["a"], vaeFile: "b" }, zFiles, flux).files, {
    textEncoderFiles: ["a"],
    vaeFile: "b",
  });
  assert.deepEqual(
    componentFieldsOwnerAfterLoad(
      { textEncoderFiles: zFiles.textEncoderFiles, vaeFile: "edited-during-load" },
      zFiles,
      { vaeFile: "old" },
      "gguf",
    ),
    { textEncoderFiles: zFiles.textEncoderFiles, vaeFile: "old" },
  );
});

test("the first loaded status forgets the remembered model's stale files and clears the untouched fields", () => {
  store.clear();
  rememberImageComponents(zImage, zFiles);
  const resident = { loaded: true, repo_id: zImage.repoId, model_kind: "gguf", gguf_filename: zImage.filename };
  rememberImageModel({ ...zImage, ...zFiles, familyOverride: "z-image" });
  assert.deepEqual(reconcileRememberedComponents({ ...zImage, ...zFiles, familyOverride: "z-image" }, { ...resident, engine: "diffusers", component_files: {} }), {
    ...zImage,
    familyOverride: "z-image",
  });
  assert.equal(readImageComponents(zImage), null);
  assert.deepEqual(readImageModel(), { ...zImage, familyOverride: "z-image" });
  assert.deepEqual(seedImageComponents(readImageModel()!), {});
  assert.deepEqual(recalledComponentFiles(readImageModel()!), {});
  rememberImageComponents(zImage, zFiles);
  rememberImageModel({ ...flux, vaeFile: "flux_ae.safetensors" });
  assert.ok(reconcileRememberedComponents(zImage, { ...resident, engine: "diffusers", component_files: {} }));
  assert.deepEqual(readImageModel(), { ...flux, vaeFile: "flux_ae.safetensors" });
  const kept: Parameters<typeof reconcileRememberedComponents>[1][] = [
    { ...resident, component_files: { text_encoder: "qwen_3_4b.safetensors", vae: "ae.safetensors" } },
    { ...resident, engine: "sd_cpp", component_files: null },
    { ...resident, gguf_filename: "z-Q8_0.gguf", component_files: {} },
    { loaded: false },
  ];
  for (const status of kept) {
    rememberImageComponents(zImage, zFiles);
    assert.equal(reconcileRememberedComponents(zImage, status), null);
    assert.deepEqual(readImageComponents(zImage), zFiles);
  }

  const source = readSrc("features/images/images-page.tsx");
  const tree = ts.createSourceFile("images-page.tsx", source, ts.ScriptTarget.Latest, true, ts.ScriptKind.TSX);
  const declarations: Record<string, string> = {};
  const visit = (node: ts.Node) => {
    if (ts.isVariableDeclaration(node) && ["pickComponentFiles", "reconcileInitialComponents"].includes(node.name.getText(tree)))
      declarations[node.name.getText(tree)] = `const ${node.getText(tree)};`;
    ts.forEachChild(node, visit);
  };
  visit(tree);
  assert.ok(declarations.reconcileInitialComponents);
  const { outputText } = ts.transpileModule(
    `${declarations.pickComponentFiles}\n${declarations.reconcileInitialComponents}\nreturn reconcileInitialComponents;`,
    { compilerOptions: { target: ts.ScriptTarget.ES2022 } },
  );
  const open = (fields: { te: string; vae: string }, status: object, initialLoadProbe = "idle") => {
    const state: { te: string; vae: string; remembered?: unknown } = { ...fields };
    const scope = {
      useCallback: (fn: unknown) => fn,
      textEncoderFiles: fields.te,
      vaeFile: fields.vae,
      componentFieldsOwner: { current: zFiles },
      componentsReconciled: { current: false },
      rememberedModel: { ...zImage, ...zFiles },
      initialLoadProbe,
      setRememberedModel: (m: unknown) => (state.remembered = m),
      splitComponentFileList,
      componentFilesForModel,
      reconcileRememberedComponents,
      setTextEncoderFiles: (v: string) => (state.te = v),
      setVaeFile: (v: string) => (state.vae = v),
    };
    const reconcile = new Function(...Object.keys(scope), outputText)(...Object.values(scope));
    reconcile(status);
    reconcile(status);
    return { state, scope };
  };
  const stale = { ...resident, engine: "diffusers", component_files: {} };
  store.clear();
  rememberImageComponents(zImage, zFiles);
  const untouched = open({ te: zFiles.textEncoderFiles[0], vae: zFiles.vaeFile }, stale);
  assert.deepEqual(untouched.state, { te: "", vae: "", remembered: { ...zImage } });
  assert.deepEqual(untouched.scope.componentFieldsOwner.current, {});
  assert.equal(untouched.scope.componentsReconciled.current, true);
  rememberImageComponents(zImage, zFiles);
  const edited = open({ te: zFiles.textEncoderFiles[0], vae: "mine.safetensors" }, stale);
  assert.deepEqual(edited.state, { te: "", vae: "mine.safetensors", remembered: { ...zImage } });
  rememberImageComponents(zImage, zFiles);
  const midReload = open({ te: zFiles.textEncoderFiles[0], vae: zFiles.vaeFile }, stale, "loading");
  assert.deepEqual(midReload.state, { te: zFiles.textEncoderFiles[0], vae: zFiles.vaeFile });
  assert.deepEqual(readImageComponents(zImage), zFiles);
  assert.equal(midReload.scope.componentsReconciled.current, true);
  const pending = open({ te: zFiles.textEncoderFiles[0], vae: zFiles.vaeFile }, stale, "pending");
  assert.deepEqual(readImageComponents(zImage), zFiles);
  assert.equal(pending.scope.componentsReconciled.current, false);
  assert.match(source, /setInitialLoadProbe\(inFlight \? "loading" : "idle"\)/);
  assert.match(
    source,
    /useEffect\(\(\) => \{\s*if \(!status\?\.loaded\) return;\s*reconcileInitialComponents\(status\);\s*reconcileResidentComponents\(status\);\s*\}, \[status, reconcileInitialComponents, reconcileResidentComponents\]\);/,
  );
});

test("every settled loaded status forgets the resident model's stale files, not only the remembered model's", () => {
  store.clear();
  rememberImageModel({ ...zImage, ...zFiles });
  rememberImageComponents(zImage, zFiles);
  rememberImageComponents(flux, { vaeFile: "../vae/flux_ae.safetensors" });
  const fluxResident = {
    loaded: true,
    repo_id: flux.repoId,
    model_kind: "single_file",
    gguf_filename: flux.filename,
    engine: "diffusers",
    component_files: {},
  };
  assert.equal(reconcileRememberedComponents(zImage, fluxResident), null);
  assert.deepEqual(readImageComponents(flux), { vaeFile: "../vae/flux_ae.safetensors" });
  forgetStaleResidentComponents(fluxResident);
  assert.equal(readImageComponents(flux), null);
  assert.deepEqual(componentFilesForModel(zFiles, zFiles, flux).files, {});
  assert.deepEqual(readImageComponents(zImage), zFiles);
  assert.deepEqual(readImageModel(), { ...zImage, ...zFiles });
  rememberImageModel({ ...flux, vaeFile: "../vae/flux_ae.safetensors", familyOverride: "flux" });
  rememberImageComponents(flux, { vaeFile: "../vae/flux_ae.safetensors" });
  assert.deepEqual(forgetStaleResidentComponents(fluxResident), { ...flux, familyOverride: "flux" });
  assert.deepEqual(readImageModel(), { ...flux, familyOverride: "flux" });
  assert.deepEqual(recalledComponentFiles(readImageModel()!), {});
  rememberImageModel({ ...flux, vaeFile: "../vae/flux_ae.safetensors" });
  assert.ok(forgetStaleResidentComponents(fluxResident));
  assert.deepEqual(readImageModel(), { ...flux });
  for (const status of [
    { ...fluxResident, component_files: { vae: "flux_ae.safetensors" } },
    { ...fluxResident, engine: "sd_cpp", component_files: null },
  ]) {
    rememberImageModel({ ...flux, vaeFile: "../vae/flux_ae.safetensors" });
    assert.equal(forgetStaleResidentComponents(status), null);
    assert.deepEqual(readImageModel(), { ...flux, vaeFile: "../vae/flux_ae.safetensors" });
  }
  rememberImageModel({ ...zImage, ...zFiles });
  const kept = [
    { ...fluxResident, component_files: { vae: "flux_ae.safetensors" } },
    { ...fluxResident, engine: "sd_cpp", component_files: null },
    { loaded: false },
  ];
  for (const status of kept) {
    rememberImageComponents(flux, { vaeFile: "../vae/flux_ae.safetensors" });
    forgetStaleResidentComponents(status);
    assert.deepEqual(readImageComponents(flux), { vaeFile: "../vae/flux_ae.safetensors" });
  }
  const pipe = { repoId: "black-forest-labs/FLUX.1-dev", kind: "pipeline" } as const;
  rememberImageComponents(pipe, { vaeFile: "/vae/ae.safetensors" });
  forgetStaleResidentComponents({ loaded: true, repo_id: pipe.repoId, model_kind: "pipeline", engine: "diffusers", component_files: {} });
  assert.equal(readImageComponents(pipe), null);

  const source = readSrc("features/images/images-page.tsx");
  const tree = ts.createSourceFile("images-page.tsx", source, ts.ScriptTarget.Latest, true, ts.ScriptKind.TSX);
  let declaration = "";
  const visit = (node: ts.Node) => {
    if (ts.isVariableDeclaration(node) && node.name.getText(tree) === "reconcileResidentComponents")
      declaration = `const ${node.getText(tree)};`;
    ts.forEachChild(node, visit);
  };
  visit(tree);
  assert.ok(declaration);
  const { outputText } = ts.transpileModule(`${declaration}\nreturn reconcileResidentComponents;`, {
    compilerOptions: { target: ts.ScriptTarget.ES2022 },
  });
  let remembered: unknown;
  const run = (initialLoadProbe: string, busy: string | null) => {
    rememberImageComponents(flux, { vaeFile: "../vae/flux_ae.safetensors" });
    const setRememberedModel = (m: unknown) => (remembered = m);
    const scope = {
      useCallback: (fn: unknown) => fn,
      residentReconciledFor: { current: null },
      initialLoadProbe,
      busy,
      forgetStaleResidentComponents,
      setRememberedModel,
    };
    new Function(...Object.keys(scope), outputText)(...Object.values(scope))(fluxResident);
    return readImageComponents(flux);
  };
  rememberImageModel({ ...flux, vaeFile: "../vae/flux_ae.safetensors" });
  assert.equal(run("idle", null), null);
  assert.deepEqual(remembered, { ...flux });
  remembered = undefined;
  rememberImageModel({ ...zImage, ...zFiles });
  assert.equal(run("idle", null), null);
  assert.equal(run("loading", null), null);
  assert.equal(run("idle", "generating"), null);
  assert.deepEqual(run("pending", null), { vaeFile: "../vae/flux_ae.safetensors" });
  assert.deepEqual(run("idle", "loading"), { vaeFile: "../vae/flux_ae.safetensors" });
  assert.equal(remembered, undefined);
});

test("a field edit reruns no resident cleanup against a cached status, so another tab's newer files survive", () => {
  const source = readSrc("features/images/images-page.tsx");
  const tree = ts.createSourceFile("images-page.tsx", source, ts.ScriptTarget.Latest, true, ts.ScriptKind.TSX);
  const names = new Set(["pickComponentFiles", "reconcileInitialComponents", "residentReconciledFor", "reconcileResidentComponents"]);
  const declarations: string[] = [];
  let effect = "";
  const visit = (node: ts.Node) => {
    if (ts.isVariableDeclaration(node) && names.has(node.name.getText(tree))) declarations.push(`const ${node.getText(tree)};`);
    if (ts.isExpressionStatement(node) && /^useEffect\(/.test(node.getText(tree)) && node.getText(tree).includes("reconcileResidentComponents(status)"))
      effect = node.getText(tree);
    ts.forEachChild(node, visit);
  };
  visit(tree);
  assert.ok(declarations.length >= 3 && effect);
  const { outputText } = ts.transpileModule([...declarations, effect].join("\n"), {
    compilerOptions: { target: ts.ScriptTarget.ES2022 },
  });
  store.clear();
  const old = { vaeFile: "/files/old-ae.safetensors" };
  rememberImageModel({ ...zImage, ...old });
  rememberImageComponents(zImage, old);
  const statusFor = (component_files: Record<string, string>) => ({
    loaded: true,
    repo_id: zImage.repoId,
    model_kind: zImage.kind,
    gguf_filename: zImage.filename,
    engine: "diffusers",
    component_files,
  });
  const hooks: { value: unknown; deps?: unknown[] }[] = [];
  let hook = 0;
  let effectDeps: unknown[] | undefined;
  const same = (a: unknown[] | undefined, b: unknown[]) => !!a && a.length === b.length && a.every((v, i) => Object.is(v, b[i]));
  const scope = {
    useRef: (value: unknown) => {
      const i = hook++;
      hooks[i] ??= { value: { current: value } };
      return hooks[i].value;
    },
    useCallback: (fn: unknown, deps: unknown[]) => {
      const i = hook++;
      if (!same(hooks[i]?.deps, deps)) hooks[i] = { value: fn, deps };
      return hooks[i].value;
    },
    useEffect: (fn: () => void, deps: unknown[]) => {
      if (same(effectDeps, deps)) return;
      effectDeps = deps;
      fn();
    },
    textEncoderFiles: "",
    vaeFile: old.vaeFile,
    componentFieldsOwner: { current: old },
    splitComponentFileList,
    componentFilesForModel,
    setTextEncoderFiles: () => {},
    setVaeFile: () => {},
    componentsReconciled: { current: false },
    initialLoadProbe: "idle",
    busy: null,
    rememberedModel: { ...zImage, ...old },
    setRememberedModel: () => {},
    reconcileRememberedComponents,
    forgetStaleResidentComponents,
    status: statusFor({ vae: "old-ae.safetensors" }) as object,
  };
  const render = () => {
    hook = 0;
    new Function(...Object.keys(scope), outputText)(...Object.values(scope));
  };
  render();
  assert.deepEqual(readImageComponents(zImage), old);
  const newer = { vaeFile: "/files/new-ae.safetensors" };
  rememberImageModel({ ...zImage, ...newer });
  rememberLoadedImageComponents({ ...zImage, fieldFiles: newer }, statusFor({ vae: "new-ae.safetensors" }));
  scope.textEncoderFiles = "/files/encoder.safetensors";
  render();
  assert.deepEqual(readImageComponents(zImage), newer);
  assert.equal(readImageModel()?.vaeFile, newer.vaeFile);
  scope.status = statusFor({ vae: "other-ae.safetensors" });
  render();
  assert.equal(readImageComponents(zImage), null);
  assert.equal(readImageModel()?.vaeFile, undefined);
});

test("a reload that changed one override keeps the other's saved path", () => {
  const resident = { loaded: true, repo_id: zImage.repoId, model_kind: "gguf", gguf_filename: zImage.filename, engine: "diffusers" };
  const vaeOnly = { ...resident, component_files: { vae: "ae.safetensors" } };
  store.clear();
  rememberImageComponents(zImage, zFiles);
  forgetStaleImageComponents(zImage, vaeOnly);
  assert.deepEqual(readImageComponents(zImage), { vaeFile: zFiles.vaeFile });
  rememberImageComponents(zImage, zFiles);
  forgetStaleImageComponents(zImage, { ...resident, component_files: { text_encoder: "qwen_3_4b.safetensors", vae: "x.safetensors" } });
  assert.deepEqual(readImageComponents(zImage), { textEncoderFiles: zFiles.textEncoderFiles });

  store.clear();
  rememberImageComponents(zImage, zFiles);
  rememberImageModel({ ...zImage, ...zFiles });
  assert.deepEqual(reconcileRememberedComponents({ ...zImage, ...zFiles }, vaeOnly), { ...zImage, vaeFile: zFiles.vaeFile });
  assert.deepEqual(readImageModel(), { ...zImage, vaeFile: zFiles.vaeFile });
  assert.deepEqual(recalledComponentFiles(readImageModel()!), { vaeFile: zFiles.vaeFile });

  store.clear();
  rememberImageModel({ ...zImage, ...zFiles });
  assert.deepEqual(forgetStaleResidentComponents(vaeOnly), { ...zImage, vaeFile: zFiles.vaeFile });
  assert.deepEqual(readImageModel(), { ...zImage, vaeFile: zFiles.vaeFile });
});

test("a recovered pending load refills the Advanced fields the user has not touched since reopening", () => {
  const source = readSrc("features/images/images-page.tsx");
  const tree = ts.createSourceFile("images-page.tsx", source, ts.ScriptTarget.Latest, true, ts.ScriptKind.TSX);
  let declaration = "";
  const visit = (node: ts.Node) => {
    if (ts.isVariableDeclaration(node) && node.name.getText(tree) === "rememberFinishedLoad")
      declaration = `const ${node.getText(tree)};`;
    ts.forEachChild(node, visit);
  };
  visit(tree);
  assert.ok(declaration);
  const { outputText } = ts.transpileModule(`${declaration}\nreturn rememberFinishedLoad;`, {
    compilerOptions: { target: ts.ScriptTarget.ES2022 },
  });
  const loaded = {
    loaded: true,
    repo_id: zImage.repoId,
    model_kind: zImage.kind,
    gguf_filename: zImage.filename,
    engine: "diffusers",
    component_files: { text_encoder: "qwen_3_4b.safetensors", vae: "ae.safetensors" },
  };
  const reopen = (shown: { textEncoderFiles?: string[]; vaeFile?: string }, owner: object, recovered = true) => {
    const state = { te: (shown.textEncoderFiles ?? []).join("\n"), vae: shown.vaeFile ?? "" };
    const scope = {
      useCallback: (fn: unknown) => fn,
      matchesRememberedModel: () => true,
      withEngagedFamily: (target: object) => target,
      rememberImageModel,
      setRememberedModel: () => {},
      rememberLoadedImageComponents,
      componentFieldsOwnerAfterLoad,
      recoveredComponentFields,
      componentFieldsShown: { current: shown },
      componentFieldsOwner: { current: owner },
      setTextEncoderFiles: (v: string) => (state.te = v),
      setVaeFile: (v: string) => (state.vae = v),
    };
    const remember = new Function(...Object.keys(scope), outputText)(...Object.values(scope));
    remember({ ...zImage, ...zFiles, fieldFiles: zFiles }, loaded, recovered);
    return { state, scope };
  };
  store.clear();
  const fresh = reopen({}, {});
  assert.deepEqual(fresh.state, { te: zFiles.textEncoderFiles.join("\n"), vae: zFiles.vaeFile });
  assert.deepEqual(fresh.scope.componentFieldsOwner.current, zFiles);
  const old = { textEncoderFiles: ["old-te.safetensors"], vaeFile: "old-ae.safetensors" };
  assert.deepEqual(reopen(old, old).state, { te: zFiles.textEncoderFiles.join("\n"), vae: zFiles.vaeFile });
  const edited = reopen({ textEncoderFiles: old.textEncoderFiles, vaeFile: "mine.safetensors" }, old);
  assert.deepEqual(edited.state, { te: zFiles.textEncoderFiles.join("\n"), vae: "mine.safetensors" });
  assert.deepEqual(edited.scope.componentFieldsOwner.current, { textEncoderFiles: zFiles.textEncoderFiles, vaeFile: old.vaeFile });
  assert.deepEqual(reopen({}, {}, false).state, { te: "", vae: "" });
});
