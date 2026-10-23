// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import ts from "@typescript/typescript6";
import {
  componentFilesForModel,
  readImageComponents,
  readImageModel,
  rememberImageComponents,
  rememberImageModel,
  sameComponentFiles,
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
  // Same repo and kind, another file: a different model.
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
  // Re-saving an old entry moves it to the front of the line.
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
  // Fields still holding what flux loaded with: picking Z-Image swaps in its own files.
  const fluxFields = { te: "clip_l.safetensors", vae: "flux_vae.safetensors" };
  const fluxOwner = { textEncoderFiles: ["clip_l.safetensors"], vaeFile: "flux_vae.safetensors" };
  const toZ = page(fluxFields, fluxOwner);
  assert.deepEqual(toZ.pick(zImage), zFiles);
  assert.deepEqual(toZ.state, { te: zFiles.textEncoderFiles.join("\n"), vae: zFiles.vaeFile });
  assert.deepEqual(toZ.scope.componentFieldsOwner.current, zFiles);

  // A model with nothing saved clears them rather than inheriting Z-Image's.
  const toFlux = page({ te: zFiles.textEncoderFiles[0], vae: zFiles.vaeFile }, zFiles);
  assert.deepEqual(toFlux.pick(flux), {});
  assert.deepEqual(toFlux.state, { te: "", vae: "" });

  // Paths the user typed since are theirs: the pick sends them and leaves the fields alone.
  const typed = page({ te: "a.safetensors, b.safetensors", vae: "" }, zFiles);
  assert.deepEqual(typed.pick(zImage), { textEncoderFiles: ["a.safetensors", "b.safetensors"], vaeFile: "" });
  assert.deepEqual(typed.state, { te: "a.safetensors, b.safetensors", vae: "" });
});

test("a successful load stores the requested files per model and the page starts on them", () => {
  const source = readSrc("features/images/images-page.tsx");
  assert.match(
    source,
    /rememberImageModel\(remembered\);\s*setRememberedModel\(remembered\);\s*const files = lastLoad\.current\.fieldFiles \?\? lastLoad\.current;\s*rememberImageComponents\(lastLoad\.current, files\);/,
  );
  // The requested files, not the load's: the native engine drops them from the request.
  assert.match(source, /const fieldFiles = componentFileFields\(opts\.kind, picked\.text_encoder_file, picked\.vae_file\);/);
  assert.match(source, /readImageComponents\(rememberedModel\)/);
  // Every load pick routes through the prefill; a download-only pick and a staged snapshot keep theirs.
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
      // handleLoad reads the live fields when nothing is pinned.
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
  // After a restart the fields start on Z-Image's files, but another client left a plain GGUF resident.
  const other = { loaded: true, repo_id: "org/other-GGUF", model_kind: "gguf", gguf_filename: "o.gguf", component_files: {} };
  const toOther = reapply({ te: zFiles.textEncoderFiles.join("\n"), vae: zFiles.vaeFile }, zFiles, other);
  assert.equal(toOther.loads.length, 1);
  assert.equal(toOther.loads[0].repoId, "org/other-GGUF");
  assert.equal(toOther.loads[0].advanced.text_encoder_file, undefined);
  assert.equal(toOther.loads[0].advanced.vae_file, undefined);
  assert.deepEqual(toOther.state, { te: "", vae: "" });

  // The model the files belong to keeps them, and paths the user typed are still sent as typed.
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
  // The fields still hold Z-Image's files: the Load the picker would fire for another GGUF sends none.
  const load = await footprint("load");
  assert.equal(load.text_encoder_file, undefined);
  assert.equal(load.vae_file, undefined);
  assert.equal(load.cpu_offload, false);
  // A download-only pick sends the fields as they are, so its footprint does too.
  const download = await footprint("download");
  assert.deepEqual(download.text_encoder_file, zFiles.textEncoderFiles);
});
