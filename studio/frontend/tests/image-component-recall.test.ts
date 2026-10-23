// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import ts from "@typescript/typescript6";
import {
  readImageComponents,
  readImageModel,
  rememberImageComponents,
  rememberImageModel,
  sameComponentFiles,
} from "../src/features/images/image-model-recall.ts";
import { splitComponentFileList } from "../src/features/images/component-files.ts";
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
