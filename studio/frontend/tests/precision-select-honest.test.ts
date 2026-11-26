// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The Precision select tells the truth on Images and Video: it snaps back after a refused or failed load, greys out
// what cannot load (with the reason), the video header names the loaded precision, and Restore brings the clip's
// or image's precision back and offers its model as a reload.

import assert from "node:assert/strict";
import test from "node:test";
import ts from "@typescript/typescript6";

import {
  pickFamily,
  precisionAdvisoryReasons,
  precisionDisabledReasons,
  precisionUnavailableReason,
} from "../src/lib/precision-availability.ts";
import {
  recipeModelToReload,
  recipeTextEncoderQuant,
  recipeTransformerQuant,
  seededTextEncoderQuant,
  seededTransformerQuant,
} from "../src/lib/recipe-precision.ts";
import { readSrc } from "./helpers/kit.ts";

const WAN = "wan2.2-ti2v-5b";
const WAN_REASON =
  "'nvfp4' is ruled out for family 'wan2.2-ti2v-5b': it was measured and failed the NVFP4 accuracy gate (LPIPS 0.29)";
const DENIALS = { [WAN]: { nvfp4: WAN_REASON } };
const OPTIONS: [string, string][] = [
  ["auto", "Auto"],
  ["none", "Off"],
  ["fp8", "FP8"],
  ["int8", "INT8"],
  ["nvfp4", "NVFP4"],
  ["mxfp8", "MXFP8"],
];

test("a family deny greys out that scheme with the backend's reason", () => {
  const ctx = { denseQuantSupported: true, denseQuantSchemes: ["int8", "fp8", "mxfp8"], family: WAN, denials: DENIALS };
  assert.deepEqual(precisionDisabledReasons(OPTIONS, ctx), { nvfp4: WAN_REASON });
  // Another family keeps every option.
  assert.deepEqual(precisionDisabledReasons(OPTIONS, { ...ctx, family: "ltx-2" }), {});
  // Unknown family: nothing is family-denied.
  assert.deepEqual(precisionDisabledReasons(OPTIONS, { ...ctx, family: null }), {});
  // The loaded model's denials as advice: the reason without any host ladder check.
  assert.deepEqual(precisionAdvisoryReasons(OPTIONS, "Wan2.2-TI2V-5B", DENIALS), { nvfp4: WAN_REASON });
  assert.deepEqual(precisionAdvisoryReasons(OPTIONS, null, DENIALS), {});
});

test("a ladder scheme the host did not report is disabled, an empty or non-CUDA list never disables", () => {
  // A Windows card whose fp8 smoke probe failed reports only int8.
  const windows = { denseQuantSupported: true, denseQuantSchemes: ["int8"], family: null, denials: null };
  const reasons = precisionDisabledReasons(OPTIONS, windows);
  assert.deepEqual(Object.keys(reasons).sort(), ["fp8", "mxfp8"]);
  assert.match(reasons.fp8, /FP8/);
  // nvfp4 is never in the ladder: its own switch decides, not this list.
  assert.equal(reasons.nvfp4, undefined);
  // MPS / CPU / an older backend: [] is unknown, and native int8 / fp8 still load there.
  assert.deepEqual(precisionDisabledReasons(OPTIONS, { ...windows, denseQuantSchemes: [] }), {});
  assert.deepEqual(precisionDisabledReasons(OPTIONS, { ...windows, denseQuantSupported: false }), {});
  assert.equal(precisionUnavailableReason("auto", windows), null);
  assert.equal(precisionUnavailableReason("none", windows), null);
  // Several cards: the list is their intersection, so it says nothing about the card the load picks.
  assert.deepEqual(precisionDisabledReasons(OPTIONS, { ...windows, gpuCount: 2 }), {});
  assert.deepEqual(Object.keys(precisionDisabledReasons(OPTIONS, { ...windows, gpuCount: 1 })).sort(), ["fp8", "mxfp8"]);
});

test("the family comes from the explicit override, else the resident model", () => {
  assert.equal(pickFamily("auto", true, "Wan2.2-TI2V-5B"), WAN);
  assert.equal(pickFamily("ltx-2", true, WAN), "ltx-2");
  assert.equal(pickFamily("auto", false, WAN), null);
  assert.equal(pickFamily(undefined, false, null), null);
});

test("seeding from a loaded record maps off to Off / Dense", () => {
  assert.equal(seededTransformerQuant({ source: "user", requested: "fp8", value: "fp8" } as never), "fp8");
  assert.equal(seededTransformerQuant({ source: "user", requested: "off", value: "off" } as never), "none");
  assert.equal(seededTextEncoderQuant({ source: "user", requested: "off", value: "off" } as never), "none");
  assert.equal(seededTransformerQuant(null), null);
});

test("a recipe restores the precision it ran with, and leaves the select alone when it cannot know", () => {
  assert.equal(recipeTransformerQuant({ transformer_quant: "fp8", model_kind: "pipeline" }), "fp8");
  assert.equal(recipeTransformerQuant({ transformer_quant: "NVFP4", model_kind: "pipeline" }), "nvfp4");
  // A dense pipeline that engaged nothing ran bf16: Off.
  assert.equal(recipeTransformerQuant({ transformer_quant: null, model_kind: "pipeline" }), "none");
  assert.equal(recipeTransformerQuant({ transformer_quant: null, model_kind: "gguf" }), null);
  assert.equal(recipeTransformerQuant({ model_kind: "pipeline" }), null);
  assert.equal(recipeTextEncoderQuant({ text_encoder_quant: "fp8_dynamic" }), "fp8_dynamic");
  assert.equal(recipeTextEncoderQuant({ text_encoder_quant: "off" }), "none");
  assert.equal(recipeTextEncoderQuant({ text_encoder_quant: null }), null);
  // A dense pipeline records a dense encoder as null: restore Dense, or a quantized pick would stay.
  assert.equal(recipeTextEncoderQuant({ text_encoder_quant: null, model_kind: "pipeline" }), "none");
  assert.equal(recipeTextEncoderQuant({ text_encoder_quant: null, model_kind: "gguf" }), null);
  assert.equal(recipeTextEncoderQuant({ model_kind: "pipeline" }), null);
});

test("a recipe from another model is offered as a reload, the resident one is not", () => {
  const loaded = { loaded: true, repo_id: "/cache/snap", display_repo_id: "Wan-AI/Wan2.2-TI2V-5B-Diffusers" };
  assert.equal(recipeModelToReload({ model: "Wan-AI/Wan2.2-TI2V-5B-Diffusers" }, loaded), null);
  assert.equal(recipeModelToReload({ model: "Lightricks/LTX-2" }, loaded), "Lightricks/LTX-2");
  assert.equal(recipeModelToReload({ model: "Lightricks/LTX-2" }, { loaded: false }), "Lightricks/LTX-2");
  assert.equal(recipeModelToReload({ model: "gpt-image", model_kind: "connection" }, null), null);
  assert.equal(recipeModelToReload({ model: null }, null), null);
});

test("a different GGUF file of the resident repo is still offered as a reload", () => {
  const repo = "QuantStack/Wan2.2-TI2V-5B-GGUF";
  const q8 = { model: repo, model_kind: "gguf", gguf_filename: "Wan2.2-TI2V-5B-Q8_0.gguf" };
  const label = (f: string) => f.replace(/\.gguf$/i, "").split("-").pop() ?? "";
  // Images status names the file.
  const q4File = { loaded: true, repo_id: repo, model_kind: "gguf", gguf_filename: "Wan2.2-TI2V-5B-Q4_K_M.gguf" };
  assert.equal(recipeModelToReload(q8, q4File), repo);
  assert.equal(recipeModelToReload(q8, { ...q4File, gguf_filename: "sub/Wan2.2-TI2V-5B-Q8_0.gguf" }), null);
  // Video status names only the quant.
  const q4Variant = { loaded: true, repo_id: repo, model_kind: "gguf", gguf_variant: "Q4_K_M" };
  assert.equal(recipeModelToReload(q8, q4Variant, label), repo);
  assert.equal(recipeModelToReload(q8, { ...q4Variant, gguf_variant: "Q8_0" }, label), null);
  // Same repo, other build kind.
  assert.equal(recipeModelToReload(q8, { loaded: true, repo_id: repo, model_kind: "pipeline" }), repo);
});

for (const page of ["features/images/images-page.tsx", "features/video/video-page.tsx"]) {
  test(`${page}: a refused or failed load reseeds Precision from the resident build`, () => {
    const src = readSrc(page);
    // Both the refused start (catch) and the phase "error" path bump the nonce.
    assert.ok((src.match(/setPrecisionReseed\(\(n\) => n \+ 1\)/g) ?? []).length >= 2);
    const effect = src.slice(src.indexOf("if (precisionReseed === 0) return;"));
    assert.ok(effect.length > 0 && src.includes("}, [precisionReseed]);"));
    assert.match(effect.slice(0, 600), /seededTransformerQuant\(record\.transformer_quant\)/);
    if (page.includes("images")) assert.match(effect.slice(0, 600), /seededTextEncoderQuant\(record\.text_encoder_quant\)/);
  });

  test(`${page}: Precision disables what cannot load, with the reason`, () => {
    const src = readSrc(page);
    const select = src.slice(src.indexOf('label="Precision"'), src.indexOf('label="Precision"') + 1200);
    assert.match(select, /disabledReasons=\{precisionDisabledReasons\(/);
    assert.match(src, /denials: status\?\.transformer_quant_denials/);
    assert.match(src, /denseQuantSupported: hostClass === "dense-quant"/);
    assert.match(src, /title=\{disabledReasons\?\.\[v\] \?\? advisoryReasons\?\.\[v\]\}/);
    // The select also configures the NEXT pick: only an explicit override disables by family; the loaded model's
    // denials are advice (tooltip and a note), never a disabled option.
    assert.match(src, /family: pickFamily\(familyOverride, false, null\),/);
    assert.match(select, /advisoryReasons=\{precisionAdvisoryReasons\(\s*\[[^\]]*\][^)]*\)?[\s\S]{0,120}status\?\.loaded \? status\.family : null,/);
    assert.match(src, /Not for the loaded model: \{selectedAdvice\}/);
  });
}

test("the video header names the loaded precision, as the Images header does", () => {
  const src = readSrc("features/video/video-page.tsx");
  const selector = src.slice(src.indexOf("<ModelSelector"), src.indexOf("<ModelSelector") + 800);
  assert.match(
    selector,
    /loadedPrecisionOverride=\{busy !== "unloading" \? loadedTransformerPrecisionLabel\(status\) : null\}/,
  );
});

test("Images Restore puts back the precision and names a different model for the user to reload", () => {
  const src = readSrc("features/images/images-page.tsx");
  const start = src.indexOf("const restoreSettings = useCallback(");
  const body = src.slice(start, src.indexOf("// Like dropping an image on ComfyUI", start));
  assert.match(body, /recipeTransformerQuant\(image\)/);
  assert.match(body, /recipeTextEncoderQuant\(image\)/);
  assert.match(body, /recipeModelToReload\(image, status, ggufQuantLabel\)/);
  assert.match(body, /toast\.info\(`This image was made with \$\{reloadModel\}`/);
  // A prompt, never a load: the picker's own load path stays the only way in.
  assert.doesNotMatch(body, /handleModelSelect/);
});

for (const page of ["images/images-page.tsx", "video/video-page.tsx"]) {
  test(`${page}: a multi-GPU host's scheme intersection never disables Precision`, () => {
    const src = readSrc(`features/${page}`);
    assert.match(src, /const gpuCount = useGpuDeviceCount\(\);/);
    assert.match(src, /denseQuantSchemes,\s*gpuCount,\s*\/\/[^\n]*\n\s*family: pickFamily\(/);
  });
}

// Runs the video page's restoreClipBuild for real, with the page's state setter and toast stubbed.
function videoRestoreClipBuild(status: unknown) {
  const source = readSrc("features/video/video-page.tsx");
  const tree = ts.createSourceFile("video-page.tsx", source, ts.ScriptTarget.Latest, true, ts.ScriptKind.TSX);
  let declaration = "";
  const visit = (node: ts.Node) => {
    if (ts.isVariableDeclaration(node) && node.name.getText(tree) === "restoreClipBuild")
      declaration = `const ${node.getText(tree)};`;
    ts.forEachChild(node, visit);
  };
  visit(tree);
  assert.ok(declaration, "restoreClipBuild");
  const { outputText } = ts.transpileModule(`${declaration}\nreturn restoreClipBuild;`, {
    compilerOptions: { target: ts.ScriptTarget.ES2022 },
  });
  const state: Record<string, unknown> = {};
  const toasts: { message: string; options: { description?: string; action?: unknown } }[] = [];
  const scope = {
    useCallback: (fn: unknown) => fn,
    setTransformerQuant: (v: unknown) => {
      state.transformerQuant = v;
    },
    restoreStatusRef: { current: status },
    recipeTransformerQuant,
    recipeModelToReload,
    // The page's ggufQuantLabel (model-identity.ts imports through the @ alias), for a plain -QUANT.gguf name.
    ggufQuantLabel: (filename: string) => filename.replace(/\.gguf$/i, "").split("-").pop(),
    toast: { info: (message: string, options: never) => toasts.push({ message, options }) },
  };
  const fn = new Function(...Object.keys(scope), outputText)(...Object.values(scope));
  return { restore: fn as (video: unknown) => void, state, toasts };
}

test("Video Restore brings an FP8 clip back as FP8 and names its model, never loading it", () => {
  const resident = { loaded: true, repo_id: "Lightricks/LTX-2", display_repo_id: null };
  const { restore, state, toasts } = videoRestoreClipBuild(resident);
  restore({ model: "Wan-AI/Wan2.2-TI2V-5B-Diffusers", model_kind: "pipeline", transformer_quant: "fp8" });
  assert.equal(state.transformerQuant, "fp8");
  assert.equal(toasts.length, 1);
  assert.match(toasts[0].message, /Wan-AI\/Wan2\.2-TI2V-5B-Diffusers/);
  assert.match(toasts[0].options.description ?? "", /model picker/);
  assert.equal(toasts[0].options.action, undefined, "a prompt, not a load");

  const same = videoRestoreClipBuild(resident);
  same.restore({ model: "Lightricks/LTX-2", model_kind: "pipeline", transformer_quant: null });
  assert.equal(same.state.transformerQuant, "none");
  assert.equal(same.toasts.length, 0);

  // Another GGUF file of the resident repo (video status names only the quant) is named, file included.
  const repo = "QuantStack/Wan2.2-TI2V-5B-GGUF";
  const gguf = videoRestoreClipBuild({ loaded: true, repo_id: repo, model_kind: "gguf", gguf_variant: "Q4_K_M" });
  gguf.restore({ model: repo, model_kind: "gguf", gguf_filename: "Wan2.2-TI2V-5B-Q8_0.gguf", transformer_quant: null });
  assert.equal(gguf.toasts.length, 1);
  assert.match(gguf.toasts[0].options.description ?? "", /Wan2\.2-TI2V-5B-Q8_0\.gguf is not loaded/);
});

test("Video restoreSettings runs the build restore", () => {
  const src = readSrc("features/video/video-page.tsx");
  const start = src.indexOf("const restoreSettings = useCallback(");
  assert.match(src.slice(start, start + 1500), /restoreClipBuild\(video\);/);
});
