// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Wan NVFP4 (and FP8 on a card whose rowwise kernel probe failed) was offered, then refused at load (NVIDIA feedback
// p.8, p.39). The backend now reports each scheme's refusal; Precision disables it with the reason on both pages.

import assert from "node:assert/strict";
import test from "node:test";

import { NVFP4_DISABLED_HINT, withNvfp4Option, type SelectOption } from "../src/lib/nvfp4-options.ts";
import {
  activePrecisionDenials,
  deniedSelectionFallback,
  gpuSelectionKey,
  precisionResidentKey,
  withPrecisionDenials,
} from "../src/lib/precision-denials.ts";
import { readSrc } from "./helpers/kit.ts";

const OPTIONS: SelectOption[] = [
  ["fp8", "FP8"],
  ["int8", "INT8"],
  ["nvfp4", "NVFP4 (Blackwell)"],
  ["mxfp8", "MXFP8 (Blackwell)"],
];
const WAN_NVFP4 = "'nvfp4' is ruled out for family 'wan2.2-ti2v': it was measured and failed the NVFP4 accuracy gate";
const WAN = { int8: null, fp8: null, nvfp4: WAN_NVFP4, mxfp8: null };

test("a denied scheme is disabled with the backend's reason; the rest stay pickable", () => {
  const options = withPrecisionDenials(withNvfp4Option(OPTIONS, true), WAN);
  assert.deepEqual(options[2], ["nvfp4", "NVFP4 (Blackwell)", WAN_NVFP4]);
  for (const i of [0, 1, 3]) assert.equal(options[i][2], undefined);
});

test("a failed FP8 kernel probe disables FP8", () => {
  const reason = "'fp8' failed this GPU's kernel probe";
  assert.equal(withPrecisionDenials(OPTIONS, { fp8: reason })[0][2], reason);
});

test("a measured family failure replaces the NVFP4 switch hint, and no denials changes nothing", () => {
  // Turning the switch on would not help Wan, so its reason wins; a family the backend leaves null keeps the hint.
  assert.equal(withPrecisionDenials(withNvfp4Option(OPTIONS, false), WAN)[2][2], WAN_NVFP4);
  assert.equal(withPrecisionDenials(withNvfp4Option(OPTIONS, false), { nvfp4: null })[2][2], NVFP4_DISABLED_HINT);
  assert.deepEqual(withPrecisionDenials(OPTIONS, null), OPTIONS);
  assert.deepEqual(withPrecisionDenials(OPTIONS, undefined), OPTIONS);
});

test("a held denied pick falls back to Auto", () => {
  assert.equal(deniedSelectionFallback("nvfp4", WAN), "auto");
  assert.equal(deniedSelectionFallback("fp8", WAN), "fp8");
  assert.equal(deniedSelectionFallback("auto", WAN), "auto");
  assert.equal(deniedSelectionFallback("nvfp4", null), "nvfp4");
});

test("the loaded build's denials apply to a load on its card; with nothing loaded the matching plan's apply", () => {
  const planned = { gpu: "auto", options: { fp8: "x" }, repos: ["Wan-AI/Wan2.2-TI2V-5B-Diffusers"] };
  // A plan for the resident model itself defers to its status.
  assert.equal(
    activePrecisionDenials(
      { loaded: true, repo_id: "Wan-AI/Wan2.2-TI2V-5B-Diffusers", precision_options: WAN },
      planned,
      "auto",
    ),
    WAN,
  );
  assert.equal(activePrecisionDenials({ loaded: false, precision_options: null }, planned, "auto"), planned.options);
  assert.equal(activePrecisionDenials({ loaded: true }, planned, "auto"), planned.options);
  assert.equal(activePrecisionDenials(null, null), null);
});

test("picking another GPU drops the resident card's probe verdicts", () => {
  // Loaded on card 0, whose FP8 probe failed; card 1 can run FP8.
  const resident = { loaded: true, precision_options: { fp8: "FP8 failed this GPU's kernel probe" }, precision_options_card: 0 };
  assert.equal(activePrecisionDenials(resident, null, "0"), resident.precision_options);
  assert.equal(activePrecisionDenials(resident, null, "1"), null);
  // A plan made for card 1 describes it; one made for another selection does not.
  const forCard1 = { gpu: "1", options: { fp8: null } };
  assert.equal(activePrecisionDenials(resident, forCard1, "1"), forCard1.options);
  assert.equal(activePrecisionDenials(resident, { gpu: "0", options: { fp8: "y" } }, "1"), null);
  // An unknown resident card never matches an explicit pick.
  assert.equal(activePrecisionDenials({ ...resident, precision_options_card: null }, null, "0"), null);
  assert.equal(gpuSelectionKey([1]), "1");
  assert.equal(gpuSelectionKey(undefined), "auto");
});

for (const path of ["features/images/images-page.tsx", "features/video/video-page.tsx"]) {
  const src = readSrc(path);
  test(`${path}: Precision applies the denials, falls back, and reads the pick's plan`, () => {
    const select = src.slice(src.indexOf('label="Precision"'));
    assert.match(select.slice(0, select.indexOf(") : (")), /withPrecisionDenials\(\s*withNvfp4Option\(/);
    assert.match(
      src,
      /const precisionDenials = activePrecisionDenials\(\s*status,\s*plannedPrecisionOptions,\s*selectedGpu !== "auto" && gpuChoices\.some/,
    );
    assert.match(src, /setTransformerQuant\(\(v\) => deniedSelectionFallback\(v, precisionDenials\)\);/);
    assert.match(
      src,
      /setPlannedPrecisionOptions\(\{\s*gpu: gpuSelectionKey\(advanced\.gpu_ids\),\s*options: plan\.precision_options \?\? null,\s*repos: \[repoId, opts\.displayRepoId\],\s*\}\);/,
    );
    // A failed load abandons the pick, so its plan goes too.
    assert.match(src, /useEffect\(\(\) => \{\s*if \(failedLoadReseed > 0\) setPlannedPrecisionOptions\(null\);\s*\}, \[failedLoadReseed\]\);/);
  });
}

// NVIDIA feedback follow-up: after loading Qwen-Image and ejecting it, its plan kept MXFP8 disabled for the next pick.
test("a plan's denials are dropped whenever the resident build changes", () => {
  assert.equal(precisionResidentKey(null), "none");
  assert.equal(precisionResidentKey({ loaded: false, repo_id: "Qwen/Qwen-Image" }), "none");
  const qwen = precisionResidentKey({ loaded: true, repo_id: "Qwen/Qwen-Image", family: "qwen-image" });
  assert.notEqual(qwen, "none");
  // Eject changes the key, as does a different model or a reload at another precision.
  assert.notEqual(precisionResidentKey({ loaded: true, repo_id: "black-forest-labs/FLUX.1-dev" }), qwen);
  assert.notEqual(
    precisionResidentKey({ loaded: true, repo_id: "Qwen/Qwen-Image", family: "qwen-image", transformer_quant: "int8" }),
    qwen,
  );
  for (const path of ["features/images/images-page.tsx", "features/video/video-page.tsx"]) {
    const src = readSrc(path);
    assert.match(
      src,
      /const residentPrecisionKey = precisionResidentKey\(status\);\s*useEffect\(\(\) => \{\s*setPlannedPrecisionOptions\(null\);\s*\}, \[residentPrecisionKey\]\);/,
      path,
    );
    // Declared after status, which it reads during render.
    assert.ok(src.indexOf("precisionResidentKey(status)") > src.indexOf("const [status, setStatus]"), path);
  }
});

// Recheck r10: with Wan resident (NVFP4 refused) a plan for FLUX on the same GPU, which permits NVFP4, lost to Wan's
// status, so NVFP4 stayed disabled and a held NVFP4 reset to Auto.
test("a plan for a different model wins over the resident family's refusals on the same GPU", () => {
  const wan = {
    loaded: true,
    repo_id: "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
    precision_options: WAN,
    precision_options_card: 0,
  };
  const flux = { gpu: "auto", options: { int8: null, fp8: null, nvfp4: null, mxfp8: null }, repos: ["black-forest-labs/FLUX.1-dev"] };
  assert.equal(activePrecisionDenials(wan, flux, "auto"), flux.options);
  assert.equal(activePrecisionDenials(wan, { ...flux, gpu: "0" }, "0"), flux.options);
  // The display id counts as the same model.
  const viaDisplay = { ...flux, repos: ["unsloth/Wan-mirror", "Wan-AI/Wan2.2-TI2V-5B-Diffusers"] };
  assert.equal(activePrecisionDenials(wan, viaDisplay, "auto"), WAN);
  // A plan for another selection does not describe this target.
  assert.equal(activePrecisionDenials(wan, { ...flux, gpu: "1" }, "auto"), WAN);
});

// Codex: canceling a staged Qwen-Image download with FLUX resident left Qwen's plan (MXFP8 disabled) over FLUX.
test("canceling a staged download drops the plan its pick made, on both pages", () => {
  for (const path of ["features/images/images-page.tsx", "features/video/video-page.tsx"]) {
    const src = readSrc(path);
    const start = src.indexOf("onCancelled: () => {");
    assert.ok(start > 0, path);
    const handler = src.slice(start, src.indexOf("usePickToastProgress(pickToast, stagedProgress);", start));
    assert.match(handler, /pendingStagedLoad\.current = null;[\s\S]*setPlannedPrecisionOptions\(null\);/, path);
  }
  // With the plan gone the resident's own status applies again.
  const flux = { loaded: true, repo_id: "black-forest-labs/FLUX.1-dev", precision_options: { mxfp8: null } };
  assert.equal(activePrecisionDenials(flux, null, "auto"), flux.precision_options);
});
