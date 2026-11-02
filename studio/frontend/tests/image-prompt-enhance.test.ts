// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { installLocalStorageFake, readSrc } from "./helpers/kit.ts";
import {
  ENHANCE_PROMPT_HINT,
  enhancePromptRequestField,
  enhanceRestoreFor,
  enhancementNotice,
  ENHANCE_UNSUPPORTED_FAMILY,
  overrideAfterFamilyChange,
  previewSignature,
  hasEnhancedPrompt,
  parsePromptUpsamplerStatus,
  PROMPT_UPSAMPLER_ROUTES,
  promptUpsamplerCapability,
  readEnhanceChoices,
  resolveEnhancePrompt,
  shouldWarmUpsampler,
  upsamplerDownloadBytes,
  WARM_RETRY_DELAY_MS,
  WARM_RETRY_LIMIT,
  WARM_RETRY_OUTCOMES,
  upsamplerDownloadFraction,
  upsamplerNeedsDownload,
  upsamplerServesEarlierQuant,
  upsamplerVariantFor,
  withEnhanceChoice,
} from "../src/features/images/lib/prompt-enhance.ts";
import { importedRecipe, recipeSettings } from "../src/features/images/lib/image-metadata.ts";

const FLUX3 = { loaded: true, family: "flux.3-klein", supports_prompt_upsampler: true, prompt_upsampler_default: true };
const SDXL = { loaded: true, family: "sdxl", supports_prompt_upsampler: false, prompt_upsampler_default: false };

test("the toggle defaults to the family default and is off for families without an upsampler", () => {
  const flux = promptUpsamplerCapability(FLUX3);
  assert.deepEqual(flux, { supported: true, defaultOn: true });
  assert.equal(resolveEnhancePrompt({ capability: flux, stored: undefined }), true);

  // A supporting family whose default is off.
  const offByDefault = promptUpsamplerCapability({ ...FLUX3, prompt_upsampler_default: false });
  assert.equal(resolveEnhancePrompt({ capability: offByDefault, stored: undefined }), false);

  for (const status of [SDXL, null, undefined, { loaded: true }, { ...FLUX3, loaded: false }]) {
    const cap = promptUpsamplerCapability(status);
    assert.equal(cap.supported, false);
    assert.equal(resolveEnhancePrompt({ capability: cap, stored: true, override: true }), false);
  }
  // An external image connection never runs the local upsampler.
  assert.equal(promptUpsamplerCapability(FLUX3, true).supported, false);
});

test("a stored choice beats the default, and a restore override beats both", () => {
  const cap = promptUpsamplerCapability(FLUX3);
  assert.equal(resolveEnhancePrompt({ capability: cap, stored: false }), false);
  assert.equal(resolveEnhancePrompt({ capability: cap, stored: false, override: true }), true);
  assert.equal(resolveEnhancePrompt({ capability: cap, stored: true, override: false }), false);
  assert.equal(resolveEnhancePrompt({ capability: cap, stored: undefined, override: null }), true);
});

test("the choice persists per family and a choice equal to the default is not stored", () => {
  const { store } = installLocalStorageFake();
  assert.deepEqual(readEnhanceChoices(), {});
  let choices = withEnhanceChoice({}, "flux.3-klein", false, true);
  assert.deepEqual(choices, { "flux.3-klein": false });
  choices = withEnhanceChoice(choices, "flux.3-klein-base", false, true);
  assert.deepEqual(readEnhanceChoices(), { "flux.3-klein": false, "flux.3-klein-base": false });
  choices = withEnhanceChoice(choices, "flux.3-klein", true, true);
  assert.deepEqual(readEnhanceChoices(), { "flux.3-klein-base": false });
  // No family key: nothing is written.
  assert.equal(withEnhanceChoice(choices, "", true, false), choices);
  // Corrupt storage reads as empty, non-boolean entries are dropped.
  store.set("unsloth:images:enhance-prompt-by-family", "not json");
  assert.deepEqual(readEnhanceChoices(), {});
  store.set("unsloth:images:enhance-prompt-by-family", JSON.stringify({ a: true, b: "yes", c: 1 }));
  assert.deepEqual(readEnhanceChoices(), { a: true });
});

test("enhance_prompt is sent only when the loaded family supports it", () => {
  assert.deepEqual(enhancePromptRequestField(promptUpsamplerCapability(FLUX3), true), { enhance_prompt: true });
  assert.deepEqual(enhancePromptRequestField(promptUpsamplerCapability(FLUX3), false), { enhance_prompt: false });
  assert.deepEqual(enhancePromptRequestField(promptUpsamplerCapability(SDXL), false), {});
  assert.deepEqual(enhancePromptRequestField(promptUpsamplerCapability(null), true), {});
  // Wired into the generate request.
  const page = readSrc("features/images/images-page.tsx");
  const call = page.slice(page.indexOf("res = await generateDiffusionImage("));
  assert.match(call.slice(0, 4000), /\.\.\.enhancePromptRequestField\(upsamplerCapability, enhanceOn\)/);
});

test("recipe restore: an applied run restores the user's prompt with the toggle on, or the caption with it off", () => {
  const applied = {
    prompt: "a cat",
    enhanced_prompt: "A photograph of a ginger cat sitting on a windowsill.",
    prompt_enhancement: "applied",
  };
  assert.deepEqual(enhanceRestoreFor(applied), { prompt: "a cat", enhance: true });
  assert.deepEqual(enhanceRestoreFor(applied, true), {
    prompt: "A photograph of a ginger cat sitting on a windowsill.",
    enhance: false,
  });
  assert.deepEqual(enhanceRestoreFor({ prompt: "a cat", prompt_enhancement: "off" }), { prompt: "a cat", enhance: false });
  assert.deepEqual(enhanceRestoreFor({ prompt: "a cat", prompt_enhancement: "unavailable" }, true), {
    prompt: "a cat",
    enhance: true,
  });
  // An older record, or another family's, leaves the toggle alone.
  assert.deepEqual(enhanceRestoreFor({ prompt: "a cat" }), { prompt: "a cat", enhance: null });
  assert.deepEqual(enhanceRestoreFor({ prompt: "a cat", prompt_enhancement: null }, true), {
    prompt: "a cat",
    enhance: null,
  });
  // Exact with no caption falls back to the user's prompt.
  assert.deepEqual(enhanceRestoreFor({ prompt: "a cat", enhanced_prompt: "  ", prompt_enhancement: "applied" }, true), {
    prompt: "a cat",
    enhance: true,
  });
  assert.equal(hasEnhancedPrompt(applied), true);
  assert.equal(hasEnhancedPrompt({ prompt: "a cat", enhanced_prompt: "a cat" }), false);
  assert.equal(hasEnhancedPrompt({ prompt: "a cat" }), false);
});

test("an embedded recipe carries the enhancement fields through to restore", () => {
  const recipe = importedRecipe({
    recipe: JSON.stringify({
      prompt: "a cat",
      enhanced_prompt: "A ginger cat.",
      prompt_enhancement: "applied",
      width: 1024,
      height: 1024,
      steps: 16,
      guidance: 1,
      seed: 3,
    }),
  });
  assert.ok(recipe);
  const settings = recipeSettings(recipe, { width: 1024, height: 1024, steps: 8, guidance: 0 });
  assert.equal(settings.enhanced_prompt, "A ginger cat.");
  assert.equal(settings.prompt_enhancement, "applied");
  const bogus = importedRecipe({ recipe: JSON.stringify({ prompt: "x", prompt_enhancement: "maybe" }) });
  assert.ok(bogus);
  assert.equal(recipeSettings(bogus, { width: 1, height: 1, steps: 1, guidance: 0 }).prompt_enhancement, null);
});

test("status parsing tolerates missing and malformed fields", () => {
  const full = parsePromptUpsamplerStatus({
    supported: true,
    downloaded: false,
    loaded: false,
    repo: "unsloth/FLUX.3-prompt-upsampler-GGUF",
    filename: "upsampler-Q4_K_M.gguf",
    size_bytes: 5_600_000_000,
    mmproj_size_bytes: 900_000_000,
    mmproj_downloaded: false,
    download: { downloading: true, model: "x", error: null, bytes_total: 200, bytes_done: 50, download_id: "d1" },
  });
  assert.equal(full.supported, true);
  assert.equal(full.size_bytes, 5_600_000_000);
  assert.equal(full.download.downloading, true);
  assert.equal(full.download.download_id, "d1");
  assert.equal(upsamplerDownloadFraction(full.download), 0.25);
  // The button fetches the vision projector too, until it is on disk.
  assert.equal(upsamplerDownloadBytes(full), 6_500_000_000);
  assert.equal(upsamplerDownloadBytes({ ...full, mmproj_downloaded: true }), 5_600_000_000);
  // Model cached, projector missing: Download stays offered and fetches only the projector.
  const noProjector = { ...full, downloaded: true };
  assert.equal(upsamplerNeedsDownload(noProjector), true);
  assert.equal(upsamplerDownloadBytes(noProjector), 900_000_000);
  assert.equal(upsamplerNeedsDownload({ ...noProjector, mmproj_downloaded: true, current_downloaded: true }), false);
  assert.equal(upsamplerNeedsDownload({ ...full, supported: false }), false);

  // An earlier quant serving keeps the control offered, sized at the current default.
  const earlier = parsePromptUpsamplerStatus({
    supported: true,
    downloaded: true,
    mmproj_downloaded: true,
    filename: "upsampler-Q4_K_M.gguf",
    size_bytes: 5_600_000_000,
    mmproj_size_bytes: 900_000_000,
    current_downloaded: false,
    current_missing_bytes: 5_900_000_000,
  });
  assert.equal(upsamplerServesEarlierQuant(earlier), true);
  assert.equal(upsamplerNeedsDownload(earlier), true);
  assert.equal(upsamplerDownloadBytes(earlier), 5_900_000_000);
  assert.equal(upsamplerNeedsDownload({ ...earlier, override: true }), false);
  const current = parsePromptUpsamplerStatus({ ...earlier, current_downloaded: true, current_missing_bytes: 0 });
  assert.equal(upsamplerNeedsDownload(current), false);
  assert.equal(upsamplerServesEarlierQuant(current), false);
  // Model-only earlier quant: the Download fetches the default AND the projector.
  const modelOnly = parsePromptUpsamplerStatus({
    ...earlier,
    mmproj_downloaded: false,
    current_filename: "upsampler-UD-Q4_K_XL.gguf",
    current_missing_bytes: 6_888_000_000,
  });
  assert.equal(upsamplerServesEarlierQuant(modelOnly), true);
  assert.equal(upsamplerNeedsDownload(modelOnly), true);
  assert.equal(upsamplerDownloadBytes(modelOnly), 6_888_000_000);
  // The current default itself with its projector missing is the projector-only case.
  const currentNoProjector = { ...modelOnly, filename: "upsampler-UD-Q4_K_XL.gguf", current_missing_bytes: 900_000_000 };
  assert.equal(upsamplerServesEarlierQuant(currentNoProjector), false);
  assert.equal(upsamplerDownloadBytes(currentNoProjector), 900_000_000);
  // Without current_filename (not reported) a model with no projector stays the projector-only case.
  assert.equal(upsamplerServesEarlierQuant({ ...modelOnly, current_filename: "" }), false);
  const olderBackend = parsePromptUpsamplerStatus({ supported: true, downloaded: true, mmproj_downloaded: true });
  assert.equal(olderBackend.current_downloaded, true);
  assert.equal(upsamplerNeedsDownload(olderBackend), false);

  const empty = parsePromptUpsamplerStatus(null);
  assert.deepEqual(
    { supported: empty.supported, downloaded: empty.downloaded, size: empty.size_bytes, repo: empty.repo },
    { supported: false, downloaded: false, size: 0, repo: "" },
  );
  assert.equal(empty.download.downloading, false);
  assert.equal(upsamplerDownloadFraction(empty.download), null);

  const junk = parsePromptUpsamplerStatus({ supported: "yes", size_bytes: -4, download: "nope" });
  assert.equal(junk.supported, false);
  assert.equal(junk.size_bytes, 0);
  assert.equal(junk.download.bytes_total, null);
  assert.equal(
    upsamplerDownloadFraction({ ...full.download, bytes_total: 100, bytes_done: 500 }),
    1,
  );
});

test("routes live in one place; variant and notices", () => {
  for (const route of Object.values(PROMPT_UPSAMPLER_ROUTES)) {
    assert.match(route, /^\/api\/inference\/images\/prompt-upsampler\//);
  }
  const api = readSrc("features/images/api.ts");
  assert.doesNotMatch(api, /"\/api\/inference\/images\/prompt-upsampler/);
  assert.equal(upsamplerVariantFor("flux.3-klein-base"), "base");
  assert.equal(upsamplerVariantFor("flux.3-klein"), "distilled");
  assert.equal(upsamplerVariantFor(null), "distilled");
  assert.equal(enhancementNotice("unavailable"), null);
  assert.ok(enhancementNotice("failed"));
  assert.equal(enhancementNotice("applied"), null);
  assert.equal(enhancementNotice("off"), null);
  assert.equal(enhancementNotice(undefined), null);
  // No em dashes in the copy.
  assert.doesNotMatch(ENHANCE_PROMPT_HINT, /—/);
  assert.doesNotMatch(readSrc("features/images/prompt-enhance-panel.tsx"), /—/);
  assert.doesNotMatch(readSrc("features/images/lib/prompt-enhance.ts"), /—/);
});

test("a preview caption goes stale when the prompt, the variant or a reference image changes", () => {
  const base = { prompt: "make it night", variant: "distilled" as const, reference_images: ["data:image/png;base64,AAAA"] };
  const sig = previewSignature(base);
  assert.equal(previewSignature({ ...base, prompt: "  make it night " }), sig);
  assert.notEqual(previewSignature({ ...base, prompt: "make it day" }), sig);
  assert.notEqual(previewSignature({ ...base, variant: "base" }), sig);
  assert.notEqual(previewSignature({ ...base, reference_images: ["data:image/png;base64,BBBB"] }), sig);
  assert.notEqual(previewSignature({ prompt: base.prompt, variant: base.variant }), sig);
  // Same length and the same 48 characters at each end: only the body differs.
  const head = "data:image/png;base64," + "A".repeat(60);
  const tail = "Z".repeat(60);
  assert.notEqual(
    previewSignature({ ...base, reference_images: [head + "QQQQ" + tail] }),
    previewSignature({ ...base, reference_images: [head + "RRRR" + tail] }),
  );
});

test("overrides offer no download", () => {
  const status = parsePromptUpsamplerStatus({ supported: true, downloaded: true, mmproj_downloaded: false, override: true });
  assert.equal(status.override, true);
  assert.equal(upsamplerNeedsDownload(status), false);
});

test("a family change consumes the one-off override instead of leaving it dormant", () => {
  const unloaded = { family: "", on: false };
  // Set before any model loaded: binds to the first FLUX.3 family that loads.
  const bound = overrideAfterFamilyChange(unloaded, "flux.3-klein");
  assert.deepEqual(bound, { family: "flux.3-klein", on: false });
  assert.equal(overrideAfterFamilyChange(bound, "flux.3-klein"), bound);
  // A switch to the other variant drops it, so switching back does not revive it.
  assert.equal(overrideAfterFamilyChange(bound, "flux.3-klein-base"), null);
  assert.equal(overrideAfterFamilyChange(null, "flux.3-klein"), null);
  // A model without an upsampler consumes a bound override but keeps one set before any model loaded.
  assert.equal(overrideAfterFamilyChange(bound, ENHANCE_UNSUPPORTED_FAMILY), null);
  assert.equal(overrideAfterFamilyChange(unloaded, ENHANCE_UNSUPPORTED_FAMILY), unloaded);
});

test("the upsampler warms only while Enhance is on and its files are ready", () => {
  const ready = parsePromptUpsamplerStatus({ supported: true, downloaded: true, mmproj_downloaded: true });
  assert.equal(shouldWarmUpsampler(true, ready), true);
  assert.equal(shouldWarmUpsampler(false, ready), false);
  assert.equal(shouldWarmUpsampler(true, null), false);
  assert.equal(shouldWarmUpsampler(true, { ...ready, downloaded: false }), false);
  assert.equal(shouldWarmUpsampler(true, { ...ready, supported: false }), false);
  assert.equal(shouldWarmUpsampler(true, { ...ready, download: { ...ready.download, downloading: true } }), false);
  assert.equal(PROMPT_UPSAMPLER_ROUTES.warm, "/api/inference/images/prompt-upsampler/warm");
  // The panel warms on the toggle, the download finishing and a model switch.
  const panel = readSrc("features/images/prompt-enhance-panel.tsx");
  assert.match(panel, /warmPromptUpsampler\(controller\.signal\)[\s\S]{0,600}\}, \[warm, modelKey\]\)/);
  // Transient declines are asked again, a bounded number of times; memory and placement ones are not.
  for (const outcome of ["generating", "loading", "gpu", "busy"]) assert.ok(WARM_RETRY_OUTCOMES.has(outcome));
  for (const outcome of ["tight", "memory", "placement", "shared", "loaded", "scheduled"]) {
    assert.ok(!WARM_RETRY_OUTCOMES.has(outcome), outcome);
  }
  assert.ok(WARM_RETRY_LIMIT * WARM_RETRY_DELAY_MS <= 5 * 60_000);
  assert.match(panel, /WARM_RETRY_OUTCOMES\.has\(outcome\)/);
  // Enhance turned off releases the server the warm start brought up.
  assert.equal(PROMPT_UPSAMPLER_ROUTES.release, "/api/inference/images/prompt-upsampler/release");
  assert.match(panel, /if \(wasEnabled\.current && !enabled\) releasePromptUpsampler\(\)/);
  // A warm start waits for a release still in flight, so a quick off / on cannot stop the server it asked for.
  const api = readSrc("features/images/prompt-upsampler-api.ts");
  assert.match(api, /warmPromptUpsampler[\s\S]{0,200}await pendingRelease;[\s\S]{0,120}authFetch\(PROMPT_UPSAMPLER_ROUTES\.warm/);
  assert.match(api, /pendingRelease = release\.catch/);
  // A reload of the same repo and file stops the upsampler, so the key carries the load too.
  const page = readSrc("features/images/images-page.tsx");
  assert.match(page, /modelKey=\{`\$\{status\?\.repo_id \?\? ""\}\|\$\{status\?\.gguf_filename \?\? ""\}\|\$\{loadEpoch\}`\}/);
  assert.match(page, /if \(wasLoading\.current && busy !== "loading"\) setLoadEpoch/);
});

