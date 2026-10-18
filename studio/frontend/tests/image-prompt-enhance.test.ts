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
  hasEnhancedPrompt,
  parsePromptUpsamplerStatus,
  PROMPT_UPSAMPLER_ROUTES,
  promptUpsamplerCapability,
  readEnhanceChoices,
  resolveEnhancePrompt,
  upsamplerDownloadBytes,
  upsamplerDownloadFraction,
  upsamplerNeedsDownload,
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
  const call = page.slice(page.indexOf("res = await generateDiffusionImage({"));
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
  assert.equal(upsamplerNeedsDownload({ ...noProjector, mmproj_downloaded: true }), false);
  assert.equal(upsamplerNeedsDownload({ ...full, supported: false }), false);

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
  assert.ok(enhancementNotice("unavailable"));
  assert.ok(enhancementNotice("failed"));
  assert.equal(enhancementNotice("applied"), null);
  assert.equal(enhancementNotice("off"), null);
  assert.equal(enhancementNotice(undefined), null);
  // No em dashes in the copy.
  assert.doesNotMatch(ENHANCE_PROMPT_HINT, /—/);
  assert.doesNotMatch(readSrc("features/images/prompt-enhance-panel.tsx"), /—/);
  assert.doesNotMatch(readSrc("features/images/lib/prompt-enhance.ts"), /—/);
});
