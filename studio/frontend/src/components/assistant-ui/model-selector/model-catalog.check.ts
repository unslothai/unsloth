// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Assertions over the model catalog: canonical keys, alias resolution, catalog
// integrity, the artifact/quant routing ladders, and search matching. Follows
// the i18n:check pattern (plain node:assert, no test framework).
//
// Run: npm run catalog:check

import assert from "node:assert/strict";

import {
  IMAGE_CATALOG,
  VIDEO_CATALOG,
  canonicalKeyFor,
  catalogGroupFitsDevice,
  catalogToModelOptions,
  classifyGgufFit,
  groupForRepoId,
  groupMatchesQuery,
  loadSpecFor,
  pickDefaultArtifact,
  pickDefaultQuant,
  stripArtifactSuffixesForDisplay,
} from "./model-catalog.ts";

// ── canonicalKeyFor: suffix stripping, owner preserved ─────────────────────────

assert.equal(canonicalKeyFor("unsloth/Qwen-Image-2512-GGUF"), "unsloth/qwen-image-2512");
assert.equal(canonicalKeyFor("unsloth/Qwen-Image-2512-FP8"), "unsloth/qwen-image-2512");
assert.equal(
  canonicalKeyFor("unsloth/Qwen-Image-2512-unsloth-bnb-4bit"),
  "unsloth/qwen-image-2512",
);
assert.equal(
  canonicalKeyFor("ideogram-ai/ideogram-4-nf4-diffusers"),
  "ideogram-ai/ideogram-4",
);
assert.equal(canonicalKeyFor("Wan-AI/Wan2.2-TI2V-5B-Diffusers"), "wan-ai/wan2.2-ti2v-5b");
assert.equal(canonicalKeyFor("lightricks/ltx-2.3-fp8"), "lightricks/ltx-2.3");
// Prequant suffixes strip regardless of case: -GGUF/-FP8/-int8/-nvfp4 all route
// to the base name.
assert.equal(canonicalKeyFor("unsloth/Qwen-Image-2512-int8"), "unsloth/qwen-image-2512");
assert.equal(canonicalKeyFor("unsloth/Qwen-Image-2512-INT8"), "unsloth/qwen-image-2512");
assert.equal(canonicalKeyFor("unsloth/Qwen-Image-2512-nvfp4"), "unsloth/qwen-image-2512");
assert.equal(canonicalKeyFor("unsloth/Qwen-Image-2512-NVFP4"), "unsloth/qwen-image-2512");
assert.equal(canonicalKeyFor("unsloth/qwen-image-2512-gguf"), "unsloth/qwen-image-2512");
assert.equal(canonicalKeyFor("unsloth/qwen-image-2512-fp8"), "unsloth/qwen-image-2512");

// ── stripArtifactSuffixesForDisplay: case-preserving base name for row labels ──

assert.equal(
  stripArtifactSuffixesForDisplay("unsloth/ERNIE-Image-Turbo-GGUF"),
  "unsloth/ERNIE-Image-Turbo",
);
assert.equal(
  stripArtifactSuffixesForDisplay("unsloth/FLUX.2-klein-base-9B-GGUF"),
  "unsloth/FLUX.2-klein-base-9B",
);
assert.equal(
  stripArtifactSuffixesForDisplay("unsloth/Qwen-Image-2512-FP8"),
  "unsloth/Qwen-Image-2512",
);
assert.equal(
  stripArtifactSuffixesForDisplay("unsloth/Some-Model-int8"),
  "unsloth/Some-Model",
);
assert.equal(
  stripArtifactSuffixesForDisplay("unsloth/Some-Model-NVFP4"),
  "unsloth/Some-Model",
);
// Non-suffixed names and suffix-only names come back unchanged, casing intact.
assert.equal(
  stripArtifactSuffixesForDisplay("krea/Krea-2-Turbo"),
  "krea/Krea-2-Turbo",
);
assert.equal(stripArtifactSuffixesForDisplay("someone/FP8"), "someone/FP8");
// Non-suffixed ids come back unchanged (lowercased).
assert.equal(canonicalKeyFor("krea/Krea-2-Turbo"), "krea/krea-2-turbo");
// Stripping never merges owners.
assert.notEqual(
  canonicalKeyFor("Qwen/Qwen-Image-2512"),
  canonicalKeyFor("unsloth/Qwen-Image-2512"),
);
// Stripping never empties a name that IS a suffix-looking token.
assert.equal(canonicalKeyFor("someone/fp8"), "someone/fp8");

// ── groupForRepoId: artifacts, aliases, canonical keys, unknowns ───────────────

const qwen2512 = groupForRepoId("unsloth/Qwen-Image-2512-GGUF", IMAGE_CATALOG);
assert.ok(qwen2512);
assert.equal(qwen2512.canonicalId, "unsloth/Qwen-Image-2512");
// Every artifact of the group resolves to the same group.
for (const artifact of qwen2512.artifacts) {
  assert.equal(groupForRepoId(artifact.repoId, IMAGE_CATALOG), qwen2512);
}
// Cross-owner aliases resolve only because they are declared.
assert.equal(groupForRepoId("Qwen/Qwen-Image-2512", IMAGE_CATALOG), qwen2512);
// Undeclared prequant variants (any case) still route to the base group via the
// stripped key, so Recommended and On Device standardize them to the base name.
assert.equal(groupForRepoId("unsloth/Qwen-Image-2512-INT8", IMAGE_CATALOG), qwen2512);
assert.equal(groupForRepoId("unsloth/Qwen-Image-2512-NVFP4", IMAGE_CATALOG), qwen2512);
assert.equal(
  groupForRepoId("Tongyi-MAI/Z-Image-Turbo", IMAGE_CATALOG)?.canonicalId,
  "unsloth/Z-Image-Turbo",
);
// A sibling artifact of an aliased owner groups via the alias' stripped key.
assert.equal(groupForRepoId("Qwen/Qwen-Image-2512-FP8", IMAGE_CATALOG), qwen2512);
// Unknown repos pass through ungrouped.
assert.equal(groupForRepoId("someone/some-model-GGUF", IMAGE_CATALOG), null);
assert.equal(groupForRepoId("unsloth/Llama-3.3-70B-GGUF", VIDEO_CATALOG), null);
// Video: the Lightricks 2.3 checkpoints group under the unsloth 2.3 release.
const ltx23 = groupForRepoId("unsloth/LTX-2.3-GGUF", VIDEO_CATALOG);
assert.ok(ltx23);
assert.equal(groupForRepoId("lightricks/ltx-2.3", VIDEO_CATALOG), ltx23);
assert.equal(groupForRepoId("lightricks/ltx-2.3-fp8", VIDEO_CATALOG), ltx23);
// ...but the LTX-2.0 base stays its own group (different model).
assert.notEqual(groupForRepoId("Lightricks/LTX-2", VIDEO_CATALOG), ltx23);
// SDXL Turbo and Base stay separate groups (different checkpoints).
assert.notEqual(
  groupForRepoId("stabilityai/sdxl-turbo", IMAGE_CATALOG),
  groupForRepoId("stabilityai/stable-diffusion-xl-base-1.0", IMAGE_CATALOG),
);
// Both HunyuanVideo resolutions land in one group.
assert.equal(
  groupForRepoId(
    "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v",
    VIDEO_CATALOG,
  ),
  groupForRepoId(
    "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-720p_t2v",
    VIDEO_CATALOG,
  ),
);

// ── catalog integrity: unique ids, artifacts resolve to exactly one group ──────

for (const catalog of [IMAGE_CATALOG, VIDEO_CATALOG]) {
  const seen = new Set<string>();
  for (const group of catalog) {
    for (const artifact of group.artifacts) {
      const lowered = artifact.repoId.toLowerCase();
      assert.ok(!seen.has(lowered), `duplicate artifact id: ${artifact.repoId}`);
      seen.add(lowered);
      assert.equal(
        groupForRepoId(artifact.repoId, catalog),
        group,
        `artifact ${artifact.repoId} resolves to a different group`,
      );
      if (artifact.loadKind === "single_file") {
        assert.ok(artifact.filename, `single_file ${artifact.repoId} needs a filename`);
      }
    }
    for (const alias of group.aliases ?? []) {
      assert.equal(
        groupForRepoId(alias, catalog),
        group,
        `alias ${alias} resolves to a different group`,
      );
    }
  }
}

// ── loadSpecFor reproduces the old page lookup tables exactly ──────────────────

const OLD_SAFETENSORS_MODELS: Record<
  string,
  { kind: "pipeline" | "single_file"; filename?: string }
> = {
  "unsloth/Z-Image-Turbo-unsloth-bnb-4bit": { kind: "pipeline" },
  "krea/Krea-2-Turbo": { kind: "pipeline" },
  "ideogram-ai/ideogram-4-fp8": { kind: "pipeline" },
  "ideogram-ai/ideogram-4-nf4-diffusers": { kind: "pipeline" },
  "unsloth/Qwen-Image-2512-unsloth-bnb-4bit": { kind: "pipeline" },
  "unsloth/Qwen-Image-2512-FP8": {
    kind: "single_file",
    filename: "qwen-image-2512-fp8.safetensors",
  },
  "stabilityai/sdxl-turbo": { kind: "pipeline" },
  "stabilityai/stable-diffusion-xl-base-1.0": { kind: "pipeline" },
};
for (const [id, spec] of Object.entries(OLD_SAFETENSORS_MODELS)) {
  const got = loadSpecFor(id, IMAGE_CATALOG);
  assert.ok(got, `missing image load spec for ${id}`);
  assert.equal(got.kind, spec.kind, id);
  assert.equal(got.filename, spec.filename, id);
}

const OLD_PIPELINE_MODELS = [
  "Lightricks/LTX-2",
  "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
  "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
  "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v",
];
for (const id of OLD_PIPELINE_MODELS) {
  const got = loadSpecFor(id, VIDEO_CATALOG);
  assert.ok(got, `missing video load spec for ${id}`);
  assert.equal(got.kind, "pipeline", id);
}
// GGUF artifacts report the gguf kind; unknown ids report null.
assert.equal(loadSpecFor("unsloth/Z-Image-Turbo-GGUF", IMAGE_CATALOG)?.kind, "gguf");
assert.equal(loadSpecFor("someone/unknown", IMAGE_CATALOG), null);

// Every old curated id is still present as an option (backwards compat).
const imageOptionIds = new Set(catalogToModelOptions(IMAGE_CATALOG).map((o) => o.id));
for (const id of [
  "unsloth/Z-Image-Turbo-GGUF",
  "unsloth/Z-Image-GGUF",
  "unsloth/Qwen-Image-2512-GGUF",
  "unsloth/Qwen-Image-GGUF",
  "unsloth/FLUX.1-schnell-GGUF",
  "unsloth/FLUX.1-dev-GGUF",
  "unsloth/FLUX.2-klein-4B-GGUF",
  "unsloth/FLUX.2-klein-9B-GGUF",
  "unsloth/Qwen-Image-Edit-2511-GGUF",
  "unsloth/FLUX.1-Kontext-dev-GGUF",
  ...Object.keys(OLD_SAFETENSORS_MODELS),
]) {
  assert.ok(imageOptionIds.has(id), `image option missing: ${id}`);
}
const videoOptionIds = new Set(catalogToModelOptions(VIDEO_CATALOG).map((o) => o.id));
for (const id of ["unsloth/LTX-2.3-GGUF", ...OLD_PIPELINE_MODELS]) {
  assert.ok(videoOptionIds.has(id), `video option missing: ${id}`);
}

// ── classifyGgufFit ────────────────────────────────────────────────────────────

const GB = 1024 ** 3;
assert.equal(classifyGgufFit(10 * GB, { gpuGb: 24, systemRamGb: 64 }), "fits");
assert.equal(classifyGgufFit(20 * GB, { gpuGb: 24, systemRamGb: 64 }), "tight");
assert.equal(classifyGgufFit(100 * GB, { gpuGb: 24, systemRamGb: 64 }), "oom");
// Unknown device: never scare with OOM.
assert.equal(classifyGgufFit(100 * GB, { gpuGb: 0, systemRamGb: 0 }), "fits");
// Unified-memory host (no GPU budget): fit-or-oom against RAM.
assert.equal(classifyGgufFit(30 * GB, { gpuGb: 0, systemRamGb: 64 }), "fits");
assert.equal(classifyGgufFit(60 * GB, { gpuGb: 0, systemRamGb: 64 }), "oom");

// ── pickDefaultQuant ───────────────────────────────────────────────────────────

const variants = [
  { quant: "Q4_K_M", filename: "m-Q4_K_M.gguf", size_bytes: 12 * GB },
  { quant: "Q8_0", filename: "m-Q8_0.gguf", size_bytes: 22 * GB },
  { quant: "BF16", filename: "m-BF16.gguf", size_bytes: 40 * GB },
];
const budget24 = { gpuGb: 24, systemRamGb: 64 };
// Repo default kept when it is not OOM.
assert.equal(pickDefaultQuant(variants, "Q4_K_M", budget24)?.quant, "Q4_K_M");
// Downloaded non-OOM quant beats the undownloaded default.
assert.equal(
  pickDefaultQuant(
    [variants[0], { ...variants[1], downloaded: true }, variants[2]],
    "Q4_K_M",
    budget24,
  )?.quant,
  "Q8_0",
);
// OOM default falls to the largest non-OOM quant (Q8_0 runs tight via RAM offload).
assert.equal(
  pickDefaultQuant(variants, "BF16", { gpuGb: 24, systemRamGb: 16 })?.quant,
  "Q8_0",
);
// Without RAM to offload into, the tight tier disappears and Q4_K_M wins.
assert.equal(
  pickDefaultQuant(variants, "BF16", { gpuGb: 24, systemRamGb: 0 })?.quant,
  "Q4_K_M",
);
// All OOM: smallest wins (closest to running).
assert.equal(
  pickDefaultQuant(variants, "BF16", { gpuGb: 4, systemRamGb: 4 })?.quant,
  "Q4_K_M",
);
// No budget knowledge: trust the repo default (expander parity).
assert.equal(
  pickDefaultQuant(variants, "Q8_0", { gpuGb: 0, systemRamGb: 0 })?.quant,
  "Q8_0",
);
assert.equal(pickDefaultQuant([], "Q4_K_M", budget24), null);

// ── pickDefaultArtifact ────────────────────────────────────────────────────────

const notDownloaded = () => false;
const qwenGroup = qwen2512;
// 8 GB consumer GPU: nothing prequant fits (fp8 24 GB, bnb 14 GB) -> GGUF.
assert.equal(
  pickDefaultArtifact(qwenGroup, { gpuGb: 8, systemRamGb: 32, isDownloaded: notDownloaded })
    .format,
  "gguf",
);
// 24 GB: bnb-4bit (14 GB) fits the 16.8 GB budget, fp8 (24 GB) does not.
assert.equal(
  pickDefaultArtifact(qwenGroup, { gpuGb: 24, systemRamGb: 64, isDownloaded: notDownloaded })
    .format,
  "bnb-4bit",
);
// 48 GB: fp8 fits -> highest quality that fits wins.
assert.equal(
  pickDefaultArtifact(qwenGroup, { gpuGb: 48, systemRamGb: 64, isDownloaded: notDownloaded })
    .format,
  "fp8",
);
// Unknown device: GGUF (the backend plans offload itself).
assert.equal(
  pickDefaultArtifact(qwenGroup, { gpuGb: 0, systemRamGb: 0, isDownloaded: notDownloaded })
    .format,
  "gguf",
);
// Downloaded-first: a downloaded bnb-4bit beats everything undownloaded.
assert.equal(
  pickDefaultArtifact(qwenGroup, {
    gpuGb: 48,
    systemRamGb: 64,
    isDownloaded: (id) => id === "unsloth/Qwen-Image-2512-unsloth-bnb-4bit",
  }).format,
  "bnb-4bit",
);
// A downloaded GGUF wins over undownloaded prequants even on a big GPU.
assert.equal(
  pickDefaultArtifact(qwenGroup, {
    gpuGb: 80,
    systemRamGb: 128,
    isDownloaded: (id) => id === "unsloth/Qwen-Image-2512-GGUF",
  }).format,
  "gguf",
);
// Ideogram on 24 GB: fp8 (46 GB) too big -> bnb-4bit (11 GB).
const ideogram = groupForRepoId("ideogram-ai/ideogram-4-fp8", IMAGE_CATALOG);
assert.ok(ideogram);
assert.equal(
  pickDefaultArtifact(ideogram, { gpuGb: 24, systemRamGb: 64, isDownloaded: notDownloaded })
    .repoId,
  "ideogram-ai/ideogram-4-nf4-diffusers",
);
// A gated BF16 artifact (FLUX.1-dev) is NOT auto-routed when undownloaded, even on a big GPU that
// fits it: the download would fail without license/token access, so route to the open GGUF.
const fluxDevRoute = groupForRepoId("unsloth/FLUX.1-dev", IMAGE_CATALOG);
assert.ok(fluxDevRoute);
assert.equal(
  pickDefaultArtifact(fluxDevRoute, { gpuGb: 80, systemRamGb: 128, isDownloaded: notDownloaded })
    .format,
  "gguf",
);
// But an already-downloaded gated BF16 (the user clearly has access) is still returned.
assert.equal(
  pickDefaultArtifact(fluxDevRoute, {
    gpuGb: 80,
    systemRamGb: 128,
    isDownloaded: (id) => id === "black-forest-labs/FLUX.1-dev",
  }).repoId,
  "black-forest-labs/FLUX.1-dev",
);
// FLUX.1 Krea dev: gated BF16 skipped when undownloaded -> the open QuantStack GGUF; the
// GGUF repo id also resolves to the group (cross-owner via the artifact list).
const kreaDevRoute = groupForRepoId("black-forest-labs/FLUX.1-Krea-dev", IMAGE_CATALOG);
assert.ok(kreaDevRoute);
assert.equal(
  pickDefaultArtifact(kreaDevRoute, { gpuGb: 80, systemRamGb: 128, isDownloaded: notDownloaded })
    .repoId,
  "QuantStack/FLUX.1-Krea-dev-GGUF",
);
assert.equal(
  groupForRepoId("QuantStack/FLUX.1-Krea-dev-GGUF", IMAGE_CATALOG),
  kreaDevRoute,
);
// Lumina Image 2.0: a single ungated bf16 pipeline artifact (11 GB) -- auto-routed on a
// 24 GB GPU (11 <= 0.7 * 24) and resolvable through its canonical id.
const lumina = groupForRepoId("Alpha-VLLM/Lumina-Image-2.0", IMAGE_CATALOG);
assert.ok(lumina);
assert.equal(
  pickDefaultArtifact(lumina, { gpuGb: 24, systemRamGb: 64, isDownloaded: notDownloaded })
    .repoId,
  "Alpha-VLLM/Lumina-Image-2.0",
);
assert.equal(loadSpecFor("Alpha-VLLM/Lumina-Image-2.0", IMAGE_CATALOG)?.kind, "pipeline");
// HunyuanImage 2.1: the 50 GB bf16 pipeline does NOT fit a 24 GB card, so a bare
// click routes to the QuantStack GGUF; on a large GPU the bf16 wins. The mirror id
// and the GGUF id resolve to one group.
const hyimage = groupForRepoId(
  "hunyuanvideo-community/HunyuanImage-2.1-Diffusers",
  IMAGE_CATALOG,
);
assert.ok(hyimage);
assert.equal(
  pickDefaultArtifact(hyimage, { gpuGb: 24, systemRamGb: 64, isDownloaded: notDownloaded })
    .repoId,
  "QuantStack/HunyuanImage-2.1-GGUF",
);
assert.equal(
  pickDefaultArtifact(hyimage, { gpuGb: 141, systemRamGb: 128, isDownloaded: notDownloaded })
    .format,
  "bf16",
);
assert.equal(groupForRepoId("QuantStack/HunyuanImage-2.1-GGUF", IMAGE_CATALOG), hyimage);
// HiDream I1: all three variants group together; a datacenter GPU auto-routes to the
// Full bf16 (catalog order wins among equal sizes), and the group is hidden by the fit
// filter on a 24 GB card (no GGUF artifact, 63 GB everywhere).
const hidream = groupForRepoId("HiDream-ai/HiDream-I1-Full", IMAGE_CATALOG);
assert.ok(hidream);
assert.equal(groupForRepoId("HiDream-ai/HiDream-I1-Dev", IMAGE_CATALOG), hidream);
assert.equal(groupForRepoId("HiDream-ai/HiDream-I1-Fast", IMAGE_CATALOG), hidream);
assert.equal(
  pickDefaultArtifact(hidream, { gpuGb: 141, systemRamGb: 128, isDownloaded: notDownloaded })
    .repoId,
  "HiDream-ai/HiDream-I1-Full",
);
assert.equal(
  catalogGroupFitsDevice(hidream, { gpuGb: 24, systemRamGb: 32 }, notDownloaded),
  false,
);
// FLUX.1-schnell is Apache-2.0 (not gated): its BF16 IS auto-routed on a GPU that fits it.
const fluxSchnellRoute = groupForRepoId("unsloth/FLUX.1-schnell", IMAGE_CATALOG);
assert.ok(fluxSchnellRoute);
assert.equal(
  pickDefaultArtifact(fluxSchnellRoute, { gpuGb: 80, systemRamGb: 128, isDownloaded: notDownloaded })
    .format,
  "bf16",
);
// HunyuanVideo on 80 GB: the highest-quality artifact that FITS (720p, 52 GB <= budget 56) wins.
const hunyuan = groupForRepoId(
  "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v",
  VIDEO_CATALOG,
);
assert.ok(hunyuan);
assert.equal(
  pickDefaultArtifact(hunyuan, { gpuGb: 80, systemRamGb: 128, isDownloaded: notDownloaded })
    .label,
  "BF16 - 720p",
);
// Same-format artifacts keep declaration order, so 720p is listed first and the fit loop returns
// it when it fits; a smaller card (budget 42) skips 720p (52 > 42) and falls back to 480p (40).
assert.equal(
  pickDefaultArtifact(hunyuan, { gpuGb: 60, systemRamGb: 128, isDownloaded: notDownloaded })
    .label,
  "BF16 - 480p",
);

// ── official BF16 artifacts (added so groups are not unsloth-quant-only) ────────
// Qwen-Image-2512 BF16 (54 GB) doesn't fit a 24/48 GB budget (bnb-4bit/fp8 win there, asserted
// above) but on an 80 GB datacenter GPU (budget 56) the official BF16 is the highest-quality
// artifact that fits and wins.
assert.equal(
  pickDefaultArtifact(qwenGroup, { gpuGb: 80, systemRamGb: 128, isDownloaded: notDownloaded })
    .format,
  "bf16",
);
assert.equal(
  pickDefaultArtifact(qwenGroup, { gpuGb: 80, systemRamGb: 128, isDownloaded: notDownloaded })
    .repoId,
  "Qwen/Qwen-Image-2512",
);
// Z-Image-Turbo BF16 (30 GB): does not fit 24 GB (bnb-4bit wins) but fits a
// 48 GB GPU (budget 33.6) where the official BF16 wins.
const zturbo = groupForRepoId("unsloth/Z-Image-Turbo", IMAGE_CATALOG);
assert.ok(zturbo);
assert.equal(
  pickDefaultArtifact(zturbo, { gpuGb: 24, systemRamGb: 64, isDownloaded: notDownloaded })
    .format,
  "bnb-4bit",
);
assert.equal(
  pickDefaultArtifact(zturbo, { gpuGb: 48, systemRamGb: 64, isDownloaded: notDownloaded })
    .format,
  "bf16",
);
// FLUX.1-dev BF16 (32 GB) fits a 48 GB GPU, but it is GATED: a bare click routes to the open
// GGUF unless the BF16 is already downloaded (see the gated-routing checks above). Small GPU -> GGUF.
const fluxDev = groupForRepoId("black-forest-labs/FLUX.1-dev", IMAGE_CATALOG);
assert.ok(fluxDev);
assert.equal(fluxDev.canonicalId, "unsloth/FLUX.1-dev");
assert.equal(
  pickDefaultArtifact(fluxDev, { gpuGb: 48, systemRamGb: 64, isDownloaded: notDownloaded })
    .format,
  "gguf",
);
assert.equal(
  pickDefaultArtifact(fluxDev, {
    gpuGb: 48,
    systemRamGb: 64,
    isDownloaded: (id) => id === "black-forest-labs/FLUX.1-dev",
  }).format,
  "bf16",
);
assert.equal(
  pickDefaultArtifact(fluxDev, { gpuGb: 24, systemRamGb: 64, isDownloaded: notDownloaded })
    .format,
  "gguf",
);
// LTX-2.3 video carries the official BF16 single-file checkpoint (no FP8: its scaled-fp8 file is
// refused by the loader), which keeps the ~50 GB Gemma3 encoder resident, so a consumer or 80 GB
// GPU routes to GGUF; only a B200-class budget picks the official BF16.
const ltxGroup = groupForRepoId("unsloth/LTX-2.3", VIDEO_CATALOG);
assert.ok(ltxGroup);
assert.equal(
  pickDefaultArtifact(ltxGroup, { gpuGb: 24, systemRamGb: 64, isDownloaded: notDownloaded })
    .format,
  "gguf",
);
assert.equal(
  pickDefaultArtifact(ltxGroup, { gpuGb: 80, systemRamGb: 128, isDownloaded: notDownloaded })
    .format,
  "gguf",
);
assert.equal(
  pickDefaultArtifact(ltxGroup, { gpuGb: 192, systemRamGb: 256, isDownloaded: notDownloaded })
    .format,
  "bf16",
);
// The LTX-2.3 official checkpoints load as single-file against the family base.
assert.equal(loadSpecFor("Lightricks/LTX-2.3", VIDEO_CATALOG)?.kind, "single_file");
assert.equal(
  loadSpecFor("Lightricks/LTX-2.3", VIDEO_CATALOG)?.filename,
  "ltx-2.3-22b-distilled.safetensors",
);
// The official image BF16 pipelines load via from_pretrained (pipeline kind).
assert.equal(loadSpecFor("Tongyi-MAI/Z-Image-Turbo", IMAGE_CATALOG)?.kind, "pipeline");
assert.equal(loadSpecFor("Qwen/Qwen-Image-2512", IMAGE_CATALOG)?.kind, "pipeline");

// ── catalogGroupFitsDevice (the fit-on-device toggle for catalog rows) ─────────

const wanA14b = groupForRepoId("Wan-AI/Wan2.2-T2V-A14B-Diffusers", VIDEO_CATALOG);
const ltxBase = groupForRepoId("Lightricks/LTX-2", VIDEO_CATALOG);
const hunyuanFit = groupForRepoId(
  "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v",
  VIDEO_CATALOG,
);
assert.ok(wanA14b && ltxBase && hunyuanFit && ltxGroup);
const consumer = { gpuGb: 24, systemRamGb: 64 }; // budget 61.6 GB
// A bare-bf16 group over budget is hidden: this is the OOM the toggle must catch.
assert.equal(catalogGroupFitsDevice(wanA14b, consumer, notDownloaded), false); // 114 GB
assert.equal(catalogGroupFitsDevice(ltxBase, consumer, notDownloaded), false); // 90 GB
// A sized bf16 group that fits the budget stays visible (Hunyuan 40/52 GB <= 61.6).
assert.equal(catalogGroupFitsDevice(hunyuanFit, consumer, notDownloaded), true);
// But on a tiny device even those are hidden.
assert.equal(
  catalogGroupFitsDevice(hunyuanFit, { gpuGb: 8, systemRamGb: 8 }, notDownloaded),
  false,
);
// A GGUF in the group is always runnable (its quant ladder self-fits + offloads),
// so LTX-2.3 stays visible even on a tiny card despite its 90 GB BF16 sibling.
assert.equal(
  catalogGroupFitsDevice(ltxGroup, { gpuGb: 4, systemRamGb: 4 }, notDownloaded),
  true,
);
// An already-downloaded artifact keeps its group visible regardless of budget.
assert.equal(
  catalogGroupFitsDevice(
    wanA14b,
    { gpuGb: 8, systemRamGb: 8 },
    (id) => id === "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
  ),
  true,
);
// Unknown device budget keeps everything (we cannot tell), even a 114 GB group.
assert.equal(catalogGroupFitsDevice(wanA14b, { gpuGb: 0, systemRamGb: 0 }, notDownloaded), true);
// On a B200-class budget the large bf16 groups fit and stay visible.
assert.equal(
  catalogGroupFitsDevice(wanA14b, { gpuGb: 192, systemRamGb: 256 }, notDownloaded),
  true,
);

// ── groupMatchesQuery ──────────────────────────────────────────────────────────

assert.ok(groupMatchesQuery(qwenGroup, "qwen"));
assert.ok(groupMatchesQuery(qwenGroup, "2512"));
assert.ok(groupMatchesQuery(qwenGroup, "gguf"));
assert.ok(groupMatchesQuery(qwenGroup, "fp8"));
assert.ok(groupMatchesQuery(qwenGroup, "4bit"));
assert.ok(groupMatchesQuery(qwenGroup, "q4_k_m"));
assert.ok(groupMatchesQuery(qwenGroup, "unsloth/qwen-image-2512-fp8"));
assert.ok(!groupMatchesQuery(qwenGroup, "mlx"));
assert.ok(!groupMatchesQuery(qwenGroup, "ideogram"));
assert.ok(groupMatchesQuery(ltx23, "ltx"));
assert.ok(groupMatchesQuery(ltx23, "lightricks/ltx-2.3"));

console.log("model-catalog check: all assertions passed");
