// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Assertions over the model catalog: keys, aliases, integrity, quant routing ladders, search matching. Plain node:assert like i18n:check. Run: npm run catalog:check
//
// `--network` adds an OPT-IN pass that asks the Hub whether every declared artifact is actually
// downloadable and whether each `gated` flag matches reality. It is deliberately not part of
// `npm run catalog:check`: it needs the network, and a Hub hiccup must never fail an unrelated PR.
// Run it on a schedule instead (.github/workflows/model-catalog-network-check.yml), or by hand
// with `npm run catalog:check:network`.

import assert from "node:assert/strict";

import type { CatalogGroup, ModelArtifact } from "./model-catalog.ts";
import {
  AUDIO_CATALOG,
  IMAGE_CATALOG,
  VIDEO_CATALOG,
  canonicalKeyFor,
  catalogGroupFitsDevice,
  catalogToModelOptions,
  classifyGgufFit,
  curatedArtifactFitsDevice,
  curatedDisplayNameFor,
  curatedRowLabelFor,
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
// Prequant suffixes strip regardless of case: -GGUF/-FP8/-int8/-nvfp4 all route to the base name.
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
// Undeclared prequant variants (any case) route to the base group via the stripped key.
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

for (const catalog of [IMAGE_CATALOG, VIDEO_CATALOG, AUDIO_CATALOG]) {
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
for (const id of [
  "unsloth/LTX-2.3-GGUF",
  "unsloth/MiniMax-H3-GGUF",
  ...OLD_PIPELINE_MODELS,
]) {
  assert.ok(videoOptionIds.has(id), `video option missing: ${id}`);
}

// H3 publishes both denoiser partitions in its official bundle. One artifact lets the lister's
// partition-aware labels expose both in Recommended and On Device without a community mirror.
const h3Group = groupForRepoId("unsloth/MiniMax-H3-GGUF", VIDEO_CATALOG);
assert.ok(h3Group);
assert.deepEqual(
  h3Group.artifacts
    .filter((artifact) => artifact.format === "gguf")
    .map((artifact) => artifact.repoId),
  ["unsloth/MiniMax-H3-GGUF"],
);
assert.equal(
  curatedDisplayNameFor("unsloth/MiniMax-H3-GGUF", VIDEO_CATALOG),
  "MiniMax H3 (GGUF)",
);

// ── curatedRowLabelFor ─────────────────────────────────────────────────────────
// The row form: format (and a resolution when that is the only difference) become chips, and only
// what actually names the variant stays in brackets.

assert.deepEqual(curatedRowLabelFor("MiniMaxAI/MiniMax-H3", VIDEO_CATALOG), {
  name: "MiniMax H3",
  tags: ["BF16"],
});
// GGUF is spelled by the repo name, like a text model's row, so no chip repeats it.
assert.deepEqual(curatedRowLabelFor("unsloth/MiniMax-H3-GGUF", VIDEO_CATALOG), {
  name: "MiniMax-H3-GGUF",
  tags: [],
});
assert.deepEqual(curatedRowLabelFor("unsloth/Z-Image-Turbo-GGUF", IMAGE_CATALOG), {
  name: "Z-Image-Turbo-GGUF",
  tags: [],
});
// Every GGUF row ends in the suffix, whoever published it.
for (const catalog of [IMAGE_CATALOG, VIDEO_CATALOG, AUDIO_CATALOG]) {
  for (const group of catalog) {
    for (const artifact of group.artifacts) {
      if (artifact.format !== "gguf") continue;
      const row = curatedRowLabelFor(artifact.repoId, catalog);
      assert.ok(row?.name.endsWith("-GGUF"), `${artifact.repoId} row reads "${row?.name}"`);
      assert.deepEqual(row?.tags, []);
    }
  }
}
// A label part that is not a format or a resolution names the variant, so it stays in the name.
assert.deepEqual(
  curatedRowLabelFor("HiDream-ai/HiDream-I1-Dev", IMAGE_CATALOG),
  { name: "HiDream I1 (Dev (distilled))", tags: ["BF16"] },
);
assert.deepEqual(
  curatedRowLabelFor(
    "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-720p_t2v",
    VIDEO_CATALOG,
  ),
  { name: "HunyuanVideo 1.5", tags: ["BF16", "720p"] },
);
assert.deepEqual(
  curatedRowLabelFor(
    "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v",
    VIDEO_CATALOG,
  ),
  { name: "HunyuanVideo 1.5", tags: ["BF16", "480p"] },
);
// One artifact means nothing to tell apart, so the row stays bare.
assert.deepEqual(curatedRowLabelFor("Lightricks/LTX-2", VIDEO_CATALOG), {
  name: "LTX 2 (base)",
  tags: [],
});
assert.equal(curatedRowLabelFor("someone/not-in-the-catalog", VIDEO_CATALOG), null);

// No two rows of a group may render identically, or the picker offers the same thing twice.
for (const catalog of [IMAGE_CATALOG, VIDEO_CATALOG]) {
  for (const group of catalog) {
    const seen = new Set<string>();
    for (const artifact of group.artifacts) {
      const row = curatedRowLabelFor(artifact.repoId, catalog);
      assert.ok(row, `${artifact.repoId} is in the catalog it came from`);
      const key = `${row.name} | ${row.tags.join(",")}`;
      assert.equal(seen.has(key), false, `${group.displayName}: two rows both read "${key}"`);
      seen.add(key);
    }
  }
}

// A non-GGUF artifact that states a parameter count must state its size too. The picker's VRAM
// badge falls back to the QLoRA estimator when it cannot size a row, and that formula reads a
// diffusion pipeline as a language model it can quantize: 5B params says 5.9 GB where the Wan 2.2
// TI2V pipeline is 30. The curated size is what keeps that fallback off these rows.
for (const catalog of [IMAGE_CATALOG, VIDEO_CATALOG, AUDIO_CATALOG]) {
  for (const group of catalog) {
    for (const artifact of group.artifacts) {
      if (!artifact.totalParams || artifact.format === "gguf") continue;
      assert.ok(
        artifact.approxSizeGb && artifact.approxSizeGb > 0,
        `${artifact.repoId} declares totalParams but no approxSizeGb`,
      );
    }
  }
}

// ── curatedArtifactFitsDevice ──────────────────────────────────────────────────

const WAN = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"; // 30 GB, no offload tiers
const H3 = "MiniMaxAI/MiniMax-H3"; // 145 GB, tiers at 74/140 and 123/80
const fitsCurated = (id: string, gpuGb: number, systemRamGb: number) =>
  curatedArtifactFitsDevice(id, VIDEO_CATALOG, { gpuGb, systemRamGb });

// The resident 70% rule, on the card alone. RAM is not a discrete GPU's budget: a pipeline with
// no measured tier is placed wholly on the card, so 64 GB of RAM does not rescue a 12 GB one.
assert.equal(fitsCurated(WAN, 48, 0), true);
assert.equal(fitsCurated(WAN, 40, 0), false);
assert.equal(fitsCurated(WAN, 12, 64), false);
// A unified-memory host reports RAM and no GPU, and there the RAM is the card.
assert.equal(fitsCurated(WAN, 0, 64), true);
// Nothing to judge against: no verdict rather than a scary one.
assert.equal(fitsCurated(WAN, 0, 0), undefined);
// Measured offload tiers override the 70% rule, in both directions. 123/80 is the case the
// generic size test gets wrong: 0.7 * (123 + 80) is 142, under 145, yet the tier was measured.
assert.equal(fitsCurated(H3, 74, 140), true);
assert.equal(fitsCurated(H3, 123, 80), true);
assert.equal(fitsCurated(H3, 74, 100), false);
// A GGUF ladder self-fits via pickDefaultQuant, and an unknown id is not ours to judge.
assert.equal(fitsCurated("unsloth/MiniMax-H3-GGUF", 12, 64), undefined);
assert.equal(fitsCurated("someone/not-in-the-catalog", 12, 64), undefined);
// Transcription retries a failed device load on CPU (stt_sidecar.py), so RAM is a real budget for
// an stt row: Whisper Large runs on a card too small to hold it. A tts load rejects CPU offload
// (inference.py raise_if_offloaded), so Orpheus is judged on the card alone.
assert.equal(
  curatedArtifactFitsDevice("unsloth/whisper-large-v3", AUDIO_CATALOG, {
    gpuGb: 4,
    systemRamGb: 32,
  }),
  true,
);
assert.equal(
  curatedArtifactFitsDevice("unsloth/whisper-large-v3", AUDIO_CATALOG, {
    gpuGb: 4,
    systemRamGb: 0,
  }),
  false,
);
// The whole model goes to whichever device takes it, so the budget is the LARGER of the two and
// never their sum: 3 GB of card and 3 GB of RAM hold a 4 GB checkpoint on neither.
assert.equal(
  curatedArtifactFitsDevice("unsloth/whisper-large-v3", AUDIO_CATALOG, {
    gpuGb: 3,
    systemRamGb: 3,
  }),
  false,
);
assert.equal(
  curatedArtifactFitsDevice("unsloth/whisper-large-v3", AUDIO_CATALOG, {
    gpuGb: 0,
    systemRamGb: 8,
  }),
  true,
);
assert.equal(
  curatedArtifactFitsDevice("unsloth/orpheus-3b-0.1-ft", AUDIO_CATALOG, {
    gpuGb: 4,
    systemRamGb: 32,
  }),
  false,
);

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
// 24 GB: bnb-4bit (14 GB) fits the 16.8 GB budget.
assert.equal(
  pickDefaultArtifact(qwenGroup, { gpuGb: 24, systemRamGb: 64, isDownloaded: notDownloaded })
    .format,
  "bnb-4bit",
);
// 48 GB: still bnb-4bit. The group has no fp8 artifact -- fp8 is family-denied for qwen-image
// (it renders black), and the -FP8 repo ships prequant .pt checkpoints, not a single-file
// safetensors, so the row that used to win here auto-routed to a download that 404s.
assert.equal(
  pickDefaultArtifact(qwenGroup, { gpuGb: 48, systemRamGb: 64, isDownloaded: notDownloaded })
    .format,
  "bnb-4bit",
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
// A gated BF16 artifact (FLUX.1-dev) is NOT auto-routed when undownloaded even on a big GPU: the download would fail without license/token access.
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
// FLUX.1 Krea dev: gated BF16 skipped when undownloaded -> the open QuantStack GGUF, whose repo id also resolves to the group.
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
// Lumina Image 2.0: one ungated bf16 pipeline (11 GB), auto-routed on a 24 GB GPU (11 <= 0.7 * 24).
const lumina = groupForRepoId("Alpha-VLLM/Lumina-Image-2.0", IMAGE_CATALOG);
assert.ok(lumina);
assert.equal(
  pickDefaultArtifact(lumina, { gpuGb: 24, systemRamGb: 64, isDownloaded: notDownloaded })
    .repoId,
  "Alpha-VLLM/Lumina-Image-2.0",
);
assert.equal(loadSpecFor("Alpha-VLLM/Lumina-Image-2.0", IMAGE_CATALOG)?.kind, "pipeline");
// HunyuanImage 2.1: the 50 GB bf16 pipeline misses a 24 GB card so a bare click routes to the QuantStack GGUF; on a large GPU bf16 wins. Both ids share one group.
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
// HiDream I1: all three variants group together, a datacenter GPU auto-routes to the Full bf16 (catalog order wins among equal sizes), and 24 GB hides the group.
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
// FLUX.1-schnell is Apache-2.0 but gated on the Hub, so a not-downloaded BF16 is skipped and the
// open GGUF is auto-routed instead, even on a GPU that fits the pipeline.
const fluxSchnellRoute = groupForRepoId("unsloth/FLUX.1-schnell", IMAGE_CATALOG);
assert.ok(fluxSchnellRoute);
assert.equal(
  pickDefaultArtifact(fluxSchnellRoute, { gpuGb: 80, systemRamGb: 128, isDownloaded: notDownloaded })
    .format,
  "gguf",
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
// Same-format artifacts keep declaration order, so 720p is listed first and the fit loop returns it; a budget of 42 skips it (52 > 42) and falls back to 480p (40).
assert.equal(
  pickDefaultArtifact(hunyuan, { gpuGb: 60, systemRamGb: 128, isDownloaded: notDownloaded })
    .label,
  "BF16 - 480p",
);

// MiniMax-H3 uses measured component-offload tiers instead of the resident 70% rule.
const h3 = groupForRepoId("MiniMaxAI/MiniMax-H3", VIDEO_CATALOG);
assert.ok(h3);
assert.equal(
  pickDefaultArtifact(h3, { gpuGb: 48, systemRamGb: 256, isDownloaded: notDownloaded })
    .format,
  "gguf",
);
assert.equal(
  pickDefaultArtifact(h3, { gpuGb: 80, systemRamGb: 128, isDownloaded: notDownloaded })
    .format,
  "gguf",
);
assert.equal(
  pickDefaultArtifact(h3, { gpuGb: 80, systemRamGb: 192, isDownloaded: notDownloaded })
    .format,
  "bf16",
);
assert.equal(
  pickDefaultArtifact(h3, { gpuGb: 122, systemRamGb: 96, isDownloaded: notDownloaded })
    .format,
  "gguf",
);
assert.equal(
  pickDefaultArtifact(h3, { gpuGb: 123, systemRamGb: 96, isDownloaded: notDownloaded })
    .format,
  "bf16",
);
// The upper tier, in the units the picker is actually handed: 132 GiB of VRAM is 141.7 decimal
// GB, past the 132 GB at which the backend estimator drops its host-RAM floor to 85 GB, and
// 85 GiB of RAM is 91.3 GB. So this host fits, and a tier table written in decimal GB would
// wrongly send it to GGUF.
assert.equal(
  pickDefaultArtifact(h3, { gpuGb: 132, systemRamGb: 85, isDownloaded: notDownloaded })
    .format,
  "bf16",
);

// ── official BF16 artifacts (added so groups are not unsloth-quant-only) ────────
// Qwen-Image-2512 BF16 (54 GB) misses a 24/48 GB budget (bnb-4bit/fp8 win there, asserted above) but on an 80 GB GPU (budget 56) it fits and wins.
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
// Z-Image-Turbo BF16 (30 GB) misses 24 GB (bnb-4bit wins) but fits a 48 GB GPU (budget 33.6) and wins.
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
// FLUX.1-dev BF16 (32 GB) fits a 48 GB GPU but is GATED, so a bare click routes to the open GGUF unless it is already downloaded. Small GPU -> GGUF.
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
// LTX-2.3 video carries the official BF16 single file (no FP8: the loader refuses its scaled-fp8 one), which keeps the ~50 GB Gemma3 encoder resident, so B200-class only.
// Looked up by the retired unsloth/LTX-2.3 id on purpose: it is no longer the canonicalId (that
// repo does not exist), and a pasted or persisted copy must still land on this group through the
// GGUF artifact's suffix-stripped key.
const ltxGroup = groupForRepoId("unsloth/LTX-2.3", VIDEO_CATALOG);
assert.ok(ltxGroup);
assert.equal(ltxGroup.canonicalId, "Lightricks/LTX-2.3");
assert.equal(groupForRepoId("Lightricks/LTX-2.3", VIDEO_CATALOG), ltxGroup);
// The retired id is not an artifact, so it can never be handed to a load as a repo to fetch.
assert.equal(loadSpecFor("unsloth/LTX-2.3", VIDEO_CATALOG), null);
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
// A GGUF in the group is always runnable (its quant ladder self-fits + offloads), so LTX-2.3 stays visible on a tiny card despite its 90 GB BF16 sibling.
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

// ── audio catalog: task tags, grouping, load specs ─────────────────────────────

// Every audio group carries a task tag; the other catalogs carry none.
for (const group of AUDIO_CATALOG) {
  assert.ok(group.task === "tts" || group.task === "stt", `audio group ${group.canonicalId} needs a task`);
  assert.equal(group.scope, "audio");
}
for (const catalog of [IMAGE_CATALOG, VIDEO_CATALOG]) {
  for (const group of catalog) {
    assert.equal(group.task, undefined, `non-audio group ${group.canonicalId} must not carry a task`);
  }
}
// The Orpheus GGUF groups with its safetensors base and reports the gguf load kind.
const orpheus = groupForRepoId("unsloth/orpheus-3b-0.1-ft-GGUF", AUDIO_CATALOG);
assert.ok(orpheus);
assert.equal(orpheus.canonicalId, "unsloth/orpheus-3b-0.1-ft");
assert.equal(orpheus.task, "tts");
assert.equal(loadSpecFor("unsloth/orpheus-3b-0.1-ft-GGUF", AUDIO_CATALOG)?.kind, "gguf");
assert.equal(loadSpecFor("unsloth/csm-1b", AUDIO_CATALOG)?.kind, "pipeline");
// The whisper sidecar repos resolve as stt groups.
assert.equal(groupForRepoId("unsloth/whisper-large-v3-turbo", AUDIO_CATALOG)?.task, "stt");
assert.equal(groupForRepoId("unslothai/Qwen3-ASR-0.6B-GGUF", AUDIO_CATALOG)?.task, "stt");
// A chat model stays unknown to the audio catalog.
assert.equal(groupForRepoId("unsloth/Llama-3.3-70B-GGUF", AUDIO_CATALOG), null);

// ── groupMatchesQuery ──────────────────────────────────────────────────────────

assert.ok(groupMatchesQuery(qwenGroup, "qwen"));
assert.ok(groupMatchesQuery(qwenGroup, "2512"));
assert.ok(groupMatchesQuery(qwenGroup, "gguf"));
// Still reachable by "fp8" and by the full prequant repo id, now through the group alias rather
// than an artifact row.
assert.ok(groupMatchesQuery(qwenGroup, "fp8"));
assert.ok(groupMatchesQuery(qwenGroup, "4bit"));
assert.ok(groupMatchesQuery(qwenGroup, "q4_k_m"));
assert.ok(groupMatchesQuery(qwenGroup, "unsloth/qwen-image-2512-fp8"));
assert.ok(!groupMatchesQuery(qwenGroup, "mlx"));
assert.ok(!groupMatchesQuery(qwenGroup, "ideogram"));
assert.ok(groupMatchesQuery(ltx23, "ltx"));
assert.ok(groupMatchesQuery(ltx23, "lightricks/ltx-2.3"));

console.log("model-catalog check: all assertions passed");

// ── opt-in: does the Hub agree? (`--network`) ─────────────────────────────────
//
// Every failure this catches was reported by hand: a preconfigured link that 401s because its
// repo is gated and the entry never said so, or 404s because the repo/file was renamed or never
// published. Neither is visible to a pure-data check, and both surface to the user as a download
// that dies partway through.
//
// Anonymous on purpose: that is what a fresh install sees, and it is exactly the request whose
// 401 the `gated` flag exists to predict. Retries cover Hub flakiness; only a definitive verdict
// (a repo that answers "gone", a filename the repo does not list, a `gated` flag that disagrees
// with the API) fails the run.

const HF_API = "https://huggingface.co/api/models";
const HF_RESOLVE = "https://huggingface.co";
const NETWORK_ATTEMPTS = 3;
/** Per-attempt wall clock, headers and body together. */
const NETWORK_TIMEOUT_MS = 20_000;
/**
 * Wall clock for the whole network pass, comfortably under the workflow's 10-minute
 * timeout-minutes.
 *
 * Bounding each attempt is not enough on its own: ~53 repos at NETWORK_BATCH 4 is 14 serial
 * batches, and a peer that stalls every request costs 3 x 20s plus 1.5s of backoff per batch,
 * so a fully stalled Hub takes about 14.3 minutes to fail open -- and the job is killed at 10,
 * which is the red run this retry logic exists to avoid. Past this deadline every remaining
 * request short-circuits to "no opinion", so the check still reaches its warnings and exits 0.
 */
const NETWORK_DEADLINE_MS = 7 * 60 * 1000;
/** Set when the network pass starts; Infinity keeps the offline assertions unbounded. */
let networkDeadlineAt = Number.POSITIVE_INFINITY;
const NETWORK_BATCH = 4;

interface HubRepo {
  /** false for an open repo; "auto" / "manual" for a gated one. */
  gated?: boolean | string;
  siblings?: { rfilename: string }[];
}

const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

/**
 * Fetch with retries. Never throws and never rejects: an answer the Hub could not give is
 * `{ response: null }`, which callers must treat as "no opinion". A rate limit, a 5xx window or a
 * DNS blip is the Hub having a moment, and turning one of those into a red nightly is exactly
 * what this whole check promises not to do.
 */
async function fetchWithRetry(
  url: string,
  init?: RequestInit,
): Promise<{ response: Response | null; body: string; why: string }> {
  let why = "unknown";
  for (let attempt = 1; attempt <= NETWORK_ATTEMPTS; attempt++) {
    if (Date.now() >= networkDeadlineAt) {
      return { response: null, body: "", why: "the network check ran out of its overall budget" };
    }
    try {
      // A per-attempt deadline, or a peer that accepts the connection and then stalls has no
      // retry and no fail-open: fetch simply never settles, and the scheduled job dies on its own
      // 10-minute timeout as a RED run. Which is the one outcome this check promises not to
      // produce for a network blip.
      const response = await fetch(url, { ...init, signal: AbortSignal.timeout(NETWORK_TIMEOUT_MS) });
      // Read the body HERE, under that same deadline. Headers can arrive promptly and the body
      // still stall, and the caller parses outside this function where an abort would surface as
      // "the Hub answered 200 with unreadable JSON" -- a catalog failure, for a stalled socket.
      // It also drains the 429/5xx responses we are about to retry, rather than leaking them.
      const body = init?.method === "HEAD" ? "" : await response.text();
      if (response.status !== 429 && response.status < 500) return { response, body, why: "" };
      why = `HTTP ${response.status}`;
    } catch (err) {
      why = String(err);
    }
    if (attempt < NETWORK_ATTEMPTS) await sleep(500 * 2 ** (attempt - 1));
  }
  return { response: null, body: "", why };
}

/** Run `work` over `items`, a few at a time: ~35 repos, and the Hub does not need a thundering herd. */
async function inBatches<T>(items: T[], work: (item: T) => Promise<void>): Promise<void> {
  for (let i = 0; i < items.length; i += NETWORK_BATCH) {
    await Promise.all(items.slice(i, i + NETWORK_BATCH).map(work));
  }
}

async function checkCatalogAgainstTheHub(catalogs: CatalogGroup[][]): Promise<string[]> {
  networkDeadlineAt = Date.now() + NETWORK_DEADLINE_MS;
  const failures: string[] = [];
  const groups = catalogs.flat();
  // One metadata call per repo id, however many artifacts share it.
  const artifactsByRepo = new Map<string, ModelArtifact[]>();
  for (const group of groups) {
    for (const artifact of group.artifacts) {
      const bucket = artifactsByRepo.get(artifact.repoId);
      if (bucket) bucket.push(artifact);
      else artifactsByRepo.set(artifact.repoId, [artifact]);
    }
  }

  await inBatches([...artifactsByRepo.entries()], async ([repoId, artifacts]) => {
    const { response, body, why } = await fetchWithRetry(`${HF_API}/${repoId}`);
    if (response === null) {
      console.warn(
        `::warning::${repoId}: the Hub did not answer after ${NETWORK_ATTEMPTS} attempts (${why}). Not a verdict about the catalog.`,
      );
      return;
    }
    if (!response.ok) {
      // 401 on the METADATA endpoint means private-or-absent (the Hub will not say which);
      // a gated-but-public repo answers 200 with gated set, so this is never just "needs a licence".
      failures.push(
        `${repoId}: HTTP ${response.status} from ${HF_API}/${repoId} -- the repo is missing, renamed or private, so no user can download it`,
      );
      return;
    }
    let repo: HubRepo;
    try {
      repo = JSON.parse(body) as HubRepo;
    } catch (err) {
      // A 200 that is not JSON is a captive portal or a proxy, not a catalog problem. Reject the
      // run rather than throwing out of Promise.all with a raw stack.
      failures.push(`${repoId}: the Hub answered 200 with unreadable JSON (${err})`);
      return;
    }
    const hubGated = Boolean(repo.gated);
    for (const artifact of artifacts) {
      const declaredGated = artifact.gated === true;
      if (hubGated && !declaredGated) {
        failures.push(
          `${repoId}: the Hub reports gated=${JSON.stringify(repo.gated)} but the catalog entry has no \`gated: true\`, so an anonymous download 401s`,
        );
      } else if (!hubGated && declaredGated) {
        failures.push(
          `${repoId}: marked \`gated: true\` but the Hub reports it open -- the router skips it for no reason`,
        );
      }
    }

    const declaredFiles = [
      ...new Set(artifacts.map((a) => a.filename).filter((f): f is string => Boolean(f))),
    ];
    if (declaredFiles.length === 0) return;
    const listed = new Set((repo.siblings ?? []).map((s) => s.rfilename));
    for (const filename of declaredFiles) {
      if (!listed.has(filename)) {
        failures.push(`${repoId}: declares '${filename}', which the repo does not contain`);
        continue;
      }
      if (hubGated) continue; // resolve/ 401s without a token; the sibling list is the check.
      // Listed is not the same as fetchable: resolve/ is the endpoint the download hits.
      const head = await fetchWithRetry(`${HF_RESOLVE}/${repoId}/resolve/main/${filename}`, {
        method: "HEAD",
      });
      if (head.response === null) {
        console.warn(
          `::warning::${repoId}: could not HEAD '${filename}' (${head.why}). Not a verdict.`,
        );
      } else if (!head.response.ok) {
        failures.push(
          `${repoId}: '${filename}' is listed but resolve/main returned HTTP ${head.response.status}`,
        );
      }
    }
  });

  // Advisory only. A canonicalId is a display/grouping key, and 15 of them are deliberately not
  // repos; but one that is BOTH `unsloth/*`-shaped and dead clears every owner guard in the app,
  // so it is worth naming without failing a scheduled job over it.
  const artifactIds = new Set([...artifactsByRepo.keys()].map((id) => id.toLowerCase()));
  const orphans = groups
    .map((g) => g.canonicalId)
    .filter((id) => !artifactIds.has(id.toLowerCase()));
  await inBatches(orphans, async (canonicalId) => {
    const { response } = await fetchWithRetry(`${HF_API}/${canonicalId}`);
    // A null response is an unreachable Hub, which is simply no advice.
    if (response !== null && !response.ok) {
      console.warn(
        `::warning::canonicalId '${canonicalId}' is not a real repo (HTTP ${response.status}). Harmless as a grouping key, but it must never reach a load.`,
      );
    }
  });

  return failures;
}

if (process.argv.includes("--network")) {
  console.log("model-catalog check: --network, asking the Hub about every declared artifact...");
  const failures = await checkCatalogAgainstTheHub([IMAGE_CATALOG, VIDEO_CATALOG]);
  if (failures.length > 0) {
    for (const failure of failures) console.error(`::error::${failure}`);
    console.error(`model-catalog network check: ${failures.length} problem(s)`);
    process.exit(1);
  }
  console.log("model-catalog network check: every declared repo, file and gated flag agrees with the Hub");
}
