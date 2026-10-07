// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// `--network` adds an opt-in Hub reachability pass, kept out of `npm run catalog:check`
// so a Hub hiccup cannot fail an unrelated PR.

import assert from "node:assert/strict";

import {
  AUDIO_CPP_MODELS,
  AUDIO_CPP_REPO,
  AUDIO_CPP_UNOFFERED_FOLDERS,
  audioCppDisplayName,
  isAudioCppFolderId,
} from "../../../audio/audio-cpp-catalog.ts";

import type { CatalogGroup, ModelArtifact } from "./model-catalog.ts";
import {
  AUDIO_CATALOG,
  artifactForRepoId,
  IMAGE_CATALOG,
  VIDEO_CATALOG,
  canonicalKeyFor,
  catalogGroupFitsDevice,
  catalogToModelOptions,
  classifyGgufFit,
  classifyMediaGgufFit,
  curatedArtifactFitsDevice,
  curatedDisplayNameFor,
  curatedArtifactTakesDenseQuant,
  curatedRowLabelFor,
  ggufFitRuns,
  groupForRepoId,
  groupMatchesQuery,
  loadSpecFor,
  pickDefaultArtifact,
  pickDefaultQuant,
  stripArtifactSuffixesForDisplay,
} from "./model-catalog.ts";


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
assert.equal(canonicalKeyFor("unsloth/Qwen-Image-2512-int8"), "unsloth/qwen-image-2512");
assert.equal(canonicalKeyFor("unsloth/Qwen-Image-2512-INT8"), "unsloth/qwen-image-2512");
assert.equal(canonicalKeyFor("unsloth/Qwen-Image-2512-nvfp4"), "unsloth/qwen-image-2512");
assert.equal(canonicalKeyFor("unsloth/Qwen-Image-2512-NVFP4"), "unsloth/qwen-image-2512");
assert.equal(canonicalKeyFor("unsloth/qwen-image-2512-gguf"), "unsloth/qwen-image-2512");
assert.equal(canonicalKeyFor("unsloth/qwen-image-2512-fp8"), "unsloth/qwen-image-2512");


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
assert.equal(
  stripArtifactSuffixesForDisplay("krea/Krea-2-Turbo"),
  "krea/Krea-2-Turbo",
);
assert.equal(stripArtifactSuffixesForDisplay("someone/FP8"), "someone/FP8");
assert.equal(canonicalKeyFor("krea/Krea-2-Turbo"), "krea/krea-2-turbo");
assert.notEqual(
  canonicalKeyFor("Qwen/Qwen-Image-2512"),
  canonicalKeyFor("unsloth/Qwen-Image-2512"),
);
assert.equal(canonicalKeyFor("someone/fp8"), "someone/fp8");


const qwen2512 = groupForRepoId("unsloth/Qwen-Image-2512-GGUF", IMAGE_CATALOG);
assert.ok(qwen2512);
assert.equal(qwen2512.canonicalId, "unsloth/Qwen-Image-2512");
for (const artifact of qwen2512.artifacts) {
  assert.equal(groupForRepoId(artifact.repoId, IMAGE_CATALOG), qwen2512);
}
assert.equal(groupForRepoId("Qwen/Qwen-Image-2512", IMAGE_CATALOG), qwen2512);
assert.equal(groupForRepoId("unsloth/Qwen-Image-2512-INT8", IMAGE_CATALOG), qwen2512);
assert.equal(groupForRepoId("unsloth/Qwen-Image-2512-NVFP4", IMAGE_CATALOG), qwen2512);
assert.equal(
  groupForRepoId("Tongyi-MAI/Z-Image-Turbo", IMAGE_CATALOG)?.canonicalId,
  "unsloth/Z-Image-Turbo",
);
assert.equal(groupForRepoId("Qwen/Qwen-Image-2512-FP8", IMAGE_CATALOG), qwen2512);
// A dotted version is a different model and must not fall back to the undotted group.
const qwen21 = groupForRepoId("Qwen/Qwen-Image-2.1", IMAGE_CATALOG);
assert.ok(qwen21);
assert.equal(qwen21.canonicalId, "unsloth/Qwen-Image-2.1");
assert.equal(groupForRepoId("unsloth/Qwen-Image-2.1-FP8", IMAGE_CATALOG), qwen21);
assert.equal(groupForRepoId("unsloth/Qwen-Image-2.1-INT8", IMAGE_CATALOG), qwen21);
assert.notEqual(groupForRepoId("Qwen/Qwen-Image", IMAGE_CATALOG), qwen21);
assert.equal(
  groupForRepoId("Qwen/Qwen-Image", IMAGE_CATALOG)?.canonicalId,
  "unsloth/Qwen-Image",
);
assert.equal(groupForRepoId("someone/some-model-GGUF", IMAGE_CATALOG), null);
assert.equal(groupForRepoId("unsloth/Llama-3.3-70B-GGUF", VIDEO_CATALOG), null);
const ltx23 = groupForRepoId("unsloth/LTX-2.3-GGUF", VIDEO_CATALOG);
assert.ok(ltx23);
assert.equal(groupForRepoId("lightricks/ltx-2.3", VIDEO_CATALOG), ltx23);
assert.equal(groupForRepoId("lightricks/ltx-2.3-fp8", VIDEO_CATALOG), ltx23);
assert.notEqual(groupForRepoId("Lightricks/LTX-2", VIDEO_CATALOG), ltx23);
assert.notEqual(
  groupForRepoId("stabilityai/sdxl-turbo", IMAGE_CATALOG),
  groupForRepoId("stabilityai/stable-diffusion-xl-base-1.0", IMAGE_CATALOG),
);
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
assert.equal(loadSpecFor("unsloth/Z-Image-Turbo-GGUF", IMAGE_CATALOG)?.kind, "gguf");
assert.equal(loadSpecFor("someone/unknown", IMAGE_CATALOG), null);

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
  const offered = artifactForRepoId(id, IMAGE_CATALOG)?.artifact.repoId ?? id;
  assert.ok(imageOptionIds.has(offered), `image option missing: ${id}`);
}
const videoOptionIds = new Set(catalogToModelOptions(VIDEO_CATALOG).map((o) => o.id));
for (const id of [
  "unsloth/LTX-2.3-GGUF",
  "unsloth/MiniMax-H3-GGUF",
  ...OLD_PIPELINE_MODELS,
]) {
  assert.ok(videoOptionIds.has(id), `video option missing: ${id}`);
}

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

assert.deepEqual(curatedRowLabelFor("MiniMaxAI/MiniMax-H3", VIDEO_CATALOG), {
  name: "MiniMax H3",
  tags: ["BF16"],
});
assert.deepEqual(curatedRowLabelFor("unsloth/MiniMax-H3-GGUF", VIDEO_CATALOG), {
  name: "MiniMax-H3-GGUF",
  tags: [],
});
assert.deepEqual(curatedRowLabelFor("unsloth/Z-Image-Turbo-GGUF", IMAGE_CATALOG), {
  name: "Z-Image-Turbo-GGUF",
  tags: [],
});

assert.deepEqual(curatedRowLabelFor("MiniMaxAI/MiniMax-H3", VIDEO_CATALOG, "accelerated"), {
  name: "MiniMax H3 (Fast)",
  tags: ["BF16"],
});
assert.deepEqual(
  curatedRowLabelFor("unsloth/MiniMax-H3-GGUF", VIDEO_CATALOG, "accelerated"),
  { name: "MiniMax-H3-GGUF (Slow)", tags: [] },
);
assert.deepEqual(
  curatedRowLabelFor("unsloth/MiniMax-H3-GGUF", VIDEO_CATALOG, "gguf-only"),
  { name: "MiniMax-H3-GGUF", tags: [] },
);
// The trigger and the row must agree, or the model renames itself as the popover opens.
assert.equal(
  curatedDisplayNameFor("MiniMaxAI/MiniMax-H3", VIDEO_CATALOG, "accelerated"),
  "MiniMax H3 (Fast)",
);
assert.equal(
  curatedDisplayNameFor("unsloth/MiniMax-H3-GGUF", VIDEO_CATALOG, "accelerated"),
  "MiniMax-H3-GGUF (Slow)",
);
assert.deepEqual(
  curatedRowLabelFor("Lightricks/LTX-2", VIDEO_CATALOG, "accelerated"),
  curatedRowLabelFor("Lightricks/LTX-2", VIDEO_CATALOG),
);
assert.deepEqual(
  curatedRowLabelFor("MiniMaxAI/MiniMax-H3", VIDEO_CATALOG, "dense-quant"),
  curatedRowLabelFor("MiniMaxAI/MiniMax-H3", VIDEO_CATALOG, "accelerated"),
);

assert.deepEqual(curatedRowLabelFor("Tongyi-MAI/Z-Image-Turbo", IMAGE_CATALOG, "dense-quant"), {
  name: "Z-Image-Turbo (Fast)",
  tags: ["BF16"],
});
assert.equal(
  curatedDisplayNameFor("Tongyi-MAI/Z-Image-Turbo", IMAGE_CATALOG, "dense-quant"),
  "Z-Image-Turbo (Fast)",
);
assert.deepEqual(
  curatedRowLabelFor("unsloth/Z-Image-Turbo-unsloth-bnb-4bit", IMAGE_CATALOG, "dense-quant"),
  { name: "Z-Image-Turbo", tags: ["bnb-4bit"] },
);
// The native engine has no low-precision tensor-core path, so every diffusion GGUF is slow.
assert.deepEqual(curatedRowLabelFor("unsloth/Z-Image-Turbo-GGUF", IMAGE_CATALOG, "dense-quant"), {
  name: "Z-Image-Turbo-GGUF (Slow)",
  tags: [],
});
for (const host of ["accelerated", "gguf-only", "unknown"] as const) {
  assert.deepEqual(
    curatedRowLabelFor("Tongyi-MAI/Z-Image-Turbo", IMAGE_CATALOG, host),
    { name: "Z-Image-Turbo", tags: ["BF16"] },
    host,
  );
}
for (const id of ["stabilityai/sdxl-turbo", "stabilityai/stable-diffusion-xl-base-1.0"]) {
  const row = curatedRowLabelFor(id, IMAGE_CATALOG, "dense-quant");
  assert.ok(row && !row.name.includes("(Fast)"), `${id} reads "${row?.name}"`);
}

for (const [label, catalog] of [
  ["image", IMAGE_CATALOG],
  ["audio", AUDIO_CATALOG],
] as const) {
  for (const group of catalog) {
    for (const artifact of group.artifacts) {
      if (artifact.format !== "bf16" || artifact.loadKind !== "pipeline") continue;
      const row = curatedRowLabelFor(artifact.repoId, catalog, "dense-quant");
      // Word boundary: `qualify` skips the bracket when the variant name already says Fast.
      const claims = /\bFast\b/.test(row?.name ?? "");
      assert.equal(
        Boolean(claims),
        label === "image" && artifact.denseQuantable === true,
        `${label}: ${artifact.repoId} reads "${row?.name}" ${JSON.stringify(row?.tags)}`,
      );
    }
  }
}

for (const [id, expected] of [
  ["unsloth/Z-Image-Turbo-GGUF", true],
  ["Tongyi-MAI/Z-Image-Turbo", true],
  ["krea/Krea-2-Turbo", true],
  ["stabilityai/sdxl-turbo", false],
  ["stabilityai/stable-diffusion-xl-base-1.0", false],
  ["unsloth/Z-Image-Turbo-unsloth-bnb-4bit", false],
  ["ideogram-ai/ideogram-4-fp8", false],
  ["ideogram-ai/ideogram-4-nf4-diffusers", false],
] as const) {
  assert.equal(curatedArtifactTakesDenseQuant(id, IMAGE_CATALOG), expected, id);
}
assert.equal(curatedArtifactTakesDenseQuant("someone/pasted", IMAGE_CATALOG), undefined);

for (const group of IMAGE_CATALOG) {
  for (const artifact of group.artifacts) {
    const row = curatedRowLabelFor(artifact.repoId, IMAGE_CATALOG, "dense-quant");
    if (!/\bFast\b/.test(row?.name ?? "")) continue;
    assert.notEqual(
      curatedArtifactTakesDenseQuant(artifact.repoId, IMAGE_CATALOG),
      false,
      artifact.repoId,
    );
  }
}

for (const id of [
  "krea/Krea-2-Turbo",
  "Alpha-VLLM/Lumina-Image-2.0",
  "hunyuanvideo-community/HunyuanImage-2.1-Diffusers",
]) {
  const group = groupForRepoId(id, IMAGE_CATALOG);
  assert.equal(group?.artifacts.length, 1, id);
  for (const host of ["dense-quant", "accelerated"] as const) {
    assert.deepEqual(curatedRowLabelFor(id, IMAGE_CATALOG, host)?.tags, [], `${id} ${host}`);
  }
  assert.ok(curatedRowLabelFor(id, IMAGE_CATALOG, "dense-quant")?.name.includes("(Fast)"), id);
  assert.equal(curatedRowLabelFor(id, IMAGE_CATALOG, "accelerated")?.name.includes("(Fast)"), false, id);
}

// The row states a host capability only; the actual precision depends on many inputs reported after load.
{
  const id = "unsloth/Z-Image-Turbo";
  assert.deepEqual(curatedRowLabelFor(id, IMAGE_CATALOG, "dense-quant"), {
    name: "Z-Image-Turbo (Fast)",
    tags: ["BF16"],
  });
  assert.deepEqual(curatedRowLabelFor(id, IMAGE_CATALOG, "accelerated"), {
    name: "Z-Image-Turbo",
    tags: ["BF16"],
  });
  for (const catalog of [IMAGE_CATALOG, VIDEO_CATALOG]) {
    for (const group of catalog) {
      for (const artifact of group.artifacts) {
        const dense = curatedRowLabelFor(artifact.repoId, catalog, "dense-quant")?.tags ?? [];
        const plain = curatedRowLabelFor(artifact.repoId, catalog, "accelerated")?.tags ?? [];
        assert.deepEqual(dense, plain, artifact.repoId);
        assert.equal(dense.includes("FP8 / INT8"), false, artifact.repoId);
      }
    }
  }
  assert.equal(
    curatedDisplayNameFor(id, IMAGE_CATALOG, "dense-quant"),
    curatedRowLabelFor(id, IMAGE_CATALOG, "dense-quant")?.name,
  );
  const rows = catalogToModelOptions(IMAGE_CATALOG, "dense-quant");
  assert.equal(rows.find((o) => o.id === id)?.name, "Z-Image-Turbo (Fast)");
}

assert.deepEqual(
  curatedRowLabelFor("HiDream-ai/HiDream-I1-Fast", IMAGE_CATALOG, "dense-quant"),
  { name: "HiDream I1 (Fast (distilled))", tags: ["BF16"] },
);

for (const id of [
  "unsloth/Z-Image-Turbo-unsloth-bnb-4bit",
  "ideogram-ai/ideogram-4-fp8",
]) {
  const row = curatedRowLabelFor(id, IMAGE_CATALOG, "dense-quant");
  assert.ok(row && !row.name.includes("(Fast)"), `${id} reads "${row?.name}"`);
}

// Non-GGUF is not the test: diffusion runs on MPS and STT via the whisper.cpp sidecar.
for (const [label, catalog, refused] of [
  ["video", VIDEO_CATALOG, ["MiniMaxAI/MiniMax-H3"]],
  ["image", IMAGE_CATALOG, []],
  ["audio", AUDIO_CATALOG, []],
] as const) {
  const all = catalogToModelOptions(catalog).map((o) => o.id);
  const offered = catalogToModelOptions(catalog, "gguf-only").map((o) => o.id);
  assert.deepEqual(
    all.filter((id) => !offered.includes(id)),
    [...refused],
    `${label}: a gguf-only host lost a row it can load`,
  );
}
// A Mac must never open the picker with a whole model family missing.
for (const [label, catalog] of [
  ["video", VIDEO_CATALOG],
  ["image", IMAGE_CATALOG],
  ["audio", AUDIO_CATALOG],
] as const) {
  const offered = new Set(catalogToModelOptions(catalog, "gguf-only").map((o) => o.id));
  for (const group of catalog) {
    assert.ok(
      group.artifacts.some((artifact) => offered.has(artifact.repoId)),
      `${label}: "${group.displayName}" vanished on a gguf-only host`,
    );
  }
}
assert.deepEqual(
  catalogToModelOptions(VIDEO_CATALOG, "unknown").map((o) => o.id),
  catalogToModelOptions(VIDEO_CATALOG).map((o) => o.id),
);
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
assert.deepEqual(curatedRowLabelFor("Lightricks/LTX-2", VIDEO_CATALOG), {
  name: "LTX 2 (base)",
  tags: [],
});
assert.equal(curatedRowLabelFor("someone/not-in-the-catalog", VIDEO_CATALOG), null);

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

for (const catalog of [IMAGE_CATALOG, VIDEO_CATALOG, AUDIO_CATALOG]) {
  for (const group of catalog) {
    for (const artifact of group.artifacts) {
      for (const text of [
        artifact.label,
        curatedDisplayNameFor(artifact.repoId, catalog) ?? "",
        ...catalogToModelOptions(catalog)
          .filter((o) => o.id === artifact.repoId)
          .map((o) => o.description ?? ""),
      ]) {
        assert.equal(
          /official/i.test(text),
          false,
          `${artifact.repoId} still reads "${text}"`,
        );
      }
    }
  }
}

// Otherwise the VRAM badge falls back to the QLoRA estimator, which misreads pipelines.
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


const WAN = "Wan-AI/Wan2.2-TI2V-5B-Diffusers";  // 30 GB, no offload tiers
const H3 = "MiniMaxAI/MiniMax-H3";  // 145 GB, tiers at 74/140 and 123/80
const fitsCurated = (id: string, gpuGb: number, systemRamGb: number) =>
  curatedArtifactFitsDevice(id, VIDEO_CATALOG, { gpuGb, systemRamGb });

// RAM is not a discrete GPU's budget: an untiered pipeline goes wholly on the card.
assert.equal(fitsCurated(WAN, 48, 0), true);
assert.equal(fitsCurated(WAN, 40, 0), false);
assert.equal(fitsCurated(WAN, 12, 64), false);
// A unified-memory host reports RAM and no GPU, and there the RAM is the card.
assert.equal(fitsCurated(WAN, 0, 64), true);
assert.equal(fitsCurated(WAN, 0, 0), undefined);
// Measured offload tiers override the 70% rule both ways.
assert.equal(fitsCurated(H3, 74, 140), true);
assert.equal(fitsCurated(H3, 123, 80), true);
assert.equal(fitsCurated(H3, 74, 100), false);
assert.equal(fitsCurated("unsloth/MiniMax-H3-GGUF", 12, 64), undefined);
assert.equal(fitsCurated("someone/not-in-the-catalog", 12, 64), undefined);
// STT retries on CPU (stt_sidecar.py) so RAM counts; TTS rejects CPU offload, judged on the card.
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
// The budget is the larger of card and RAM, never their sum.
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


const GB = 1024 ** 3;
// Delegated to lib/gguf-fit, the Hub badge formula.
assert.equal(classifyGgufFit(10 * GB, { gpuGb: 24, systemRamGb: 64 }), "fits");
assert.equal(classifyGgufFit(20 * GB, { gpuGb: 24, systemRamGb: 64 }), "marginal");
assert.equal(classifyGgufFit(40 * GB, { gpuGb: 24, systemRamGb: 64 }), "partial");
assert.equal(classifyGgufFit(100 * GB, { gpuGb: 24, systemRamGb: 64 }), "oom");
assert.equal(classifyGgufFit(100 * GB, { gpuGb: 0, systemRamGb: 0 }), "fits");
assert.equal(classifyGgufFit(20 * GB, { gpuGb: 0, systemRamGb: 64 }), "ram");
assert.equal(classifyGgufFit(60 * GB, { gpuGb: 0, systemRamGb: 64 }), "oom");
assert.equal(
  classifyGgufFit(16 * GB, { gpuGb: 24, systemRamGb: 0, budgetFraction: 0.97 }),
  "fits",
);
assert.equal(
  classifyGgufFit(16 * GB, { gpuGb: 24, systemRamGb: 0, budgetFraction: 0.8 }),
  "marginal",
);
// At the top of the slider the loader keeps its 512 MiB floor.
assert.equal(
  classifyGgufFit(20 * GB, { gpuGb: 24, systemRamGb: 0, budgetFraction: 1 }),
  "marginal",
);
assert.equal(
  classifyGgufFit(19 * GB, { gpuGb: 24, systemRamGb: 0, budgetFraction: 0.97 }),
  "fits",
);
// The floor is charged once per card (_select_gpus sums per-device usable MiB).
assert.equal(
  classifyGgufFit(40.2 * GB, {
    gpuGb: 48,
    systemRamGb: 0,
    budgetFraction: 1,
    gpuCount: 2,
  }),
  "marginal",
);
assert.equal(
  classifyGgufFit(40.2 * GB, { gpuGb: 48, systemRamGb: 0, budgetFraction: 1 }),
  "fits",
);
for (const gpuCount of [1, 2, 4]) {
  assert.equal(
    classifyGgufFit(39 * GB, {
      gpuGb: 48,
      systemRamGb: 0,
      budgetFraction: 0.97,
      gpuCount,
    }),
    "fits",
  );
}

// Media GGUFs use the diffusion planner budget (diffusion_memory.py), not the llama.cpp rule.
assert.equal(classifyMediaGgufFit(40 * GB, 64, 0), "fits");  // 40 <= 44.8
assert.equal(classifyMediaGgufFit(50 * GB, 64, 0), "oom");  // past 44.8, no RAM tier
assert.equal(classifyGgufFit(50 * GB, { gpuGb: 64, systemRamGb: 0 }), "fits");
assert.equal(classifyMediaGgufFit(20 * GB, 24, 64), "partial");
assert.equal(classifyMediaGgufFit(100 * GB, 24, 64), "oom");
assert.equal(classifyMediaGgufFit(30 * GB, 0, 64), "fits");
assert.equal(classifyMediaGgufFit(60 * GB, 0, 64), "oom");

assert.equal(ggufFitRuns("partial"), true);
assert.equal(ggufFitRuns("oom"), false);


const variants = [
  { quant: "Q4_K_M", filename: "m-Q4_K_M.gguf", size_bytes: 12 * GB },
  { quant: "Q8_0", filename: "m-Q8_0.gguf", size_bytes: 22 * GB },
  { quant: "BF16", filename: "m-BF16.gguf", size_bytes: 40 * GB },
];
const budget24 = { gpuGb: 24, systemRamGb: 64 };
assert.equal(pickDefaultQuant(variants, "Q4_K_M", budget24)?.quant, "Q4_K_M");
assert.equal(
  pickDefaultQuant(
    [variants[0], { ...variants[1], downloaded: true }, variants[2]],
    "Q4_K_M",
    budget24,
  )?.quant,
  "Q8_0",
);
assert.equal(
  pickDefaultQuant(variants, "BF16", { gpuGb: 24, systemRamGb: 16 })?.quant,
  "Q8_0",
);
assert.equal(
  pickDefaultQuant(variants, "BF16", { gpuGb: 24, systemRamGb: 0 })?.quant,
  "Q4_K_M",
);
assert.equal(
  pickDefaultQuant(variants, "BF16", { gpuGb: 4, systemRamGb: 4 })?.quant,
  "Q4_K_M",
);
assert.equal(
  pickDefaultQuant(variants, "Q8_0", { gpuGb: 0, systemRamGb: 0 })?.quant,
  "Q8_0",
);
assert.equal(pickDefaultQuant([], "Q4_K_M", budget24), null);


const notDownloaded = () => false;
const qwenGroup = qwen2512;
assert.equal(
  pickDefaultArtifact(qwenGroup, { gpuGb: 8, systemRamGb: 32, isDownloaded: notDownloaded })
    .format,
  "gguf",
);
assert.equal(
  pickDefaultArtifact(qwenGroup, { gpuGb: 24, systemRamGb: 64, isDownloaded: notDownloaded })
    .format,
  "bnb-4bit",
);
// fp8 is family-denied for qwen-image (renders black) and its repo ships .pt, not safetensors.
assert.equal(
  pickDefaultArtifact(qwenGroup, { gpuGb: 48, systemRamGb: 64, isDownloaded: notDownloaded })
    .format,
  "bnb-4bit",
);
assert.equal(
  pickDefaultArtifact(qwenGroup, { gpuGb: 0, systemRamGb: 0, isDownloaded: notDownloaded })
    .format,
  "gguf",
);
assert.equal(
  pickDefaultArtifact(qwenGroup, {
    gpuGb: 48,
    systemRamGb: 64,
    isDownloaded: (id) => id === "unsloth/Qwen-Image-2512-unsloth-bnb-4bit",
  }).format,
  "bnb-4bit",
);
assert.equal(
  pickDefaultArtifact(qwenGroup, {
    gpuGb: 80,
    systemRamGb: 128,
    isDownloaded: (id) => id === "unsloth/Qwen-Image-2512-GGUF",
  }).format,
  "gguf",
);
const ideogram = groupForRepoId("ideogram-ai/ideogram-4-fp8", IMAGE_CATALOG);
assert.ok(ideogram);
assert.equal(
  pickDefaultArtifact(ideogram, { gpuGb: 24, systemRamGb: 64, isDownloaded: notDownloaded })
    .repoId,
  "ideogram-ai/ideogram-4-nf4-diffusers",
);
const fluxDevRoute = groupForRepoId("unsloth/FLUX.1-dev", IMAGE_CATALOG);
assert.ok(fluxDevRoute);
assert.equal(
  pickDefaultArtifact(fluxDevRoute, { gpuGb: 80, systemRamGb: 128, isDownloaded: notDownloaded })
    .repoId,
  "unsloth/FLUX.1-dev",
);
const kreaDevRoute = groupForRepoId("black-forest-labs/FLUX.1-Krea-dev", IMAGE_CATALOG);
assert.ok(kreaDevRoute);
assert.equal(
  pickDefaultArtifact(kreaDevRoute, { gpuGb: 80, systemRamGb: 128, isDownloaded: notDownloaded })
    .repoId,
  "unsloth/FLUX.1-Krea-dev",
);
assert.equal(
  groupForRepoId("QuantStack/FLUX.1-Krea-dev-GGUF", IMAGE_CATALOG),
  kreaDevRoute,
);
const lumina = groupForRepoId("Alpha-VLLM/Lumina-Image-2.0", IMAGE_CATALOG);
assert.ok(lumina);
assert.equal(
  pickDefaultArtifact(lumina, { gpuGb: 24, systemRamGb: 64, isDownloaded: notDownloaded })
    .repoId,
  "unsloth/Lumina-Image-2.0",
);
assert.equal(loadSpecFor("Alpha-VLLM/Lumina-Image-2.0", IMAGE_CATALOG)?.kind, "pipeline");
const hyimage = groupForRepoId(
  "hunyuanvideo-community/HunyuanImage-2.1-Diffusers",
  IMAGE_CATALOG,
);
assert.ok(hyimage);
// The QuantStack GGUF was unpublished; assert the group did not vanish with it.
assert.equal(
  pickDefaultArtifact(hyimage, { gpuGb: 24, systemRamGb: 64, isDownloaded: notDownloaded })
    .repoId,
  "hunyuanvideo-community/HunyuanImage-2.1-Diffusers",
);
assert.equal(
  pickDefaultArtifact(hyimage, { gpuGb: 141, systemRamGb: 128, isDownloaded: notDownloaded })
    .format,
  "bf16",
);
const hidream = groupForRepoId("HiDream-ai/HiDream-I1-Full", IMAGE_CATALOG);
assert.ok(hidream);
assert.equal(groupForRepoId("HiDream-ai/HiDream-I1-Dev", IMAGE_CATALOG), hidream);
assert.equal(groupForRepoId("HiDream-ai/HiDream-I1-Fast", IMAGE_CATALOG), hidream);
assert.equal(
  pickDefaultArtifact(hidream, { gpuGb: 141, systemRamGb: 128, isDownloaded: notDownloaded })
    .repoId,
  "unsloth/HiDream-I1-Full",
);
assert.equal(
  catalogGroupFitsDevice(hidream, { gpuGb: 24, systemRamGb: 32 }, notDownloaded),
  false,
);
const fluxSchnellRoute = groupForRepoId("unsloth/FLUX.1-schnell", IMAGE_CATALOG);
assert.ok(fluxSchnellRoute);
assert.equal(
  pickDefaultArtifact(fluxSchnellRoute, { gpuGb: 80, systemRamGb: 128, isDownloaded: notDownloaded })
    .format,
  "bf16",
);
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
// Same-format artifacts keep declaration order.
assert.equal(
  pickDefaultArtifact(hunyuan, { gpuGb: 60, systemRamGb: 128, isDownloaded: notDownloaded })
    .label,
  "BF16 - 480p",
);

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
// Tiers are GiB while the estimator is decimal GB; a decimal tier table would misroute this host.
assert.equal(
  pickDefaultArtifact(h3, { gpuGb: 132, systemRamGb: 85, isDownloaded: notDownloaded })
    .format,
  "bf16",
);
// The 30 GiB tier streams an int8 denoiser, which needs torchao group offload (diffusers >= 0.40).
for (const quantisedStreaming of [undefined, false]) {
  assert.equal(
    pickDefaultArtifact(h3, {
      gpuGb: 32,
      systemRamGb: 80,
      quantisedStreaming,
      isDownloaded: notDownloaded,
    }).format,
    "gguf",
  );
  assert.equal(
    curatedArtifactFitsDevice(H3, VIDEO_CATALOG, {
      gpuGb: 32,
      systemRamGb: 80,
      quantisedStreaming,
    }),
    false,
  );
}
assert.equal(
  pickDefaultArtifact(h3, {
    gpuGb: 32,
    systemRamGb: 80,
    quantisedStreaming: true,
    isDownloaded: notDownloaded,
  }).format,
  "bf16",
);
assert.equal(
  curatedArtifactFitsDevice(H3, VIDEO_CATALOG, {
    gpuGb: 32,
    systemRamGb: 80,
    quantisedStreaming: true,
  }),
  true,
);
assert.equal(
  curatedArtifactFitsDevice(H3, VIDEO_CATALOG, {
    gpuGb: 74,
    systemRamGb: 140,
    quantisedStreaming: false,
  }),
  true,
);

assert.equal(
  pickDefaultArtifact(qwenGroup, { gpuGb: 80, systemRamGb: 128, isDownloaded: notDownloaded })
    .format,
  "bf16",
);
assert.equal(
  pickDefaultArtifact(qwenGroup, { gpuGb: 80, systemRamGb: 128, isDownloaded: notDownloaded })
    .repoId,
  "unsloth/Qwen-Image-2512",
);
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
const fluxDev = groupForRepoId("black-forest-labs/FLUX.1-dev", IMAGE_CATALOG);
assert.ok(fluxDev);
assert.equal(fluxDev.canonicalId, "unsloth/FLUX.1-dev");
assert.equal(
  pickDefaultArtifact(fluxDev, { gpuGb: 48, systemRamGb: 64, isDownloaded: notDownloaded })
    .format,
  "bf16",
);
assert.equal(
  pickDefaultArtifact(fluxDev, { gpuGb: 24, systemRamGb: 64, isDownloaded: notDownloaded })
    .format,
  "gguf",
);
// Looked up by the retired unsloth/LTX-2.3 id on purpose: a pasted copy must still resolve.
const ltxGroup = groupForRepoId("unsloth/LTX-2.3", VIDEO_CATALOG);
assert.ok(ltxGroup);
assert.equal(ltxGroup.canonicalId, "Lightricks/LTX-2.3");
assert.equal(groupForRepoId("Lightricks/LTX-2.3", VIDEO_CATALOG), ltxGroup);
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
assert.equal(loadSpecFor("Lightricks/LTX-2.3", VIDEO_CATALOG)?.kind, "single_file");
assert.equal(
  loadSpecFor("Lightricks/LTX-2.3", VIDEO_CATALOG)?.filename,
  "ltx-2.3-22b-distilled.safetensors",
);
assert.equal(loadSpecFor("Tongyi-MAI/Z-Image-Turbo", IMAGE_CATALOG)?.kind, "pipeline");
assert.equal(loadSpecFor("Qwen/Qwen-Image-2512", IMAGE_CATALOG)?.kind, "pipeline");


const wanA14b = groupForRepoId("Wan-AI/Wan2.2-T2V-A14B-Diffusers", VIDEO_CATALOG);
const ltxBase = groupForRepoId("Lightricks/LTX-2", VIDEO_CATALOG);
const hunyuanFit = groupForRepoId(
  "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v",
  VIDEO_CATALOG,
);
assert.ok(wanA14b && ltxBase && hunyuanFit && ltxGroup);
const consumer = { gpuGb: 24, systemRamGb: 64 };  // budget 61.6 GB
assert.equal(catalogGroupFitsDevice(wanA14b, consumer, notDownloaded), false);  // 114 GB
assert.equal(catalogGroupFitsDevice(ltxBase, consumer, notDownloaded), false);  // 90 GB
assert.equal(catalogGroupFitsDevice(hunyuanFit, consumer, notDownloaded), true);
assert.equal(
  catalogGroupFitsDevice(hunyuanFit, { gpuGb: 8, systemRamGb: 8 }, notDownloaded),
  false,
);
assert.equal(
  catalogGroupFitsDevice(ltxGroup, { gpuGb: 4, systemRamGb: 4 }, notDownloaded),
  true,
);
assert.equal(
  catalogGroupFitsDevice(
    wanA14b,
    { gpuGb: 8, systemRamGb: 8 },
    (id) => id === "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
  ),
  true,
);
assert.equal(catalogGroupFitsDevice(wanA14b, { gpuGb: 0, systemRamGb: 0 }, notDownloaded), true);
assert.equal(
  catalogGroupFitsDevice(wanA14b, { gpuGb: 192, systemRamGb: 256 }, notDownloaded),
  true,
);


for (const group of AUDIO_CATALOG) {
  assert.ok(group.task === "tts" || group.task === "stt", `audio group ${group.canonicalId} needs a task`);
  assert.equal(group.scope, "audio");
}
for (const catalog of [IMAGE_CATALOG, VIDEO_CATALOG]) {
  for (const group of catalog) {
    assert.equal(group.task, undefined, `non-audio group ${group.canonicalId} must not carry a task`);
  }
}
const orpheus = groupForRepoId("unsloth/orpheus-3b-0.1-ft-GGUF", AUDIO_CATALOG);
assert.ok(orpheus);
assert.equal(orpheus.canonicalId, "unsloth/orpheus-3b-0.1-ft");
assert.equal(orpheus.task, "tts");
assert.equal(loadSpecFor("unsloth/orpheus-3b-0.1-ft-GGUF", AUDIO_CATALOG)?.kind, "gguf");
assert.equal(loadSpecFor("unsloth/csm-1b", AUDIO_CATALOG)?.kind, "pipeline");
assert.equal(groupForRepoId("unsloth/whisper-large-v3-turbo", AUDIO_CATALOG)?.task, "stt");
assert.equal(groupForRepoId("unslothai/Qwen3-ASR-0.6B-GGUF", AUDIO_CATALOG)?.task, "stt");
assert.equal(groupForRepoId("unsloth/Llama-3.3-70B-GGUF", AUDIO_CATALOG), null);
// 67 GB is the download size; the BF16 modular loader fits a 24 GB card.
const minimaxMusic = groupForRepoId("MiniMaxAI/MiniMax-Music3", AUDIO_CATALOG);
assert.ok(minimaxMusic);
assert.equal(
  curatedArtifactFitsDevice(
    "MiniMaxAI/MiniMax-Music3",
    AUDIO_CATALOG,
    { gpuGb: 95, systemRamGb: 0 },
  ),
  true,
);
assert.equal(
  curatedArtifactFitsDevice(
    "MiniMaxAI/MiniMax-Music3",
    AUDIO_CATALOG,
    { gpuGb: 23, systemRamGb: 256 },
  ),
  false,
);


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

const PREQUANT_ROWS = [
  ["Tongyi-MAI/Z-Image-Turbo", "unsloth/Z-Image-Turbo-FP8"],
  ["Qwen/Qwen-Image", "unsloth/Qwen-Image-FP8"],
  ["Qwen/Qwen-Image-2512", "unsloth/Qwen-Image-2512-FP8"],
  ["black-forest-labs/FLUX.1-schnell", "unsloth/FLUX.1-schnell-FP8"],
  ["krea/Krea-2-Turbo", "unsloth/Krea-2-Turbo-FP8"],
] as const;

for (const [id, repo] of PREQUANT_ROWS) {
  const hit = artifactForRepoId(id, IMAGE_CATALOG);
  assert.ok(hit, id);
  assert.equal(hit.artifact.prequantRepo, repo, id);
  // Else the fit rule sizes a load by an artifact the backend never fetches.
  assert.equal(curatedArtifactTakesDenseQuant(id, IMAGE_CATALOG), true, id);
  for (const scheme of ["fp8", "int8"] as const) {
    const size = hit.artifact.prequantSizeGb?.[scheme];
    assert.ok(typeof size === "number" && size > 0, `${id} ${scheme}`);
    assert.ok((size as number) < (hit.artifact.approxSizeGb ?? 0), `${id} ${scheme}`);
  }
  assert.ok(hit.artifact.totalParams, id);
  assert.ok(hit.artifact.approxSizeGb, id);
}

for (const catalog of [IMAGE_CATALOG, VIDEO_CATALOG, AUDIO_CATALOG]) {
  for (const group of catalog) {
    for (const artifact of group.artifacts) {
      if (!artifact.prequantRepo && !artifact.prequantSizeGb) continue;
      assert.equal(group.scope, "image", artifact.repoId);
      assert.equal(artifact.format, "bf16", artifact.repoId);
      assert.equal(artifact.loadKind, "pipeline", artifact.repoId);
      assert.equal(artifact.denseQuantable, true, artifact.repoId);
      assert.ok(artifact.prequantRepo, artifact.repoId);
      assert.ok(artifact.prequantSizeGb, artifact.repoId);
    }
  }
}

const zTurboId = "unsloth/Z-Image-Turbo";
assert.equal(
  curatedArtifactFitsDevice(zTurboId, IMAGE_CATALOG, { gpuGb: 24, systemRamGb: 128 }),
  false,
);
for (const schemes of [["fp8"], ["int8"]]) {
  assert.equal(
    curatedArtifactFitsDevice(zTurboId, IMAGE_CATALOG, {
      gpuGb: 24,
      systemRamGb: 128,
      denseQuantSchemes: schemes,
    }),
    true,
    schemes[0],
  );
}
assert.equal(
  curatedArtifactFitsDevice(zTurboId, IMAGE_CATALOG, {
    gpuGb: 24,
    systemRamGb: 128,
    denseQuantSchemes: [],
  }),
  false,
);
assert.equal(
  curatedArtifactFitsDevice(zTurboId, IMAGE_CATALOG, { gpuGb: 32, systemRamGb: 128 }),
  true,
);
assert.equal(
  curatedArtifactFitsDevice("Qwen/Qwen-Image", IMAGE_CATALOG, {
    gpuGb: 64,
    systemRamGb: 128,
    denseQuantSchemes: ["fp8"],
  }),
  true,
);
assert.equal(
  curatedArtifactFitsDevice("Qwen/Qwen-Image", IMAGE_CATALOG, {
    gpuGb: 64,
    systemRamGb: 128,
    denseQuantSchemes: ["int8"],
  }),
  false,
);
// The reported schemes are a ladder; sizing by schemes[0] alone refuses cards the pipeline runs on.
const qwenImageGroup = groupForRepoId("Qwen/Qwen-Image", IMAGE_CATALOG);
assert.ok(qwenImageGroup);
assert.equal(
  curatedArtifactFitsDevice("Qwen/Qwen-Image", IMAGE_CATALOG, {
    gpuGb: 64,
    systemRamGb: 128,
    denseQuantSchemes: ["int8", "fp8"],
  }),
  true,
);
assert.equal(
  pickDefaultArtifact(qwenImageGroup, {
    gpuGb: 64,
    systemRamGb: 128,
    denseQuantSchemes: ["int8", "fp8"],
    isDownloaded: notDownloaded,
  }).repoId,
  "unsloth/Qwen-Image",
);
assert.equal(
  curatedArtifactFitsDevice("Qwen/Qwen-Image", IMAGE_CATALOG, {
    gpuGb: 24,
    systemRamGb: 128,
    denseQuantSchemes: ["int8", "fp8"],
  }),
  false,
);
for (const id of ["black-forest-labs/FLUX.1-dev", "stabilityai/sdxl-turbo"]) {
  assert.equal(
    curatedArtifactFitsDevice(id, IMAGE_CATALOG, {
      gpuGb: 40,
      systemRamGb: 128,
      denseQuantSchemes: ["fp8"],
    }),
    curatedArtifactFitsDevice(id, IMAGE_CATALOG, { gpuGb: 40, systemRamGb: 128 }),
    id,
  );
}

// The group filter must size rows like the badge and router, by the hosted quantised artifact.
const kreaTurboGroup = groupForRepoId("krea/Krea-2-Turbo", IMAGE_CATALOG);
assert.ok(kreaTurboGroup);
assert.equal(
  catalogGroupFitsDevice(kreaTurboGroup, { gpuGb: 24, systemRamGb: 0 }, notDownloaded),
  false,
);
assert.equal(
  catalogGroupFitsDevice(
    kreaTurboGroup,
    { gpuGb: 24, systemRamGb: 0, denseQuantSchemes: ["int8", "fp8"] },
    notDownloaded,
  ),
  true,
);
assert.equal(
  catalogGroupFitsDevice(
    kreaTurboGroup,
    { gpuGb: 16, systemRamGb: 0, denseQuantSchemes: ["int8", "fp8"] },
    notDownloaded,
  ),
  false,
);

const zTurboGroup = groupForRepoId(zTurboId, IMAGE_CATALOG);
assert.ok(zTurboGroup);
assert.equal(
  pickDefaultArtifact(zTurboGroup, {
    gpuGb: 24,
    systemRamGb: 128,
    isDownloaded: notDownloaded,
  }).format,
  "bnb-4bit",
);
assert.equal(
  pickDefaultArtifact(zTurboGroup, {
    gpuGb: 40,
    systemRamGb: 128,
    denseQuantSchemes: ["fp8"],
    isDownloaded: notDownloaded,
  }).repoId,
  zTurboId,
);
assert.equal(
  pickDefaultArtifact(zTurboGroup, {
    gpuGb: 24,
    systemRamGb: 128,
    denseQuantSchemes: ["fp8"],
    isDownloaded: notDownloaded,
  }).repoId,
  zTurboId,
);
assert.equal(
  pickDefaultArtifact(zTurboGroup, {
    gpuGb: 16,
    systemRamGb: 128,
    denseQuantSchemes: ["fp8"],
    isDownloaded: notDownloaded,
  }).format,
  "bnb-4bit",
);

assert.deepEqual(curatedRowLabelFor(zTurboId, IMAGE_CATALOG, "dense-quant", ["fp8"]), {
  name: "Z-Image-Turbo (Fast FP8)",
  tags: ["BF16"],
});
assert.deepEqual(curatedRowLabelFor(zTurboId, IMAGE_CATALOG, "dense-quant", ["int8"]), {
  name: "Z-Image-Turbo (Fast FP8)",
  tags: ["BF16"],
});
assert.equal(
  curatedRowLabelFor(zTurboId, IMAGE_CATALOG, "dense-quant", ["fp8", "int8"])?.name,
  "Z-Image-Turbo (Fast FP8)",
);
assert.equal(
  curatedRowLabelFor(zTurboId, IMAGE_CATALOG, "dense-quant", [])?.name,
  "Z-Image-Turbo (Fast)",
);
assert.equal(
  curatedDisplayNameFor(zTurboId, IMAGE_CATALOG, "dense-quant", ["fp8"]),
  "Z-Image-Turbo (Fast FP8)",
);
assert.equal(
  catalogToModelOptions(IMAGE_CATALOG, "dense-quant", ["int8"]).find((o) => o.id === zTurboId)
    ?.name,
  "Z-Image-Turbo (Fast FP8)",
);
assert.deepEqual(
  curatedRowLabelFor("HiDream-ai/HiDream-I1-Fast", IMAGE_CATALOG, "dense-quant", ["fp8"]),
  { name: "HiDream I1 (Fast (distilled))", tags: ["BF16"] },
);
assert.equal(
  curatedRowLabelFor("HiDream-ai/HiDream-I1-Dev", IMAGE_CATALOG, "dense-quant", ["fp8"])?.name,
  "HiDream I1 (Dev (distilled)) (Fast FP8)",
);

assert.deepEqual(
  curatedRowLabelFor("MiniMaxAI/MiniMax-H3", VIDEO_CATALOG, "dense-quant", ["fp8"]),
  { name: "MiniMax H3 (Fast FP8)", tags: ["BF16"] },
);
assert.deepEqual(
  curatedRowLabelFor("MiniMaxAI/MiniMax-H3", VIDEO_CATALOG, "dense-quant", ["int8"]),
  { name: "MiniMax H3 (Fast FP8)", tags: ["BF16"] },
);
assert.equal(
  curatedRowLabelFor("unsloth/MiniMax-H3-GGUF", VIDEO_CATALOG, "dense-quant", ["fp8"])?.name,
  "MiniMax-H3-GGUF (Slow)",
);
for (const schemes of [[], ["fp8"], ["int8"]]) {
  for (const catalog of [IMAGE_CATALOG, VIDEO_CATALOG]) {
    for (const group of catalog) {
      for (const artifact of group.artifacts) {
        assert.deepEqual(
          curatedRowLabelFor(artifact.repoId, catalog, "dense-quant", schemes)?.tags ?? [],
          curatedRowLabelFor(artifact.repoId, catalog, "accelerated")?.tags ?? [],
          `${artifact.repoId} ${schemes.join(",") || "none"}`,
        );
      }
    }
  }
}

console.log("model-catalog check: all assertions passed");

// Opt-in `--network` pass; anonymous on purpose (what a fresh install sees). Only definitive verdicts fail.

const HF_API = "https://huggingface.co/api/models";
const HF_RESOLVE = "https://huggingface.co";
const NETWORK_ATTEMPTS = 3;
/** Per-attempt wall clock, headers and body together. */
const NETWORK_TIMEOUT_MS = 20_000;
/** Whole-pass wall clock, under the workflow's 10-minute timeout; past it requests are no opinion. */
const NETWORK_DEADLINE_MS = 7 * 60 * 1000;
let networkDeadlineAt = Number.POSITIVE_INFINITY;
const NETWORK_BATCH = 4;

interface HubRepo {
  /** false for an open repo; "auto" / "manual" for a gated one. */
  gated?: boolean | string;
  siblings?: { rfilename: string }[];
}

const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

/** Never throws: an unanswered request is `{ response: null }` (no opinion), so blips stay green. */
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
      // Per-attempt deadline, or a stalled peer leaves fetch unsettled and the job dies red.
      const response = await fetch(url, { ...init, signal: AbortSignal.timeout(NETWORK_TIMEOUT_MS) });
      // Read the body under the same deadline: headers can arrive while the body stalls.
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

/** Run `work` over `items` a few at a time to avoid hammering the Hub. */
async function inBatches<T>(items: T[], work: (item: T) => Promise<void>): Promise<void> {
  for (let i = 0; i < items.length; i += NETWORK_BATCH) {
    await Promise.all(items.slice(i, i + NETWORK_BATCH).map(work));
  }
}

async function checkCatalogAgainstTheHub(catalogs: CatalogGroup[][]): Promise<string[]> {
  networkDeadlineAt = Date.now() + NETWORK_DEADLINE_MS;
  const failures: string[] = [];
  const groups = catalogs.flat();
  // One metadata call per repo id; a package folder id is checked against its shared repo.
  const artifactsByRepo = new Map<string, ModelArtifact[]>();
  for (const group of groups) {
    for (const artifact of group.artifacts) {
      const hubRepo = isAudioCppFolderId(artifact.repoId) ? AUDIO_CPP_REPO : artifact.repoId;
      const bucket = artifactsByRepo.get(hubRepo);
      if (bucket) bucket.push(artifact);
      else artifactsByRepo.set(hubRepo, [artifact]);
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
      // 401 on the metadata endpoint means private-or-absent; gated-but-public repos answer 200.
      failures.push(
        `${repoId}: HTTP ${response.status} from ${HF_API}/${repoId} -- the repo is missing, renamed or private, so no user can download it`,
      );
      return;
    }
    let repo: HubRepo;
    try {
      repo = JSON.parse(body) as HubRepo;
    } catch (err) {
      // A 200 that is not JSON is a captive portal or proxy.
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

    for (const artifact of artifacts) {
      if (!isAudioCppFolderId(artifact.repoId)) continue;
      const folder = artifact.repoId.slice(AUDIO_CPP_REPO.length + 1).split("/")[0];
      if (!(repo.siblings ?? []).some((s) => s.rfilename.startsWith(`${folder}/`))) {
        failures.push(`${artifact.repoId}: ${repoId} has no '${folder}/' folder`);
      }
    }

    if (repoId === AUDIO_CPP_REPO) {
      const folders = new Set(
        (repo.siblings ?? [])
          .map((s) => s.rfilename.split("/"))
          .filter((parts) => parts.length > 1)
          .map((parts) => parts[0]),
      );
      const offered = AUDIO_CPP_MODELS.filter((m) => isAudioCppFolderId(m.id)).map((m) =>
        audioCppDisplayName(m.id),
      );
      const unoffered = Object.keys(AUDIO_CPP_UNOFFERED_FOLDERS);
      for (const folder of folders) {
        if (!offered.includes(folder) && !unoffered.includes(folder)) {
          failures.push(
            `${repoId}: '${folder}/' is in neither AUDIO_CPP_MODELS nor AUDIO_CPP_UNOFFERED_FOLDERS -- classify it`,
          );
        }
      }
      for (const folder of unoffered) {
        if (!folders.has(folder)) {
          failures.push(`${repoId}: AUDIO_CPP_UNOFFERED_FOLDERS lists '${folder}', which the repo no longer has`);
        }
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
      if (hubGated) continue;  // resolve/ 401s without a token; the sibling list is the check.
      // Listed is not fetchable: resolve/ is the endpoint downloads hit.
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

  // Advisory only: some canonicalIds are deliberately not repos.
  const artifactIds = new Set(
    groups.flatMap((g) => g.artifacts.map((a) => a.repoId.toLowerCase())),
  );
  const orphans = groups
    .map((g) => g.canonicalId)
    .filter((id) => !artifactIds.has(id.toLowerCase()));
  await inBatches(orphans, async (canonicalId) => {
    const { response } = await fetchWithRetry(`${HF_API}/${canonicalId}`);
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
  const audioCppGroups = AUDIO_CATALOG.filter((group) =>
    group.artifacts.some((artifact) => isAudioCppFolderId(artifact.repoId)),
  );
  const failures = await checkCatalogAgainstTheHub([IMAGE_CATALOG, VIDEO_CATALOG, audioCppGroups]);
  if (failures.length > 0) {
    for (const failure of failures) console.error(`::error::${failure}`);
    console.error(`model-catalog network check: ${failures.length} problem(s)`);
    process.exit(1);
  }
  console.log("model-catalog network check: every declared repo, file and gated flag agrees with the Hub");
}
