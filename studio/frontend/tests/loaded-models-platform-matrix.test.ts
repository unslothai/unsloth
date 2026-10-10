// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Covers payloads each OS x accelerator produces: ROCm reports cuda, Apple mps, Intel XPU
// reports cpu for STT, and sd.cpp omits model_kind with dtype gguf.

import assert from "node:assert/strict";
import test from "node:test";

import type { InferenceStatusResponse } from "../src/features/chat/types/api.ts";
import type { DiffusionStatus } from "../src/features/images/api.ts";
import {
  type SttStatusResponse,
  describeDiffusionStatus,
  describeInferenceStatus,
  describeSttStatus,
  describeVideoStatus,
  mergeLoadedModels,
} from "../src/features/loaded-models/loaded-models-sources.ts";
import type { VideoStatus } from "../src/features/video/api.ts";

function chat(
  overrides: Partial<InferenceStatusResponse>,
): InferenceStatusResponse {
  return {
    active_model: null,
    loaded: [],
    is_gguf: false,
    is_mlx: false,
    is_vision: false,
    is_audio: false,
    audio_type: null,
    gguf_variant: null,
    ...overrides,
  } as InferenceStatusResponse;
}

function diffusion(overrides: Partial<DiffusionStatus>): DiffusionStatus {
  return {
    loaded: false,
    repo_id: null,
    family: null,
    device: null,
    dtype: null,
    model_kind: null,
    ...overrides,
  } as DiffusionStatus;
}

function video(overrides: Partial<VideoStatus>): VideoStatus {
  return {
    loaded: false,
    repo_id: null,
    family: null,
    device: null,
    dtype: null,
    model_kind: null,
    transformer_quant: null,
    ...overrides,
  } as VideoStatus;
}

test("a GGUF chat model reads the same on every accelerator", () => {
  // is_mlx is forced false on the GGUF branch.
  for (const host of ["nvidia", "rocm", "xpu", "cpu-only", "apple"]) {
    const rows = describeInferenceStatus(
      chat({
        active_model: "unsloth/Qwen3-4B-GGUF",
        loaded: ["unsloth/Qwen3-4B-GGUF"],
        is_gguf: true,
        gguf_variant: "Q4_K_M",
      }),
    );
    assert.equal(rows.length, 1, host);
    assert.equal(rows[0].kind, "text", host);
    assert.equal(rows[0].detail, "GGUF · Q4_K_M", host);
  }
});

test("Apple Silicon MLX is labelled MLX, and only there", () => {
  const mlx = describeInferenceStatus(
    chat({ active_model: "mlx-community/Qwen3-4B-4bit", is_mlx: true }),
  );
  assert.equal(mlx[0].detail, "MLX");
  // An Intel Mac, or an unusable MLX stack, reports is_mlx false and must read as Transformers.
  const intelMac = describeInferenceStatus(
    chat({ active_model: "unsloth/Qwen3-4B", is_mlx: false }),
  );
  assert.equal(intelMac[0].detail, "Transformers");
});

test("GGUF wins over MLX if a payload ever claims both", () => {
  const rows = describeInferenceStatus(
    chat({
      active_model: "unsloth/Qwen3-4B-GGUF",
      is_gguf: true,
      is_mlx: true,
      gguf_variant: "UD-Q4_K_XL",
    }),
  );
  assert.equal(rows[0].detail, "GGUF · UD-Q4_K_XL");
});

test("a vision model is marked on any backend", () => {
  const rows = describeInferenceStatus(
    chat({
      active_model: "unsloth/gemma-3-4b-it",
      is_vision: true,
    }),
  );
  assert.equal(rows[0].detail, "Transformers · Vision");
  assert.equal(rows[0].kind, "text", "vision is still a chat row");
});

test("audio models split three ways, not two", () => {
  // VALID_AUDIO_TYPES in model_config.py; is_audio means TTS.
  const speaks = ["snac", "csm", "bicodec", "dac"];
  for (const audioType of speaks) {
    const rows = describeInferenceStatus(
      chat({ active_model: `m/${audioType}`, is_audio: true, audio_type: audioType }),
    );
    assert.equal(rows[0].kind, "tts", `${audioType} produces audio`);
  }
  const whisper = describeInferenceStatus(
    chat({ active_model: "m/whisper", is_audio: true, audio_type: "whisper" }),
  );
  assert.equal(whisper[0].kind, "stt", "whisper transcribes, it does not answer");
  // Transformers can report is_audio true for Gemma 3n.
  const vlm = describeInferenceStatus(
    chat({ active_model: "unsloth/gemma-3n-E4B-it", is_audio: true, audio_type: "audio_vlm" }),
  );
  assert.equal(vlm[0].kind, "text", "an audio VLM is a chat model that listens");
});

test("an audio flag with no type still reads as speech", () => {
  const rows = describeInferenceStatus(
    chat({ active_model: "m/unknown", is_audio: true, audio_type: null }),
  );
  assert.equal(rows[0].kind, "tts");
});

test("audio_type without is_audio does not make an audio row", () => {
  const rows = describeInferenceStatus(
    chat({ active_model: "m/x", is_audio: false, audio_type: "whisper" }),
  );
  assert.equal(rows[0].kind, "text");
});

test("every device the diffusion resolver can report renders", () => {
  // resolve_diffusion_device_target() emits exactly these four.
  const expected: Record<string, string> = {
    cuda: "flux · BF16 · cuda", // also AMD ROCm
    xpu: "flux · BF16 · xpu",
    mps: "flux · BF16 · mps",
    cpu: "flux · BF16 · cpu",
  };
  for (const [device, detail] of Object.entries(expected)) {
    const rows = describeDiffusionStatus(
      diffusion({
        loaded: true,
        repo_id: "black-forest-labs/FLUX.1-dev",
        family: "flux",
        device,
        dtype: "bfloat16",
        model_kind: "pipeline",
      }),
    );
    assert.equal(rows[0].detail, detail, device);
    assert.equal(rows[0].source, "image", device);
  }
});

test("the sd.cpp engine still says GGUF without a model_kind", () => {
  // sd.cpp status has no model_kind and puts gguf in dtype.
  const rows = describeDiffusionStatus(
    diffusion({
      loaded: true,
      repo_id: "unsloth/FLUX.1-dev-GGUF",
      family: "flux",
      device: "cpu",
      dtype: "gguf",
    }),
  );
  assert.equal(rows[0].detail, "flux · GGUF · cpu");
});

test("a GGUF image model under the diffusers engine is not doubled", () => {
  const rows = describeDiffusionStatus(
    diffusion({
      loaded: true,
      repo_id: "unsloth/FLUX.1-dev-GGUF",
      family: "flux",
      device: "cuda",
      dtype: "gguf",
      model_kind: "gguf",
    }),
  );
  assert.equal(rows[0].detail, "flux · GGUF · cuda");
});

test("a diffusion runtime with no repo id yields no row", () => {
  assert.deepEqual(
    describeDiffusionStatus(diffusion({ loaded: true, repo_id: null })),
    [],
  );
});

test("video precision prefers the transformer quant over the dtype", () => {
  const rows = describeVideoStatus(
    video({
      loaded: true,
      repo_id: "Wan-AI/Wan2.2-T2V-A14B",
      family: "wan",
      device: "cuda",
      dtype: "bfloat16",
      transformer_quant: "fp8",
    }),
  );
  assert.equal(rows[0].detail, "wan · FP8 · cuda");
  assert.equal(rows[0].kind, "video");
});

test("video falls back to the dtype when unquantised", () => {
  const rows = describeVideoStatus(
    video({
      loaded: true,
      repo_id: "Wan-AI/Wan2.2-T2V-A14B",
      family: "wan",
      device: "cuda",
      dtype: "bfloat16",
      transformer_quant: null,
    }),
  );
  assert.equal(rows[0].detail, "wan · BF16 · cuda");
});

test('a "none" quant is not printed as a precision', () => {
  const rows = describeVideoStatus(
    video({
      loaded: true,
      repo_id: "Wan-AI/Wan2.2-T2V-A14B",
      family: "wan",
      device: "cuda",
      dtype: "none",
      transformer_quant: "none",
    }),
  );
  assert.equal(rows[0].detail, "wan · cuda");
});

test("a host that can never run video reports an empty runtime, not an error", () => {
  // The video router is always registered, so hosts without torch get loaded:false, not 404.
  assert.deepEqual(describeVideoStatus(video({ loaded: false })), []);
});

test("each dictation engine reports its own row", () => {
  const rows = describeSttStatus({
    transformers: { loaded_model: "large-v3", device: "cuda" },
    mtmd: { loaded_model: "qwen3-asr-0.6b", device: "llama.cpp" },
    gguf: { loaded_model: "ggml-base.en", device: "whisper.cpp" },
  } as SttStatusResponse);
  assert.deepEqual(
    rows.map((row) => row.detail),
    ["Transformers · cuda", "llama.cpp", "whisper.cpp"],
  );
  assert.deepEqual(
    rows.map((row) => row.sttEngine),
    ["transformers", "mtmd", "gguf"],
  );
});

test("dictation on Apple and on CPU-only hosts", () => {
  // stt_sidecar._pick_device never checks xpu.
  for (const device of ["mps", "cpu"]) {
    const rows = describeSttStatus({
      transformers: { loaded_model: "small", device },
    } as SttStatusResponse);
    assert.equal(rows[0].detail, `Transformers · ${device}`, device);
  }
});

test("an engine holding nothing contributes no row", () => {
  const rows = describeSttStatus({
    transformers: { loaded_model: null, device: null },
    mtmd: { loaded_model: null, device: null },
    gguf: { loaded_model: null, device: null },
  } as SttStatusResponse);
  assert.deepEqual(rows, []);
});

test("a host holding all four runtimes lists them in a fixed order", () => {
  const rows = mergeLoadedModels([
    describeInferenceStatus(
      chat({
        active_model: "unsloth/Qwen3-4B-GGUF",
        is_gguf: true,
        gguf_variant: "Q4_K_M",
      }),
    ),
    describeDiffusionStatus(
      diffusion({
        loaded: true,
        repo_id: "black-forest-labs/FLUX.1-dev",
        family: "flux",
        device: "cuda",
        dtype: "bfloat16",
      }),
    ),
    describeVideoStatus(
      video({
        loaded: true,
        repo_id: "Wan-AI/Wan2.2-T2V-A14B",
        family: "wan",
        device: "cuda",
        dtype: "bfloat16",
      }),
    ),
    describeSttStatus({
      transformers: { loaded_model: "large-v3", device: "cuda" },
    } as SttStatusResponse),
  ]);
  assert.deepEqual(
    rows.map((row) => row.source),
    ["chat", "image", "video", "stt"],
    "a stable order stops rows jumping between polls",
  );
  assert.equal(new Set(rows.map((row) => row.id)).size, 4, "ids are unique");
});

test("a whisper model in chat and in dictation is two rows, not one", () => {
  // Chat and dictation hold separate copies, so both rows must show.
  const rows = mergeLoadedModels([
    describeInferenceStatus(
      chat({
        active_model: "openai/whisper-large-v3",
        is_audio: true,
        audio_type: "whisper",
      }),
    ),
    describeSttStatus({
      transformers: { loaded_model: "openai/whisper-large-v3", device: "cuda" },
    } as SttStatusResponse),
  ]);
  assert.equal(rows.length, 2);
  assert.deepEqual(
    rows.map((row) => row.source),
    ["chat", "stt"],
    "each row must eject through its own runtime",
  );
});
