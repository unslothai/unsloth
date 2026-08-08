// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The indicator is the only screen that reads all four runtimes at once, so it
// is also the only one that sees every hardware shape the backend can report.
// None of that is visible from a CUDA dev box: ROCm reports itself as "cuda"
// (diffusion_device.py, "Apple Silicon maps to MPS and ROCm to cuda"), Apple
// reports "mps" and never "mlx", Intel XPU reports "xpu" for images but "cpu"
// for dictation because stt_sidecar._pick_device never checks torch.xpu, and the
// sd.cpp engine Macs and CPU-only hosts fall back to omits model_kind entirely
// and puts "gguf" in dtype instead.
//
// So this walks the payloads [Windows, Linux, WSL, macOS] x [NVIDIA, AMD ROCm,
// Intel XPU, CPU-only, Apple] actually produce, and asserts the row each one
// renders. The OS axis is not a separate assertion for the chat/images/video
// rows because the backend emits identical JSON for it -- what the OS changes is
// which accelerator branch is reachable, and that is the axis enumerated here.

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

// The handler returns a bare model with every flag false when nothing is
// loaded, so a partial fixture stands in for the full response.
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

// ── Chat runtime: the accelerator axis ──────────────────────────────────

test("a GGUF chat model reads the same on every accelerator", () => {
  // llama.cpp runs on NVIDIA, AMD, Intel, Apple and plain CPU, and the GGUF
  // branch is chosen purely by llama_backend.is_loaded with no hardware gate.
  // is_mlx is force-set false there, so it must never win the runtime ladder.
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
  // An Intel Mac, or Apple Silicon whose MLX stack is unusable, falls back to
  // DeviceType.CPU and reports is_mlx false -- so it must read as Transformers.
  const intelMac = describeInferenceStatus(
    chat({ active_model: "unsloth/Qwen3-4B", is_mlx: false }),
  );
  assert.equal(intelMac[0].detail, "Transformers");
});

test("GGUF wins over MLX if a payload ever claims both", () => {
  // Defensive: the backend force-sets is_mlx false on the GGUF branch, but the
  // ladder must not depend on that to avoid mislabelling the runtime.
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

// ── Chat runtime: every audio_type the backend can emit ─────────────────

test("audio models split by direction, not by name", () => {
  // VALID_AUDIO_TYPES in model_config.py, against is_audio_input_type(): only
  // whisper and audio_vlm take audio IN. Anything else speaks.
  const speaks = ["snac", "csm", "bicodec", "dac"];
  const listens = ["whisper", "audio_vlm"];
  for (const audioType of speaks) {
    const rows = describeInferenceStatus(
      chat({ active_model: `m/${audioType}`, is_audio: true, audio_type: audioType }),
    );
    assert.equal(rows[0].kind, "tts", `${audioType} produces audio`);
  }
  for (const audioType of listens) {
    const rows = describeInferenceStatus(
      chat({ active_model: `m/${audioType}`, is_audio: true, audio_type: audioType }),
    );
    assert.equal(rows[0].kind, "stt", `${audioType} consumes audio`);
  }
});

test("an audio flag with no type still reads as speech", () => {
  // audio_type detection can come back null on a model the tokenizer scan could
  // not classify. It is not an input type, so the TTS default is right.
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

// ── Diffusion: the device vocabulary, and the two engines ───────────────

test("every device the diffusion resolver can report renders", () => {
  // resolve_diffusion_device_target() emits exactly these four, never cuda:0.
  const expected: Record<string, string> = {
    cuda: "flux · BF16 · cuda", // NVIDIA, and AMD ROCm, which reports cuda too
    xpu: "flux · BF16 · xpu", // Intel Arc / Data Center GPU
    mps: "flux · BF16 · mps", // Apple Silicon under the diffusers engine
    cpu: "flux · BF16 · cpu", // no accelerator, or no torch at all
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
  // sd_cpp_backend.status() has no model_kind key at all and puts the literal
  // "gguf" in dtype. This is the Mac / CPU-only shape, so it is the one most
  // likely to go untested on a CUDA box.
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
  // Here model_kind IS "gguf" and dtype is a real precision, so both the kind
  // and the precision have something to say and neither should repeat.
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
  // loaded and repo_id are always set together, but a row with no name would be
  // unejectable, so refusing it is worth pinning.
  assert.deepEqual(
    describeDiffusionStatus(diffusion({ loaded: true, repo_id: null })),
    [],
  );
});

// ── Video: NVIDIA/Intel only in practice, but the payload is the contract ──

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
  // The video router is registered unconditionally and imports torch lazily, so
  // macOS and CPU-only hosts get a clean loaded:false rather than a 404.
  assert.deepEqual(describeVideoStatus(video({ loaded: false })), []);
});

// ── Dictation: three engines, and a "device" that is not a device ────────

test("each dictation engine reports its own row", () => {
  const rows = describeSttStatus({
    transformers: { loaded_model: "large-v3", device: "cuda" },
    mtmd: { loaded_model: "qwen3-asr-0.6b", device: "llama.cpp" },
    gguf: { loaded_model: "ggml-base.en", device: "whisper.cpp" },
  } as SttStatusResponse);
  assert.deepEqual(
    rows.map((row) => row.detail),
    // The sidecars put their engine name in device, so it must not print twice.
    ["Transformers · cuda", "llama.cpp", "whisper.cpp"],
  );
  assert.deepEqual(
    rows.map((row) => row.sttEngine),
    ["transformers", "mtmd", "gguf"],
  );
});

test("dictation on Apple and on CPU-only hosts", () => {
  // _pick_device() in stt_sidecar.py knows only cuda/mps/cpu -- notably NOT xpu,
  // so an Intel GPU host reports cpu here while images report xpu.
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

// ── The merge, across a fully loaded host ───────────────────────────────

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
  // These really are two resident copies: chat loads it inside the inference
  // subprocess, dictation in its own in-process sidecar. Collapsing them would
  // hide a copy the user cannot then free.
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
