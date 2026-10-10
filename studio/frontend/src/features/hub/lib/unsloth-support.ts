// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

const EXCLUDED_TAGS_GPU = new Set([
  "gptq",
  "awq",
  "exl2",
  "mlx",
  "onnx",
  "openvino",
  "coreml",
  "tflite",
  "ctranslate2",
]);

const EXCLUDED_TAGS_MLX = new Set([
  "gptq",
  "awq",
  "exl2",
  "onnx",
  "openvino",
  "coreml",
  "tflite",
  "ctranslate2",
]);

const UNSUPPORTED_PIPELINE_TAGS: ReadonlySet<string> = new Set([
  "text-to-image",
  "image-to-image",
  "image-text-to-image",
  "text-to-video",
  "video-to-video",
  "image-to-video",
  "image-text-to-video",
  "video-text-to-text",
  "video-classification",
  "unconditional-image-generation",
  "text-to-3d",
  "image-to-3d",
  "image-segmentation",
  "object-detection",
  "depth-estimation",
  "mask-generation",
  "zero-shot-image-classification",
  "zero-shot-object-detection",
  "image-classification",
  "keypoint-detection",
  "image-feature-extraction",
  "robotics",
  "reinforcement-learning",
  "graph-ml",
  "tabular-classification",
  "tabular-regression",
  "time-series-forecasting",
]);

const UNSUPPORTED_LIBRARY_TAGS: ReadonlySet<string> = new Set([
  "diffusers",
  "stable-diffusion",
  "stable-diffusion-xl",
  "flux",
  "controlnet",
  "lora-diffusers",
  // Checked AFTER the pipeline tag: only taskless single-file mirrors land here, else they would
  // read as chat models.
  "diffusion-single-file",
]);

const FORMAT_TAG_LABEL: Record<string, string> = {
  gptq: "GPTQ quantization",
  awq: "AWQ quantization",
  exl2: "EXL2 quantization",
  mlx: "MLX-format weights",
  onnx: "ONNX-format weights",
  openvino: "OpenVINO-format weights",
  coreml: "Core ML-format weights",
  tflite: "TensorFlow Lite-format weights",
  ctranslate2: "CTranslate2-format weights",
};

const FORMAT_ALIAS_TAGS: Record<string, string> = {
  "auto-gptq": "gptq",
};

const FORMAT_NAME_PATTERNS: ReadonlyArray<{ key: string; pattern: RegExp }> = [
  { key: "awq", pattern: /(?:^|[-_./])awq(?:\d+(?:bit)?)?(?:$|[-_./])/i },
  { key: "gptq", pattern: /(?:^|[-_./])gptq(?:\d+(?:bit)?)?(?:$|[-_./])/i },
  { key: "exl2", pattern: /(?:^|[-_./])exl2(?:$|[-_./])/i },
  { key: "mlx", pattern: /(?:^|[-_./])mlx(?:$|[-_./])/i },
  { key: "onnx", pattern: /(?:^|[-_./])onnx(?:$|[-_./])/i },
  { key: "openvino", pattern: /(?:^|[-_./])openvino(?:$|[-_./])/i },
  { key: "coreml", pattern: /(?:^|[-_./])coreml(?:$|[-_./])/i },
  { key: "tflite", pattern: /(?:^|[-_./])tflite(?:$|[-_./])/i },
  { key: "ctranslate2", pattern: /(?:^|[-_./])ctranslate2(?:$|[-_./])/i },
];

const UNSUPPORTED_QUANT_METHODS: Record<string, string> = {
  awq: "AWQ quantization",
  gptq: "GPTQ quantization",
  exl2: "EXL2 quantization",
  "compressed-tensors": "compressed-tensors quantization",
  aqlm: "AQLM quantization",
  eetq: "EETQ quantization",
  hqq: "HQQ quantization",
  fbgemm_fp8: "FBGEMM FP8 quantization",
  finegrained_fp8: "fine-grained FP8 quantization",
  quark: "Quark quantization",
  vptq: "VPTQ quantization",
  spqr: "SpQR quantization",
  higgs: "HIGGS quantization",
  sinq: "SINQ quantization",
  torchao: "torchao quantization",
  quanto: "optimum-quanto quantization",
  auto_round: "AutoRound quantization",
  autoround: "AutoRound quantization",
  metal: "Metal-kernel quantization",
  fouroversix: "Four Over Six quantization",
  fp_quant: "FP-Quant quantization",
};

export type UnslothSupportStatus = "supported" | "unsupported";

export interface UnslothSupport {
  status: UnslothSupportStatus;
  reason: string | null;
  /** Set when Unsloth runs this model on a dedicated page rather than in chat. The status stays "unsupported" because the chat pickers gate on it, but the UI must not call it unsupported: the Images and Video pages load it. "vllm": only the optional vLLM engine can (#11728). */
  supportedIn?: "images" | "video" | "vllm";
}

const VLLM_QUANT_METHODS: ReadonlySet<string> = new Set([
  "compressed-tensors",
  "awq",
  "gptq",
]);
const VLLM_FORMAT_KEYS: ReadonlySet<string> = new Set(["awq", "gptq"]);
// Chat only: vLLM refuses audio (_reject_unsupported_managed_kind); untagged repos count as chat.
const VLLM_CHAT_TASKS: ReadonlySet<string> = new Set(["text-generation", "image-text-to-text"]);

// Generation tasks the Images / Video pages handle. Mirrors IMAGE_GEN_TASKS and the video picker's tasks; image-to-video is included for LTX-2.3.
const IMAGE_PAGE_TASKS: ReadonlySet<string> = new Set([
  "text-to-image",
  "image-to-image",
  "image-text-to-image",
]);
const VIDEO_PAGE_TASKS: ReadonlySet<string> = new Set([
  "text-to-video",
  "image-to-video",
  "image-text-to-video",
]);

export function studioPageForTask(
  pipelineTag?: string | null,
): "images" | "video" | undefined {
  const tag = pipelineTag?.toLowerCase().trim();
  if (!tag) return undefined;
  if (IMAGE_PAGE_TASKS.has(tag)) return "images";
  if (VIDEO_PAGE_TASKS.has(tag)) return "video";
  return undefined;
}

export function excludedFormatTagsForDevice(
  deviceType?: string | null,
): ReadonlySet<string> {
  return deviceType?.toLowerCase() === "mac"
    ? EXCLUDED_TAGS_MLX
    : EXCLUDED_TAGS_GPU;
}

function normalizeQuantMethod(value: string | null | undefined): string | null {
  if (typeof value !== "string") return null;
  const trimmed = value.toLowerCase().trim();
  return trimmed.length > 0 ? trimmed : null;
}

function repoLeaf(modelId: string): string {
  const parts = modelId.trim().split("/").filter(Boolean);
  return parts.at(-1) ?? modelId;
}

// Export-format tags alone must not hide a repo: native checkpoints often ship ONNX/TFLite exports too.
const EXPORT_FORMAT_TAGS: ReadonlySet<string> = new Set([
  "onnx",
  "openvino",
  "tflite",
  "coreml",
]);

function detectUnsupportedFormatKey(
  modelId: string | null | undefined,
  lowerTags: ReadonlySet<string>,
  excludedFormats: ReadonlySet<string>,
): string | null {
  const hasNativeWeights = lowerTags.has("pytorch") || lowerTags.has("safetensors");
  for (const tag of lowerTags) {
    if (hasNativeWeights && EXPORT_FORMAT_TAGS.has(tag)) continue;
    if (FORMAT_TAG_LABEL[tag]) {
      if (excludedFormats.has(tag)) return tag;
      continue;
    }
    const alias = FORMAT_ALIAS_TAGS[tag];
    if (alias && excludedFormats.has(alias)) return alias;
  }
  if (modelId) {
    // Mirrors the backend's _looks_like_mlx_repo heuristic.
    if (excludedFormats.has("mlx") && modelId.trim().toLowerCase().startsWith("mlx-community/")) return "mlx";
    const name = repoLeaf(modelId);
    for (const { key, pattern } of FORMAT_NAME_PATTERNS) {
      if (excludedFormats.has(key) && pattern.test(name)) return key;
    }
  }
  return null;
}

export function classifyUnslothSupport({
  modelId,
  pipelineTag,
  tags,
  libraryName,
  deviceType,
  quantMethod,
  vllmAvailable = false,
}: {
  modelId?: string | null;
  pipelineTag?: string | null;
  tags?: readonly string[] | null;
  libraryName?: string | null;
  deviceType?: string | null;
  quantMethod?: string | null;
  vllmAvailable?: boolean;
}): UnslothSupport {
  const pipeline = pipelineTag?.toLowerCase().trim() || null;
  const lowerTags = new Set(
    (tags ?? []).map((tag) => tag.toLowerCase().trim()).filter(Boolean),
  );
  const library = libraryName?.toLowerCase().trim() || null;
  const formatTags = excludedFormatTagsForDevice(deviceType);
  const normalizedQuant = normalizeQuantMethod(quantMethod);
  const vllmRuns = vllmAvailable && (!pipeline || VLLM_CHAT_TASKS.has(pipeline));

  // GGUF runs through llama.cpp, so the HF quant_method must not disqualify it.
  const isGguf =
    lowerTags.has("gguf") ||
    library === "gguf" ||
    (modelId ? /(?:^|[-_.])gguf$/i.test(repoLeaf(modelId)) : false);

  if (normalizedQuant && !isGguf) {
    if (Object.hasOwn(UNSUPPORTED_QUANT_METHODS, normalizedQuant)) {
      const reason = `Detected ${UNSUPPORTED_QUANT_METHODS[normalizedQuant]}.`;
      // vLLM excuses only the quantization; any other rejection keeps the old answer.
      if (vllmRuns && VLLM_QUANT_METHODS.has(normalizedQuant)) {
        const rest = classifyUnslothSupport({ modelId, pipelineTag, tags, libraryName, deviceType, vllmAvailable });
        if (rest.status === "supported" || rest.supportedIn === "vllm") {
          return { status: "unsupported", reason, supportedIn: "vllm" };
        }
      }
      return { status: "unsupported", reason };
    }
  }

  if (pipeline && UNSUPPORTED_PIPELINE_TAGS.has(pipeline)) {
    return {
      status: "unsupported",
      reason: `Pipeline task: ${pipeline}.`,
      supportedIn: studioPageForTask(pipeline),
    };
  }
  for (const tag of lowerTags) {
    if (UNSUPPORTED_LIBRARY_TAGS.has(tag)) {
      return {
        status: "unsupported",
        reason: `Library: ${tag}.`,
      };
    }
  }
  if (library && UNSUPPORTED_LIBRARY_TAGS.has(library)) {
    return {
      status: "unsupported",
      reason: `Library: ${library}.`,
    };
  }
  const formatKey = detectUnsupportedFormatKey(modelId, lowerTags, formatTags);
  if (formatKey) {
    const label = FORMAT_TAG_LABEL[formatKey] ?? `${formatKey.toUpperCase()} weights`;
    const reason = `Detected ${label}.`;
    // AWQ / GPTQ must be the only format objection, else tag order would decide.
    if (
      vllmRuns &&
      !isGguf &&
      VLLM_FORMAT_KEYS.has(formatKey) &&
      !detectUnsupportedFormatKey(
        modelId,
        lowerTags,
        new Set([...formatTags].filter((tag) => !VLLM_FORMAT_KEYS.has(tag))),
      )
    ) {
      return { status: "unsupported", reason, supportedIn: "vllm" };
    }
    return { status: "unsupported", reason };
  }
  return { status: "supported", reason: null };
}
