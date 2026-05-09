// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type {
  DocExtractSettings,
  OcrModelPresetId,
  OcrModelSelection,
} from "../stores/chat-runtime-store";

export type { OcrModelPresetId, OcrModelSelection };

/**
 * A built-in OCR model preset advertised in the Document Extraction settings
 * sheet. The HF id is used verbatim for `validateModel` and `loadModel`
 * requests; the orchestrator never substitutes another id server-side.
 */
export interface OcrModelPreset {
  id: OcrModelPresetId;
  label: string;
  modelId: string;
  requiresTrustRemoteCode: boolean;
  defaultMaxSeqLength: number;
  hint: string;
}

/**
 * The resolved load target for an OCR run. Built from a preset OR from the
 * user's custom path. `null` means "no dedicated OCR model swap" — the
 * extraction route can still use the loaded chat VLM when one is active.
 */
export interface OcrModelTarget {
  source: "preset" | "custom";
  label: string;
  modelId: string;
  ggufVariant: string | null;
  requiresTrustRemoteCode: boolean;
  defaultMaxSeqLength: number;
}

export const OCR_MODEL_PRESETS: readonly OcrModelPreset[] = [
  {
    id: "deepseek-ocr",
    label: "DeepSeek-OCR",
    modelId: "deepseek-ai/DeepSeek-OCR",
    requiresTrustRemoteCode: true,
    defaultMaxSeqLength: 8192,
    hint: "Custom-code vision model. Trust remote code must be enabled.",
  },
  {
    id: "glm-ocr",
    label: "GLM-OCR",
    modelId: "zai-org/GLM-OCR",
    requiresTrustRemoteCode: true,
    defaultMaxSeqLength: 8192,
    hint: "GLM OCR vision model. Trust remote code must be enabled.",
  },
  {
    id: "paddleocr-vl",
    label: "PaddleOCR-VL",
    modelId: "unsloth/PaddleOCR-VL",
    requiresTrustRemoteCode: true,
    defaultMaxSeqLength: 4096,
    hint: "Layout-aware OCR VLM.",
  },
];

/**
 * Heuristic for whether a custom HF id is likely to need `trust_remote_code`.
 * Conservative — flips on for known OCR repo prefixes; falls back to false.
 * The validate route is still authoritative; this only seeds the UI hint.
 */
function looksLikeTrcModel(id: string): boolean {
  const normalized = id.trim().toLowerCase();
  if (!normalized) return false;
  // Match path segments to avoid false-positives like "myorg/non-ocr-vlm".
  // The validate route is still authoritative; this only seeds the UI hint.
  const segments = normalized.split(/[/\-_]/).filter(Boolean);
  return (
    normalized.startsWith("deepseek-ai/") ||
    normalized.startsWith("zai-org/") ||
    normalized.includes("/glm-") ||
    segments.includes("paddleocr") ||
    (segments.includes("ocr") &&
      (segments.includes("vl") || segments.includes("vlm"))) ||
    /(^|[/_-])ocr([-_/]|$)/.test(normalized)
  );
}

export function resolveOcrModelTarget(
  settings: DocExtractSettings,
): OcrModelTarget | null {
  if (settings.ocrModel === "default" || settings.ocrModel === "none") {
    return null;
  }
  if (settings.ocrModel === "custom") {
    const id = settings.customOcrModelId.trim();
    if (!id) return null;
    return {
      source: "custom",
      label: id,
      modelId: id,
      ggufVariant: settings.customOcrGgufVariant,
      requiresTrustRemoteCode: looksLikeTrcModel(id),
      defaultMaxSeqLength: 8192,
    };
  }
  const preset = OCR_MODEL_PRESETS.find((it) => it.id === settings.ocrModel);
  if (!preset) return null;
  return {
    source: "preset",
    label: preset.label,
    modelId: preset.modelId,
    ggufVariant: null,
    requiresTrustRemoteCode: preset.requiresTrustRemoteCode,
    defaultMaxSeqLength: preset.defaultMaxSeqLength,
  };
}

export function hasSelectedOcrModel(settings: DocExtractSettings): boolean {
  return resolveOcrModelTarget(settings) !== null;
}
