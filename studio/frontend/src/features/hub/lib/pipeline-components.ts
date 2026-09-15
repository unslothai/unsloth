// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

const COMPONENT_LABELS: Record<string, string> = {
  vae: "VAE",
  text_encoder: "Text encoder",
  text_encoder_2: "Text encoder 2",
  tokenizer: "Tokenizer",
  tokenizer_2: "Tokenizer 2",
  scheduler: "Scheduler",
  feature_extractor: "Feature extractor",
  image_encoder: "Image encoder",
};

export function formatPipelineComponentLabel(component: string): string {
  const normalized = component.trim().toLowerCase();
  if (COMPONENT_LABELS[normalized]) {
    return COMPONENT_LABELS[normalized];
  }
  return component
    .replace(/_/g, " ")
    .replace(/\b\w/g, (char) => char.toUpperCase());
}

export function formatCachedComponentsSummary(
  components: readonly string[] | null | undefined,
): string | null {
  if (!components?.length) {
    return null;
  }
  return components.map(formatPipelineComponentLabel).join(", ");
}

export function companionPrefetchDownloadHint(
  components: readonly string[] | null | undefined,
): string {
  const summary = formatCachedComponentsSummary(components);
  if (summary) {
    return `${summary} cached for a quantized image model. Full pipeline weights are not installed. Click Download to fetch the complete model.`;
  }
  return "Some pipeline components are cached for a quantized image model. Full pipeline weights are not installed. Click Download to fetch the complete model.";
}
