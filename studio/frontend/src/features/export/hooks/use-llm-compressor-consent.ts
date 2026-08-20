// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { MERGED_FORMATS } from "../constants";
import { fetchLlmCompressorProbe } from "../api/llm-compressor-consent-api";
import { useLlmCompressorConsentStore } from "../stores/llm-compressor-consent-store";

export function mergedFormatNeedsLlmCompressor(formatValue: string): boolean {
  const fmt = MERGED_FORMATS.find((f) => f.value === formatValue);
  return fmt?.backend === "compressed";
}

/** Pause export when FP8/FP4 compressed-tensors needs llm-compressor. */
export async function confirmLlmCompressorInstallIfNeeded(
  selectedFormatValues: string[],
): Promise<{ ok: boolean; installMissingDependencies: boolean }> {
  const needsCompressed = selectedFormatValues.some(mergedFormatNeedsLlmCompressor);
  if (!needsCompressed) {
    return { ok: true, installMissingDependencies: false };
  }

  const probe = await fetchLlmCompressorProbe();
  if (probe.ready) {
    return { ok: true, installMissingDependencies: false };
  }
  if (probe.blocked_reason || probe.needs_consent) {
    const consented = await useLlmCompressorConsentStore
      .getState()
      .requestConsent(probe);
    return { ok: consented, installMissingDependencies: consented };
  }
  return { ok: false, installMissingDependencies: false };
}
