// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useEffect, useState } from "react";
import { type ExportSizeEstimate, fetchExportSize } from "../api/export-api";

export interface ExportSizeState {
  data: ExportSizeEstimate | null;
  loading: boolean;
  /** Real FP16/BF16-equivalent bytes, or null when unknown. */
  fp16Bytes: number | null;
}

const EMPTY: ExportSizeState = { data: null, loading: false, fp16Bytes: null };

// Export page's "no model" sentinel (em dash U+2014); char code keeps this file ASCII.
const EMPTY_MODEL_SENTINEL = String.fromCharCode(0x2014);

function normalizeModelId(modelId: string | null | undefined): string {
  const id = (modelId ?? "").trim();
  return id && id !== EMPTY_MODEL_SENTINEL ? id : "";
}

/**
 * Fetch the selected model's fp16-equivalent size so the GGUF picker can scale
 * its per-quant estimates. Null size on any failure; refetches on token change.
 */
export function useExportSizeEstimate(
  modelId: string | null | undefined,
  hfToken?: string | null,
): ExportSizeState {
  const modelKey = normalizeModelId(modelId);
  const token = hfToken?.trim() || "";
  // Refetch when model or token changes; "|" can't appear in an id/token.
  const key = modelKey ? `${modelKey}|${token}` : "";
  const [state, setState] = useState<{ key: string; value: ExportSizeState }>(
    () => ({ key: "", value: EMPTY }),
  );

  useEffect(() => {
    if (!modelKey) {
      setState({ key: "", value: EMPTY });
      return;
    }
    const controller = new AbortController();
    setState({ key, value: { ...EMPTY, loading: true } });
    void fetchExportSize(modelKey, token, controller.signal)
      .then((data) => {
        if (controller.signal.aborted) {
          return;
        }
        setState({
          key,
          value: { data, loading: false, fp16Bytes: data.fp16_bytes ?? null },
        });
      })
      .catch(() => {
        if (controller.signal.aborted) {
          return;
        }
        setState({ key, value: EMPTY });
      });
    return () => controller.abort();
  }, [key, modelKey, token]);

  // Guard against a stale result rendering for a newer key.
  return state.key === key ? state.value : EMPTY;
}
