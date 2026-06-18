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

// The "no selection" sentinel the Export page uses for the base model name
// (an em dash, U+2014). Built from a char code so this source stays ASCII.
const EMPTY_MODEL_SENTINEL = String.fromCharCode(0x2014);

/** Dropdown sentinels / blanks mean "no model selected". */
function normalizeModelId(modelId: string | null | undefined): string {
  const id = (modelId ?? "").trim();
  return id && id !== EMPTY_MODEL_SENTINEL ? id : "";
}

/**
 * Fetch the selected model's FP16/BF16-equivalent size so the Export GGUF
 * picker can scale its per-quant size estimates. On any failure (offline,
 * gated, unresolved id) returns a null size, so the picker shows no estimate
 * instead of a misleading fixed number. The token is forwarded so private or
 * gated repos can still be sized, and a token change triggers a refetch.
 */
export function useExportSizeEstimate(
  modelId: string | null | undefined,
  hfToken?: string | null,
): ExportSizeState {
  const modelKey = normalizeModelId(modelId);
  const token = hfToken?.trim() || "";
  // Composite key so the effect refetches when either the model or the token
  // changes. "|" cannot appear in an HF repo id or token. Never rendered.
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
