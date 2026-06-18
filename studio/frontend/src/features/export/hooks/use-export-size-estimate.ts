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

/** Dropdown sentinels / blanks mean "no model selected". */
function normalizeModelId(modelId: string | null | undefined): string {
  const id = (modelId ?? "").trim();
  return id && id !== "—" ? id : "";
}

/**
 * Fetch the selected model's FP16/BF16-equivalent size so the Export GGUF
 * picker can scale its per-quant size estimates. On any failure (offline,
 * gated, unresolved id) returns a null size, so the picker shows no estimate
 * instead of a misleading fixed number.
 */
export function useExportSizeEstimate(
  modelId: string | null | undefined,
): ExportSizeState {
  const key = normalizeModelId(modelId);
  const [state, setState] = useState<{ key: string; value: ExportSizeState }>(
    () => ({ key: "", value: EMPTY }),
  );

  useEffect(() => {
    if (!key) {
      setState({ key: "", value: EMPTY });
      return;
    }
    const controller = new AbortController();
    setState({ key, value: { ...EMPTY, loading: true } });
    void fetchExportSize(key, controller.signal)
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
  }, [key]);

  // Guard against a stale result rendering for a newer key.
  return state.key === key ? state.value : EMPTY;
}
