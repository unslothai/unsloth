// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useEffect, useState } from "react";
import { type ExportSizeEstimate, fetchExportSize } from "../api/export-api";

export interface ExportSizeState {
  data: ExportSizeEstimate | null;
  loading: boolean;
  fp16Bytes: number | null;
}

const EMPTY: ExportSizeState = { data: null, loading: false, fp16Bytes: null };

// Em dash U+2014 "no model" sentinel; char code keeps this file ASCII.
const EMPTY_MODEL_SENTINEL = String.fromCharCode(0x2014);

function normalizeModelId(modelId: string | null | undefined): string {
  const id = (modelId ?? "").trim();
  return id && id !== EMPTY_MODEL_SENTINEL ? id : "";
}

export function useExportSizeEstimate(
  modelId: string | null | undefined,
  hfToken?: string | null,
): ExportSizeState {
  const modelKey = normalizeModelId(modelId);
  const token = hfToken?.trim() || "";
  // "|" cannot appear in an id or token.
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

  return state.key === key ? state.value : EMPTY;
}
