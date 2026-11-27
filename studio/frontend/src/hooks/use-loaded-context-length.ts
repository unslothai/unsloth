// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { getInferenceStatus } from "@/features/chat/api/chat-api";
import { useChatRuntimeStore } from "@/features/chat/stores/chat-runtime-store";
import { loadedContextFields } from "@/features/model-picker/model-config/per-model-config";
import { useEffect, useState } from "react";

/** The loaded model's context length; read from the server when Chat has not loaded it into the store yet. */
export function useLoadedContextLength(): number | null {
  const storeContextLength = useChatRuntimeStore((s) => s.loadedContextLength);
  const [statusContextLength, setStatusContextLength] = useState<number | null>(null);
  useEffect(() => {
    if (storeContextLength != null) return;
    const controller = new AbortController();
    getInferenceStatus(controller.signal)
      .then((status) => setStatusContextLength(loadedContextFields(status).loadedContextLength))
      .catch(() => undefined);
    return () => controller.abort();
  }, [storeContextLength]);
  return storeContextLength ?? statusContextLength;
}
