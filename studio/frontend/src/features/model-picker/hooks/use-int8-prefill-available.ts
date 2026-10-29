// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useEffect, useState } from "react";
import { fetchInt8PrefillAvailable } from "../api/int8-prefill";

export function useInt8PrefillAvailable(
  modelPath: string | null,
  hfToken: string | null,
): boolean {
  const key = modelPath == null ? null : `${modelPath}\n${hfToken ?? ""}`;
  const [answer, setAnswer] = useState<{
    key: string;
    available: boolean;
  } | null>(null);
  useEffect(() => {
    if (key == null || modelPath == null) {
      return;
    }
    const controller = new AbortController();
    fetchInt8PrefillAvailable(modelPath, hfToken, controller.signal)
      .then((available) => {
        if (!controller.signal.aborted) {
          setAnswer({ key, available });
        }
      })
      .catch(() => {
        if (!controller.signal.aborted) {
          setAnswer({ key, available: false });
        }
      });
    return () => controller.abort();
  }, [key, modelPath, hfToken]);
  return answer != null && answer.key === key && answer.available;
}
