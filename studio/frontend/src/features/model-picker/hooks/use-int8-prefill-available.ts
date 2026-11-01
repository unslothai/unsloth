// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useEffect, useState } from "react";
import { fetchPrefillOptionAvailable, type PrefillOption } from "../api/int8-prefill";

export function usePrefillOptionAvailable(
  option: PrefillOption,
  modelPath: string | null,
  hfToken: string | null,
): boolean {
  const key = modelPath == null ? null : `${option}\n${modelPath}\n${hfToken ?? ""}`;
  const [answer, setAnswer] = useState<{
    key: string;
    available: boolean;
  } | null>(null);
  useEffect(() => {
    if (key == null || modelPath == null) {
      return;
    }
    const controller = new AbortController();
    fetchPrefillOptionAvailable(option, modelPath, hfToken, controller.signal)
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
  }, [key, option, modelPath, hfToken]);
  return answer != null && answer.key === key && answer.available;
}
