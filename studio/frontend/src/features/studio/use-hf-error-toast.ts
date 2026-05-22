// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useEffect, useRef } from "react";
import { toast } from "sonner";
import { isHfAuthError } from "./picker-tab-toggle";

export function useHfErrorToast(
  error: string | null,
  entity: "models" | "datasets",
) {
  const lastToastedRef = useRef<string | null>(null);
  useEffect(() => {
    if (!error) {
      lastToastedRef.current = null;
      return;
    }
    if (lastToastedRef.current === error) return;
    lastToastedRef.current = error;
    if (isHfAuthError(error)) {
      toast.error("Hugging Face token rejected", {
        id: "hf-token-rejected",
        description: `Your token was refused. Update it in Settings → Hugging Face to search ${entity}.`,
      });
    } else {
      toast.error("Couldn't reach Hugging Face", {
        id: "hf-search-failed",
        description: error,
      });
    }
  }, [error, entity]);
}
