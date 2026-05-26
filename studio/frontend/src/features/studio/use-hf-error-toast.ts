// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useEffect, useRef } from "react";
import { toast } from "sonner";
import { isHfAuthError } from "./picker-tab-toggle";

const HF_RATE_LIMIT_RE = /\b429\b|rate.?limit|too many requests/i;
const HF_NETWORK_RE =
  /failed to fetch|network|offline|timeout|timed out|connection|econn|enotfound/i;
const HF_STATUS_RE = /\b([45]\d\d)\b/;

type HfErrorCategory = "auth" | "rate-limit" | "network" | "other";

function errorCategory(error: string): HfErrorCategory {
  if (isHfAuthError(error)) return "auth";
  if (HF_RATE_LIMIT_RE.test(error)) return "rate-limit";
  if (HF_NETWORK_RE.test(error)) return "network";
  return "other";
}

function toastKey(error: string): string {
  const category = errorCategory(error);
  if (category !== "other") return category;
  const status = error.match(HF_STATUS_RE)?.[1];
  return status ? `status:${status}` : "other";
}

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
    const keyPart = toastKey(error);
    const key = `${entity}:${keyPart}`;
    if (lastToastedRef.current === key) return;
    lastToastedRef.current = key;
    const category = errorCategory(error);
    if (category === "auth") {
      toast.error("Hugging Face token rejected", {
        id: "hf-token-rejected",
        description: `Your token was refused. Update it in Settings → Hugging Face to search ${entity}.`,
      });
    } else if (category === "rate-limit") {
      toast.error("Hugging Face rate limit reached", {
        id: `hf-search-${entity}-rate-limit`,
        description: `Wait a moment, then retry searching ${entity}.`,
      });
    } else {
      toast.error("Couldn't reach Hugging Face", {
        id: `hf-search-${entity}-${keyPart.replace(":", "-")}`,
        description: error,
      });
    }
  }, [error, entity]);
}
