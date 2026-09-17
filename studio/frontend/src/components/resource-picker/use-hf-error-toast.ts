// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { type TranslationKey, useT } from "@/i18n";
import { toast } from "@/lib/toast";
import { useEffect, useRef } from "react";
import { isHfAuthError } from "./hf-error";

const HF_RATE_LIMIT_RE = /\b429\b|rate.?limit|too many requests/i;
const HF_NETWORK_RE =
  /failed to fetch|network|offline|timeout|timed out|connection|econn|enotfound/i;
const HF_STATUS_RE = /\b([45]\d\d)\b/;

type HfErrorCategory = "auth" | "rate-limit" | "network" | "other";

const ENTITY_COPY = {
  models: {
    tokenRejectedTitle: "studio.modelPicker.tokenRejectedTitle",
    tokenRejectedBody: "studio.modelPicker.tokenRejectedBody",
    hubUnreachable: "studio.modelPicker.hubUnreachable",
    noun: "studio.modelPicker.noun",
  },
  datasets: {
    tokenRejectedTitle: "studio.datasetPicker.tokenRejectedTitle",
    tokenRejectedBody: "studio.datasetPicker.tokenRejectedBody",
    hubUnreachable: "studio.datasetPicker.hubUnreachable",
    noun: "studio.datasetPicker.noun",
  },
} as const satisfies Record<
  "models" | "datasets",
  Record<string, TranslationKey>
>;

function errorCategory(error: string): HfErrorCategory {
  if (isHfAuthError(error)) {
    return "auth";
  }
  if (HF_RATE_LIMIT_RE.test(error)) {
    return "rate-limit";
  }
  if (HF_NETWORK_RE.test(error)) {
    return "network";
  }
  return "other";
}

function toastKey(error: string): string {
  const category = errorCategory(error);
  if (category !== "other") {
    return category;
  }
  const status = error.match(HF_STATUS_RE)?.[1];
  return status ? `status:${status}` : "other";
}

export function useHfErrorToast(
  error: string | null,
  entity: "models" | "datasets",
) {
  const t = useT();
  const lastToastedRef = useRef<string | null>(null);
  useEffect(() => {
    if (!error) {
      lastToastedRef.current = null;
      return;
    }
    const keyPart = toastKey(error);
    const key = `${entity}:${keyPart}`;
    if (lastToastedRef.current === key) {
      return;
    }
    lastToastedRef.current = key;
    const category = errorCategory(error);
    const copy = ENTITY_COPY[entity];
    if (category === "auth") {
      toast.error(t(copy.tokenRejectedTitle), {
        id: "hf-token-rejected",
        description: t(copy.tokenRejectedBody),
      });
    } else if (category === "rate-limit") {
      toast.error(t("picker.rateLimitedTitle"), {
        id: `hf-search-${entity}-rate-limit`,
        description: t("picker.rateLimitedBody", { noun: t(copy.noun) }),
      });
    } else {
      toast.error(t(copy.hubUnreachable), {
        id: `hf-search-${entity}-${keyPart.replace(":", "-")}`,
        description: error,
      });
    }
  }, [error, entity, t]);
}
