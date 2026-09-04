// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { translate } from "@/i18n";
import type { TranslationKey } from "@/i18n";

const TRAINING_START_ERROR_KEYS = new Map<string, TranslationKey>([
  ["hf_model_access_denied", "studio.training.hfModelAccessDenied"],
  [
    "hf_model_verification_rate_limited",
    "studio.training.hfModelVerificationRateLimited",
  ],
  ["hf_model_verification_failed", "studio.training.hfModelVerificationFailed"],
  [
    "hf_model_metadata_unavailable",
    "studio.training.hfModelMetadataUnavailable",
  ],
]);

function errorDetails(
  error: unknown,
  explicitErrorCode?: string | null,
): { message: string; errorCode: string | null } {
  if (typeof error === "string") {
    return { message: error, errorCode: explicitErrorCode ?? null };
  }
  if (error && typeof error === "object") {
    const candidate = error as {
      message?: unknown;
      errorCode?: unknown;
      code?: unknown;
    };
    const errorCode =
      explicitErrorCode ??
      (typeof candidate.errorCode === "string"
        ? candidate.errorCode
        : typeof candidate.code === "string"
          ? candidate.code
          : null);
    return {
      message:
        typeof candidate.message === "string"
          ? candidate.message
          : String(error),
      errorCode,
    };
  }
  return { message: String(error), errorCode: explicitErrorCode ?? null };
}

export function normalizeTrainingStartError(
  error: unknown,
  explicitErrorCode?: string | null,
): string {
  const { message, errorCode } = errorDetails(error, explicitErrorCode);
  const translationKey = errorCode
    ? TRAINING_START_ERROR_KEYS.get(errorCode)
    : undefined;
  if (translationKey) {
    return translate(translationKey);
  }
  const normalized = message.toLowerCase();
  if (
    normalized.includes("failed to check dataset format") &&
    normalized.includes("dataset scripts are no longer supported")
  ) {
    return translate("studio.training.legacyDatasetScriptUnsupported");
  }
  return message;
}
