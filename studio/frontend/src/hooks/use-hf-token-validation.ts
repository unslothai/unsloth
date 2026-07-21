// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { validateHfToken } from "@/features/hf-auth";
import { useEffect, useRef, useState } from "react";
import { useDebouncedValue } from "./use-debounced-value";

export interface HfTokenValidationState {
  isValid: boolean | null;
  error: string | null;
  isChecking: boolean;
}

const INITIAL: HfTokenValidationState = {
  isValid: null,
  error: null,
  isChecking: false,
};

interface CompletedValidation extends HfTokenValidationState {
  token: string;
}

const NO_COMPLETED_VALIDATION: CompletedValidation = {
  ...INITIAL,
  token: "",
};

// Current user access tokens contain 34 characters after the hf_ prefix.
// Action-time validation still accepts legacy shapes without spending quota
// on every intermediate value typed into a live form field.
const COMPLETE_HF_TOKEN = /^hf_[A-Za-z0-9]{34}$/;

/**
 * Validates the HF token via the whoami-v2 API, debounced to avoid excessive
 * requests while typing. isValid is null until checked.
 */
export function useHfTokenValidation(token: string): HfTokenValidationState {
  const debouncedToken = useDebouncedValue(
    token.trim().replace(/^["']+|["']+$/g, ""),
    500,
  );
  const [completed, setCompleted] = useState<CompletedValidation>(
    NO_COMPLETED_VALIDATION,
  );
  const versionRef = useRef(0);
  const shouldValidate = COMPLETE_HF_TOKEN.test(debouncedToken);

  useEffect(() => {
    if (!shouldValidate) {
      versionRef.current += 1;
      return;
    }
    const version = ++versionRef.current;
    void validateHfToken(debouncedToken).then(
      (result) => {
        if (versionRef.current !== version) return;
        if (result.status === "valid") {
          setCompleted({
            token: debouncedToken,
            isValid: true,
            error: null,
            isChecking: false,
          });
        } else if (result.status === "invalid") {
          setCompleted({
            token: debouncedToken,
            isValid: false,
            error: "invalid or expired token",
            isChecking: false,
          });
        } else if (result.status === "rate_limited") {
          const wait = result.retryAfterSeconds
            ? ` Try again in about ${Math.ceil(result.retryAfterSeconds / 60)} minute(s).`
            : " Try again later.";
          setCompleted({
            token: debouncedToken,
            isValid: null,
            error: `Token verification is rate limited.${wait}`,
            isChecking: false,
          });
        } else {
          setCompleted({
            token: debouncedToken,
            isValid: null,
            error: "Could not verify the token. Check your connection and try again.",
            isChecking: false,
          });
        }
      },
      () => {
        if (versionRef.current !== version) return;
        setCompleted({
          token: debouncedToken,
          isValid: null,
          error: "Could not verify the token. Check your connection and try again.",
          isChecking: false,
        });
      },
    );
  }, [debouncedToken, shouldValidate]);

  if (!shouldValidate) return INITIAL;
  if (completed.token !== debouncedToken) {
    return { isValid: null, error: null, isChecking: true };
  }
  return {
    isValid: completed.isValid,
    error: completed.error,
    isChecking: false,
  };
}
