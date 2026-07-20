// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { validateHfToken } from "@/features/hf-auth";
import { useCallback, useEffect, useRef, useState } from "react";
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
  const [state, setState] = useState<HfTokenValidationState>(INITIAL);
  const versionRef = useRef(0);

  const runCheck = useCallback(async (t: string) => {
    if (!t) {
      setState({ isValid: null, error: null, isChecking: false });
      return;
    }

    const v = ++versionRef.current;
    setState((prev) => ({ ...prev, isChecking: true, error: null }));

    try {
      const result = await validateHfToken(t);
      if (versionRef.current !== v) return;
      if (result.status === "valid") {
        setState({ isValid: true, error: null, isChecking: false });
      } else if (result.status === "invalid") {
        setState({
          isValid: false,
          error: "invalid or expired token",
          isChecking: false,
        });
      } else if (result.status === "rate_limited") {
        const wait = result.retryAfterSeconds
          ? ` Try again in about ${Math.ceil(result.retryAfterSeconds / 60)} minute(s).`
          : " Try again later.";
        setState({
          isValid: null,
          error: `Token verification is rate limited.${wait}`,
          isChecking: false,
        });
      } else {
        setState({
          isValid: null,
          error: "Could not verify the token. Check your connection and try again.",
          isChecking: false,
        });
      }
    } catch {
      if (versionRef.current !== v) return;
      setState({
        isValid: null,
        error: "Could not verify the token. Check your connection and try again.",
        isChecking: false,
      });
    }
  }, []);

  useEffect(() => {
    if (!debouncedToken) {
      setState(INITIAL);
      return;
    }
    if (!COMPLETE_HF_TOKEN.test(debouncedToken)) {
      setState(INITIAL);
      return;
    }
    runCheck(debouncedToken);
  }, [debouncedToken, runCheck]);

  return state;
}
