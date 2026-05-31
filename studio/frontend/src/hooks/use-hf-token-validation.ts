// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { whoAmI } from "@huggingface/hub";
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

/**
 * Validates the Hugging Face token by calling the whoami-v2 API.
 * Debounces the token to avoid excessive requests while typing.
 * Returns validation state: isValid (null = not checked), error message, and isChecking.
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
      await whoAmI({ accessToken: t });
      if (versionRef.current !== v) return;
      setState({ isValid: true, error: null, isChecking: false });
    } catch {
      if (versionRef.current !== v) return;
      setState({
        isValid: false,
        error: "invalid or expired token",
        isChecking: false,
      });
    }
  }, []);

  useEffect(() => {
    if (!debouncedToken) {
      setState(INITIAL);
      return;
    }
    runCheck(debouncedToken);
  }, [debouncedToken, runCheck]);

  return state;
}
