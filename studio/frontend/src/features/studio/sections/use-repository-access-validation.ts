// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  type RepositoryAccessStatus,
  validateRepositoryAccess,
} from "@/features/hf-auth";
import { hfApiToken } from "@/features/hub";
import { useCallback, useEffect, useRef, useState } from "react";
import { isRepositoryId } from "./repository-id";

export type RepositoryValidationState =
  | "idle"
  | "checking"
  | RepositoryAccessStatus
  | "invalid_syntax";

const DEBOUNCE_MS = 650;

export function useRepositoryAccessValidation(repoId: string, token: string) {
  const [completed, setCompleted] = useState<{
    repoId: string;
    token: string;
    state: RepositoryValidationState;
  } | null>(null);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const controllerRef = useRef<AbortController | null>(null);
  const sequenceRef = useRef(0);
  const normalizedRepoId = repoId.trim();
  const normalizedToken = hfApiToken(token) ?? "";

  const cancelPending = useCallback(() => {
    if (timerRef.current) clearTimeout(timerRef.current);
    timerRef.current = null;
    controllerRef.current?.abort();
    controllerRef.current = null;
    sequenceRef.current += 1;
  }, []);

  const validate = useCallback(() => {
    cancelPending();
    if (!normalizedRepoId) return;
    if (!isRepositoryId(normalizedRepoId)) {
      setCompleted({
        repoId: normalizedRepoId,
        token: normalizedToken,
        state: "invalid_syntax",
      });
      return;
    }
    if (!normalizedToken) {
      setCompleted({
        repoId: normalizedRepoId,
        token: normalizedToken,
        state: "authentication_required",
      });
      return;
    }
    const sequence = sequenceRef.current;
    const controller = new AbortController();
    controllerRef.current = controller;
    setCompleted({
      repoId: normalizedRepoId,
      token: normalizedToken,
      state: "checking",
    });
    void validateRepositoryAccess(
      normalizedRepoId,
      normalizedToken,
      controller.signal,
    ).then(
      (state) => {
        if (sequenceRef.current !== sequence || controller.signal.aborted)
          return;
        setCompleted({
          repoId: normalizedRepoId,
          token: normalizedToken,
          state,
        });
      },
      (error: unknown) => {
        if (
          sequenceRef.current !== sequence ||
          controller.signal.aborted ||
          (error instanceof DOMException && error.name === "AbortError")
        )
          return;
        setCompleted({
          repoId: normalizedRepoId,
          token: normalizedToken,
          state: "unavailable",
        });
      },
    );
  }, [cancelPending, normalizedRepoId, normalizedToken]);

  useEffect(() => {
    cancelPending();
    if (!normalizedRepoId) return;
    timerRef.current = setTimeout(validate, DEBOUNCE_MS);
    return cancelPending;
  }, [cancelPending, normalizedRepoId, validate]);

  const state =
    completed?.repoId === normalizedRepoId &&
    completed.token === normalizedToken
      ? completed.state
      : "idle";
  return { state, validateNow: validate };
}
