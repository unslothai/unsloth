// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// These stores are used outside React and are not part of their features'
// React-facing public barrels.
// eslint-disable-next-line no-restricted-imports
import { useHfTokenStore } from "@/features/hub/stores/hf-token-store";
// eslint-disable-next-line no-restricted-imports
import { useSettingsDialogStore } from "@/features/settings/stores/settings-dialog-store";
import { validateHfToken } from "./api";
import { useHfTokenWarningStore } from "./store";

export interface PreparedHfToken {
  proceed: boolean;
  token: string | null;
}

// A caller can retain the pre-dialog payload while the shared store is cleared.
// Remember that one-session choice so a follow-up /load does not prompt again
// after its preceding /validate already continued anonymously.
const anonymousForSession = new Set<string>();

export async function prepareHfTokenForUse(
  token: string | null | undefined,
): Promise<PreparedHfToken> {
  const normalized = token?.trim() ?? "";
  if (!normalized) return { proceed: true, token: null };
  if (anonymousForSession.has(normalized)) {
    return { proceed: true, token: null };
  }

  let validation;
  try {
    validation = await validateHfToken(normalized);
  } catch {
    // Validation is advisory. Let the real operation retain its own error.
    return { proceed: true, token: normalized };
  }
  if (validation.status !== "invalid") {
    // A connectivity failure or rate limit cannot prove that a token is bad.
    // Let the real operation proceed and retain its repository-specific error.
    return { proceed: true, token: normalized };
  }

  const decision = await useHfTokenWarningStore.getState().requestDecision();
  if (decision === "anonymous") {
    anonymousForSession.add(normalized);
    useHfTokenStore.getState().clearToken();
    return { proceed: true, token: null };
  }
  if (decision === "replace") {
    useSettingsDialogStore.getState().openDialog("general");
  }
  return { proceed: false, token: normalized };
}
