


// These stores are used outside React and are not part of their features' public barrels.
// eslint-disable-next-line no-restricted-imports
import { useHfTokenStore } from "@/features/hub/stores/hf-token-store";
// eslint-disable-next-line no-restricted-imports
import { useSettingsDialogStore } from "@/features/settings/stores/settings-dialog-store";
import { type HfTokenValidationResult, validateHfToken } from "./api";
import { useHfTokenWarningStore } from "./store";

export interface PreparedHfToken {
  proceed: boolean;
  token: string | null;
}

interface PrepareHfTokenOptions {
  allowAnonymous?: boolean;
}

// A caller can retain the pre-dialog payload while the shared store is cleared. Remember that
// one-session choice so a follow-up /load does not prompt again after an anonymous /validate.
const anonymousForSession = new Set<string>();

export async function prepareHfTokenForUse(
  token: string | null | undefined,
  options: PrepareHfTokenOptions = {},
): Promise<PreparedHfToken> {
  const normalized = token?.trim() ?? "";
  if (!normalized) {
    return { proceed: true, token: null };
  }
  const allowAnonymous = options.allowAnonymous ?? true;
  if (allowAnonymous && anonymousForSession.has(normalized)) {
    return { proceed: true, token: null };
  }

  let validation: HfTokenValidationResult;
  try {
    validation = await validateHfToken(normalized);
  } catch {
    // Validation is advisory. Let the real operation retain its own error.
    return { proceed: true, token: normalized };
  }
  if (validation.status !== "invalid") {
    // A connectivity failure or rate limit cannot prove a token is bad; let the real operation run.
    return { proceed: true, token: normalized };
  }

  const decision = await useHfTokenWarningStore
    .getState()
    .requestDecision(allowAnonymous);
  if (decision === "anonymous") {
    anonymousForSession.add(normalized);
    const tokenStore = useHfTokenStore.getState();
    if (tokenStore.token === normalized) {
      tokenStore.clearToken();
    }
    return { proceed: true, token: null };
  }
  if (decision === "replace") {
    useSettingsDialogStore.getState().openDialog("general");
  }
  return { proceed: false, token: normalized };
}
