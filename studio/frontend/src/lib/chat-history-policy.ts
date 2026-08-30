// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export const CHAT_HISTORY_DISABLED_ATTRIBUTE = "data-unsloth-no-chat-history";
const CHAT_HISTORY_DISABLED_GLOBAL = "__UNSLOTH_NO_CHAT_HISTORY__";
const CHAT_DRAFT_PREFIXES = ["chat-draft:", "chat-draft-pastes:"];
const CHAT_HISTORY_KEYS = ["unsloth_chat_auto_continue_leases"];
type HostPolicyWindow = Window & {
  [CHAT_HISTORY_DISABLED_GLOBAL]?: boolean;
};

let policyPurgedStorage: Storage | null = null;

/** Delete persisted chat browser state without disturbing unrelated local preferences. */
export function clearPersistedChatDrafts(): void {
  if (typeof window === "undefined") return;
  try {
    const storage = window.localStorage;
    const keys: string[] = [];
    for (let index = 0; index < storage.length; index += 1) {
      const key = storage.key(index);
      if (
        key &&
        (CHAT_HISTORY_KEYS.includes(key) ||
          CHAT_DRAFT_PREFIXES.some((prefix) => key.startsWith(prefix)))
      ) {
        keys.push(key);
      }
    }
    for (const key of keys) storage.removeItem(key);
  } catch {
    // Storage can be unavailable in private or locked-down browser contexts.
  }
}

/** Immutable host policy injected before boot scripts run. */
export function isChatHistoryDisabled(): boolean {
  const disabled =
    (typeof window !== "undefined" &&
      (window as HostPolicyWindow)[CHAT_HISTORY_DISABLED_GLOBAL] === true) ||
    (typeof document !== "undefined" &&
      document.documentElement?.getAttribute(
        CHAT_HISTORY_DISABLED_ATTRIBUTE,
      ) === "true");
  if (disabled && typeof window !== "undefined") {
    try {
      if (policyPurgedStorage !== window.localStorage) {
        clearPersistedChatDrafts();
        policyPurgedStorage = window.localStorage;
      }
    } catch {
      // The policy still applies when storage cannot be inspected.
    }
  }
  return disabled;
}

export function chatHistoryDisabledError(): Error {
  return new Error("Chat history is disabled by the server operator.");
}

/** Remove chat dictation transcripts while retaining unrelated voice preferences. */
export function stripPersistedDictationHistory(value: string): string {
  if (!isChatHistoryDisabled()) return value;
  try {
    const parsed = JSON.parse(value) as {
      state?: { recentDictations?: unknown };
    };
    if (!parsed.state || !("recentDictations" in parsed.state)) return value;
    parsed.state.recentDictations = [];
    return JSON.stringify(parsed);
  } catch {
    return value;
  }
}
