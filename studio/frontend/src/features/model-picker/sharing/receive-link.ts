// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  AUTH_SESSION_CLEARED_EVENT,
  AUTH_SESSION_MARK_KEY,
  AUTH_SESSION_STORED_EVENT,
  hasAuthToken,
} from "@/features/auth";
import {
  createDeepLinkIntentGate,
  parseUnslothDeepLink,
} from "@/features/deep-links";
import { isTauri } from "@/lib/api-base";
import { toast } from "@/lib/toast";
import { markModelConfigDraftEdited } from "../model-config/model-config-draft";
import {
  clearModelConfigHandoff,
  createModelConfigHandoffRequestId,
} from "../model-config/model-config-handoff";
import { runConfigInbox } from "./inbox";
import {
  type RunConfigLinkResult,
  createRunConfigLink,
  parseRunConfigLink,
} from "./links";

const acceptNativeIntent = createDeepLinkIntentGate(2_000);
const nativeScheme = /^unsloth:/i;
// Only the startup URL is eligible; hash changes during this session are ignored.
let startupUrl = typeof window === "undefined" ? "" : window.location.href;
const recoveryKey = "unsloth.run-config-login.v1";
let awaitingLogin = false;
let recoveryExpiresAt = 0;

function clearRecovery() {
  try {
    sessionStorage.removeItem(recoveryKey);
  } catch {
    return;
  }
}

function saveRecovery() {
  const pending = runConfigInbox.getSnapshot();
  if (!pending || pending.draftKey) {
    return;
  }
  recoveryExpiresAt ||= Date.now() + 10 * 60_000;
  try {
    sessionStorage.setItem(
      recoveryKey,
      JSON.stringify({
        url: createRunConfigLink(pending.value),
        replaceHistory: pending.replaceHistory,
        session: localStorage.getItem(AUTH_SESSION_MARK_KEY),
        expiresAt: recoveryExpiresAt,
      }),
    );
  } catch {
    toast.error("Reopen the run settings link after signing in.");
  }
}

export function subscribeRunConfigSession(onChange: () => void): () => void {
  const onCleared = () => {
    startupUrl = "";
    acceptNativeIntent.clear();
    if (awaitingLogin) {
      saveRecovery();
    } else {
      clearPendingImport();
    }
    onChange();
  };
  const onStored = () => {
    if (awaitingLogin) {
      saveRecovery();
      awaitingLogin = false;
    }
    onChange();
  };
  const unsubscribe = runConfigInbox.subscribe(() => {
    const pending = runConfigInbox.getSnapshot();
    if (!pending || pending.draftKey) {
      awaitingLogin = false;
      recoveryExpiresAt = 0;
      clearRecovery();
    }
  });
  window.addEventListener(AUTH_SESSION_CLEARED_EVENT, onCleared);
  window.addEventListener(AUTH_SESSION_STORED_EVENT, onStored);
  return () => {
    unsubscribe();
    window.removeEventListener(AUTH_SESSION_CLEARED_EVENT, onCleared);
    window.removeEventListener(AUTH_SESSION_STORED_EVENT, onStored);
  };
}

function clearPendingImport() {
  awaitingLogin = false;
  recoveryExpiresAt = 0;
  clearRecovery();
  const pending = runConfigInbox.getSnapshot();
  if (pending) {
    clearModelConfigHandoff(pending.id);
    runConfigInbox.clear(pending.id);
  }
}

export function cancelRunConfigImportForEdit(draftKey: string): void {
  const pending = runConfigInbox.getSnapshot();
  if (pending?.draftKey !== draftKey) {
    return;
  }
  clearPendingImport();
  if (Object.keys(pending.value.config).length > 0) {
    markModelConfigDraftEdited(draftKey);
    toast.info("Run settings import cancelled", {
      description:
        "Your newer edits were kept. Reopen the link to import its settings.",
    });
  }
}

function receiveParsedLink(
  parsed: RunConfigLinkResult,
  replaceHistory = false,
  expiresAt = 0,
): boolean {
  if (parsed.kind === "unrelated") {
    return false;
  }
  startupUrl = "";
  clearPendingImport();
  if (parsed.kind === "invalid") {
    toast.error("Could not open shared run settings", {
      description: parsed.error,
    });
    return true;
  }
  runConfigInbox.submit({
    id: createModelConfigHandoffRequestId(),
    value: parsed.value,
    replaceHistory,
  });
  awaitingLogin = !hasAuthToken();
  if (awaitingLogin) {
    recoveryExpiresAt = expiresAt;
    saveRecovery();
  }
  return true;
}

export function receiveStartupRunConfigUrl(): void {
  const initial = startupUrl;
  startupUrl = "";
  if (!initial) {
    return;
  }
  if (!isTauri) {
    const parsed = parseRunConfigLink(initial);
    if (parsed.kind !== "unrelated") {
      const url = new URL(window.location.href);
      if (
        /^#run(?:\?|$)/.test(url.hash) &&
        new URLSearchParams(url.hash.slice(4)).toString() ===
          new URLSearchParams(new URL(initial).hash.slice(4)).toString()
      ) {
        if (url.searchParams.get("run") === "1") {
          url.searchParams.delete("run");
        }
        url.hash = "";
        window.history.replaceState(window.history.state, "", url.href);
      }
      receiveParsedLink(parsed, true);
      return;
    }
  }
  let recovery: {
    url?: unknown;
    replaceHistory?: boolean;
    session?: unknown;
    expiresAt?: unknown;
  } | null = null;
  try {
    recovery = JSON.parse(sessionStorage.getItem(recoveryKey) ?? "null");
  } catch {
    recovery = null;
  }
  clearRecovery();
  if (
    typeof recovery?.url === "string" &&
    typeof recovery.expiresAt === "number" &&
    recovery.expiresAt > Date.now() &&
    recovery.session === localStorage.getItem(AUTH_SESSION_MARK_KEY)
  ) {
    receiveParsedLink(
      parseRunConfigLink(recovery.url),
      recovery.replaceHistory === true,
      recovery.expiresAt,
    );
  }
}

export function receiveSharedRunConfigUrls(urls: string[]): boolean {
  for (let index = urls.length - 1; index >= 0; index -= 1) {
    const url = urls[index];
    if (!nativeScheme.test(url)) {
      continue;
    }
    const parsed = parseRunConfigLink(url);
    if (parsed.kind === "unrelated" && parseUnslothDeepLink(url)) {
      startupUrl = "";
      clearPendingImport();
      acceptNativeIntent.clear();
      return false;
    }
    if (parsed.kind === "unrelated") {
      continue;
    }
    if (acceptNativeIntent(url) === null) {
      return true;
    }
    return receiveParsedLink(parsed);
  }
  return false;
}
