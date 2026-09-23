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
import {
  markModelConfigDraftEdited,
  modelConfigDraftKey,
} from "../model-config/model-config-draft";
import {
  clearModelConfigHandoff,
  createModelConfigHandoffRequestId,
} from "../model-config/model-config-handoff";
import { runConfigInbox } from "./inbox";
import {
  MAX_RUN_CONFIG_URL_LENGTH,
  isRunConfigLink,
  runConfigHash,
} from "./link-address";

const acceptNativeIntent = createDeepLinkIntentGate(2_000);
const nativeScheme = /^unsloth:/i;
// Only the startup URL is eligible; hash changes during this session are ignored.
let startupUrl = typeof window === "undefined" ? "" : window.location.href;
const recoveryKey = "unsloth.run-config-login.v1";
const handledNativeUrlKey = "unsloth.run-config-native-handled.v1";
let handledNativeUrl = readHandledNativeUrl();
let awaitingLogin = false;
let recoveryExpiresAt = 0;
let recoveryUrl = "";
let recoveryReplaceHistory = false;
let intakeRevision = 0;

function readHandledNativeUrl(): string | null {
  try {
    return sessionStorage.getItem(handledNativeUrlKey);
  } catch {
    return null;
  }
}

function saveHandledNativeUrl(): void {
  if (!handledNativeUrl) {
    return;
  }
  try {
    sessionStorage.setItem(handledNativeUrlKey, handledNativeUrl);
  } catch {
    return;
  }
}

function clearRecovery() {
  try {
    sessionStorage.removeItem(recoveryKey);
  } catch {
    return;
  }
}

function saveRecovery() {
  const pending = runConfigInbox.getSnapshot();
  if (!recoveryUrl || pending?.draftKey) {
    return;
  }
  recoveryExpiresAt ||= Date.now() + 10 * 60_000;
  try {
    sessionStorage.setItem(
      recoveryKey,
      JSON.stringify({
        url: recoveryUrl,
        replaceHistory: recoveryReplaceHistory,
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
    saveHandledNativeUrl();
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
      recoveryUrl = "";
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
  intakeRevision += 1;
  recoveryUrl = "";
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
  runConfigInbox.recordEdit(draftKey);
  const pending = runConfigInbox.getSnapshot();
  const targetKey = pending?.target
    ? modelConfigDraftKey(pending.target.id, pending.target.meta.ggufVariant)
    : undefined;
  if (pending?.draftKey !== draftKey && targetKey !== draftKey) {
    return;
  }
  cancelEditedRunConfigImport(draftKey);
}

export function cancelEditedRunConfigImport(draftKey: string): void {
  const pending = runConfigInbox.getSnapshot();
  if (!pending) return;
  clearPendingImport();
  if (Object.keys(pending.value.config).length > 0) {
    markModelConfigDraftEdited(draftKey);
    toast.info("Run settings import cancelled", {
      description:
        "Your newer edits were kept. Reopen the link to import its settings.",
    });
  }
}

async function receiveRunConfigUrl(
  url: string,
  replaceHistory = false,
  expiresAt = 0,
): Promise<void> {
  startupUrl = "";
  clearPendingImport();
  const revision = intakeRevision;
  awaitingLogin = !hasAuthToken();
  recoveryUrl = url.length <= MAX_RUN_CONFIG_URL_LENGTH ? url : "";
  recoveryReplaceHistory = replaceHistory;
  recoveryExpiresAt = expiresAt;
  if (awaitingLogin) {
    saveRecovery();
  }
  try {
    const { parseRunConfigLink } = await import("./runtime");
    if (revision !== intakeRevision) {
      return;
    }
    const parsed = parseRunConfigLink(url);
    if (parsed.kind !== "valid") {
      clearPendingImport();
      if (parsed.kind === "invalid") {
        toast.error("Could not open shared run settings", {
          description: parsed.error,
        });
      }
      return;
    }
    runConfigInbox.submit({
      id: createModelConfigHandoffRequestId(),
      value: parsed.value,
      replaceHistory,
    });
  } catch {
    if (revision !== intakeRevision) {
      return;
    }
    clearPendingImport();
    acceptNativeIntent.clear();
    toast.error("Could not open shared run settings", {
      description: "Reopen the link to try again.",
    });
  }
}

export async function receiveStartupRunConfigUrl(): Promise<void> {
  const initial = startupUrl;
  startupUrl = "";
  if (!initial) {
    return;
  }
  if (!isTauri && isRunConfigLink(initial)) {
    const url = new URL(window.location.href);
    if (
      runConfigHash.test(url.hash) &&
      new URLSearchParams(url.hash.slice(4)).toString() ===
        new URLSearchParams(new URL(initial).hash.slice(4)).toString()
    ) {
      if (url.searchParams.get("run") === "1") {
        url.searchParams.delete("run");
      }
      url.hash = "";
      window.history.replaceState(window.history.state, "", url.href);
    }
    await receiveRunConfigUrl(initial, true);
    return;
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
    isRunConfigLink(recovery.url) &&
    typeof recovery.expiresAt === "number" &&
    recovery.expiresAt > Date.now() &&
    recovery.session === localStorage.getItem(AUTH_SESSION_MARK_KEY)
  ) {
    await receiveRunConfigUrl(
      recovery.url,
      recovery.replaceHistory === true,
      recovery.expiresAt,
    );
  }
}

export function receiveSharedRunConfigUrls(
  urls: string[],
  source: "startup" | "event" = "event",
): boolean | "ignored" {
  for (let index = urls.length - 1; index >= 0; index -= 1) {
    const url = urls[index];
    if (!nativeScheme.test(url)) {
      continue;
    }
    const candidate = isRunConfigLink(url);
    if (!candidate && parseUnslothDeepLink(url)) {
      startupUrl = "";
      clearPendingImport();
      acceptNativeIntent.clear();
      return false;
    }
    if (!candidate) {
      continue;
    }
    if (source === "startup" && handledNativeUrl === url) {
      return "ignored";
    }
    handledNativeUrl = url;
    saveHandledNativeUrl();
    if (acceptNativeIntent(url) === null) {
      return true;
    }
    void receiveRunConfigUrl(url);
    return true;
  }
  return false;
}
