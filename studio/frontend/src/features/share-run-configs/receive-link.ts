// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { toast } from "@/lib/toast";
import { parseUnslothDeepLink } from "../deep-links/parse-deep-link";
import { markModelConfigDraftEdited } from "../model-picker/model-config/model-config-draft";
import {
  clearModelConfigHandoff,
  createModelConfigHandoffRequestId,
} from "../model-picker/model-config/model-config-handoff";
import { runConfigInbox } from "./inbox";
import { type RunConfigLinkResult, parseRunConfigLink } from "./links";

let lastNativeIntent = { url: "", at: 0 };
let startupUrl = typeof window === "undefined" ? "" : window.location.href;

function clearPendingImport() {
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

function receiveParsedLink(parsed: RunConfigLinkResult): boolean {
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
  });
  return true;
}

export function receiveRunConfigUrl(url: string): boolean {
  const parsed = parseRunConfigLink(url);
  if (parsed.kind !== "unrelated") {
    lastNativeIntent = { url: "", at: 0 };
  }
  return receiveParsedLink(parsed);
}

export function receiveStartupRunConfigUrl(currentUrl: string): void {
  const initial = startupUrl;
  startupUrl = "";
  if (initial && !receiveRunConfigUrl(currentUrl)) {
    receiveRunConfigUrl(initial);
  }
}

export function receiveSharedRunConfigUrls(urls: string[]): boolean {
  for (let index = urls.length - 1; index >= 0; index -= 1) {
    const url = urls[index];
    const parsed = parseRunConfigLink(url);
    if (parsed.kind === "unrelated" && parseUnslothDeepLink(url)) {
      startupUrl = "";
      clearPendingImport();
      lastNativeIntent = { url: "", at: 0 };
      return false;
    }
    if (parsed.kind === "unrelated") {
      continue;
    }
    const now = Date.now();
    if (lastNativeIntent.url === url && now - lastNativeIntent.at < 2_000) {
      return true;
    }
    lastNativeIntent = { url, at: now };
    return receiveParsedLink(parsed);
  }
  return false;
}
