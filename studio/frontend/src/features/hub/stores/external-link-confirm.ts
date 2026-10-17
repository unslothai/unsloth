// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { openLink } from "@/lib/open-link";
import { create } from "zustand";

interface ExternalLinkConfirmStore {
  pendingUrl: string | null;
  request: (url: string) => void;
  dismiss: () => void;
}

export const useExternalLinkConfirm = create<ExternalLinkConfirmStore>((set) => ({
  pendingUrl: null,
  request: (url) => set({ pendingUrl: url }),
  dismiss: () => set({ pendingUrl: null }),
}));

/** Unsloth's own site: unsloth.ai and its subdomains, over http(s). */
export function isUnslothSiteUrl(url: string): boolean {
  try {
    const { protocol, hostname } = new URL(url);
    const host = hostname.toLowerCase();
    return (
      (protocol === "https:" || protocol === "http:") &&
      (host === "unsloth.ai" || host.endsWith(".unsloth.ai"))
    );
  } catch {
    return false;
  }
}

export function confirmExternalLink(url: string): boolean {
  const trimmed = url?.trim() ?? "";
  if (!trimmed || trimmed.startsWith("#")) return false;
  const lower = trimmed.toLowerCase();
  if (
    !lower.includes("://") &&
    !lower.startsWith("//") &&
    !lower.startsWith("mailto:")
  ) {
    return false;
  }
  if (isUnslothSiteUrl(trimmed)) {
    return openLink(trimmed);
  }
  useExternalLinkConfirm.getState().request(trimmed);
  return true;
}
