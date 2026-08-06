// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { create } from "zustand";
import { type PreviewLink, loadPreviewLink } from "../api/preview-sharing";

// Shared so the history grid's copy button sees a link opened from Settings.
interface PreviewLinkState {
  url: string | null;
}

export const usePreviewLinkStore = create<PreviewLinkState>()(() => ({
  url: null,
}));

export function setPreviewLink(link: PreviewLink): void {
  usePreviewLinkStore.setState({ url: link.url });
}

export async function refreshPreviewLink(): Promise<void> {
  try {
    setPreviewLink(await loadPreviewLink());
  } catch {
    // Non-fatal: callers fall back to the local base and say so.
  }
}
