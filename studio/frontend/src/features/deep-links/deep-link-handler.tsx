// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { isTauri } from "@/lib/api-base";
import { useNavigate } from "@tanstack/react-router";
import { useEffect } from "react";

import { createDeepLinkIntentGate } from "./deep-link-intent";
import { parseUnslothDeepLink } from "./parse-deep-link";

const acceptIntent = createDeepLinkIntentGate(2_000);

// Via Rust so a hidden login start also gets its Dock icon restored.
async function restoreMainWindow(): Promise<void> {
  const { invoke } = await import("@tauri-apps/api/core");
  await invoke("reveal_main_window");
}

export function DeepLinkHandler() {
  const navigate = useNavigate();

  useEffect(() => {
    if (!isTauri) return;

    let disposed = false;
    let receivedLiveIntent = false;
    let unlisten: (() => void) | undefined;

    const handleUrls = (urls: string[]): boolean => {
      if (disposed) return false;

      let hasValidIntent = false;
      let intent: ReturnType<typeof parseUnslothDeepLink> = null;

      let intentSequence: number | null = null;
      for (const rawUrl of urls) {
        const parsed = parseUnslothDeepLink(rawUrl);
        if (!parsed) continue;
        hasValidIntent = true;
        const sequence = acceptIntent(parsed.model, parsed.file);
        if (sequence !== null) {
          intent = parsed;
          intentSequence = sequence;
        }
      }
      if (!intent || intentSequence === null) return hasValidIntent;

      void restoreMainWindow().catch(() => undefined);
      void navigate({
        to: "/hub",
        search: {
          tab: "discover",
          kind: "models",
          model: intent.model,
          file: intent.file,

          intent: intentSequence,
        },
      });
      return true;
    };

    async function subscribe() {
      const { getCurrent, onOpenUrl } =
        await import("@tauri-apps/plugin-deep-link");
      if (disposed) return;

      const cleanup = await onOpenUrl((urls) => {
        if (handleUrls(urls)) receivedLiveIntent = true;
      });
      if (disposed) {
        cleanup();
        return;
      }
      unlisten = cleanup;

      const currentUrls = await getCurrent();
      if (currentUrls && !receivedLiveIntent) handleUrls(currentUrls);
    }

    void subscribe().catch(() => undefined);

    return () => {
      disposed = true;
      unlisten?.();
    };
  }, [navigate]);

  return null;
}
