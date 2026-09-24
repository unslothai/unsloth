// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { usePlatformStore } from "@/config/env";
import { useIsAccountOwner } from "@/features/auth";
import { isTauri } from "@/lib/api-base";
import { toast } from "@/lib/toast";
import { type LibraryItem, revealLibraryItem } from "./api";

const LOOPBACK_HOST = /^(localhost|127(\.\d{1,3}){3}|\[::1\]|::1)$|\.localhost$/;

/**
 * Which file manager Reveal opens, or null where it cannot help. It opens the file manager of the
 * machine running Studio, so it is offered only there (the desktop app, or a browser on this
 * machine), and only to the installation owner, as the backend requires.
 */
export function useRevealPlatform(): "finder" | "folder" | null {
  const owner = useIsAccountOwner();
  const deviceType = usePlatformStore((s) => s.deviceType);
  const local = isTauri || LOOPBACK_HOST.test(window.location.hostname);
  if (!owner || !local) return null;
  return deviceType === "mac" ? "finder" : "folder";
}

export function useRevealLabel(): string | null {
  const platform = useRevealPlatform();
  if (!platform) return null;
  return platform === "finder" ? "Reveal in Finder" : "Reveal in Folder";
}

/** Chat attachments live inside their messages, so there is no file to show. */
export function canReveal(item: LibraryItem): boolean {
  return !item.id.startsWith("attachment:");
}

export function revealInFolder(id: string): void {
  revealLibraryItem(id).catch((error: unknown) =>
    toast.error("Could not open the file manager", {
      description: error instanceof Error ? error.message : String(error),
    }),
  );
}
