// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { usePlatformStore } from "@/config/env";
import { useIsAccountOwner } from "@/features/auth";
import { isTauri } from "@/lib/api-base";
import { translate, useT } from "@/i18n";
import { toast } from "@/lib/toast";
import { type LibraryItem, errorMessage, revealLibraryItem } from "./api";
import { isLoopbackHost, revealLabelFor } from "./reveal-label";

/**
 * The Reveal label, or null where Reveal cannot help. It opens the file manager of the machine
 * running Studio, so it is offered only there (the desktop app, or a browser on this machine), only
 * where that machine can show one (not a container or a headless server, as the server reports),
 * and only to the installation owner, as the backend requires. Named as the server's platform
 * names it.
 */
export function useRevealLabel(): string | null {
  const t = useT();
  const owner = useIsAccountOwner();
  const deviceType = usePlatformStore((s) => s.deviceType);
  const fileManager = usePlatformStore((s) => s.fileManager);
  const local = isTauri || isLoopbackHost(window.location.hostname);
  if (!owner || !local) return null;
  const key = revealLabelFor(fileManager, deviceType);
  return key && t(key);
}

/** Chat attachments live inside their messages, so there is no file to show. */
export function canReveal(item: LibraryItem): boolean {
  return !item.id.startsWith("attachment:");
}

export function revealInFolder(id: string): void {
  revealLibraryItem(id).catch((error: unknown) =>
    toast.error(translate("library.toast.revealFailed"), {
      description: errorMessage(error),
    }),
  );
}
