// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { usePlatformStore } from "@/config/env";
import { useIsAccountOwner } from "@/features/auth";
import { isTauri } from "@/lib/api-base";
import { translate, useT } from "@/i18n";
import { toast } from "@/lib/toast";
import { type LibraryItem, errorMessage, revealLibraryItem } from "./api";
import { isLoopbackHost, revealLabelFor } from "./reveal-label";

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
