// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { pickNativeDocumentFolder } from "@/features/native-intents";
import { toast } from "@/lib/toast";
import { useSyncExternalStore } from "react";
import { createChatProject } from "../hooks/use-chat-projects";
import type { ProjectRecord } from "../types";

let opening = false;
const openingListeners = new Set<() => void>();

function setOpening(value: boolean): void {
  opening = value;
  openingListeners.forEach((listener) => listener());
}

export function useOpeningFolder(): boolean {
  return useSyncExternalStore(
    (listener) => {
      openingListeners.add(listener);
      return () => openingListeners.delete(listener);
    },
    () => opening,
  );
}

export async function openFolderAsProject(): Promise<ProjectRecord | null> {
  if (opening) return null;
  setOpening(true);
  try {
    const selected = await pickNativeDocumentFolder();
    if (!selected) return null;
    return await createChatProject(
      selected.displayName,
      selected.token,
      undefined,
      { reuseExisting: true },
    );
  } catch (error) {
    toast.error("Could not open folder", {
      description: error instanceof Error ? error.message : String(error),
    });
    return null;
  } finally {
    setOpening(false);
  }
}
