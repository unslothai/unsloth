// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useTrainingRuntimeStore } from "@/features/training";
import type { TranslationKey } from "@/i18n";
import { useNavigate } from "@tanstack/react-router";
import type { LibraryItem } from "./api";

export interface LibraryOrigin {
  label: TranslationKey;
  open: () => void;
}

/** Where an item was made or shared: its chat, project, training run or media history.
 *  Direct Library uploads have none. */
export function useLibraryOrigin(): (item: LibraryItem) => LibraryOrigin | null {
  const navigate = useNavigate();
  return (item) => {
    if (item.threadId) {
      const { threadId, pairId } = item;
      return {
        label: "library.preview.viewOriginalChat",
        open: () =>
          void navigate({
            to: "/chat",
            search: pairId ? { compare: pairId } : { thread: threadId },
          }),
      };
    }
    if (item.projectId) {
      const project = item.projectId;
      return {
        label: "library.preview.viewOriginalProject",
        open: () => void navigate({ to: "/chat", search: { project } }),
      };
    }
    if (item.runId) {
      const runId = item.runId;
      return {
        label: "library.preview.viewTrainingRun",
        open: () => {
          useTrainingRuntimeStore.getState().setSelectedHistoryRunId(runId);
          void navigate({ to: "/studio" });
        },
      };
    }
    if (item.archived) return null;
    const [kind, ...rest] = item.id.split(":");
    const search = { item: rest.join(":") };
    if (kind === "image") {
      return {
        label: "library.preview.viewInImages",
        open: () => void navigate({ to: "/images", search }),
      };
    }
    if (kind === "video") {
      return {
        label: "library.preview.viewInVideo",
        open: () => void navigate({ to: "/video", search }),
      };
    }
    if (kind === "audio") {
      return {
        label: "library.preview.viewInAudio",
        open: () => void navigate({ to: "/audio", search: { ...search, task: "text-to-speech" } }),
      };
    }
    return null;
  };
}
