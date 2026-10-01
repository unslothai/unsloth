// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { LinkedInstance } from "@/features/settings/api/linked-instances";
import { toast } from "sonner";
import { type LinkedLoadStage, loadChatOnLinked } from "./linked-api";
import type { LinkedChatPick } from "./linked-models-panel";

const GB = 1024 ** 3;

function stageText(stage: LinkedLoadStage, where: string): string {
  if (stage.stage === "loading") {
    return stage.fraction
      ? `Loading on ${where} · ${Math.round(stage.fraction * 100)}%`
      : `Loading on ${where}`;
  }
  const size =
    stage.total > 0
      ? ` · ${(stage.bytes / GB).toFixed(1)} of ${(stage.total / GB).toFixed(1)} GB`
      : "";
  const pct =
    stage.fraction != null ? ` ${Math.round(stage.fraction * 100)}%` : "";
  return `Downloading on ${where}${pct}${size}`;
}

/** Runs the pick on the linked machine behind a progress toast; `onReady` gets the routed id. */
export async function loadLinkedChatWithToast(
  instance: LinkedInstance,
  pick: LinkedChatPick,
  onReady: (modelId: string) => void,
): Promise<void> {
  const where = `@${instance.name}`;
  const name = `${pick.repoId.slice(pick.repoId.lastIndexOf("/") + 1)}${pick.variant ? ` ${pick.variant}` : ""}`;
  const id = toast.loading(name, { description: `Starting on ${where}` });
  try {
    const modelId = await loadChatOnLinked(instance, pick, (stage) =>
      toast.loading(name, { id, description: stageText(stage, where) }),
    );
    toast.success(name, { id, description: `Ready on ${where}` });
    onReady(modelId);
  } catch (error) {
    toast.error(name, {
      id,
      description: `Couldn't load on ${where}: ${error instanceof Error ? error.message : String(error)}`,
    });
  }
}
