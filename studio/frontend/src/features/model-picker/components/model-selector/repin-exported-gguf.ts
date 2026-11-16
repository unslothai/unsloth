// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { listLoras } from "@/features/chat/api/chat-api";
import { pinKey, usePinnedModelsStore } from "./pinned-models";

/** An exported GGUF is keyed by one of its files: after a quant delete, move its pin to a surviving file. */
export async function repinExportedGgufAfterDelete(
  oldId: string,
): Promise<void> {
  const state = usePinnedModelsStore.getState();
  if (!state.pinned.includes(pinKey(oldId))) return;
  try {
    const folder = oldId.slice(
      0,
      Math.max(oldId.lastIndexOf("/"), oldId.lastIndexOf("\\")),
    );
    const { loras } = await listLoras();
    const next = loras.find(
      (lora) =>
        lora.source === "exported" &&
        lora.export_type === "gguf" &&
        (lora.adapter_path.startsWith(`${folder}/`) ||
          lora.adapter_path.startsWith(`${folder}\\`)),
    );
    if (next) state.replacePinned(pinKey(oldId), pinKey(next.adapter_path));
    else state.unpinRepo(oldId);
  } catch {
    state.unpinRepo(oldId);
  }
}
