// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Reads and ejects for the indicator. Reads are best-effort and independent: a
// chat-only host has no video runtime, and that must not blank the other rows.
// images/video/api-monitor are reached directly, as api-monitor-page.tsx does:
// their indexes re-export pages __root.tsx keeps out of the eager bundle.

import { unloadResident } from "@/features/api-monitor/unload-resident";
import { authFetch } from "@/features/auth";
import {
  getInferenceStatus,
  isExternalModelId,
  resolveInferenceCheckpointId,
  unloadModel,
  useChatRuntimeStore,
} from "@/features/chat";
import { modelIdsMatch } from "@/features/hub/lib/model-identity";
import {
  getDiffusionStatus,
  unloadDiffusionModel,
} from "@/features/images/api";
import { getVideoStatus, unloadVideoModel } from "@/features/video/api";
import {
  type LoadedModelEntry,
  type SttStatusResponse,
  describeDiffusionStatus,
  describeInferenceStatus,
  describeSttStatus,
  describeVideoStatus,
  mergeLoadedModels,
} from "./loaded-models-sources";

async function readSttStatus(): Promise<SttStatusResponse | null> {
  const response = await authFetch("/api/inference/audio/stt/status");
  if (!response.ok) return null;
  return (await response.json()) as SttStatusResponse;
}

/** null on failure: one unreachable runtime must not empty the list. */
async function settled<T>(read: () => Promise<T>): Promise<T | null> {
  try {
    return await read();
  } catch {
    return null;
  }
}

/** Everything resident right now, across all four runtimes. */
export async function readLoadedModels(): Promise<LoadedModelEntry[]> {
  const [inference, diffusion, video, stt] = await Promise.all([
    settled(getInferenceStatus),
    settled(getDiffusionStatus),
    settled(getVideoStatus),
    settled(readSttStatus),
  ]);
  return mergeLoadedModels([
    describeInferenceStatus(inference),
    describeDiffusionStatus(diffusion),
    describeVideoStatus(video),
    describeSttStatus(stt),
  ]);
}

/** Read, unload, re-read, as the API monitor's Unload does: an auto-switch can
 *  replace the model mid-eject, and /unload naming a replaced model is a no-op. */
async function ejectActiveChatModel(): Promise<string | null> {
  const { unloadedAliases, stillResident } = await unloadResident({
    readResident: async () => {
      const status = await getInferenceStatus();
      const checkpoint = resolveInferenceCheckpointId(status);
      if (!checkpoint) return null;
      // Both spellings: status reports the load path, the store the repo id.
      return {
        checkpoint,
        aliases: [checkpoint, status.active_model].filter(
          (alias): alias is string => alias != null,
        ),
      };
    },
    unload: (checkpoint) => unloadModel({ model_path: checkpoint }),
  });
  clearChatSelectionFor(unloadedAliases);
  return stillResident;
}

/** Clear the picker only when it names something this eject released: chat can
 *  hold an external selection while a local model is resident. */
function clearChatSelectionFor(aliases: string[]): void {
  const store = useChatRuntimeStore.getState();
  const selected = store.params.checkpoint;
  if (!selected || isExternalModelId(selected)) return;
  if (aliases.some((alias) => modelIdsMatch(selected, alias))) {
    store.clearCheckpoint();
  }
}

/** Release one row. Resolves to a model still resident, or null once freed. */
export async function ejectLoadedModel(
  entry: LoadedModelEntry,
): Promise<string | null> {
  switch (entry.source) {
    case "chat": {
      if (entry.inactive) {
        // Not active, so no auto-switch can swap it under the call.
        await unloadModel({ model_path: entry.name });
        clearChatSelectionFor([entry.name]);
        return null;
      }
      return ejectActiveChatModel();
    }
    case "image": {
      const status = await unloadDiffusionModel();
      return status.loaded ? (status.repo_id ?? entry.name) : null;
    }
    case "video": {
      const status = await unloadVideoModel();
      return status.loaded ? (status.repo_id ?? entry.name) : null;
    }
    case "stt": {
      const query = entry.sttEngine
        ? `?${new URLSearchParams({ engine: entry.sttEngine }).toString()}`
        : "";
      const response = await authFetch(
        `/api/inference/audio/stt/unload${query}`,
        { method: "POST" },
      );
      if (!response.ok) {
        throw new Error(await readErrorDetail(response));
      }
      return null;
    }
  }
}

async function readErrorDetail(response: Response): Promise<string> {
  try {
    const body = (await response.json()) as { detail?: unknown };
    if (typeof body.detail === "string") return body.detail;
  } catch {
    // non-JSON error body
  }
  return `Request failed (${response.status})`;
}
