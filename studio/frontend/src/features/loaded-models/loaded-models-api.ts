// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Reads and ejects for the indicator. Reads are best-effort and independent: a
// chat-only host has no video runtime, and that must not blank the other rows.
// images/video/api-monitor are reached directly, as api-monitor-page.tsx does:
// their indexes re-export pages __root.tsx keeps out of the eager bundle.

import { authFetch } from "@/features/auth";
import {
  getInferenceStatus,
  isExternalModelId,
  resolveInferenceCheckpointId,
  unloadModel,
  useChatRuntimeStore,
} from "@/features/chat";
import { disposableTimeoutSignal } from "@/features/hub/lib/abort-signals";
import { modelIdsMatch } from "@/features/hub/lib/model-identity";
import {
  getDiffusionStatus,
  unloadDiffusionModel,
} from "@/features/images/api";
import { getVideoStatus, unloadVideoModel } from "@/features/video/api";
import { notifyModelEjected } from "@/lib/model-eject-events";
import { ejectChatModel } from "./eject-chat-model";
import {
  type LoadedModelEntry,
  type SttStatusResponse,
  describeDiffusionStatus,
  describeInferenceStatus,
  describeSttStatus,
  describeVideoStatus,
  mergeLoadedModels,
  sttEngineStatus,
  verifyResident,
} from "./loaded-models-sources";

async function readSttStatus(
  signal?: AbortSignal,
): Promise<SttStatusResponse | null> {
  const response = await authFetch("/api/inference/audio/stt/status", {
    signal,
  });
  if (!response.ok) return null;
  return (await response.json()) as SttStatusResponse;
}

// A runtime that accepts the connection and never answers would leave the whole
// batch pending forever, and with it the in-flight guard gating every later
// refresh. Well past a cold status probe, so a slow-but-healthy read still lands
// and only a real hang trips it.
const READ_TIMEOUT_MS = 10_000;

/** null on failure or timeout: one stalled runtime must not empty the list. */
async function settled<T>(
  read: (signal: AbortSignal) => Promise<T>,
): Promise<T | null> {
  const timeout = disposableTimeoutSignal(READ_TIMEOUT_MS);
  try {
    return await read(timeout.signal);
  } catch {
    return null;
  } finally {
    timeout.dispose();
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

/** Release the model this row names, and only that one. See eject-chat-model.ts. */
async function ejectChatRow(entry: LoadedModelEntry): Promise<string | null> {
  const { unloadedAliases, stillResident } = await ejectChatModel(entry.name, {
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
    unload: (modelPath) => unloadModel({ model_path: modelPath }),
    matches: modelIdsMatch,
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

/**
 * How an eject ended. `replaced` means the row named a model the runtime no
 * longer holds, so nothing was unloaded: these endpoints carry no model id and
 * would have released whatever took its place.
 */
export type EjectOutcome =
  | { status: "ejected" }
  | { status: "replaced"; resident: string }
  | { status: "stillResident"; model: string };

/**
 * The three identity-less unloads, guarded by a fresh read of their runtime.
 * `unload` resolves to a model still resident afterwards, or null once free.
 *
 * This narrows the window to the round trip rather than closing it, since only
 * a backend that took the model id could do that, but the row itself is up to a
 * whole poll old and that is the part worth not trusting.
 */
async function ejectRuntimeRow(
  entry: LoadedModelEntry,
  resident: string | null,
  unload: () => Promise<string | null>,
): Promise<EjectOutcome> {
  const verdict = verifyResident(entry.name, resident, modelIdsMatch);
  if (verdict === "replaced" && resident) {
    return { status: "replaced", resident };
  }
  // Nothing resident: the row is stale and its memory is already free.
  if (verdict !== "match") return { status: "ejected" };
  const stillResident = await unload();
  return stillResident
    ? { status: "stillResident", model: stillResident }
    : { status: "ejected" };
}

/** Release one row, after checking the runtime still holds what the row names. */
export async function ejectLoadedModel(
  entry: LoadedModelEntry,
): Promise<EjectOutcome> {
  switch (entry.source) {
    case "chat": {
      // /unload matches on the model id, so this one needs no pre-check.
      const stillResident = await ejectChatRow(entry);
      return stillResident
        ? { status: "stillResident", model: stillResident }
        : { status: "ejected" };
    }
    case "image": {
      const before = await getDiffusionStatus();
      return ejectRuntimeRow(
        entry,
        before.loaded ? before.repo_id : null,
        async () => {
          const after = await unloadDiffusionModel();
          // The page owning this runtime keeps its own copy of the status.
          notifyModelEjected("image");
          return after.loaded ? (after.repo_id ?? entry.name) : null;
        },
      );
    }
    case "video": {
      const before = await getVideoStatus();
      return ejectRuntimeRow(
        entry,
        before.loaded ? before.repo_id : null,
        async () => {
          const after = await unloadVideoModel();
          notifyModelEjected("video");
          return after.loaded ? (after.repo_id ?? entry.name) : null;
        },
      );
    }
    case "stt": {
      const engine = entry.sttEngine;
      if (!engine) throw new Error("This row names no dictation engine.");
      // Dictation loads on demand and releases when idle, so the engine's
      // resident model can change with no user action at all.
      const before = await readSttStatus();
      // Unreadable status: say so rather than unload blind or claim success.
      if (!before) {
        throw new Error(
          "Could not read dictation status, so nothing was ejected.",
        );
      }
      const resident = sttEngineStatus(before, engine);
      return ejectRuntimeRow(
        entry,
        resident?.loaded_model ?? null,
        async () => {
          const query = new URLSearchParams({ engine }).toString();
          const response = await authFetch(
            `/api/inference/audio/stt/unload?${query}`,
            { method: "POST" },
          );
          if (!response.ok) throw new Error(await readErrorDetail(response));
          return null;
        },
      );
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
