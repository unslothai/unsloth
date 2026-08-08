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
import { notifyModelEjected } from "@/lib/model-lifecycle-events";
import { ejectChatModel } from "./eject-chat-model";
import {
  type LoadedModelEntry,
  type LoadedModelSource,
  type SttStatusResponse,
  describeDiffusionStatus,
  describeInferenceStatus,
  describeSttStatus,
  describeVideoStatus,
  mergeLoadedModels,
  sttEngineStatus,
  verifyResident,
} from "./loaded-models-sources";

/** What describeInferenceStatus reads, without importing the chat types path. */
type InferenceStatus = NonNullable<
  Parameters<typeof describeInferenceStatus>[0]
>;

/** Read the chat runtime directly, as the dictation read below does, rather
 *  than through the chat barrel. */
async function readInferenceStatus(
  signal?: AbortSignal,
): Promise<InferenceStatus | null> {
  const response = await authFetch("/api/inference/status", { signal });
  if (!response.ok) return null;
  return (await response.json()) as InferenceStatus;
}

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

/**
 * The same bound for the eject path's *reads*, but the failure is raised rather
 * than swallowed. Without it a runtime that accepts the connection and never
 * answers leaves the eject pending forever, so the row keeps its spinner and
 * stays disabled until the page is reloaded.
 *
 * Deliberately not applied to the unloads themselves. Those block on the
 * runtime's generate lock while the in-flight denoise or clip winds down, which
 * is routinely tens of seconds, and aborting the fetch would not cancel the
 * teardown: the eject would report a failure while the memory was being freed.
 * A stale read is cheap to retry; a half-reported unload is not.
 */
async function bounded<T>(
  read: (signal: AbortSignal) => Promise<T>,
): Promise<T> {
  const timeout = disposableTimeoutSignal(READ_TIMEOUT_MS);
  try {
    return await read(timeout.signal);
  } finally {
    timeout.dispose();
  }
}

/**
 * Everything resident right now, across all four runtimes.
 *
 * `previous` is what the card is showing. A read that failed or timed out comes
 * back as null, which is not evidence the runtime is empty: dropping its rows
 * would take the model off the card, and with all four failing on one blip of a
 * remote Studio the whole card would vanish while everything stayed loaded. So
 * an unreadable source keeps what it last showed and a readable one is always
 * replaced, including by an empty answer, which is how an unload still clears.
 */
export async function readLoadedModels(
  previous: readonly LoadedModelEntry[] = [],
): Promise<LoadedModelEntry[]> {
  const [inference, diffusion, video, stt] = await Promise.all([
    settled(readInferenceStatus),
    settled(getDiffusionStatus),
    settled(getVideoStatus),
    settled(readSttStatus),
  ]);
  const kept = (source: LoadedModelSource) =>
    previous.filter((row) => row.source === source);
  return mergeLoadedModels([
    inference === null ? kept("chat") : describeInferenceStatus(inference),
    diffusion === null ? kept("image") : describeDiffusionStatus(diffusion),
    video === null ? kept("video") : describeVideoStatus(video),
    stt === null ? kept("stt") : describeSttStatus(stt),
  ]);
}

/** Release the model this row names, and only that one. See eject-chat-model.ts. */
async function ejectChatRow(entry: LoadedModelEntry): Promise<EjectOutcome> {
  const { unloadedAliases, stillResident, replacedBy } = await ejectChatModel(
    entry.name,
    {
      readResident: async () => {
        const status = await bounded(getInferenceStatus);
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
      cachedRow: entry.inactive === true,
      readCached: async () => (await bounded(getInferenceStatus)).loaded ?? [],
    },
  );
  if (stillResident) return { status: "stillResident", model: stillResident };
  if (replacedBy) return { status: "replaced", resident: replacedBy };
  // Nothing released and nothing in its place: the runtime was already idle.
  if (unloadedAliases.length === 0) return { status: "alreadyFree" };
  // Only when the row's model is really gone. A reload during the run leaves it
  // resident and still usable, so emptying the picker would be wrong.
  clearChatSelectionFor(unloadedAliases);
  return { status: "ejected" };
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
 * would have released whatever took its place. `alreadyFree` is the same read
 * of a runtime holding nothing at all, kept apart from `ejected` so a stale row
 * cannot report an unload that never ran.
 */
export type EjectOutcome =
  | { status: "ejected" }
  | { status: "alreadyFree" }
  | { status: "replaced"; resident: string }
  | { status: "stillResident"; model: string }
  // The unload was accepted but the read that confirms it did not answer, so
  // neither "done" nor "failed" is true. Reported as its own outcome rather
  // than collapsed into either, since the whole point of the confirming read is
  // that a 200 from this endpoint is not evidence.
  | { status: "unverified" };

/** `unload` could not confirm what the runtime holds, which is not the same as
 *  confirming it holds nothing. */
const UNVERIFIED = Symbol("unverified");

/**
 * The three identity-less unloads, guarded by a fresh read of their runtime.
 * `unload` resolves to a model still resident afterwards, null once free, or
 * UNVERIFIED when the confirming read did not answer.
 *
 * This narrows the window to the round trip rather than closing it, since only
 * a backend that took the model id could do that, but the row itself is up to a
 * whole poll old and that is the part worth not trusting.
 */
async function ejectRuntimeRow(
  entry: LoadedModelEntry,
  resident: string | null,
  unload: () => Promise<string | null | typeof UNVERIFIED>,
): Promise<EjectOutcome> {
  const verdict = verifyResident(entry.name, resident, modelIdsMatch);
  // Nothing resident: the row is stale and its memory is already free. Said
  // plainly rather than as an eject, which would claim an unload never issued.
  if (!resident) return { status: "alreadyFree" };
  if (verdict !== "match") return { status: "replaced", resident };
  const stillResident = await unload();
  if (stillResident === UNVERIFIED) return { status: "unverified" };
  return stillResident
    ? { status: "stillResident", model: stillResident }
    : { status: "ejected" };
}

/** Release one row, after checking the runtime still holds what the row names. */
export async function ejectLoadedModel(
  entry: LoadedModelEntry,
): Promise<EjectOutcome> {
  switch (entry.source) {
    case "chat":
      return ejectChatRow(entry);
    case "image": {
      const before = await bounded(getDiffusionStatus);
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
      const before = await bounded(getVideoStatus);
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
      const before = await bounded(readSttStatus);
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
          // The unload response body is a fixed {loaded_model: null}, and the
          // backend silently serves `gguf` from the transformers engine when
          // whisper-server is absent, so a 200 is not evidence this engine let
          // go. Re-read and report what it actually holds.
          const after = await bounded(readSttStatus);
          // A non-2xx read is null too, and reading that as "nothing left"
          // would toast success and drop the row for a model still holding
          // memory, which is the exact case this re-read exists to catch.
          if (!after) return UNVERIFIED;
          return sttEngineStatus(after, engine)?.loaded_model ?? null;
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
