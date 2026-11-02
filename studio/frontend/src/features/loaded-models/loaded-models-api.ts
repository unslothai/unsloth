// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Reads are independent so a missing runtime does not blank other rows. Feature modules are
// imported directly because their indexes re-export pages kept out of the eager bundle.

import { authFetch, isAccountOwner } from "@/features/auth";
import {
  confirmStopRunningChatsIfNeeded,
  getInferenceStatus,
  isExternalModelId,
  requestScopedLocalPromptQueueStop,
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
import { unloadEmbeddingModel } from "@/features/settings/api/embedding-model";
import { useEmbeddingModelStore } from "@/features/settings/stores/embedding-model-store";
import { notifyModelEjected } from "@/lib/model-lifecycle-events";
import { ejectChatModel } from "./eject-chat-model";
import {
  embeddingRowName,
  type EmbeddingStatusResponse,
  type LoadedModelEntry,
  type LoadedModelSource,
  type SttStatusResponse,
  describeDiffusionStatus,
  describeEmbeddingStatus,
  describeInferenceStatus,
  describeSttStatus,
  describeVideoStatus,
  heldByAnotherAccount,
  mergeLoadedModels,
  sttEngineStatus,
  verifyResident,
} from "./loaded-models-sources";

type InferenceStatus = NonNullable<
  Parameters<typeof describeInferenceStatus>[0]
>;

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

// Owner-only route: a managed account would only collect 403s, so it is not asked.
async function readEmbeddingStatus(
  signal?: AbortSignal,
): Promise<EmbeddingStatusResponse | null> {
  const response = await authFetch("/api/settings/embedding-model", {
    signal,
  });
  if (!response.ok) return null;
  return (await response.json()) as EmbeddingStatusResponse;
}

// A runtime that accepts the connection and never answers would leave the whole batch pending
// forever, and with it the in-flight guard gating every later refresh. Well past a cold status
// probe, so a slow-but-healthy read still lands and only a real hang trips it.
const READ_TIMEOUT_MS = 10_000;

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
 * Bounds eject reads but rethrows. Not applied to unloads: they wait on the generate lock
 * for tens of seconds, and aborting would not cancel the teardown.
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
 * A failed read is not evidence the runtime is empty: an unreadable source keeps its last
 * rows, a readable one is always replaced (even by an empty answer).
 */
export type LoadedModelsRead = {
  entries: LoadedModelEntry[];
  unreadable: LoadedModelSource[];
  heldByOther: boolean;
};

export async function readLoadedModels(
  previous: readonly LoadedModelEntry[] = [],
): Promise<LoadedModelsRead> {
  const owner = isAccountOwner();
  const [inference, diffusion, video, stt, embedding] = await Promise.all([
    settled(readInferenceStatus),
    settled(getDiffusionStatus),
    settled(getVideoStatus),
    settled(readSttStatus),
    owner ? settled(readEmbeddingStatus) : Promise.resolve(null),
  ]);
  const kept = (source: LoadedModelSource) =>
    previous.filter((row) => row.source === source);
  const unreadable: LoadedModelSource[] = [];
  const group = <T>(
    source: LoadedModelSource,
    status: T | null,
    describe: (value: T) => LoadedModelEntry[],
  ) => {
    if (status !== null) return describe(status);
    unreadable.push(source);
    return kept(source);
  };
  const entries = mergeLoadedModels([
    group("chat", inference, describeInferenceStatus),
    group("image", diffusion, describeDiffusionStatus),
    group("video", video, describeVideoStatus),
    group("stt", stt, describeSttStatus),
    owner ? group("embedding", embedding, describeEmbeddingStatus) : [],
  ]);
  const heldByOther = heldByAnotherAccount([
    inference as { yours?: boolean } | null,
    diffusion as { yours?: boolean } | null,
    video as { yours?: boolean } | null,
  ]);
  return { entries, unreadable, heldByOther };
}

async function ejectChatRow(entry: LoadedModelEntry): Promise<EjectOutcome> {
  if (entry.servingCheckpoint) return ejectServingRow(entry);
  const decision = await confirmStopRunningChatsIfNeeded(
    "Unloading this model",
    "unloadOne",
    entry.name,
  );
  if (!decision.proceed) return { status: "declined" };
  const { unloadedAliases, stillResident, replacedBy } = await ejectChatModel(
    entry.name,
    {
      readResident: async () => {
        const status = await bounded(getInferenceStatus);
        const checkpoint = resolveInferenceCheckpointId(status);
        if (!checkpoint) return null;
        return {
          checkpoint,
          aliases: [checkpoint, status.active_model].filter(
            (alias): alias is string => alias != null,
          ),
        };
      },
      // Only once an unload is really sent: a row replaced since the poll unloads nothing.
      unload: (modelPath) => {
        requestScopedLocalPromptQueueStop(decision.promptQueueThreadIds);
        return unloadModel({
          model_path: modelPath,
          force_cancel_active: decision.forceCancelActive,
        });
      },
      matches: modelIdsMatch,
      cachedRow: entry.inactive === true,
      readCached: async () => (await bounded(getInferenceStatus)).loaded ?? [],
    },
  );
  if (stillResident) return { status: "stillResident", model: stillResident };
  if (replacedBy) return { status: "replaced", resident: replacedBy };
  if (unloadedAliases.length === 0) return { status: "alreadyFree" };
  // Only when the model is really gone: a reload during the run leaves it usable.
  clearChatSelectionFor(unloadedAliases);
  return { status: "ejected" };
}

async function ejectServingRow(entry: LoadedModelEntry): Promise<EjectOutcome> {
  const checkpoint = entry.servingCheckpoint ?? entry.name;
  const decision = await confirmStopRunningChatsIfNeeded(
    "Unloading this model",
    "unloadOne",
    checkpoint,
  );
  if (!decision.proceed) return { status: "declined" };
  const servedCheckpoints = async () =>
    (await bounded(getInferenceStatus)).serving_checkpoints ?? [];
  const { unloadedAliases, stillResident } = await ejectChatModel(checkpoint, {
    // Never the active model: that row has no servingCheckpoint.
    readResident: async () => null,
    unload: (modelPath) => {
      requestScopedLocalPromptQueueStop(decision.promptQueueThreadIds);
      return unloadModel({
        model_path: modelPath,
        force_cancel_active: decision.forceCancelActive,
      });
    },
    matches: (left, right) => left === right,
    cachedRow: true,
    readCached: servedCheckpoints,
  });
  if (stillResident) return { status: "stillResident", model: stillResident };
  if (unloadedAliases.length === 0) return { status: "alreadyFree" };
  clearChatSelectionFor([checkpoint, entry.name]);
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
 * `replaced`: the runtime holds another model, so nothing was unloaded (these endpoints
 * carry no model id). `alreadyFree`: it held nothing.
 */
export type EjectOutcome =
  | { status: "ejected" }
  | { status: "alreadyFree" }
  | { status: "declined" }
  | { status: "replaced"; resident: string }
  | { status: "stillResident"; model: string }
  // Unload accepted but the confirming read did not answer: neither done nor failed.
  | { status: "unverified" };

const UNVERIFIED = Symbol("unverified");

/** Narrows the stale-row window to the round trip; only a model-id unload could close it. */
async function ejectRuntimeRow(
  entry: LoadedModelEntry,
  resident: string | null,
  unload: () => Promise<string | null | typeof UNVERIFIED>,
): Promise<EjectOutcome> {
  const verdict = verifyResident(entry.name, resident, modelIdsMatch);
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
    case "embedding": {
      const before = await bounded(readEmbeddingStatus);
      if (!before) {
        throw new Error(
          "Could not read the embedding model status, so nothing was ejected.",
        );
      }
      if (!(before.backend_loaded ?? before.loaded))
        return { status: "alreadyFree" };
      // The unload route releases whatever is resident: a model swapped in since the poll stays.
      const resident = embeddingRowName(before);
      if (resident !== entry.name) return { status: "replaced", resident };
      await useEmbeddingModelStore
        .getState()
        .applyResidency(unloadEmbeddingModel);
      const after = await bounded(readEmbeddingStatus);
      if (!after) return { status: "unverified" };
      return (after.backend_loaded ?? after.loaded)
        ? { status: "stillResident", model: entry.name }
        : { status: "ejected" };
    }
    case "stt": {
      const engine = entry.sttEngine;
      if (!engine) throw new Error("This row names no dictation engine.");
      // Dictation loads and releases on its own, so the resident model can change unprompted.
      const before = await bounded(readSttStatus);
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
          // The unload body is fixed and the backend may serve `gguf` from transformers, so re-read.
          const after = await bounded(readSttStatus);
          // A failed read is null too; treating it as empty would hide a model still holding memory.
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
