// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { ExportedMessageRepository, ThreadMessage } from "@assistant-ui/react";

import type { MessageRecord } from "../types";
import { restoredAssistantStatus } from "./continuation";
import { resolveSavedBranchHead } from "./message-order";
import {
  generationNeedsRecovery,
  generationRawContent,
  isLiveGenerationRun,
  recoveredContentToImport,
} from "./chat-generation-recovery";

const refreshGenerations = new Map<string, number>();

export function beginSavedHistoryReconciliation(threadId: string): number {
  const generation = (refreshGenerations.get(threadId) ?? 0) + 1;
  refreshGenerations.set(threadId, generation);
  return generation;
}

export function isSavedHistoryReconciliationSuperseded(
  threadId: string,
  generation: number,
): boolean {
  return refreshGenerations.get(threadId) !== generation;
}

function contentProjectionKey(content: unknown): string {
  const { raw, carried } = generationRawContent(content);
  return `${raw}\0${JSON.stringify(carried)}`;
}

function runtimeCustom(
  message: ThreadMessage,
): Record<string, unknown> {
  const metadata = message.metadata as { custom?: Record<string, unknown> } | undefined;
  return metadata?.custom ?? {};
}

export function shouldReconcileOrdinaryStoredAssistant(options: {
  stored: MessageRecord;
  runtime: ThreadMessage;
  editingMessageId: string | null;
}): boolean {
  const { stored, runtime, editingMessageId } = options;
  if (stored.role !== "assistant" || runtime.role !== "assistant") {
    return false;
  }
  if (editingMessageId === stored.id) {
    return false;
  }
  if (runtime.status?.type === "running") {
    return false;
  }

  const storedMeta = (stored.metadata ?? {}) as Record<string, unknown>;
  const storedRunId = storedMeta.generationRunId;
  if (typeof storedRunId === "string") {
    if (isLiveGenerationRun(storedRunId)) {
      return false;
    }
    if (generationNeedsRecovery(storedMeta)) {
      return false;
    }
  }

  const runtimeMeta = runtimeCustom(runtime);
  const runtimeRunId = runtimeMeta.generationRunId;
  if (typeof runtimeRunId === "string") {
    if (isLiveGenerationRun(runtimeRunId)) {
      return false;
    }
    if (generationNeedsRecovery(runtimeMeta)) {
      return false;
    }
  }

  const storedKey = contentProjectionKey(stored.content);
  const runtimeKey = contentProjectionKey(runtime.content);
  return storedKey !== runtimeKey;
}

export function reconcileOrdinarySavedMessagesInExport(
  exported: ExportedMessageRepository,
  storedMessages: readonly MessageRecord[],
  options: { editingMessageId: string | null },
): { messages: ExportedMessageRepository["messages"]; changed: boolean } {
  const storedById = new Map(storedMessages.map((message) => [message.id, message]));
  let changed = false;
  const messages = exported.messages.map((item) => {
    const stored = storedById.get(item.message.id);
    if (!stored) {
      return item;
    }
    if (
      !shouldReconcileOrdinaryStoredAssistant({
        stored,
        runtime: item.message,
        editingMessageId: options.editingMessageId,
      })
    ) {
      return item;
    }

    const imported = recoveredContentToImport(item.message.content, stored.content);
    if (contentProjectionKey(imported) === contentProjectionKey(item.message.content)) {
      return item;
    }

    changed = true;
    const custom = runtimeCustom(item.message);
    const status = restoredAssistantStatus({ custom });
    return {
      ...item,
      message: {
        ...item.message,
        content: imported as ThreadMessage["content"],
        status,
      } as ThreadMessage,
    };
  });

  return { messages, changed };
}

export function appendExternalStoredTurns(
  exported: ExportedMessageRepository,
  storedMessages: readonly MessageRecord[],
  toThreadMessage: (record: MessageRecord) => ThreadMessage,
  idsAtRead?: ReadonlySet<string>,
): ExportedMessageRepository | null {
  if (exported.messages.some((item) => item.message.status?.type === "running")) {
    return null;
  }
  const known = new Set(exported.messages.map((item) => item.message.id));
  // A local delete raced the read, so the snapshot still holds the removed rows.
  if (idsAtRead && [...idsAtRead].some((id) => !known.has(id))) return null;
  // In-flight runs are added like a reload adds them; generation recovery then streams them.
  const pending = storedMessages
    .filter((message) => !known.has(message.id))
    .sort((a, b) => a.createdAt - b.createdAt);
  const appended: ExportedMessageRepository["messages"][number][] = [];
  let progressed = true;
  while (progressed) {
    progressed = false;
    for (const message of pending) {
      if (known.has(message.id)) continue;
      if (!message.parentId || !known.has(message.parentId)) continue;
      known.add(message.id);
      appended.push({ parentId: message.parentId, message: toThreadMessage(message) });
      progressed = true;
    }
  }
  if (appended.length === 0) return null;

  // Same leaf policy as the history loader; a head on another branch stays put.
  const leaf = resolveSavedBranchHead([...storedMessages], exported.headId);
  const headId = leaf && known.has(leaf) ? leaf : (exported.headId ?? null);
  return { ...exported, headId, messages: [...exported.messages, ...appended] };
}

type ThreadImportExport = {
  export: () => ExportedMessageRepository;
  import: (data: ExportedMessageRepository) => void;
  getState?: () => { messages: readonly { composer?: { isEditing?: boolean } }[] };
};

type ReconcileView = {
  threadListItem: () => { getState: () => { remoteId?: string | null } };
  thread: () => ThreadImportExport;
};

export function reconcileOrdinarySavedMessagesInView(
  view: ReconcileView,
  threadId: string,
  storedMessages: readonly MessageRecord[],
  options: {
    editingMessageId: string | null;
    toThreadMessage: (record: MessageRecord) => ThreadMessage;
    idsAtRead?: ReadonlySet<string>;
  },
): boolean {
  if (view.threadListItem().getState().remoteId !== threadId) {
    return false;
  }
  try {
    const thread = view.thread();
    // An import resets an open prompt-edit composer and loses its draft.
    if (thread.getState?.().messages.some((message) => message.composer?.isEditing)) {
      return false;
    }
    const exported = thread.export();
    const { messages, changed } = reconcileOrdinarySavedMessagesInExport(
      exported,
      storedMessages,
      options,
    );
    const patched = { ...exported, messages };
    const extended =
      options.editingMessageId === null
        ? appendExternalStoredTurns(
            patched,
            storedMessages,
            options.toThreadMessage,
            options.idsAtRead,
          )
        : null;
    if (!changed && !extended) {
      return false;
    }
    thread.import(extended ?? patched);
    return true;
  } catch {
    return false;
  }
}

/** Reconciles when this view's thread becomes the active one; other mounted views stay idle. */
export function subscribeThreadSwitchRecovery(
  store: {
    subscribe: (
      listener: (
        state: { activeThreadId: string | null },
        prev: { activeThreadId: string | null },
      ) => void,
    ) => () => void;
  },
  ownsThread: (threadId: string) => boolean,
  recover: () => void,
): () => void {
  let timer: ReturnType<typeof setTimeout> | undefined;
  const unsubscribe = store.subscribe((state, prev) => {
    const next = state.activeThreadId;
    if (!next || next === prev.activeThreadId || !ownsThread(next)) return;
    clearTimeout(timer);
    timer = setTimeout(recover, 0);
  });
  return () => {
    clearTimeout(timer);
    unsubscribe();
  };
}
