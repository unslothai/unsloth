// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { ExportedMessageRepository, ThreadMessage } from "@assistant-ui/react";

import type { MessageRecord } from "../types";
import { restoredAssistantStatus } from "./continuation";
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

type ThreadImportExport = {
  export: () => ExportedMessageRepository;
  import: (data: ExportedMessageRepository) => void;
};

type ReconcileView = {
  threadListItem: () => { getState: () => { remoteId?: string | null } };
  thread: () => ThreadImportExport;
};

export function reconcileOrdinarySavedMessagesInView(
  view: ReconcileView,
  threadId: string,
  storedMessages: readonly MessageRecord[],
  options: { editingMessageId: string | null },
): boolean {
  if (view.threadListItem().getState().remoteId !== threadId) {
    return false;
  }
  try {
    const thread = view.thread();
    const exported = thread.export();
    const { messages, changed } = reconcileOrdinarySavedMessagesInExport(
      exported,
      storedMessages,
      options,
    );
    if (!changed) {
      return false;
    }
    thread.import({ ...exported, messages });
    return true;
  } catch {
    return false;
  }
}
