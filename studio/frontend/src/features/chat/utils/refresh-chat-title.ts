// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// eslint-disable-next-line no-restricted-imports
import { disposableTimeoutSignal } from "@/features/hub/lib/abort-signals";
import { parseExternalModelId } from "../external-providers";
import { updateChatThread } from "../api/chat-api";
import type { SidebarItem } from "../hooks/use-chat-sidebar-items";
import { useChatRuntimeStore } from "../stores/chat-runtime-store";
import {
  getStoredChatThread,
  listStoredChatMessages,
  listStoredChatThreads,
} from "./chat-history-storage";
import { generateChatTitle } from "./generate-chat-title";
import { liveThreadBranch } from "./live-thread-head";
import { orderByParentChain } from "./message-order";
import { attachmentsSample } from "./pasted-text";

const pending = new Map<string, Promise<void>>();

export function refreshChatTitle(item: SidebarItem): Promise<void> {
  const key = `${item.type}:${item.id}`;
  const existing = pending.get(key);
  if (existing) return existing;
  const request = refresh(item).finally(() => pending.delete(key));
  pending.set(key, request);
  return request;
}

function budgetTranscript(conversation: string, contextLength: number): string {
  // reserve prompt/output space and budget utf-8 bytes conservatively for multilingual text.
  const budget = Math.max(128, Math.min(12_000, contextLength - 512));
  const bytes = new TextEncoder().encode(conversation);
  if (bytes.length <= budget) return conversation;
  const marker = "\n[Earlier conversation abbreviated]\n";
  const available = budget - new TextEncoder().encode(marker).length;
  let headEnd = Math.floor(available / 4);
  let tailStart = bytes.length - (available - headEnd);
  while ((bytes[headEnd] & 0xc0) === 0x80) headEnd -= 1;
  while ((bytes[tailStart] & 0xc0) === 0x80) tailStart += 1;
  const decoder = new TextDecoder();
  return (
    decoder.decode(bytes.subarray(0, headEnd)) +
    marker +
    decoder.decode(bytes.subarray(tailStart))
  );
}

async function refresh(item: SidebarItem): Promise<void> {
  const runtime = useChatRuntimeStore.getState();
  const model = runtime.params.checkpoint;
  const contextLength = parseExternalModelId(model)
    ? 4096
    : (runtime.loadedCustomContextLength ??
      runtime.loadedContextLength ??
      (runtime.params.maxSeqLength || 4096));
  if (!model) throw new Error("Select a model to refresh the chat title.");
  const threads =
    item.type === "compare"
      ? await listStoredChatThreads({ pairId: item.id, includeArchived: true })
      : [await getStoredChatThread(item.id)].filter(
          (thread) => thread !== undefined,
        );
  if (threads.length === 0) throw new Error("This chat no longer exists.");
  const conversations = await Promise.all(
    threads.map(async (thread) => {
      const liveBranch = liveThreadBranch(thread.id);
      const raw = await listStoredChatMessages(thread.id);
      const storedIds = new Set(raw.map((message) => message.id));
      const messages = raw.some((message) => message.parentId != null)
        ? orderByParentChain(raw, {
            includeSiblings: false,
            headId: liveBranch
              ? ([...liveBranch].reverse().find((id) => storedIds.has(id)) ??
                null)
              : undefined,
          })
        : raw;
      return messages
        .flatMap((message) => {
          if (message.role !== "user" && message.role !== "assistant")
            return [];
          const text = message.content
            .filter((part) => part.type === "text")
            .map((part) => part.text)
            .join("\n");
          const sample =
            message.role === "user"
              ? attachmentsSample(message.attachments)
              : "";
          const content = [text, sample].filter(Boolean).join("\n\n").trim();
          return content ? [`${message.role}: ${content}`] : [];
        })
        .join("\n\n");
    }),
  );
  const conversation = conversations
    .filter(Boolean)
    .join("\n\nAnother comparison pane:\n\n");
  if (!conversation) throw new Error("This chat has no text to summarize.");
  const timeout = disposableTimeoutSignal(60_000);
  let title: string | null;
  try {
    title = await generateChatTitle(
      budgetTranscript(conversation, contextLength),
      model,
      timeout.signal,
    );
  } finally {
    timeout.dispose();
  }
  if (!title) throw new Error("The model did not return a title. Try again.");
  await Promise.all(
    threads.map((thread) =>
      updateChatThread(thread.id, { title }, { expectedTitle: thread.title }),
    ),
  );
}
