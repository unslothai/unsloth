// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

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

async function refresh(item: SidebarItem): Promise<void> {
  const model = useChatRuntimeStore.getState().params.checkpoint;
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
  const title = await generateChatTitle(
    conversation,
    model,
    AbortSignal.timeout(60_000),
  );
  if (!title) throw new Error("The model did not return a title. Try again.");
  await Promise.all(
    threads.map((thread) =>
      updateChatThread(thread.id, { title }, { expectedTitle: thread.title }),
    ),
  );
}
