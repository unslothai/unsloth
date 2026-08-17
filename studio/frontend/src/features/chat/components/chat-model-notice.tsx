// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useEffect, useState } from "react";

import { Button } from "@/components/ui/button";
import { compareModelDisplayName } from "../lib/external-model-label";
import { getStoredChatThread } from "../utils/chat-history-storage";

/**
 * The model a saved chat was started on, read once per chat.
 *
 * Every thread row has carried `modelId` since long before this notice, so an
 * existing chat knows its model without any migration. A chat with no row yet
 * (a New Chat that has not been sent) answers null and shows nothing.
 */
export function useChatCreatedModel(
  threadId: string | undefined,
): string | null {
  const [modelId, setModelId] = useState<string | null>(null);
  useEffect(() => {
    if (!threadId) {
      setModelId(null);
      return;
    }
    let cancelled = false;
    // Cleared first: without this the outgoing chat's model is on screen until
    // the next read lands, which is the wrong model against the new chat.
    setModelId(null);
    void getStoredChatThread(threadId)
      .then((thread) => {
        if (!cancelled) setModelId(thread?.modelId || null);
      })
      .catch(() => {
        // A failed read is not worth a message: the notice is an offer, not a
        // gate, and the chat runs on whatever is loaded either way.
      });
    return () => {
      cancelled = true;
    };
  }, [threadId]);
  return modelId;
}

type ChatModelNoticeProps = {
  /** The saved chat on screen, or undefined for an unsent New Chat. */
  threadId: string | undefined;
  /** The model the composer will actually send to. */
  checkpoint: string;
  /** Every model that can be selected right now, by id. */
  selectableModelIds: ReadonlySet<string>;
  onSwitch: (modelId: string) => void;
};

/**
 * Offers to put a chat back on the model it was started on.
 *
 * Deliberately an offer and not an automatic switch: for a local model that
 * would evict whatever is resident and spend a multi-gigabyte load on opening a
 * chat, which is not what clicking a row in the sidebar should cost.
 */
export function ChatModelNotice({
  threadId,
  checkpoint,
  selectableModelIds,
  onSwitch,
}: ChatModelNoticeProps) {
  const createdModelId = useChatCreatedModel(threadId);
  if (!createdModelId || createdModelId === checkpoint) return null;
  // A model that has since been deleted, or a connection that is gone: the
  // switch could not be honoured, and saying so on every open is just noise.
  if (!selectableModelIds.has(createdModelId)) return null;
  const label = compareModelDisplayName(createdModelId);
  return (
    <div className="flex items-center gap-2 border-b border-border/60 bg-muted/40 px-4 py-1.5 text-ui-12 text-muted-foreground">
      <span className="min-w-0 truncate">
        This chat was started on <span className="font-medium">{label}</span>.
      </span>
      <Button
        variant="ghost"
        size="sm"
        className="ml-auto h-6 shrink-0 px-2 text-ui-12"
        onClick={() => onSwitch(createdModelId)}
      >
        Switch back
      </Button>
    </div>
  );
}
