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
  // Keyed by the chat it was read for, not a bare value. Clearing it in the
  // effect is a frame late: the effect is passive, so the first render for the
  // incoming chat has already committed with the outgoing chat's model, which
  // paints the wrong notice and moves the viewport padding under it. Answering
  // only for a matching key makes that render impossible rather than brief.
  const [read, setRead] = useState<{
    threadId: string;
    modelId: string | null;
  } | null>(null);
  useEffect(() => {
    if (!threadId) return;
    let cancelled = false;
    void getStoredChatThread(threadId)
      .then((thread) => {
        if (!cancelled) setRead({ threadId, modelId: thread?.modelId || null });
      })
      .catch(() => {
        // A failed read is not worth a message: the notice is an offer, not a
        // gate, and the chat runs on whatever is loaded either way.
      });
    return () => {
      cancelled = true;
    };
  }, [threadId]);
  return read && read.threadId === threadId ? read.modelId : null;
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
    // Positioned, not in flow. The chat header is `absolute ... z-40` with an
    // opaque `bg-background`, so an in-flow sibling starts at y=0 UNDER it and
    // the whole bar is invisible bar the 10px the header's `right-[10px]`
    // leaves uncovered. Offset by the same header height the drop overlay and
    // the header fade use, above the fade (z-20) and below the header (z-40).
    <div
      data-chat-model-notice=""
      className="absolute left-0 right-[10px] top-[calc(var(--studio-content-top-inset,0px)+var(--studio-chat-header-height,48px))] z-30 flex h-[var(--studio-chat-notice-height,2.25rem)] items-center gap-2 border-b border-border/60 bg-muted px-4 text-ui-12 text-muted-foreground"
    >
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
