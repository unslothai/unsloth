// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useEffect, useState } from "react";

import { Button } from "@/components/ui/button";
import { modelDisplayName } from "../../model-picker/model-config/model-identity";
import {
  CHAT_HISTORY_UPDATED_EVENT,
  type ChatHistoryUpdatedDetail,
} from "../api/chat-api";
import { externalModelLabel } from "../lib/external-model-label";
import { getStoredChatThread } from "../utils/chat-history-storage";
import {
  type ChatModelSwitchTarget,
  chatModelIsResident,
  chatModelSelectableId,
  createChatModelHistoryReader,
} from "./chat-model-notice-switch";

export function useChatCreatedModel(
  threadId: string | undefined,
): ChatModelSwitchTarget | null {
  // Keyed by chat: clearing in a passive effect is a frame late and paints the old chat's model.
  const [read, setRead] = useState<{
    threadId: string;
    model: ChatModelSwitchTarget | null;
  } | null>(null);
  useEffect(() => {
    if (!threadId) return;
    const reader = createChatModelHistoryReader(threadId, (model) =>
      setRead({ threadId, model }),
    );
    const onHistoryUpdated = (event: Event): void => {
      const updatedThread = (event as CustomEvent<ChatHistoryUpdatedDetail>)
        .detail?.thread;
      if (updatedThread) reader.applyUpdate(updatedThread);
    };
    window.addEventListener(CHAT_HISTORY_UPDATED_EVENT, onHistoryUpdated);
    void getStoredChatThread(threadId)
      .then((thread) => reader.applyInitial(thread))
      .catch(() => undefined);
    return () => {
      reader.dispose();
      window.removeEventListener(CHAT_HISTORY_UPDATED_EVENT, onHistoryUpdated);
    };
  }, [threadId]);
  return read && read.threadId === threadId ? read.model : null;
}

type ChatModelNoticeProps = {
  threadId: string | undefined;
  checkpoint: string;
  activeGgufVariant: string | null;
  selectableModelIds: ReadonlySet<string>;
  onSwitch: (target: ChatModelSwitchTarget) => void;
};

/** An offer, not an automatic switch: switching a local model evicts the resident and reloads. */
export function ChatModelNotice({
  threadId,
  checkpoint,
  activeGgufVariant,
  selectableModelIds,
  onSwitch,
}: ChatModelNoticeProps) {
  const createdModel = useChatCreatedModel(threadId);
  if (!createdModel) return null;
  if (chatModelIsResident(createdModel, checkpoint, activeGgufVariant)) {
    return null;
  }
  // Unselectable models are not offered; a gone snapshot loads its repo's live row.
  const selectableId = chatModelSelectableId(
    createdModel.modelId,
    selectableModelIds,
  );
  if (!selectableId) {
    return null;
  }
  const switchTarget = { ...createdModel, modelId: selectableId };
  const label =
    externalModelLabel(createdModel.modelId) ??
    modelDisplayName(createdModel.modelId);
  return (
    // Positioned: the header is absolute z-40 and opaque, so in flow this would sit under it.
    <div
      data-chat-model-notice=""
      data-side-panel-inset=""
      className="absolute left-[var(--studio-side-panel-left,0px)] right-[calc(var(--thread-scrollbar-gutter,10px)+var(--studio-side-panel-width,0px))] top-[calc(var(--studio-content-top-inset,0px)+var(--studio-chat-header-height,48px))] z-30 flex h-[var(--studio-chat-notice-height,2.25rem)] items-center gap-2 border-b border-border/60 bg-muted px-4 text-ui-12 text-muted-foreground"
    >
      <span className="min-w-0 truncate">
        This chat was started on <span className="font-medium">{label}</span>.
      </span>
      <Button
        variant="ghost"
        size="sm"
        className="ml-auto h-6 shrink-0 px-2 text-ui-12"
        onClick={() => onSwitch(switchTarget)}
      >
        Switch back
      </Button>
    </div>
  );
}
