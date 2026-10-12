// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { ArrowTurnBackwardIcon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { Popover as PopoverPrimitive } from "radix-ui";
import { useEffect, useState } from "react";

import { Button } from "@/components/ui/button";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
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

/** reads the saved chat's starting model and waits for queued initialization. */
export function useChatCreatedModel(
  threadId: string | undefined,
): ChatModelSwitchTarget | null {
  // Keyed by the chat it was read for, not a bare value. Clearing it in the effect is a frame late:
  // the effect is passive, so the first render for the incoming chat has already committed with
  // the outgoing chat's model, painting the wrong notice. Answering only for a matching key makes
  // that render impossible rather than brief.
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
  /** Keep the history reader mounted while the chat route is hidden. */
  active: boolean;
  /** The saved chat on screen, or undefined for an unsent New Chat. */
  threadId: string | undefined;
  /** The model the composer will actually send to. */
  checkpoint: string;
  activeGgufVariant: string | null;
  /** Every model that can be selected right now, by id. */
  selectableModelIds: ReadonlySet<string>;
  onSwitch: (target: ChatModelSwitchTarget) => void;
};

/** Offers to put a chat back on the model it was started on. Deliberately an offer and not an
 *  automatic switch: for a local model that would evict whatever is resident and spend a
 *  multi-gigabyte load on opening a chat. */
export function ChatModelNotice({
  active,
  threadId,
  checkpoint,
  activeGgufVariant,
  selectableModelIds,
  onSwitch,
}: ChatModelNoticeProps) {
  const createdModel = useChatCreatedModel(threadId);
  // Drop the portaled UI off-route, retaining the successful read and history subscription.
  if (!active) return null;
  if (!createdModel) return null;
  if (chatModelIsResident(createdModel, checkpoint, activeGgufVariant)) {
    return null;
  }
  // A model that has since been deleted, or a connection that is gone: the switch could not be
  // honoured, and saying so on every open is just noise. A gone snapshot loads its repo's live row.
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
    <Popover key={threadId}>
      <Tooltip>
        <TooltipTrigger asChild={true} disableClickToggle={true}>
          <PopoverTrigger asChild={true}>
            <button
              type="button"
              data-chat-model-notice=""
              aria-label={`Original model: ${label}`}
              className="flex h-[var(--studio-chat-control-height,34px)] min-w-[calc(30px*var(--ui-space-scale,1))] max-w-56 shrink cursor-pointer items-center rounded-[10px] px-2 text-ui-13 text-nav-fg transition-colors hover:bg-nav-surface-hover hover:text-black focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring aria-expanded:bg-nav-surface-hover aria-expanded:text-black dark:hover:text-white dark:aria-expanded:text-white max-sm:min-w-[calc(22px*var(--ui-space-scale,1))] max-sm:px-1"
            >
              <HugeiconsIcon
                icon={ArrowTurnBackwardIcon}
                strokeWidth={1.75}
                className="size-3.5 shrink-0"
                aria-hidden={true}
              />
              {/* Keep the label's spacing inside its clipped box so an icon-only trigger fits. */}
              <span className="min-w-0 truncate before:inline-block before:w-1.5 before:content-['']">
                {label}
              </span>
            </button>
          </PopoverTrigger>
        </TooltipTrigger>
        <TooltipContent side="bottom" sideOffset={6}>
          Original model: {label}
        </TooltipContent>
      </Tooltip>
      <PopoverContent
        align="start"
        sideOffset={6}
        collisionPadding={12}
        aria-label="Original model"
        className="menu-soft-surface w-64 gap-1 rounded-[14px] p-1"
      >
        <div className="space-y-1 px-3 py-2">
          <p className="text-ui-12 text-muted-foreground">Original model</p>
          <p className="break-words text-ui-13 font-medium">{label}</p>
        </div>
        <PopoverPrimitive.Close asChild={true}>
          <Button
            type="button"
            variant="ghost"
            size="sm"
            className="h-8 justify-start gap-2.5 rounded-[10px] px-3 text-ui-13 font-normal"
            onClick={() => onSwitch(switchTarget)}
          >
            <HugeiconsIcon
              icon={ArrowTurnBackwardIcon}
              strokeWidth={1.75}
              className="size-icon"
              aria-hidden={true}
            />
            Switch back
          </Button>
        </PopoverPrimitive.Close>
      </PopoverContent>
    </Popover>
  );
}
