// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { ArrowTurnBackwardIcon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { Popover as PopoverPrimitive } from "radix-ui";
import { useEffect, useMemo, useRef, useState } from "react";

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
import { useChatRuntimeStore } from "../stores/chat-runtime-store";
import { getStoredChatThread } from "../utils/chat-history-storage";
import {
  type ChatModelAutoSwitch,
  type ChatModelSwitchTarget,
  type LoadedChatModelSlot,
  chatModelIsResident,
  chatModelLoadedTarget,
  chatModelSelectableId,
  createChatModelHistoryReader,
  stepChatModelAutoSwitch,
} from "./chat-model-notice-switch";

/** reads the saved chat's last used model and waits for queued initialization. */
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
    // Bumped by every newer answer, so a slower re-read never lands over it.
    let readGeneration = 0;
    const onHistoryUpdated = (event: Event): void => {
      const updatedThread = (event as CustomEvent<ChatHistoryUpdatedDetail>)
        .detail?.thread;
      if (updatedThread) {
        if (updatedThread.id === threadId) readGeneration += 1;
        reader.applyUpdate(updatedThread);
        return;
      }
      // Another tab's change arrives as a plain Event; this tab's thread-less ones (chunks) are CustomEvents.
      if (event instanceof CustomEvent) return;
      reader.invalidate();
      const generation = ++readGeneration;
      void getStoredChatThread(threadId)
        .then((thread) => {
          if (thread && generation === readGeneration) reader.applyUpdate(thread);
        })
        .catch(() => undefined);
    };
    window.addEventListener(CHAT_HISTORY_UPDATED_EVENT, onHistoryUpdated);
    void getStoredChatThread(threadId)
      .then((thread) => reader.applyInitial(thread))
      .catch(() => undefined);
    return () => {
      reader.dispose();
      window.removeEventListener(CHAT_HISTORY_UPDATED_EVENT, onHistoryUpdated);
      // Unsubscribed from here on, so coming back must wait for a fresh read.
      setRead(null);
    };
  }, [threadId]);
  return read && read.threadId === threadId ? read.model : null;
}

type ChatModelNoticeProps = {
  /** Keep the history reader mounted while the chat route is hidden. */
  active: boolean;
  /** The saved chat on screen, or undefined for an unsent New Chat. */
  threadId: string | undefined;
  checkpoint: string;
  activeGgufVariant: string | null;
  selectableModelIds: ReadonlySet<string>;
  loadedModels: readonly LoadedChatModelSlot[];
  onSwitch: (target: ChatModelSwitchTarget) => void;
  onAutoSwitch: (
    target: ChatModelSwitchTarget,
    stillWanted: () => boolean,
  ) => void;
};

/** Adopts a chat's last used model when it is still loaded, else only offers it: a local load evicts. */
export function ChatModelNotice({
  active,
  threadId,
  checkpoint,
  activeGgufVariant,
  selectableModelIds,
  loadedModels,
  onSwitch,
  onAutoSwitch,
}: ChatModelNoticeProps) {
  const createdModel = useChatCreatedModel(threadId);
  const resident =
    createdModel != null &&
    chatModelIsResident(createdModel, checkpoint, activeGgufVariant);
  const loadedTarget = useMemo(
    () =>
      createdModel && !resident
        ? chatModelLoadedTarget(createdModel, loadedModels)
        : null,
    [createdModel, resident, loadedModels],
  );
  // A send that moves the chat to another model before the switch lands drops the switch.
  const createdKey = createdModel
    ? `${createdModel.modelId}\0${createdModel.ggufVariant ?? ""}`
    : null;
  const createdKeyRef = useRef(createdKey);
  useEffect(() => {
    createdKeyRef.current = createdKey;
  }, [createdKey]);
  const autoSwitchRef = useRef<ChatModelAutoSwitch>({ settled: false });
  // Unmounting (Compare, a Project) ends the open too, so a pending switch must not land after it.
  useEffect(
    () => () => {
      autoSwitchRef.current = { settled: false };
    },
    [],
  );
  useEffect(() => {
    const { state, switchTo } = stepChatModelAutoSwitch(autoSwitchRef.current, {
      threadId,
      active,
      resident,
      loadedTarget,
      modelLoading: useChatRuntimeStore.getState().modelLoading,
    });
    autoSwitchRef.current = state;
    if (switchTo) {
      const key = createdKeyRef.current;
      onAutoSwitch(
        switchTo,
        () => autoSwitchRef.current === state && createdKeyRef.current === key,
      );
    }
  }, [active, threadId, resident, loadedTarget, onAutoSwitch]);
  // Drop the portaled UI off-route, retaining the successful read and history subscription.
  if (!active) return null;
  if (!createdModel || resident) return null;
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
              aria-label={`Last used model: ${label}`}
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
          Last used model: {label}
        </TooltipContent>
      </Tooltip>
      <PopoverContent
        align="start"
        sideOffset={6}
        collisionPadding={12}
        aria-label="Last used model"
        className="menu-soft-surface w-64 gap-1 rounded-lg p-1"
      >
        <div className="space-y-1 px-3 py-2">
          <p className="text-ui-12 text-muted-foreground">Last used model</p>
          <p className="break-words text-ui-13 font-medium">{label}</p>
        </div>
        <PopoverPrimitive.Close asChild={true}>
          <Button
            type="button"
            variant="ghost"
            size="sm"
            className="h-8 justify-start gap-2.5 rounded-md px-3 text-ui-13 font-normal"
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
