// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Tooltip, TooltipContent } from "@/components/ui/tooltip";
import { copyToClipboard } from "@/lib/copy-to-clipboard";
import { toast } from "@/lib/toast";
import { cn } from "@/lib/utils";
import { useAui, useAuiState } from "@assistant-ui/react";
import {
  Bookmark02Icon,
  Copy01Icon,
  Delete02Icon,
  MoreHorizontalIcon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { Tooltip as TooltipPrimitive } from "radix-ui";
import { useEffect, useId, useState } from "react";
import { create } from "zustand";
import { persistTemporaryThread } from "../runtime-provider";
import { useChatRuntimeStore } from "../stores/chat-runtime-store";
import { usePromptQueueUI } from "../stores/prompt-queue-ui-store";
import { isThreadIncognito } from "../utils/chat-history-storage";
import {
  buildConversationMarkdown,
  contentBlocksToMarkdownBlocks,
  renderConversationBlocks,
} from "../utils/conversation-markdown";
import { CHAT_MENU, CHAT_MENU_TRIGGER } from "./chat-header-menu";

/** Skip the confirmation once the user said so. Per browser. */
export const SKIP_SAVE_TEMPORARY_CONFIRM_KEY = "unsloth_chat_skip_save_temporary_confirm";

function skipConfirm(): boolean {
  try {
    return localStorage.getItem(SKIP_SAVE_TEMPORARY_CONFIRM_KEY) === "1";
  } catch {
    return false;
  }
}

function rememberSkipConfirm(): void {
  try {
    localStorage.setItem(SKIP_SAVE_TEMPORARY_CONFIRM_KEY, "1");
  } catch {
    // Private mode: the popup just keeps showing.
  }
}

type SaveTarget = {
  hasMessages: boolean;
  running: boolean;
  queued: boolean;
  save: () => Promise<void>;
  /** The messages on screen as markdown: nothing of a temporary chat is stored to read back. */
  markdown: () => string;
};

/** The open temporary chat, published from inside its runtime for the header, which sits outside. */
const useSaveTarget = create<{ target: SaveTarget | null }>(() => ({ target: null }));

/** Mount inside the single-chat runtime. */
export function TemporaryChatSaveBridge() {
  const aui = useAui();
  const incognito = useChatRuntimeStore((s) => s.incognito);
  const remoteId = useAuiState(({ threadListItem }) => threadListItem.remoteId);
  const hasMessages = useAuiState(({ thread }) => thread.messages.length > 0);
  const running = useAuiState(({ thread }) => thread.isRunning);
  // A queue keeps its temporary tag, so leaving a saved chat would still discard it.
  const queued = usePromptQueueUI((s) =>
    Object.values(s.byThreadId).some((entry) => entry.temporary),
  );

  useEffect(() => {
    if (!incognito) return;
    useSaveTarget.setState({
      target: {
        hasMessages: hasMessages && Boolean(remoteId),
        running,
        queued,
        save: async () => {
          const threadId = aui.threadListItem().getState().remoteId;
          if (!threadId || !isThreadIncognito(threadId)) return;
          await persistTemporaryThread({
            threadId,
            modelType: "base",
            messages: aui.thread().export().messages,
          });
          useChatRuntimeStore.getState().setIncognito(false);
          aui.threadListItem().generateTitle();
        },
        markdown: () =>
          buildConversationMarkdown(
            aui
              .thread()
              .getState()
              .messages.map((message) => ({
                role: message.role,
                content: renderConversationBlocks(
                  contentBlocksToMarkdownBlocks(message.content),
                ),
              })),
          ),
      },
    });
    return () => useSaveTarget.setState({ target: null });
  }, [aui, incognito, remoteId, hasMessages, running, queued]);

  return null;
}

async function copyMarkdown(markdown: string): Promise<void> {
  if (!markdown) {
    toast.info("No exportable content.");
    return;
  }
  if (await copyToClipboard(markdown)) toast.success("Chat copied as Markdown");
  else toast.error("Could not copy this chat.");
}

/** The header's "…" menu in a temporary chat with messages: only what works without saving it. */
export function SaveTemporaryChatMenu({
  className,
  onDiscard,
}: {
  className?: string;
  onDiscard: () => void;
}) {
  const target = useSaveTarget((s) => s.target);
  const [open, setOpen] = useState(false);
  const [dontShowAgain, setDontShowAgain] = useState(false);
  const [saving, setSaving] = useState(false);
  const checkboxId = useId();

  if (!target) return null;
  // Saving waits for a finished reply and for queued prompts, whose temporary tag would discard them.
  const canSave =
    target.hasMessages && !target.running && !target.queued && !saving;
  const label = "Save chat to history";

  const save = async () => {
    setSaving(true);
    try {
      await target.save();
      if (dontShowAgain) rememberSkipConfirm();
      setOpen(false);
      toast.success("Chat saved to history");
    } catch (error) {
      toast.error("Could not save chat", {
        description: error instanceof Error ? error.message : String(error),
      });
    } finally {
      setSaving(false);
    }
  };

  return (
    <>
      {target.hasMessages ? (
        <DropdownMenu>
          <Tooltip>
            <TooltipPrimitive.Trigger asChild={true}>
              <DropdownMenuTrigger asChild={true}>
                <button
                  type="button"
                  className={cn(CHAT_MENU_TRIGGER, className)}
                  aria-label="Chat options"
                >
                  <HugeiconsIcon
                    icon={MoreHorizontalIcon}
                    strokeWidth={1.75}
                    className="size-icon"
                  />
                </button>
              </DropdownMenuTrigger>
            </TooltipPrimitive.Trigger>
            <TooltipContent
              side="bottom"
              sideOffset={6}
              className="tooltip-compact"
            >
              Chat options
            </TooltipContent>
          </Tooltip>
          <DropdownMenuContent
            align="end"
            sideOffset={6}
            className={cn(CHAT_MENU, "w-56")}
          >
            {canSave ? (
              <DropdownMenuItem
                onSelect={() => {
                  if (skipConfirm()) void save();
                  else setOpen(true);
                }}
              >
                <HugeiconsIcon
                  icon={Bookmark02Icon}
                  strokeWidth={1.75}
                  className="size-icon"
                />
                {label}
              </DropdownMenuItem>
            ) : null}
            <DropdownMenuItem
              onSelect={() => void copyMarkdown(target.markdown())}
            >
              <HugeiconsIcon
                icon={Copy01Icon}
                strokeWidth={1.75}
                className="size-icon"
              />
              Copy as Markdown
            </DropdownMenuItem>
            <DropdownMenuSeparator className="mx-3" />
            <DropdownMenuItem variant="destructive" onSelect={onDiscard}>
              <HugeiconsIcon
                icon={Delete02Icon}
                strokeWidth={1.75}
                className="size-icon"
              />
              Delete
            </DropdownMenuItem>
          </DropdownMenuContent>
        </DropdownMenu>
      ) : null}
      <Dialog
        open={open}
        onOpenChange={(next) => {
          if (!saving) setOpen(next);
          if (!next) setDontShowAgain(false);
        }}
      >
        <DialogContent
          className="dialog-soft-surface sm:max-w-md"
          showCloseButton={!saving}
        >
          <DialogHeader className="gap-3">
            <DialogTitle className="flex items-center gap-2">
              <HugeiconsIcon
                icon={Bookmark02Icon}
                strokeWidth={2}
                className="text-foreground/80 size-4.5 shrink-0"
              />
              Save this chat to history?
            </DialogTitle>
            <DialogDescription>
              The whole conversation is saved to your history, and new messages
              are saved too.
            </DialogDescription>
          </DialogHeader>
          <DialogFooter className="flex-wrap items-center gap-2 sm:justify-between">
            <label
              htmlFor={checkboxId}
              className="text-muted-foreground flex cursor-pointer items-center gap-2 text-sm"
            >
              <Checkbox
                id={checkboxId}
                checked={dontShowAgain}
                onCheckedChange={(checked) =>
                  setDontShowAgain(checked === true)
                }
                disabled={saving}
              />
              Don&apos;t show again
            </label>
            <div className="flex gap-2">
              <Button
                type="button"
                variant="ghost"
                onClick={() => setOpen(false)}
                disabled={saving}
              >
                Keep temporary
              </Button>
              <Button
                type="button"
                onClick={() => void save()}
                disabled={saving}
              >
                {saving ? "Saving…" : "Save chat"}
              </Button>
            </div>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </>
  );
}
