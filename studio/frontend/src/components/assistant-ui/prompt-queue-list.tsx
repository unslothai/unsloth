// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useEffect, useId, useRef, useState } from "react";
import {
  CornerDownRightIcon,
  GripVerticalIcon,
  ListEndIcon,
  MoreHorizontalIcon,
  PlayIcon,
} from "lucide-react";
import {
  Copy01Icon,
  Delete02Icon,
  Edit03Icon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { Button } from "@/components/ui/button";
import {
  DropdownMenuItem,
  DropdownMenuSeparator,
} from "@/components/ui/dropdown-menu";
import { NonModalDropdownMenu } from "@/components/ui/non-modal-dropdown-menu";
import { usePromptQueueReorder } from "./use-prompt-queue-reorder";
import { TooltipIconButton } from "@/components/assistant-ui/tooltip-icon-button";
import {
  type PromptQueueUIEntry,
  type PromptQueueUIItem,
  useChatPreferencesStore,
} from "@/features/chat";
import { copyToClipboard } from "@/lib/copy-to-clipboard";
import { cn } from "@/lib/utils";

type PromptQueueListProps = {
  entry: PromptQueueUIEntry;
  items: PromptQueueUIItem[];
  onEdit: (id: string, prompt: string) => boolean;
  onRemove: (id: string) => boolean;
  onMove: (id: string, targetId: string) => boolean;
  onSteer: (id: string) => boolean;
  onResume: () => void;
};

/** The queue engine owns dispatch and validates every mutation against live IDs. */
export function PromptQueueList({
  entry,
  items,
  onEdit,
  onRemove,
  onMove,
  onSteer,
  onResume,
}: PromptQueueListProps) {
  const followUpBehavior = useChatPreferencesStore((s) => s.followUpBehavior);
  const setFollowUpBehavior = useChatPreferencesStore((s) => s.setFollowUpBehavior);
  const [editingId, setEditingId] = useState<string | null>(null);
  const [draft, setDraft] = useState("");
  const [announcement, setAnnouncement] = useState("");
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const editFromMenuRef = useRef(false);
  const instructionsId = useId();
  const editingItem = items.find(
    (item) => item.id === editingId && item.canEdit,
  );
  const activeEditingId = editingItem?.id;
  const movableItems = items.filter((item) => item.canEdit && item.canRemove);
  const {
    listRef,
    draggingId,
    move,
    cancelDrag,
    onPointerDown,
    onPointerMove,
    onPointerUp,
  } = usePromptQueueReorder(
    items,
    Boolean(activeEditingId),
    onMove,
    setAnnouncement,
  );

  useEffect(() => {
    if (!activeEditingId) return;
    inputRef.current?.focus();
    inputRef.current?.select();
  }, [activeEditingId]);

  function startEditing(item: PromptQueueUIItem) {
    setDraft(item.prompt);
    setEditingId(item.id);
  }

  function finishEditing() {
    const id = editingId;
    setEditingId(null);
    setDraft("");
    // The editor replaces its row controls. Restore focus once they return.
    requestAnimationFrame(() => {
      const row = Array.from(
        listRef.current?.querySelectorAll<HTMLElement>(
          "[data-queue-item-id]",
        ) ?? [],
      ).find((element) => element.dataset.queueItemId === id);
      row?.querySelector<HTMLButtonElement>("[data-queue-menu]")?.focus();
    });
  }

  function saveEditing() {
    if (!editingItem || !draft.trim()) return;
    if (onEdit(editingItem.id, draft)) {
      setAnnouncement("Queued prompt updated.");
      finishEditing();
    } else {
      setAnnouncement(
        "This prompt can no longer be edited because the queue changed.",
      );
    }
  }

  return (
    <div
      ref={listRef}
      className="relative z-0 mx-3 mb-[-8px] max-h-[28dvh] overflow-y-auto rounded-t-[20px] border border-border/60 bg-background px-1.5 pt-1 pb-3 text-muted-foreground sm:mx-5 sm:px-2 dark:bg-card [&_button]:border-0 [&_button]:shadow-none [&_.aui-button-icon:focus-visible]:bg-accent"
      aria-label={`Prompt queue, ${entry.current} of ${entry.total}`}
    >
      <p id={instructionsId} className="sr-only">
        Drag the handle to reorder. With the handle focused, use Up or Down to
        move one position, or Home or End to move to the front or end.
      </p>
      <div role="status" aria-live="polite" className="sr-only">
        {announcement}
      </div>
      <div role="list" aria-label="Queued prompts">
        {items.map((item, index) => {
          const isEditing = editingItem?.id === item.id;
          const position = index + 1;
          const moveIndex = movableItems.findIndex(
            (candidate) => candidate.id === item.id,
          );
          const canMove =
            !activeEditingId && moveIndex >= 0 && movableItems.length > 1;
          const previous = movableItems[moveIndex - 1];
          const next = movableItems[moveIndex + 1];
          return (
            <div
              key={item.id}
              role="listitem"
              data-queue-item-id={item.id}
              data-queue-dragging={draggingId === item.id || undefined}
              aria-label={`Queued prompt ${position} of ${items.length}: ${item.prompt}`}
              className={cn(
                "group relative rounded-lg transition-colors",
                draggingId && "will-change-transform",
                draggingId === item.id && "z-10 opacity-50",
              )}
            >
              {isEditing ? (
                <div className="flex flex-wrap items-center justify-end gap-2 px-1 py-2">
                  <textarea
                    ref={inputRef}
                    value={draft}
                    rows={2}
                    onChange={(event) => setDraft(event.currentTarget.value)}
                    onKeyDown={(event) => {
                      if (
                        event.nativeEvent.isComposing ||
                        event.nativeEvent.keyCode === 229 ||
                        event.repeat
                      ) return;
                      if (
                        event.key === "Enter" &&
                        (event.metaKey || event.ctrlKey)
                      ) {
                        event.preventDefault();
                        event.stopPropagation();
                        saveEditing();
                      } else if (event.key === "Escape") {
                        event.preventDefault();
                        event.stopPropagation();
                        finishEditing();
                      }
                    }}
                    className="max-h-40 min-h-16 w-full resize-y rounded-lg border border-border bg-background px-3 py-2 text-sm text-foreground focus-visible:outline focus-visible:outline-ring"
                    aria-label={`Edit queued prompt ${position}`}
                  />
                  <Button
                    type="button"
                    variant="ghost"
                    size="sm"
                    className="focus-visible:bg-accent"
                    onClick={finishEditing}
                  >
                    Cancel
                  </Button>
                  <Button
                    type="button"
                    size="sm"
                    className="focus-visible:bg-primary/80"
                    disabled={!draft.trim()}
                    onClick={saveEditing}
                  >
                    Save
                  </Button>
                </div>
              ) : (
                <div className="flex min-h-11 items-center gap-1 sm:min-h-12">
                  <TooltipIconButton
                    type="button"
                    tooltip="Drag to reorder"
                    aria-label={`Reorder queued prompt ${position} of ${items.length}`}
                    aria-describedby={instructionsId}
                    className="h-8 w-4 shrink-0 touch-none cursor-grab text-muted-foreground/50 hover:text-muted-foreground active:cursor-grabbing pointer-coarse:h-11 pointer-coarse:w-8"
                    disabled={!canMove}
                    onPointerDown={(event) => onPointerDown(event, item.id)}
                    onPointerMove={onPointerMove}
                    onPointerUp={onPointerUp}
                    onPointerCancel={cancelDrag}
                    onLostPointerCapture={cancelDrag}
                    onKeyDown={(event) => {
                      if (event.key === "Escape" && cancelDrag()) {
                        event.preventDefault();
                        event.stopPropagation();
                        return;
                      }
                      const targets: Record<string, string | undefined> = {
                        ArrowUp: previous?.id,
                        ArrowDown: next?.id,
                        Home: movableItems[0]?.id,
                        End: movableItems.at(-1)?.id,
                      };
                      if (!(event.key in targets)) return;
                      event.preventDefault();
                      event.stopPropagation();
                      move(item.id, targets[event.key]);
                    }}
                  >
                    <GripVerticalIcon className="size-4" />
                  </TooltipIconButton>
                  <ListEndIcon
                    className="hidden size-4 shrink-0 text-muted-foreground/60 sm:block"
                    aria-hidden="true"
                  />
                  <span className="min-w-0 flex-1 truncate px-1.5 text-sm text-foreground/80">
                    {item.prompt}
                  </span>
                  {index === 0 && entry.paused && (
                    <span className="hidden shrink-0 px-1 text-xs text-muted-foreground sm:inline">
                      Paused
                    </span>
                  )}
                  <TooltipIconButton
                    type="button"
                    tooltip="Interrupt the response and send this prompt next"
                    aria-label={`Steer with queued prompt ${position}`}
                    disabled={!item.canEdit || !item.canRemove}
                    className="h-8 w-auto shrink-0 gap-1.5 px-2 font-normal text-muted-foreground hover:text-foreground pointer-coarse:h-11"
                    onClick={() => {
                      setAnnouncement(
                        onSteer(item.id)
                          ? "This prompt will steer the response next."
                          : "This prompt could not steer the response. Check the queue and try again.",
                      );
                    }}
                  >
                    <CornerDownRightIcon className="size-4" />
                    <span>Steer</span>
                  </TooltipIconButton>
                  <TooltipIconButton
                    type="button"
                    tooltip="Remove from queue"
                    aria-label={`Remove queued prompt ${position}`}
                    disabled={!item.canRemove}
                    className="size-8 shrink-0 text-muted-foreground hover:text-destructive pointer-coarse:size-11"
                    onClick={() => {
                      if (onRemove(item.id))
                        setAnnouncement("Prompt removed from queue.");
                    }}
                  >
                    <HugeiconsIcon icon={Delete02Icon} strokeWidth={1.8} />
                  </TooltipIconButton>
                  <NonModalDropdownMenu
                    trigger={(triggerRef) => (
                      <TooltipIconButton
                        ref={triggerRef}
                        tooltip="More options"
                        data-queue-menu
                        aria-label={`More options for queued prompt ${position}`}
                        className="size-8 shrink-0 text-muted-foreground hover:text-foreground data-[state=open]:bg-accent data-[state=open]:text-foreground pointer-coarse:size-11"
                      >
                        <MoreHorizontalIcon className="size-4" />
                      </TooltipIconButton>
                    )}
                    align="end"
                    side="bottom"
                    sideOffset={6}
                    className="w-56 rounded-2xl border border-border/60 p-1.5 shadow-lg"
                    // Opening the menu must not select an item on pointer release.
                    onPointerUpCapture={(event) => event.preventDefault()}
                    onCloseAutoFocus={(event) => {
                      if (!editFromMenuRef.current) return;
                      event.preventDefault();
                      editFromMenuRef.current = false;
                      inputRef.current?.focus();
                    }}
                  >
                    <DropdownMenuItem
                      disabled={!item.canEdit}
                      onSelect={() => {
                        editFromMenuRef.current = true;
                        startEditing(item);
                      }}
                    >
                      <HugeiconsIcon icon={Edit03Icon} strokeWidth={1.8} /> Edit
                      message
                    </DropdownMenuItem>
                    <DropdownMenuItem
                      onSelect={() => {
                        void copyToClipboard(item.prompt).then((copied) =>
                          setAnnouncement(
                            copied
                              ? "Prompt copied."
                              : "Could not copy this prompt. Try again.",
                          ),
                        );
                      }}
                    >
                      <HugeiconsIcon icon={Copy01Icon} strokeWidth={1.8} /> Copy
                      message
                    </DropdownMenuItem>
                    <DropdownMenuSeparator />
                    <DropdownMenuItem
                      onSelect={() => {
                        const behavior = followUpBehavior === "queue" ? "steer" : "queue";
                        setFollowUpBehavior(behavior);
                        setAnnouncement(
                          behavior === "queue"
                            ? "New follow-ups will queue after the current response."
                            : "New follow-ups will steer the current response.",
                        );
                      }}
                    >
                      <ListEndIcon />
                      {followUpBehavior === "queue" ? "Turn off queueing" : "Turn on queueing"}
                    </DropdownMenuItem>
                    {entry.paused && (
                      <DropdownMenuItem onSelect={onResume}>
                        <PlayIcon /> Resume queue
                      </DropdownMenuItem>
                    )}
                  </NonModalDropdownMenu>
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
