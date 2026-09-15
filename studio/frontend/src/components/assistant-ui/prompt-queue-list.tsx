// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useEffect, useId, useRef, useState } from "react";
import {
  ArrowDownIcon,
  ArrowDownToLineIcon,
  ArrowUpIcon,
  ArrowUpToLineIcon,
  CornerDownRightIcon,
  GripVerticalIcon,
  MoreHorizontalIcon,
  PauseIcon,
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
import { TooltipIconButton } from "@/components/assistant-ui/tooltip-icon-button";
import {
  type PromptQueueUIEntry,
  type PromptQueueUIItem,
} from "@/features/chat";
import { copyToClipboard } from "@/lib/copy-to-clipboard";
import { cn } from "@/lib/utils";

type PromptQueueListProps = {
  entry: PromptQueueUIEntry;
  items: PromptQueueUIItem[];
  onEdit: (id: string, prompt: string) => boolean;
  onRemove: (id: string) => boolean;
  onMove: (id: string, targetId: string) => boolean;
  onPause: () => void;
  onResume: () => void;
};

/** The queue engine owns dispatch and validates every mutation against live IDs. */
export function PromptQueueList({
  entry,
  items,
  onEdit,
  onRemove,
  onMove,
  onPause,
  onResume,
}: PromptQueueListProps) {
  const [editingId, setEditingId] = useState<string | null>(null);
  const [draft, setDraft] = useState("");
  const [draggingId, setDraggingId] = useState<string | null>(null);
  const [dropTargetId, setDropTargetId] = useState<string | null>(null);
  const [announcement, setAnnouncement] = useState("");
  const listRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const editFromMenuRef = useRef(false);
  const pointerDragRef = useRef<{
    id: string;
    pointerId: number;
    y: number;
    moved: boolean;
  } | null>(null);
  const instructionsId = useId();
  const editingItem = items.find(
    (item) => item.id === editingId && item.canEdit,
  );
  const activeEditingId = editingItem?.id;
  const movableItems = items.filter((item) => item.canEdit && item.canRemove);
  const draggingIndex = items.findIndex((item) => item.id === draggingId);

  useEffect(() => {
    if (!activeEditingId) return;
    inputRef.current?.focus();
    inputRef.current?.select();
  }, [activeEditingId]);

  function endDrag() {
    pointerDragRef.current = null;
    setDraggingId(null);
    setDropTargetId(null);
  }

  function move(id: string, targetId: string | undefined) {
    if (!targetId || id === targetId) return;
    if (onMove(id, targetId)) {
      const position = items.findIndex((item) => item.id === targetId) + 1;
      setAnnouncement(
        `Prompt moved to position ${position} of ${items.length}.`,
      );
    } else {
      setAnnouncement(
        "The queue changed before this prompt could be moved. Try again.",
      );
    }
  }

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
      row?.querySelector<HTMLButtonElement>("[data-queue-edit]")?.focus();
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

  function canDropOn(id: string) {
    return id !== draggingId && movableItems.some((item) => item.id === id);
  }

  function pointerTarget(clientX: number, clientY: number) {
    const list = listRef.current;
    const row = document
      .elementFromPoint(clientX, clientY)
      ?.closest<HTMLElement>("[data-queue-item-id]");
    const id = row && list?.contains(row) ? row.dataset.queueItemId : undefined;
    return id && canDropOn(id) ? id : null;
  }

  return (
    <div
      ref={listRef}
      className="relative z-0 mx-3 mb-[-8px] max-h-[28dvh] overflow-y-auto rounded-t-[18px] border border-border/45 bg-background px-2 pt-1.5 pb-3 text-muted-foreground sm:mx-7 sm:px-3 dark:bg-card"
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
          const canMove = moveIndex >= 0 && movableItems.length > 1;
          const previous = movableItems[moveIndex - 1];
          const next = movableItems[moveIndex + 1];
          const marker = dropTargetId === item.id && draggingIndex >= 0;
          return (
            <div
              key={item.id}
              role="listitem"
              data-queue-item-id={item.id}
              aria-label={`Queued prompt ${position} of ${items.length}: ${item.prompt}`}
              className={cn(
                "group relative rounded-lg transition-colors",
                draggingId === item.id && "opacity-40",
                marker &&
                  "bg-accent/60 after:pointer-events-none after:absolute after:inset-x-1 after:h-0.5 after:rounded-full after:bg-primary",
                marker &&
                  (draggingIndex < index ? "after:bottom-0" : "after:top-0"),
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
                      if (event.nativeEvent.isComposing) return;
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
                    onClick={finishEditing}
                  >
                    Cancel
                  </Button>
                  <Button
                    type="button"
                    size="sm"
                    disabled={!draft.trim()}
                    onClick={saveEditing}
                  >
                    Save
                  </Button>
                </div>
              ) : (
                <div className="flex min-h-11 items-center gap-1">
                  <TooltipIconButton
                    tooltip="Drag to reorder"
                    aria-label={`Reorder queued prompt ${position} of ${items.length}`}
                    aria-describedby={instructionsId}
                    className="size-7 shrink-0 touch-none cursor-grab text-muted-foreground/70 active:cursor-grabbing pointer-coarse:h-11 pointer-coarse:w-8"
                    disabled={!canMove}
                    onPointerDown={(event) => {
                      if (event.button !== 0 || !canMove || !event.isPrimary)
                        return;
                      event.preventDefault();
                      event.currentTarget.focus({ preventScroll: true });
                      pointerDragRef.current = {
                        id: item.id,
                        pointerId: event.pointerId,
                        y: event.clientY,
                        moved: false,
                      };
                      event.currentTarget.setPointerCapture(event.pointerId);
                    }}
                    onPointerMove={(event) => {
                      const drag = pointerDragRef.current;
                      if (!drag || drag.pointerId !== event.pointerId) return;
                      if (!drag.moved && Math.abs(event.clientY - drag.y) < 5)
                        return;
                      drag.moved = true;
                      setDraggingId(drag.id);
                      const list = listRef.current;
                      const bounds = list?.getBoundingClientRect();
                      if (list && bounds) {
                        if (event.clientY < bounds.top + 28)
                          list.scrollTop -= 12;
                        if (event.clientY > bounds.bottom - 28)
                          list.scrollTop += 12;
                      }
                      setDropTargetId(
                        pointerTarget(event.clientX, event.clientY),
                      );
                    }}
                    onPointerUp={(event) => {
                      const drag = pointerDragRef.current;
                      if (!drag || drag.pointerId !== event.pointerId) return;
                      const target = pointerTarget(
                        event.clientX,
                        event.clientY,
                      );
                      if (drag.moved && target) move(drag.id, target);
                      endDrag();
                    }}
                    onPointerCancel={() => {
                      if (pointerDragRef.current) endDrag();
                    }}
                    onLostPointerCapture={() => {
                      if (pointerDragRef.current) endDrag();
                    }}
                    onKeyDown={(event) => {
                      if (pointerDragRef.current && event.key === "Escape") {
                        event.preventDefault();
                        event.stopPropagation();
                        endDrag();
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
                  <CornerDownRightIcon
                    className="hidden size-4 shrink-0 text-muted-foreground/60 sm:block"
                    aria-hidden="true"
                  />
                  <span className="min-w-0 flex-1 truncate px-1.5 text-sm text-foreground/80">
                    {item.prompt}
                  </span>
                  {index === 0 && (
                    <span className="hidden shrink-0 px-1 text-xs text-muted-foreground sm:inline">
                      {entry.paused ? "Paused" : "Next"}
                    </span>
                  )}
                  <TooltipIconButton
                    tooltip="Edit message"
                    data-queue-edit
                    aria-label={`Edit queued prompt ${position}`}
                    disabled={!item.canEdit}
                    className="size-8 shrink-0 text-muted-foreground hover:text-foreground pointer-coarse:size-11"
                    onClick={() => startEditing(item)}
                  >
                    <HugeiconsIcon icon={Edit03Icon} strokeWidth={1.8} />
                  </TooltipIconButton>
                  <TooltipIconButton
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
                      disabled={!canMove || !previous}
                      onSelect={() => move(item.id, movableItems[0]?.id)}
                    >
                      <ArrowUpToLineIcon /> Move to front
                    </DropdownMenuItem>
                    <DropdownMenuItem
                      disabled={!canMove || !previous}
                      onSelect={() => move(item.id, previous?.id)}
                    >
                      <ArrowUpIcon /> Move up
                    </DropdownMenuItem>
                    <DropdownMenuItem
                      disabled={!canMove || !next}
                      onSelect={() => move(item.id, next?.id)}
                    >
                      <ArrowDownIcon /> Move down
                    </DropdownMenuItem>
                    <DropdownMenuItem
                      disabled={!canMove || !next}
                      onSelect={() => move(item.id, movableItems.at(-1)?.id)}
                    >
                      <ArrowDownToLineIcon /> Move to end
                    </DropdownMenuItem>
                    <DropdownMenuSeparator />
                    <DropdownMenuItem
                      onSelect={entry.paused ? onResume : onPause}
                    >
                      {entry.paused ? <PlayIcon /> : <PauseIcon />}
                      {entry.paused ? "Resume queue" : "Stop and pause queue"}
                    </DropdownMenuItem>
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
