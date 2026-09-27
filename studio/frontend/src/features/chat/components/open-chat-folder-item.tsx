// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { sandboxSessionIdFor } from "@/components/assistant-ui/sandbox-files";
import { revealSandbox } from "@/components/assistant-ui/sandbox-reveal";
import { DropdownMenuItem } from "@/components/ui/dropdown-menu";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { FolderOpenIcon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { isTauri } from "@/lib/api-base";
import { toast } from "@/lib/toast";
import { useState, type ComponentProps, type ComponentType } from "react";
import type { SidebarItem } from "../hooks/use-chat-sidebar-items";
import { getSidebarItemThreadIds, sandboxSessionIdsHolding } from "./chat-row-menu";

/**
 * "Open chat folder" (desktop app only). A compare chat outside a project spans two
 * sandboxes, so it gets no item.
 */
export function OpenChatFolderItem({
  item,
  Item = DropdownMenuItem,
}: {
  item: SidebarItem;
  Item?: ComponentType<ComponentProps<typeof DropdownMenuItem>>;
}) {
  const threadIds = getSidebarItemThreadIds(item);
  const sandboxSessionId =
    item.type === "single" || item.projectId
      ? sandboxSessionIdFor(threadIds[0] ?? item.id, item.projectId)
      : undefined;
  if (!sandboxSessionId) return null;
  if (!isTauri) return <OpenChatFolderUnavailableItem Item={Item} />;
  return (
    <Item
      title="Open the folder this chat's tool calls read and write"
      onSelect={() => {
        void (async () => {
          try {
            // Read from history: a chat moved between projects keeps its old sandbox.
            const ids = threadIds.length > 0 ? threadIds : [item.id];
            const distinct = await sandboxSessionIdsHolding(ids);
            if (distinct.length > 1) {
              toast.error("This chat wrote to more than one folder.", {
                description:
                  "It ran tools on both sides of a move, so open the folder from a tool card instead.",
              });
              return;
            }
            await revealSandbox(distinct[0] ?? sandboxSessionId);
          } catch (error) {
            toast.error("Could not open the chat folder.", {
              description: error instanceof Error ? error.message : String(error),
            });
          }
        })();
      }}
    >
      <HugeiconsIcon icon={FolderOpenIcon} strokeWidth={1.75} className="size-icon" />
      <span>Open chat folder</span>
    </Item>
  );
}

const CHAT_FOLDER_HINT =
  "Only the desktop app can open a chat's files folder. In a browser, download a file from the tool result that wrote it.";
const PROJECT_FOLDER_HINT =
  "Only the desktop app can open a project's folder. In a browser, download files from the tool results that wrote them.";

/** "Open project folder": the workspace shared by the project's chats (desktop app only). */
export function OpenProjectFolderItem({
  projectId,
  Item = DropdownMenuItem,
}: {
  projectId: string;
  Item?: ComponentType<ComponentProps<typeof DropdownMenuItem>>;
}) {
  const sandboxSessionId = sandboxSessionIdFor(undefined, projectId);
  if (!sandboxSessionId) return null;
  if (!isTauri) {
    return (
      <OpenChatFolderUnavailableItem Item={Item} label="Open project folder" hint={PROJECT_FOLDER_HINT} />
    );
  }
  return (
    <Item
      title="Open the folder this project's chats read and write"
      onSelect={() => {
        void revealSandbox(sandboxSessionId).catch((error: unknown) => {
          toast.error("Could not open the project folder.", {
            description: error instanceof Error ? error.message : String(error),
          });
        });
      }}
    >
      <HugeiconsIcon icon={FolderOpenIcon} strokeWidth={1.75} className="size-icon" />
      <span>Open project folder</span>
    </Item>
  );
}

/**
 * "Open chat folder" for a browser session, where the backend's file manager is not the user's.
 * Radix's `disabled` takes the row's pointer events away and a tooltip is blocked while the menu
 * owns the screen, so the row stays enabled, refuses the select itself, and drives a controlled
 * tooltip off a pointer-events-none anchor (as the MCP rows do). The reason is carried twice: that
 * tooltip opens on hover, which a screen reader never reaches and a touch device does not have, so
 * `title` describes the row and selecting it opens the hint rather than doing nothing.
 */
export function OpenChatFolderUnavailableItem({
  // The sidebar renders this row into its right-click menu too, which is a different Radix set.
  Item = DropdownMenuItem,
  label = "Open chat folder",
  hint = CHAT_FOLDER_HINT,
}: {
  Item?: ComponentType<ComponentProps<typeof DropdownMenuItem>>;
  label?: string;
  hint?: string;
} = {}) {
  const [hintOpen, setHintOpen] = useState(false);

  return (
    <Item
      aria-disabled={true}
      title={hint}
      className="relative opacity-50"
      onSelect={(event) => {
        event.preventDefault();
        setHintOpen(true);
      }}
      onPointerEnter={() => setHintOpen(true)}
      onPointerLeave={() => setHintOpen(false)}
      onFocus={() => setHintOpen(true)}
      onBlur={() => setHintOpen(false)}
    >
      <HugeiconsIcon icon={FolderOpenIcon} strokeWidth={1.75} className="size-icon" />
      <span>{label}</span>
      <Tooltip open={hintOpen}>
        {/* Our wrapper, not the raw primitive: it registers the trigger element,
            without which the tooltip counts itself blocked by the open menu. */}
        <TooltipTrigger asChild={true}>
          <span
            aria-hidden={true}
            className="pointer-events-none absolute inset-y-0 right-0 w-0"
          />
        </TooltipTrigger>
        <TooltipContent side="right" className="max-w-[calc(220px*var(--ui-space-scale,1))]">
          {hint}
        </TooltipContent>
      </Tooltip>
    </Item>
  );
}
