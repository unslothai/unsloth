// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { DropdownMenuItem } from "@/components/ui/dropdown-menu";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { FolderOpenIcon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useState, type ComponentProps, type ComponentType } from "react";

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
}: {
  Item?: ComponentType<ComponentProps<typeof DropdownMenuItem>>;
} = {}) {
  const [hintOpen, setHintOpen] = useState(false);

  return (
    <Item
      aria-disabled={true}
      title="Only the desktop app can open a chat's files folder. In a browser, download a file from the tool result that wrote it."
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
      <span>Open chat folder</span>
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
          Only the desktop app can open a chat&apos;s files folder. In a browser, download a
          file from the tool result that wrote it.
        </TooltipContent>
      </Tooltip>
    </Item>
  );
}
