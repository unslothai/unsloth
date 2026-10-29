// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The assistant message action bar's icon row, with production TooltipIconButtons and styling.
import { TooltipIconButton } from "@/components/assistant-ui/tooltip-icon-button";
import { TooltipProvider } from "@/components/ui/tooltip";
import {
  CopyIcon,
  EllipsisIcon,
  GitForkIcon,
  PencilIcon,
  RefreshCwIcon,
  Trash2Icon,
  Volume2Icon,
} from "lucide-react";
import { createRoot } from "react-dom/client";
import "./src/index.css";

const BUTTONS = [
  ["Copy", CopyIcon],
  ["Edit response", PencilIcon],
  ["Refresh", RefreshCwIcon],
  ["Read aloud", Volume2Icon],
  ["Fork in new chat", GitForkIcon],
  ["Delete message", Trash2Icon],
  ["More", EllipsisIcon],
] as const;

createRoot(document.getElementById("root") as HTMLElement).render(
  <TooltipProvider>
    <div className="aui-root" style={{ padding: 48 }}>
      <div className="aui-assistant-action-bar-root flex min-w-0 flex-wrap items-center gap-1 text-chat-icon-fg [&_button]:size-8 [&_button]:!rounded-full [&_button:hover]:bg-chat-icon-bg-hover">
        {BUTTONS.map(([label, Icon]) => (
          <TooltipIconButton key={label} tooltip={label} data-case={label}>
            <Icon className="size-icon" />
          </TooltipIconButton>
        ))}
      </div>
    </div>
  </TooltipProvider>,
);
