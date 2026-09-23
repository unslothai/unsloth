// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Render production buttons and icons without a backend.
import { TooltipIconButton } from "@/components/assistant-ui/tooltip-icon-button";
import { Button } from "@/components/ui/button";
import { TooltipProvider } from "@/components/ui/tooltip";
import { ArrowUpIcon, SquareIcon } from "lucide-react";
import { createRoot } from "react-dom/client";
import "./src/index.css";

createRoot(document.getElementById("root")!).render(
  <TooltipProvider>
    <div className="aui-root" style={{ padding: 32.25 }}>
      <div className="aui-composer-action-wrapper flex items-center gap-2.5">
        <TooltipIconButton
          data-case="chat-send"
          tooltip="Send message"
          variant="default"
          className="aui-composer-send ml-1.5 size-9 rounded-full"
        >
          <ArrowUpIcon className="unsloth-send-icon aui-composer-send-icon size-[21px] stroke-2" />
        </TooltipIconButton>
        <Button
          data-case="chat-stop"
          aria-label="Stop generation"
          size="icon"
          className="aui-composer-cancel ml-1.5 size-9 rounded-full"
        >
          <SquareIcon className="size-3 fill-current" />
        </Button>
        <TooltipIconButton
          data-case="dictation-stop"
          tooltip="Stop recording"
          variant="ghost"
          className="size-9 rounded-full bg-accent text-foreground"
        >
          <SquareIcon className="size-3 fill-current" />
        </TooltipIconButton>
      </div>
    </div>
    <div
      className="composer-action-wrapper flex items-center gap-2.5"
      style={{ padding: 32.5 }}
    >
      <TooltipIconButton
        data-case="comparison-send"
        tooltip="Send message"
        variant="default"
        className="ml-1.5 size-9 rounded-full"
      >
        <ArrowUpIcon className="unsloth-send-icon size-[22px] stroke-2" />
      </TooltipIconButton>
      <Button
        data-case="comparison-stop"
        aria-label="Stop generation"
        variant="default"
        size="icon"
        className="ml-1.5 size-9 rounded-full"
      >
        <SquareIcon className="size-3 fill-current" />
      </Button>
      <TooltipIconButton
        data-case="transcription-stop"
        tooltip="Cancel transcription"
        variant="default"
        className="ml-1.5 size-9 rounded-full"
      >
        <SquareIcon className="size-3 animate-pulse fill-current" />
      </TooltipIconButton>
    </div>
  </TooltipProvider>,
);
