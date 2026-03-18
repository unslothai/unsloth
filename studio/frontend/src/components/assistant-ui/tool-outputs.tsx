// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"use client";

import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import { cn } from "@/lib/utils";
import { copyToClipboard } from "@/lib/copy-to-clipboard";
import type { ToolOutput } from "@/features/chat/api/chat-adapter";
import { useAuiState } from "@assistant-ui/react";
import { ChevronDownIcon, WrenchIcon, CopyIcon, CheckIcon } from "lucide-react";
import { type CSSProperties, memo, useState, useCallback, useRef } from "react";

const ANIMATION_DURATION = 200;
const MAX_DISPLAY_LENGTH = 10000;
const COPY_RESET_MS = 2000;

function truncate(text: string): string {
  if (text.length <= MAX_DISPLAY_LENGTH) return text;
  return text.slice(0, MAX_DISPLAY_LENGTH) + "\n... (truncated)";
}

function formatInput(tool: ToolOutput): string {
  if (tool.toolName === "python" && typeof tool.input?.code === "string") {
    return tool.input.code;
  }
  if (tool.toolName === "web_search" && typeof tool.input?.query === "string") {
    return tool.input.query;
  }
  if (tool.toolName === "terminal" && typeof tool.input?.command === "string") {
    return tool.input.command;
  }
  return JSON.stringify(tool.input, null, 2);
}

function toolLabel(name: string): string {
  switch (name) {
    case "web_search":
      return "Web Search";
    case "python":
      return "Python";
    case "terminal":
      return "Terminal";
    default:
      return name;
  }
}

function CopyButton({ text, className }: { text: string; className?: string }) {
  const [copied, setCopied] = useState(false);
  const resetRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const handleCopy = useCallback(() => {
    if (copyToClipboard(text)) {
      setCopied(true);
      if (resetRef.current) clearTimeout(resetRef.current);
      resetRef.current = setTimeout(() => setCopied(false), COPY_RESET_MS);
    }
  }, [text]);

  return (
    <button
      type="button"
      onClick={handleCopy}
      className={cn(
        "inline-flex items-center gap-1 rounded px-1.5 py-0.5 text-xs text-muted-foreground transition-colors hover:text-foreground hover:bg-muted",
        className,
      )}
      aria-label="Copy to clipboard"
    >
      {copied ? (
        <CheckIcon className="size-3" />
      ) : (
        <CopyIcon className="size-3" />
      )}
      {copied ? "Copied" : "Copy"}
    </button>
  );
}

function ToolItemCard({ tool }: { tool: ToolOutput }) {
  const [isItemOpen, setIsItemOpen] = useState(false);
  const inputText = formatInput(tool);
  const outputText = tool.result;

  return (
    <Collapsible
      open={isItemOpen}
      onOpenChange={setIsItemOpen}
      className="rounded-md border bg-muted/30"
    >
      <CollapsibleTrigger className="flex w-full items-center gap-2 px-3 py-2 text-left text-xs transition-colors hover:bg-muted/50">
        <span className="inline-block rounded-full bg-muted px-2 py-0.5 font-medium text-muted-foreground">
          {toolLabel(tool.toolName)}
        </span>
        <span className="flex-1 truncate text-muted-foreground">
          {tool.toolName === "web_search"
            ? tool.input?.query as string ?? ""
            : tool.toolName === "python"
              ? (tool.input?.code as string ?? "").split("\n")[0].slice(0, 60)
              : tool.toolName === "terminal"
                ? (tool.input?.command as string ?? "").slice(0, 60)
                : ""}
        </span>
        <ChevronDownIcon
          className={cn(
            "size-3.5 shrink-0 text-muted-foreground transition-transform ease-out",
            isItemOpen ? "rotate-0" : "-rotate-90",
          )}
          style={{ transitionDuration: `${ANIMATION_DURATION}ms` }}
        />
      </CollapsibleTrigger>

      <CollapsibleContent
        className={cn(
          "overflow-hidden text-sm",
          "data-[state=closed]:animate-collapsible-up",
          "data-[state=open]:animate-collapsible-down",
        )}
      >
        <div className="border-t px-3 py-2 space-y-2">
          <div>
            <div className="flex items-center justify-between mb-1">
              <span className="text-xs font-medium text-muted-foreground">Input</span>
              <CopyButton text={inputText} />
            </div>
            <pre className="max-h-40 overflow-auto whitespace-pre-wrap break-words rounded bg-muted/50 p-2 text-xs">
              {truncate(inputText)}
            </pre>
          </div>

          <div>
            <div className="flex items-center justify-between mb-1">
              <span className="text-xs font-medium text-muted-foreground">Output</span>
              <CopyButton text={outputText} />
            </div>
            <pre className="max-h-60 overflow-auto whitespace-pre-wrap break-words rounded bg-muted/50 p-2 text-xs">
              {truncate(outputText)}
            </pre>
          </div>
        </div>
      </CollapsibleContent>
    </Collapsible>
  );
}

const ToolOutputsGroupImpl = () => {
  const toolOutputs = useAuiState(({ message }) => {
    const custom = message.metadata?.custom as
      | Record<string, unknown>
      | undefined;
    return custom?.toolOutputs as ToolOutput[] | undefined;
  });

  const isRunning = useAuiState(
    ({ message }) => message.status?.type === "running",
  );

  const [isOpen, setIsOpen] = useState(false);

  if (isRunning || !toolOutputs || toolOutputs.length === 0) {
    return null;
  }

  return (
    <Collapsible
      open={isOpen}
      onOpenChange={setIsOpen}
      className={cn(
        "mb-4 w-full rounded-lg",
        isOpen ? "border px-3 py-2" : "",
      )}
      style={{ "--animation-duration": `${ANIMATION_DURATION}ms` } as CSSProperties}
    >
      <CollapsibleTrigger
        className={cn(
          "group/trigger flex max-w-[75%] items-center gap-2 py-1 text-muted-foreground text-sm transition-colors hover:text-foreground",
          !isOpen && "px-0",
        )}
      >
        <WrenchIcon className="size-4 shrink-0" />
        <span className="leading-none">
          Tool Outputs ({toolOutputs.length})
        </span>
        <ChevronDownIcon
          className={cn(
            "mt-0.5 size-4 shrink-0",
            "transition-transform ease-out",
            isOpen ? "rotate-0" : "-rotate-90",
          )}
          style={{ transitionDuration: `${ANIMATION_DURATION}ms` }}
        />
      </CollapsibleTrigger>

      <CollapsibleContent
        className={cn(
          "overflow-hidden text-sm",
          "data-[state=closed]:animate-collapsible-up",
          "data-[state=open]:animate-collapsible-down",
        )}
      >
        <div className="mt-2 space-y-2">
          {toolOutputs.map((tool, idx) => (
            <ToolItemCard key={tool.toolCallId || idx} tool={tool} />
          ))}
        </div>
      </CollapsibleContent>
    </Collapsible>
  );
};

export const ToolOutputsGroup = memo(ToolOutputsGroupImpl);
ToolOutputsGroup.displayName = "ToolOutputsGroup";
