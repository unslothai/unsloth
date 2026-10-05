// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"use client";

import { Button } from "@/components/ui/button";
import {
  type BrowserApprovalChoice,
  resultSummary,
  siteOf,
  useDesktopBrowserStore,
} from "@/features/desktop-browser";
import { stringifyToolResult } from "@/lib/strip-ansi";
import {
  type ToolCallMessagePartComponent,
  useAuiState,
} from "@assistant-ui/react";
import { BrowserIcon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { type ComponentProps, memo, useState } from "react";
import { ScrollPane } from "./scroll-pane";
import { isToolCallRunning, toolArgText } from "./tool-arg-text";
import {
  ToolFallbackContent,
  ToolFallbackRoot,
  ToolFallbackTrigger,
} from "./tool-fallback";
import { useToolActivityOpen } from "./use-tool-activity-open";

function BrowserGlyph(
  props: Omit<ComponentProps<typeof HugeiconsIcon>, "icon">,
) {
  return <HugeiconsIcon icon={BrowserIcon} strokeWidth={2} {...props} />;
}

type BrowserImage = { data: string; mimeType: string };

const IMAGE_MIME = /^image\/(png|jpeg|webp)$/;
// the wrapper is for the model; the card shows the page text without it.
const PAGE_TAGS = /^<\/?browser_page[^>]*>\n?/gm;

function resultParts(result: unknown): {
  text: string;
  images: BrowserImage[];
} {
  if (result && typeof result === "object" && "text" in result) {
    const raw = result as { text?: unknown; images?: unknown };
    const images = Array.isArray(raw.images)
      ? raw.images.filter(
          (image): image is BrowserImage =>
            !!image &&
            typeof image === "object" &&
            typeof (image as BrowserImage).data === "string" &&
            IMAGE_MIME.test(String((image as BrowserImage).mimeType)),
        )
      : [];
    return { text: stringifyToolResult(raw.text), images };
  }
  return {
    text: result == null ? "" : stringifyToolResult(result),
    images: [],
  };
}

function runningName(toolName: string, args: Record<string, unknown>): string {
  const ref = toolArgText(args.ref).trim();
  switch (toolName) {
    case "browser_navigate": {
      const url = toolArgText(args.url).trim();
      const lower = url.toLowerCase();
      if (lower === "back" || lower === "forward") return `Going ${lower}…`;
      if (lower === "reload") return "Reloading the page…";
      return `Opening ${siteOf(url) ?? siteOf(`https://${url}`) ?? (url || "page")}…`;
    }
    case "browser_click":
      return ref ? `Clicking ${ref}…` : "Clicking…";
    case "browser_type":
      return ref ? `Typing into ${ref}…` : "Typing…";
    case "browser_select":
      return `Choosing "${toolArgText(args.option)}"…`;
    case "browser_press_key":
      return `Pressing ${toolArgText(args.key) || "a key"}…`;
    case "browser_scroll":
      return `Scrolling ${toolArgText(args.direction) === "up" ? "up" : "down"}…`;
    case "browser_read":
      return "Reading the page…";
    case "browser_find":
      return `Finding "${toolArgText(args.text)}"…`;
    case "browser_screenshot":
      return "Taking a screenshot…";
    case "browser_handoff":
      return "Handing the browser to you…";
    default:
      return "Looking at the page…";
  }
}

function BrowserApprovalControls({ toolCallId }: { toolCallId: string }) {
  const request = useDesktopBrowserStore(
    (state) => state.approvals[toolCallId],
  );
  const [answered, setAnswered] = useState(false);
  if (!request || answered) return null;
  const answer = (choice: BrowserApprovalChoice) => {
    setAnswered(true);
    request.resolve(choice);
  };
  return (
    <div
      data-testid="browser-approval"
      className="flex flex-col gap-2 rounded-xl bg-muted/40 p-3 text-xs"
    >
      <p className="font-medium text-foreground">{request.question}</p>
      {request.detail ? (
        <p className="text-muted-foreground">{request.detail}</p>
      ) : null}
      <div className="flex flex-wrap items-center gap-2">
        <Button size="xs" onClick={() => answer("allow")}>
          Allow
        </Button>
        {request.site ? (
          <Button
            size="xs"
            variant="outline"
            onClick={() => answer("allow-site")}
          >
            Allow on {request.site}
          </Button>
        ) : null}
        <Button size="xs" variant="destructive" onClick={() => answer("deny")}>
          Deny
        </Button>
      </div>
    </div>
  );
}

function BrowserHandoffNote({
  toolCallId,
  reason,
}: {
  toolCallId: string | undefined;
  reason: string;
}) {
  const handoff = useDesktopBrowserStore((state) =>
    toolCallId ? state.handoffs[toolCallId] : undefined,
  );
  return (
    <div
      data-testid="browser-handoff"
      className="flex flex-col gap-2 rounded-xl bg-primary/10 p-3 text-xs text-foreground"
    >
      <p className="font-medium text-primary">Your turn in the browser</p>
      <p className="whitespace-pre-wrap break-words">{reason}</p>
      {handoff ? (
        <div>
          <Button size="xs" onClick={handoff.done}>
            I&apos;m done
          </Button>
        </div>
      ) : null}
    </div>
  );
}

const BrowserToolUIImpl: ToolCallMessagePartComponent = ({
  toolName,
  args,
  result,
  status,
  toolCallId,
}) => {
  const isRunning = isToolCallRunning(status);
  const argMap = (args ?? {}) as Record<string, unknown>;
  const { text, images } = resultParts(result);
  const awaitingApproval = useDesktopBrowserStore((state) =>
    Boolean(
      toolCallId && (state.approvals[toolCallId] || state.handoffs[toolCallId]),
    ),
  );
  const hasText = useAuiState(({ message }) =>
    message.content.some(
      (p) =>
        p.type === "text" &&
        "text" in p &&
        (p as { text: string }).text.length > 0,
    ),
  );
  const [open, setOpen] = useToolActivityOpen(isRunning, hasText);
  const summary = result == null ? "" : resultSummary(result);
  const name = isRunning || !summary ? runningName(toolName, argMap) : summary;
  const reason = toolArgText(argMap.reason).trim();

  return (
    <ToolFallbackRoot
      open={open}
      onOpenChange={setOpen}
      awaitingApproval={awaitingApproval}
    >
      <ToolFallbackTrigger
        toolName={name}
        status={status}
        icon={BrowserGlyph}
      />
      <ToolFallbackContent>
        {toolCallId ? (
          <BrowserApprovalControls toolCallId={toolCallId} />
        ) : null}
        {toolName === "browser_handoff" && reason ? (
          <BrowserHandoffNote toolCallId={toolCallId} reason={reason} />
        ) : null}
        {images.map((image) => (
          <img
            key={image.data.slice(-32)}
            src={`data:${image.mimeType};base64,${image.data}`}
            alt="Browser screenshot"
            className="max-h-64 w-fit max-w-full rounded-lg border border-border object-contain"
          />
        ))}
        {!isRunning && text ? (
          <ScrollPane
            className="rounded bg-muted/50 p-2"
            scrollerClassName="max-h-40 overflow-auto whitespace-pre-wrap break-words text-xs"
          >
            {text.replace(PAGE_TAGS, "")}
          </ScrollPane>
        ) : null}
      </ToolFallbackContent>
    </ToolFallbackRoot>
  );
};

export const BrowserToolUI = memo(
  BrowserToolUIImpl,
) as unknown as ToolCallMessagePartComponent;
BrowserToolUI.displayName = "BrowserToolUI";
