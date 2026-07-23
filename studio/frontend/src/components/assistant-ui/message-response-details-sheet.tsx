// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"use client";

import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
} from "@/components/ui/sheet";
import {
  customProviderDisplayName,
  parseExternalModelId,
  useChatPreferencesStore,
  useChatRuntimeStore,
  useExternalProvidersStore,
} from "@/features/chat";
import { cn } from "@/lib/utils";
import { HelpCircleIcon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useMessage, useMessageTiming } from "@assistant-ui/react";
import type { FC, ReactNode } from "react";

type ResponseDetailsMetadata = {
  modelId?: string;
  modelLabel?: string;
  responseModelId?: string;
  providerId?: string;
  providerName?: string;
  providerType?: string;
  startedAt?: number;
  finishedAt?: number;
  durationMs?: number;
  sessionId?: string | null;
  cancelId?: string;
  toolCalls?: string[];
  tools?: Record<string, boolean | undefined>;
};

type ContextUsageMetadata = {
  promptTokens?: number;
  completionTokens?: number;
  totalTokens?: number;
  cachedTokens?: number;
  cacheWriteTokens?: number;
  modelId?: string;
};

type MessageCustomMetadata = {
  responseDetails?: ResponseDetailsMetadata;
  contextUsage?: ContextUsageMetadata;
  serverTimings?: Record<string, unknown>;
  reasoningDuration?: number;
};

function asNumber(value: unknown): number | undefined {
  return typeof value === "number" && Number.isFinite(value)
    ? value
    : undefined;
}

function formatNumber(value: number | undefined): string | null {
  return value == null ? null : value.toLocaleString();
}

function formatMs(value: number | undefined): string | null {
  if (value == null) return null;
  if (value < 1000) return `${Math.round(value)}ms`;
  return `${(value / 1000).toFixed(2)}s`;
}

function formatRate(value: number | undefined): string | null {
  if (value == null) return null;
  return `${value.toFixed(1)} tok/s`;
}

function formatDate(value: Date | number | string | undefined): string | null {
  if (value == null) return null;
  const date = value instanceof Date ? value : new Date(value);
  if (Number.isNaN(date.getTime())) return null;
  return new Intl.DateTimeFormat(undefined, {
    dateStyle: "medium",
    timeStyle: "medium",
  }).format(date);
}

const TOOL_CATEGORY_LABELS: Record<string, string> = {
  search: "Search",
  fetch: "Fetch",
  code: "Code",
  images: "Images",
  mcp: "MCP",
  docs: "Docs",
  artifacts: "Canvas",
};

const TOOL_CALL_LABELS: Record<string, string> = {
  web_search: "Search",
  web_fetch: "Fetch",
  code_execution: "Code",
  python: "Python",
  terminal: "Terminal",
  image_generation: "Images",
  search_knowledge_base: "Docs",
  render_html: "Canvas",
};

function uniqueValues(values: string[]): string[] {
  return Array.from(new Set(values));
}

function toolCategoryFromCall(toolName: string): string | null {
  const normalized = toolName.toLowerCase();
  if (normalized === "web_search") return "search";
  if (normalized === "web_fetch") return "fetch";
  if (
    normalized === "code_execution" ||
    normalized === "python" ||
    normalized === "terminal"
  ) {
    return "code";
  }
  if (normalized === "image_generation") return "images";
  if (normalized === "search_knowledge_base") return "docs";
  if (normalized === "render_html") return "artifacts";
  if (normalized.startsWith("mcp__")) return "mcp";
  return null;
}

function formatToolCallName(toolName: string): string {
  const normalized = toolName.toLowerCase();
  if (TOOL_CALL_LABELS[normalized]) return TOOL_CALL_LABELS[normalized];
  if (normalized.startsWith("mcp__")) return `MCP: ${toolName.slice(5)}`;
  return toolName
    .replace(/[_-]+/g, " ")
    .replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function toolCallsFromContent(content: unknown): string[] {
  if (!Array.isArray(content)) return [];
  return uniqueValues(
    content
      .map((part) =>
        part && typeof part === "object" && "type" in part
          ? (part as { type?: unknown; toolName?: unknown })
          : null,
      )
      .filter(
        (part): part is { type: "tool-call"; toolName: string } =>
          part?.type === "tool-call" &&
          typeof part.toolName === "string" &&
          part.toolName.length > 0,
      )
      .map((part) => part.toolName),
  );
}

function enabledTools(
  tools: Record<string, boolean | undefined> | undefined,
  toolCalls: string[],
): string | null {
  if (!tools && toolCalls.length === 0) return null;
  const activeKeys = new Set<string>();
  for (const key of Object.keys(TOOL_CATEGORY_LABELS)) {
    if (tools?.[key] === true) activeKeys.add(key);
  }
  for (const toolName of toolCalls) {
    const key = toolCategoryFromCall(toolName);
    if (key) activeKeys.add(key);
  }
  const active = Object.keys(TOOL_CATEGORY_LABELS)
    .filter((key) => activeKeys.has(key))
    .map((key) => TOOL_CATEGORY_LABELS[key]);
  return active.length > 0 ? active.join(", ") : "None";
}

function calledTools(toolCalls: string[]): string | null {
  if (toolCalls.length === 0) return null;
  return uniqueValues(toolCalls.map(formatToolCallName)).join(", ");
}

function DetailSection({
  title,
  children,
}: {
  title: string;
  children: ReactNode;
}) {
  return (
    <section className="rounded-md bg-muted/45 p-3">
      <h3 className="mb-2 font-heading text-foreground text-sm">{title}</h3>
      <div className="grid gap-2">{children}</div>
    </section>
  );
}

function DetailRow({
  label,
  value,
  mono = false,
}: {
  label: string;
  value: ReactNode | null | undefined;
  mono?: boolean;
}) {
  if (value == null || value === "") return null;
  return (
    <div className="grid grid-cols-[8.5rem_minmax(0,1fr)] items-start gap-3 text-[0.8125rem]">
      <span className="text-muted-foreground">{label}</span>
      <span
        className={cn(
          "min-w-0 break-words text-right text-foreground",
          mono && "font-mono tabular-nums",
        )}
      >
        {value}
      </span>
    </div>
  );
}

function useResponseModelDisplay() {
  const message = useMessage();
  const models = useChatRuntimeStore((s) => s.models);
  const providers = useExternalProvidersStore((s) => s.providers);

  const custom = (
    message.metadata as Record<string, unknown> | undefined
  )?.custom as MessageCustomMetadata | undefined;
  const responseDetails = custom?.responseDetails;
  const usage = custom?.contextUsage;
  const serverTimings = custom?.serverTimings;

  const recordedModelId =
    responseDetails?.responseModelId ??
    responseDetails?.modelId ??
    usage?.modelId;
  const parsedExternal = parseExternalModelId(recordedModelId);
  const provider = parsedExternal
    ? providers.find((candidate) => candidate.id === parsedExternal.providerId)
    : null;
  const modelSummary = models.find(
    (candidate) => candidate.id === recordedModelId,
  );
  const modelLabel =
    responseDetails?.modelLabel ??
    responseDetails?.responseModelId ??
    parsedExternal?.modelId ??
    modelSummary?.name ??
    recordedModelId ??
    "Not recorded";
  const providerLabel =
    responseDetails?.providerName ??
    provider?.name ??
    (responseDetails?.providerType
      ? customProviderDisplayName(responseDetails.providerType)
      : parsedExternal
        ? customProviderDisplayName(provider?.providerType)
        : recordedModelId
          ? "Local model"
          : null);

  return {
    message,
    custom,
    responseDetails,
    usage,
    serverTimings,
    modelLabel,
    providerLabel,
  };
}

export const MessageResponseModelBadge: FC<{ className?: string }> = ({
  className,
}) => {
  const showResponseModel = useChatPreferencesStore(
    (state) => state.showResponseModel,
  );
  const { modelLabel, providerLabel } = useResponseModelDisplay();

  if (!showResponseModel || modelLabel === "Not recorded") {
    return null;
  }

  return (
    <span
      className={cn(
        "aui-response-model-badge pointer-events-none relative inline-flex min-h-5 max-w-full cursor-text select-text items-center text-muted-foreground/80 text-xs font-medium leading-5 opacity-0 transition-opacity duration-150 after:absolute after:inset-x-0 after:top-full after:h-1 after:content-[''] hover:opacity-100 group-hover/assistant-message:pointer-events-auto group-hover/assistant-message:opacity-100 group-focus-within/assistant-message:pointer-events-auto group-focus-within/assistant-message:opacity-100",
        className,
      )}
      title={providerLabel ? `${modelLabel} - ${providerLabel}` : modelLabel}
    >
      <span className="min-w-0 truncate align-middle">{modelLabel}</span>
    </span>
  );
};

export const MessageResponseDetailsSheet: FC<{
  open: boolean;
  onOpenChange: (open: boolean) => void;
}> = ({ open, onOpenChange }) => {
  const timing = useMessageTiming();
  const {
    message,
    responseDetails,
    usage,
    serverTimings,
    modelLabel,
    providerLabel,
  } = useResponseModelDisplay();
  const promptTokens =
    usage?.promptTokens ?? asNumber(serverTimings?.prompt_n);
  const completionTokens =
    usage?.completionTokens ??
    timing?.tokenCount ??
    asNumber(serverTimings?.predicted_n);
  const totalTokens =
    usage?.totalTokens ??
    (promptTokens != null && completionTokens != null
      ? promptTokens + completionTokens
      : undefined);
  const totalTime =
    responseDetails?.durationMs ?? timing?.totalStreamTime ?? undefined;
  const summaryLabel =
    modelLabel === "Not recorded" ? "Model not recorded" : `Used ${modelLabel}`;
  const messageToolCalls = toolCallsFromContent(message.content);
  const toolCalls =
    responseDetails?.toolCalls && responseDetails.toolCalls.length > 0
      ? responseDetails.toolCalls
      : messageToolCalls;

  return (
    <Sheet open={open} onOpenChange={onOpenChange}>
      <SheetContent
        side="right"
        className="w-[min(448px,100vw)] p-0 sm:max-w-[448px]"
      >
        <SheetHeader className="border-b p-4">
          <SheetTitle className="flex items-center gap-2 pr-10 font-heading text-base">
            <HugeiconsIcon
              icon={HelpCircleIcon}
              strokeWidth={1.75}
              className="size-icon text-chat-icon-fg"
            />
            Response details
          </SheetTitle>
          <SheetDescription className="sr-only">
            Timing, model, token, and tool details for this response.
          </SheetDescription>
        </SheetHeader>

        <div className="flex min-h-0 flex-1 flex-col gap-4 overflow-y-auto p-4">
          <div className="min-w-0 rounded-md border border-border/70 bg-card p-3">
            <p className="min-w-0 break-words font-heading text-foreground text-sm">
              {summaryLabel}
            </p>
            {providerLabel ? (
              <p className="mt-1 min-w-0 break-words text-muted-foreground text-xs">
                {providerLabel}
              </p>
            ) : null}
          </div>

          <DetailSection title="Response">
            <DetailRow label="Model" value={modelLabel} />
            <DetailRow
              label="Requested"
              value={
                responseDetails?.modelId &&
                responseDetails.modelId !== responseDetails.responseModelId
                  ? responseDetails.modelId
                  : null
              }
            />
            <DetailRow label="Provider" value={providerLabel} />
            <DetailRow label="Message ID" value={message.id} mono={true} />
            <DetailRow label="Created" value={formatDate(message.createdAt)} />
            <DetailRow
              label="Started"
              value={formatDate(responseDetails?.startedAt)}
            />
            <DetailRow
              label="Finished"
              value={formatDate(responseDetails?.finishedAt)}
            />
          </DetailSection>

          <DetailSection title="Tokens">
            <DetailRow label="Prompt" value={formatNumber(promptTokens)} mono />
            <DetailRow
              label="Output"
              value={formatNumber(completionTokens)}
              mono
            />
            <DetailRow label="Total" value={formatNumber(totalTokens)} mono />
            <DetailRow
              label="Cache hits"
              value={formatNumber(
                usage?.cachedTokens ?? asNumber(serverTimings?.cache_n),
              )}
              mono
            />
            <DetailRow
              label="Cache writes"
              value={formatNumber(usage?.cacheWriteTokens)}
              mono
            />
          </DetailSection>

          <DetailSection title="Timing">
            <DetailRow label="Total" value={formatMs(totalTime)} mono />
            <DetailRow
              label="First token"
              value={formatMs(timing?.firstTokenTime)}
              mono
            />
            <DetailRow
              label="Prompt eval"
              value={formatMs(asNumber(serverTimings?.prompt_ms))}
              mono
            />
            <DetailRow
              label="Generation"
              value={formatMs(asNumber(serverTimings?.predicted_ms))}
              mono
            />
            <DetailRow
              label="Speed"
              value={formatRate(
                asNumber(serverTimings?.predicted_per_second) ??
                  timing?.tokensPerSecond,
              )}
              mono
            />
            <DetailRow
              label="Chunks"
              value={formatNumber(timing?.totalChunks)}
              mono
            />
            <DetailRow
              label="Tool calls"
              value={formatNumber(timing?.toolCallCount)}
              mono
            />
          </DetailSection>

          <DetailSection title="Tools">
            <DetailRow
              label="Enabled"
              value={enabledTools(responseDetails?.tools, toolCalls)}
            />
            <DetailRow label="Called" value={calledTools(toolCalls)} />
            <DetailRow
              label="Confirmation"
              value={
                responseDetails?.tools?.confirmToolCalls === true
                  ? "On"
                  : responseDetails?.tools?.confirmToolCalls === false
                    ? "Off"
                    : null
              }
            />
            <DetailRow
              label="Bypass"
              value={
                responseDetails?.tools?.bypassPermissions === true
                  ? "On"
                  : responseDetails?.tools?.bypassPermissions === false
                    ? "Off"
                    : null
              }
            />
            <DetailRow label="Session" value={responseDetails?.sessionId} mono />
            <DetailRow label="Run ID" value={responseDetails?.cancelId} mono />
          </DetailSection>
        </div>
      </SheetContent>
    </Sheet>
  );
};
