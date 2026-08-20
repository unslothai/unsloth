


"use client";

import {
  Sheet,
  SheetCloseButton,
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
  formatMcpToolName,
  mcpServerFromProvenance,
} from "@/features/chat";
import { cn } from "@/lib/utils";
import { useMessage, useMessageTiming } from "@assistant-ui/react";
import { HelpCircleIcon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
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
  platformUsage?: ContextUsageMetadata | null;
  serverTimings?: Record<string, unknown>;
  reasoningDuration?: number;
  platformChatId?: string | null;
  platformSessionId?: string | null;
  platformMessageId?: string | null;
  platformReference?: unknown;
  platformCitations?: unknown;
  platformStreamCompleted?: boolean;
  platformThumbup?: boolean;
  platformFeedback?: string;
  [key: string]: unknown;
};

const SAFE_USAGE_KEYS = new Set([
  "prompttokens",
  "completiontokens",
  "totaltokens",
  "cachedtokens",
  "cachewritetokens",
  "tokencount",
]);

function isSensitiveMetadataKey(key: string): boolean {
  const normalized = key.replace(/[^a-z0-9]/gi, "").toLowerCase();
  if (SAFE_USAGE_KEYS.has(normalized)) return false;
  return (
    normalized === "token" ||
    /(?:authorization|cookie|password|passwd|secret|apikey|providerkey|accesstoken|refreshtoken|idtoken|authtoken|bearertoken|csrftoken|credential|privatekey)/.test(
      normalized,
    )
  );
}

function safeMetadataValue(
  value: unknown,
  seen = new WeakSet<object>(),
): unknown {
  if (
    value == null ||
    typeof value === "string" ||
    typeof value === "number" ||
    typeof value === "boolean"
  ) {
    return value;
  }
  if (typeof value === "bigint") return value.toString();
  if (typeof value === "function" || typeof value === "symbol") {
    return `[${typeof value}]`;
  }
  if (value instanceof Date) return value.toISOString();
  if (typeof value !== "object") return String(value);
  if (seen.has(value)) return "[Circular]";
  seen.add(value);
  if (Array.isArray(value)) {
    return value.map((item) => safeMetadataValue(item, seen));
  }
  return Object.fromEntries(
    Object.entries(value as Record<string, unknown>).map(([key, item]) => [
      key,
      isSensitiveMetadataKey(key)
        ? "[REDACTED]"
        : safeMetadataValue(item, seen),
    ]),
  );
}

function metadataJson(custom: MessageCustomMetadata | undefined): string | null {
  if (!custom || Object.keys(custom).length === 0) return null;
  return JSON.stringify(safeMetadataValue(custom), null, 2);
}

function platformReferenceCounts(value: unknown): {
  chunks: number | undefined;
  documents: number | undefined;
} {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    return { chunks: undefined, documents: undefined };
  }
  const reference = value as Record<string, unknown>;
  return {
    chunks: Array.isArray(reference.chunks) ? reference.chunks.length : undefined,
    documents: Array.isArray(reference.documentAggregations)
      ? reference.documentAggregations.length
      : Array.isArray(reference.doc_aggs)
        ? reference.doc_aggs.length
        : undefined,
  };
}

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
  edit_file: "Edit",
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
    normalized === "terminal" ||
    normalized === "edit_file"
  ) {
    return "code";
  }
  if (normalized === "image_generation") return "images";
  if (normalized === "search_knowledge_base") return "docs";
  if (normalized === "render_html") return "artifacts";
  if (normalized.startsWith("mcp__")) return "mcp";
  return null;
}

function formatToolCallName(toolName: string, mcpServer?: string): string {
  const normalized = toolName.toLowerCase();
  if (TOOL_CALL_LABELS[normalized]) return TOOL_CALL_LABELS[normalized];
  const mcpLabel = formatMcpToolName(toolName, mcpServer);
  if (mcpLabel) return `MCP: ${mcpLabel}`;
  // Malformed but still MCP: keep the prefix so the category check agrees.
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

function mcpServersFromContent(content: unknown): Map<string, string> {
  const servers = new Map<string, string>();
  if (!Array.isArray(content)) return servers;
  for (const part of content) {
    if (!part || typeof part !== "object") continue;
    const p = part as { type?: unknown; toolName?: unknown; provenance?: unknown };
    if (p.type !== "tool-call" || typeof p.toolName !== "string") continue;
    const server = mcpServerFromProvenance(p.provenance);
    if (server) servers.set(p.toolName, server);
  }
  return servers;
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

function calledTools(
  toolCalls: string[],
  mcpServers: Map<string, string>,
): string | null {
  if (toolCalls.length === 0) return null;
  return uniqueValues(
    toolCalls.map((name) => formatToolCallName(name, mcpServers.get(name))),
  ).join(", ");
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
    <div className="grid grid-cols-[8.5rem_minmax(0,1fr)] items-start gap-3 text-ui-13">
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
  const usage = custom?.contextUsage ?? custom?.platformUsage ?? undefined;
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
    custom,
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
  // Same gate as the timing tooltip: a one-token prompt or a missing rate has no speed to show.
  const promptRate = asNumber(serverTimings?.prompt_per_second);
  const promptSpeed =
    (asNumber(serverTimings?.prompt_n) ?? 0) > 1 && (promptRate ?? 0) > 0
      ? promptRate
      : undefined;
  const totalTime =
    responseDetails?.durationMs ?? timing?.totalStreamTime ?? undefined;
  const summaryLabel =
    modelLabel === "Not recorded"
      ? providerLabel
        ? `${providerLabel} response`
        : "Model not recorded"
      : `Used ${modelLabel}`;
  const messageToolCalls = toolCallsFromContent(message.content);
  const mcpServers = mcpServersFromContent(message.content);
  const toolCalls =
    responseDetails?.toolCalls && responseDetails.toolCalls.length > 0
      ? responseDetails.toolCalls
      : messageToolCalls;
  const referenceCounts = platformReferenceCounts(custom?.platformReference);
  const rawMetadata = open ? metadataJson(custom) : null;
  const hasPlatformDetails = Boolean(
    custom?.platformChatId ||
      custom?.platformSessionId ||
      custom?.platformMessageId ||
      custom?.platformReference ||
      custom?.platformStreamCompleted !== undefined,
  );

  return (
    <Sheet open={open} onOpenChange={onOpenChange}>
      <SheetContent
        side="right"
        className="w-[min(28rem,100vw)] p-0 sm:max-w-[28rem]"
        showCloseButton={false}
      >
        <SheetHeader className="border-b p-4">
          <div className="relative">
            <SheetTitle className="flex items-center gap-2 pr-10 font-heading text-base">
              <HugeiconsIcon
                icon={HelpCircleIcon}
                strokeWidth={1.75}
                className="size-icon text-chat-icon-fg"
              />
              Response details
            </SheetTitle>
            <SheetCloseButton className="absolute top-1/2 right-0 -translate-y-1/2" />
          </div>
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
            <DetailRow
              label="Backend message ID"
              value={custom?.platformMessageId}
              mono={true}
            />
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

          {hasPlatformDetails ? (
            <DetailSection title="Rag Platform">
              <DetailRow label="Chat ID" value={custom?.platformChatId} mono />
              <DetailRow
                label="Session ID"
                value={custom?.platformSessionId}
                mono
              />
              <DetailRow
                label="Message ID"
                value={custom?.platformMessageId}
                mono
              />
              <DetailRow
                label="Stream"
                value={
                  custom?.platformStreamCompleted === true
                    ? "Completed"
                    : custom?.platformStreamCompleted === false
                      ? "Incomplete"
                      : null
                }
              />
              <DetailRow
                label="Reference chunks"
                value={formatNumber(referenceCounts.chunks)}
                mono
              />
              <DetailRow
                label="Reference documents"
                value={formatNumber(referenceCounts.documents)}
                mono
              />
              <DetailRow
                label="Reasoning"
                value={formatMs(custom?.reasoningDuration)}
                mono
              />
              <DetailRow
                label="Feedback"
                value={custom?.platformFeedback}
              />
              <DetailRow
                label="Thumbs up"
                value={
                  custom?.platformThumbup === true
                    ? "Yes"
                    : custom?.platformThumbup === false
                      ? "No"
                      : null
                }
              />
            </DetailSection>
          ) : null}

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
              label="Prompt speed"
              value={formatRate(promptSpeed)}
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
            <DetailRow label="Called" value={calledTools(toolCalls, mcpServers)} />
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

          {rawMetadata ? (
            <DetailSection title="Metadata (sensitive values redacted)">
              <pre
                aria-label="Response metadata"
                className="max-h-96 overflow-auto whitespace-pre-wrap break-words rounded-md border border-border/70 bg-background p-3 font-mono text-[11px] leading-5 text-foreground"
              >
                {rawMetadata}
              </pre>
            </DetailSection>
          ) : null}
        </div>
      </SheetContent>
    </Sheet>
  );
};
