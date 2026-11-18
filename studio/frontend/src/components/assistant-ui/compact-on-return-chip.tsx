// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"use client";

import { useAuiState } from "@assistant-ui/react";
import { Tick02Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { type FC, useEffect, useRef, useState } from "react";

import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import {
  isExternalModelId,
  parseExternalModelId,
} from "@/features/chat/external-providers";
import { formatTokenCount } from "@/features/chat/lib/context-usage-bar-state";
import { useChatRuntimeStore } from "@/features/chat/stores/chat-runtime-store";
import { useCompactRequestStore } from "@/features/chat/stores/compact-request-store";
import { useExternalProvidersStore } from "@/features/chat/stores/external-providers-store";
import {
  promptCacheLifetimeMs,
  shouldOfferCompactOnReturn,
} from "@/features/chat/utils/compact-on-return";
import { isThreadIncognito } from "@/features/chat";
import { cn } from "@/lib/utils";

const RECHECK_MS = 30_000;

/**
 * Next to send on a large cloud thread the user has come back to after the provider's prompt
 * cache lapsed. Only a suggestion: the full history is sent unless a click queues the same
 * summary /compact makes, and a second click withdraws it.
 */
export const CompactOnReturnChip: FC = () => {
  const threadId = useChatRuntimeStore((s) => s.activeThreadId);
  const checkpoint = useChatRuntimeStore((s) => s.params.checkpoint);
  const incognito = useChatRuntimeStore((s) => s.incognito);
  const threadTokens = useChatRuntimeStore((s) =>
    s.contextUsage ? (s.contextUsage.contextTokens ?? s.contextUsage.totalTokens) : null,
  );
  const provider = useExternalProvidersStore((s) => {
    const selection = parseExternalModelId(checkpoint);
    return selection ? s.providers.find((p) => p.id === selection.providerId) : undefined;
  });
  const running = useAuiState(({ thread }) => thread.isRunning);
  const lastReplyAtMs = useAuiState(({ thread }) => {
    const last = thread.messages[thread.messages.length - 1];
    return last?.role === "assistant" ? (last.createdAt?.getTime() ?? null) : null;
  });
  const requested = useCompactRequestStore((s) =>
    threadId ? s.requested[threadId] === true : false,
  );
  const [nowMs, setNowMs] = useState(() => Date.now());
  useEffect(() => {
    const timer = window.setInterval(() => setNowMs(Date.now()), RECHECK_MS);
    return () => window.clearInterval(timer);
  }, []);
  // The thread whose request this chip queued, so leaving eligibility withdraws only its own.
  const queuedForRef = useRef<string | null>(null);

  const eligible =
    threadId !== null &&
    lastReplyAtMs !== null &&
    shouldOfferCompactOnReturn({
      isCloudModel: isExternalModelId(checkpoint) && provider?.providerType !== "openai_codex",
      threadTokens,
      lastReplyAtMs,
      nowMs,
      cacheLifetimeMs: promptCacheLifetimeMs({
        providerType: provider?.providerType,
        promptCacheTtl: provider?.promptCacheTtl,
      }),
      blocked: running || incognito || (threadId ? isThreadIncognito(threadId) : true),
    });

  useEffect(() => {
    const queuedFor = queuedForRef.current;
    if (eligible || queuedFor === null || queuedFor !== threadId || running) return;
    // A switch to a local model or a thread that shrank must not leave a summary queued.
    queuedForRef.current = null;
    useCompactRequestStore.getState().clear(queuedFor);
  }, [eligible, running, threadId]);

  if (!eligible || !threadId) return null;
  const tokens = formatTokenCount(threadTokens ?? 0);
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <button
          type="button"
          data-testid="compact-on-return-chip"
          aria-pressed={requested}
          onClick={() => {
            const store = useCompactRequestStore.getState();
            if (requested) {
              queuedForRef.current = null;
              store.clear(threadId);
            } else {
              queuedForRef.current = threadId;
              store.request(threadId);
            }
          }}
          className={cn(
            "inline-flex h-7 shrink-0 items-center gap-1 rounded-full border px-2.5 font-medium text-xs transition-colors focus-visible:outline-2 focus-visible:outline-ring",
            requested
              ? "border-control-accent/40 bg-control-accent/10 text-foreground"
              : "border-border text-muted-foreground hover:text-foreground",
          )}
        >
          {requested && (
            <HugeiconsIcon icon={Tick02Icon} strokeWidth={2} className="-ml-0.5 size-3.5 text-control-accent" />
          )}
          Compact
          <span className="font-mono tabular-nums text-muted-foreground">{tokens}</span>
        </button>
      </TooltipTrigger>
      <TooltipContent side="top" sideOffset={8} variant="rich">
        <p className="max-w-64 text-xs leading-snug">
          {requested
            ? "Your next message will be sent with a summary in place of the older messages, which stay visible and saved here. Click to send the full history instead."
            : `This chat holds about ${tokens} tokens and the provider's prompt cache has likely expired, so your next message re-reads all of it at full price. Click to summarize the older messages first.`}
        </p>
      </TooltipContent>
    </Tooltip>
  );
};
