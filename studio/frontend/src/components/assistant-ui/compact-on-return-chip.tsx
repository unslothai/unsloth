// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"use client";

import { useAuiState } from "@assistant-ui/react";
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
 * cache lapsed. On by default when auto-compaction is on: the next message is sent after a summary
 * of the history, through the same request /compact makes. A click switches it off or on.
 */
export const CompactOnReturnChip: FC = () => {
  const threadId = useChatRuntimeStore((s) => s.activeThreadId);
  const checkpoint = useChatRuntimeStore((s) => s.params.checkpoint);
  const incognito = useChatRuntimeStore((s) => s.incognito);
  // With auto-compaction off the chip still shows, but only a click queues the summary.
  const autoCompactEnabled = useChatRuntimeStore((s) => s.autoCompactEnabled);
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
  // The reply this offer was made (or declined) for: a new reply starts a new offer.
  const [keptFor, setKeptFor] = useState<string | null>(null);
  const offeredRef = useRef<string | null>(null);

  const offerKey = threadId && lastReplyAtMs !== null ? `${threadId}:${lastReplyAtMs}` : null;
  const eligible =
    offerKey !== null &&
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
  const kept = eligible && keptFor === offerKey;

  useEffect(() => {
    if (!threadId) return;
    const store = useCompactRequestStore.getState();
    if (eligible && !kept && autoCompactEnabled && offeredRef.current !== offerKey) {
      offeredRef.current = offerKey;
      store.request(threadId);
    } else if (!eligible && offeredRef.current?.startsWith(`${threadId}:`)) {
      // Only a request this chip made, and only while it is still unsent: a switch to a local
      // model or a thread that shrank must not leave a compaction queued.
      offeredRef.current = null;
      if (!running) store.clear(threadId);
    }
  }, [autoCompactEnabled, eligible, kept, offerKey, running, threadId]);

  if (!eligible || !threadId) return null;
  const on = requested && !kept;
  const tokens = formatTokenCount(threadTokens ?? 0);
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <button
          type="button"
          data-testid="compact-on-return-chip"
          aria-pressed={on}
          onClick={() => {
            const store = useCompactRequestStore.getState();
            if (on) {
              setKeptFor(offerKey);
              store.clear(threadId);
            } else {
              setKeptFor(null);
              offeredRef.current = offerKey;
              store.request(threadId);
            }
          }}
          className={cn(
            "inline-flex h-7 shrink-0 items-center gap-1 rounded-full border px-2.5 font-medium text-xs transition-colors focus-visible:outline-2 focus-visible:outline-ring",
            on
              ? "border-control-accent/40 bg-control-accent/10 text-foreground"
              : "border-border text-muted-foreground hover:text-foreground",
          )}
        >
          {on ? "Compact" : "Keep"}
          <span className="font-mono tabular-nums text-muted-foreground">{tokens}</span>
        </button>
      </TooltipTrigger>
      <TooltipContent side="top" sideOffset={8} variant="rich">
        <p className="max-w-64 text-xs leading-snug">
          {on
            ? `This chat holds about ${tokens} tokens and the provider's prompt cache has likely expired, so sending now re-reads all of it at full price. Your message will be sent after a summary of the history. Click to keep the full history for this message.`
            : "The full history will be sent with this message. Click to summarize it first."}
        </p>
      </TooltipContent>
    </Tooltip>
  );
};
