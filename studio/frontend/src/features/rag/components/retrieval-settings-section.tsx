// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useEffect, useState } from "react";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Slider } from "@/components/ui/slider";
import { Button } from "@/components/ui/button";
import {
  type RagMode,
  type RagSource,
  useChatRuntimeStore,
} from "@/features/chat/stores/chat-runtime-store";
import { listKnowledgeBases } from "../api/rag-api";
import type { KnowledgeBase } from "../types/rag";
import { KnowledgeBaseDialog } from "./knowledge-base-dialog";

const THREAD_VALUE = "__thread__";

const MODE_LABEL: Record<RagMode, string> = {
  hybrid: "Hybrid",
  dense: "Semantic only",
  lexical: "BM25 only",
};

/**
 * Retrieval settings: pick where search_knowledge_base reads from (this
 * thread's own documents or a knowledge base), the search backend, and
 * the number of passages to retrieve. Lives in the chat settings sheet
 * next to Tools / MCP Servers.
 */
export function RetrievalSettingsSection() {
  const ragSource = useChatRuntimeStore((s) => s.ragSource);
  const setRagSource = useChatRuntimeStore((s) => s.setRagSource);
  const ragMode = useChatRuntimeStore((s) => s.ragMode);
  const setRagMode = useChatRuntimeStore((s) => s.setRagMode);
  const ragTopK = useChatRuntimeStore((s) => s.ragTopK);
  const setRagTopK = useChatRuntimeStore((s) => s.setRagTopK);

  const [kbs, setKbs] = useState<KnowledgeBase[]>([]);
  const [dialogOpen, setDialogOpen] = useState(false);
  const [refreshTick, setRefreshTick] = useState(0);

  useEffect(() => {
    let cancelled = false;
    listKnowledgeBases()
      .then((rows) => {
        if (!cancelled) setKbs(rows);
      })
      .catch(() => {
        if (!cancelled) setKbs([]);
      });
    return () => {
      cancelled = true;
    };
  }, [refreshTick]);

  // If the selected KB disappears (deleted in the manager), fall back to
  // the thread source so the request builder never sends a stale kb_id.
  useEffect(() => {
    if (
      ragSource.type === "kb" &&
      kbs.length > 0 &&
      !kbs.some((kb) => kb.id === ragSource.kbId)
    ) {
      setRagSource({ type: "thread" });
    }
  }, [kbs, ragSource, setRagSource]);

  const sourceValue =
    ragSource.type === "kb" ? ragSource.kbId : THREAD_VALUE;

  const onSourceChange = (value: string) => {
    const next: RagSource =
      value === THREAD_VALUE ? { type: "thread" } : { type: "kb", kbId: value };
    setRagSource(next);
  };

  return (
    <div className="flex flex-col gap-5 pt-1">
      <div className="flex flex-col gap-2">
        <span className="text-[13px] font-medium leading-[1.25] tracking-nav text-nav-fg">
          Source
        </span>
        <Select value={sourceValue} onValueChange={onSourceChange}>
          <SelectTrigger
            className="panel-select-trigger h-8 w-full"
            aria-label="Retrieval source"
          >
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value={THREAD_VALUE}>This thread's documents</SelectItem>
            {kbs.map((kb) => (
              <SelectItem key={kb.id} value={kb.id}>
                {kb.name}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>

      <div className="flex flex-col gap-2">
        <span className="text-[13px] font-medium leading-[1.25] tracking-nav text-nav-fg">
          Search mode
        </span>
        <Select
          value={ragMode}
          onValueChange={(value) => setRagMode(value as RagMode)}
        >
          <SelectTrigger
            className="panel-select-trigger h-8 w-full"
            aria-label="Search mode"
          >
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="hybrid">{MODE_LABEL.hybrid}</SelectItem>
            <SelectItem value="dense">{MODE_LABEL.dense}</SelectItem>
            <SelectItem value="lexical">{MODE_LABEL.lexical}</SelectItem>
          </SelectContent>
        </Select>
      </div>

      <div className="flex flex-col gap-2">
        <div className="flex items-center justify-between">
          <span className="text-[13px] font-medium leading-[1.25] tracking-nav text-nav-fg">
            Passages (top K)
          </span>
          <span className="text-[13px] tabular-nums text-muted-foreground">
            {ragTopK}
          </span>
        </div>
        <Slider
          value={[ragTopK]}
          min={1}
          max={20}
          step={1}
          onValueChange={([value]) => setRagTopK(value)}
          aria-label="Number of passages to retrieve"
        />
      </div>

      <div className="flex justify-end">
        <Button
          variant="ghost"
          size="sm"
          onClick={() => setDialogOpen(true)}
        >
          Manage knowledge bases…
        </Button>
      </div>

      <KnowledgeBaseDialog
        open={dialogOpen}
        onOpenChange={(next) => {
          setDialogOpen(next);
          if (!next) setRefreshTick((tick) => tick + 1);
        }}
      />
    </div>
  );
}
