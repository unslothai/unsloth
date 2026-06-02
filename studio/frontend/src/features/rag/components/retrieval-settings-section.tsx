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
import { Switch } from "@/components/ui/switch";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import { Button } from "@/components/ui/button";
import { ChevronDownIcon } from "lucide-react";
import { cn } from "@/lib/utils";
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

/** Labeled slider with a value readout, matching the section's other rows. */
function SliderRow({
  label,
  value,
  min,
  max,
  step,
  onChange,
  disabled = false,
  format = (v: number) => String(v),
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (v: number) => void;
  disabled?: boolean;
  format?: (v: number) => string;
}) {
  return (
    <div
      className={cn(
        "flex flex-col gap-2",
        disabled && "pointer-events-none opacity-50",
      )}
    >
      <div className="flex items-center justify-between">
        <span className="text-[13px] font-medium leading-[1.25] tracking-nav text-nav-fg">
          {label}
        </span>
        <span className="text-[13px] tabular-nums text-muted-foreground">
          {format(value)}
        </span>
      </div>
      <Slider
        value={[value]}
        min={min}
        max={max}
        step={step}
        disabled={disabled}
        onValueChange={([v]) => onChange(v)}
        aria-label={label}
      />
    </div>
  );
}

/**
 * Retrieval settings for search_knowledge_base: source (thread docs or a KB),
 * search backend, and passage count.
 */
export function RetrievalSettingsSection() {
  const ragSource = useChatRuntimeStore((s) => s.ragSource);
  const setRagSource = useChatRuntimeStore((s) => s.setRagSource);
  const ragMode = useChatRuntimeStore((s) => s.ragMode);
  const setRagMode = useChatRuntimeStore((s) => s.setRagMode);
  const ragTopK = useChatRuntimeStore((s) => s.ragTopK);
  const setRagTopK = useChatRuntimeStore((s) => s.setRagTopK);
  const ragAutoInject = useChatRuntimeStore((s) => s.ragAutoInject);
  const setRagAutoInject = useChatRuntimeStore((s) => s.setRagAutoInject);
  const ragAutoInjectMinScore = useChatRuntimeStore(
    (s) => s.ragAutoInjectMinScore,
  );
  const setRagAutoInjectMinScore = useChatRuntimeStore(
    (s) => s.setRagAutoInjectMinScore,
  );
  const ragMinScore = useChatRuntimeStore((s) => s.ragMinScore);
  const setRagMinScore = useChatRuntimeStore((s) => s.setRagMinScore);
  const ragRrfK = useChatRuntimeStore((s) => s.ragRrfK);
  const setRagRrfK = useChatRuntimeStore((s) => s.setRagRrfK);
  const ragTopKLexical = useChatRuntimeStore((s) => s.ragTopKLexical);
  const setRagTopKLexical = useChatRuntimeStore((s) => s.setRagTopKLexical);
  const ragTopKDense = useChatRuntimeStore((s) => s.ragTopKDense);
  const setRagTopKDense = useChatRuntimeStore((s) => s.setRagTopKDense);
  // RRF + candidate pools apply only to hybrid; dim them otherwise.
  const hybrid = ragMode === "hybrid";

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

  // If the selected KB was deleted, fall back to the thread source so we never
  // send a stale kb_id.
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

      <div className="flex flex-col gap-3">
        <div className="flex items-center justify-between gap-3">
          <div className="flex flex-col">
            <span className="text-[13px] font-medium leading-[1.25] tracking-nav text-nav-fg">
              Auto-retrieve documents
            </span>
            <span className="text-[12px] leading-[1.3] text-muted-foreground">
              Always search attached documents before answering.
            </span>
          </div>
          <Switch
            checked={ragAutoInject}
            onCheckedChange={setRagAutoInject}
            aria-label="Auto-retrieve documents"
          />
        </div>
        <SliderRow
          label="Auto-retrieve threshold"
          value={ragAutoInjectMinScore}
          min={0}
          max={1}
          step={0.05}
          disabled={!ragAutoInject}
          onChange={setRagAutoInjectMinScore}
          format={(v) => v.toFixed(2)}
        />
      </div>

      <Collapsible className="flex flex-col gap-5">
        <CollapsibleTrigger className="group flex items-center justify-between text-[13px] font-medium leading-[1.25] tracking-nav text-nav-fg">
          Advanced
          <ChevronDownIcon className="size-4 text-muted-foreground transition-transform group-data-[state=open]:rotate-180" />
        </CollapsibleTrigger>
        <CollapsibleContent className="flex flex-col gap-5">
          <SliderRow
            label="Minimum score"
            value={ragMinScore}
            min={0}
            max={1}
            step={0.05}
            onChange={setRagMinScore}
            format={(v) => v.toFixed(2)}
          />
          <SliderRow
            label="Fusion constant (RRF k)"
            value={ragRrfK}
            min={1}
            max={120}
            step={1}
            disabled={!hybrid}
            onChange={setRagRrfK}
          />
          <SliderRow
            label="Lexical candidates"
            value={ragTopKLexical}
            min={1}
            max={100}
            step={1}
            disabled={!hybrid}
            onChange={setRagTopKLexical}
          />
          <SliderRow
            label="Dense candidates"
            value={ragTopKDense}
            min={1}
            max={100}
            step={1}
            disabled={!hybrid}
            onChange={setRagTopKDense}
          />
        </CollapsibleContent>
      </Collapsible>

      <p className="text-[12px] leading-[1.4] text-muted-foreground">
        Document tool-calling works best with capable models (roughly 4B
        parameters or more). Smaller models often answer from memory instead of
        searching, so keep Auto-retrieve on to consult attachments either way.
      </p>

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
