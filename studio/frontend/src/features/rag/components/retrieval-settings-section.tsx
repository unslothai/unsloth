// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import {
  type RagAutoInject,
  type RagMode,
  useChatRuntimeStore,
} from "@/features/chat/stores/chat-runtime-store";
import { cn } from "@/lib/utils";
import { InfoIcon } from "lucide-react";
import type { ReactNode } from "react";

const MODE_LABEL: Record<RagMode, string> = {
  hybrid: "Hybrid",
  dense: "Semantic only",
  lexical: "BM25 only",
};

function InfoHint({ children }: { children: ReactNode }) {
  return (
    <Tooltip>
      <TooltipTrigger asChild={true}>
        <button
          type="button"
          aria-label="More info"
          className="text-muted-foreground/50 hover:text-muted-foreground"
        >
          <InfoIcon className="size-3.5" />
        </button>
      </TooltipTrigger>
      <TooltipContent className="max-w-xs">{children}</TooltipContent>
    </Tooltip>
  );
}

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
        className="panel-slider"
      />
    </div>
  );
}

// Retrieval settings; the source itself is picked from the composer dropdown.
export function RetrievalSettingsSection() {
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
  const ragOcrScanned = useChatRuntimeStore((s) => s.ragOcrScanned);
  const setRagOcrScanned = useChatRuntimeStore((s) => s.setRagOcrScanned);
  const ragCaptionFigures = useChatRuntimeStore((s) => s.ragCaptionFigures);
  const setRagCaptionFigures = useChatRuntimeStore(
    (s) => s.setRagCaptionFigures,
  );

  return (
    <div className="flex flex-col gap-5 pt-1">
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
          className="panel-slider"
        />
      </div>

      <div className="flex flex-col gap-3">
        <div className="flex flex-col">
          <span className="flex items-center gap-1.5 text-[13px] font-medium leading-[1.25] tracking-nav text-nav-fg">
            Auto-retrieve documents
            <InfoHint>
              Auto turns retrieval on for smaller models (9B and below), which
              tend to answer from memory instead of searching, and leaves it to
              larger ones. On and Off force it either way.
            </InfoHint>
          </span>
          <span className="text-[12px] leading-[1.3] text-muted-foreground">
            Search attached documents before answering.
          </span>
        </div>
        <ToggleGroup
          type="single"
          variant="outline"
          value={ragAutoInject}
          onValueChange={(value) => {
            // Radix clears on re-click; ignore empty so one stays selected.
            if (value) {
              setRagAutoInject(value as RagAutoInject);
            }
          }}
          className="w-full"
          aria-label="Auto-retrieve documents"
        >
          <ToggleGroupItem value="auto" className="flex-1">
            Auto
          </ToggleGroupItem>
          <ToggleGroupItem value="on" className="flex-1">
            On
          </ToggleGroupItem>
          <ToggleGroupItem value="off" className="flex-1">
            Off
          </ToggleGroupItem>
        </ToggleGroup>
        <SliderRow
          label="Auto-retrieve threshold"
          value={ragAutoInjectMinScore}
          min={0}
          max={1}
          step={0.05}
          disabled={ragAutoInject === "off"}
          onChange={setRagAutoInjectMinScore}
          format={(v) => v.toFixed(2)}
        />
      </div>

      <div className="flex items-start justify-between gap-3">
        <div className="flex flex-col">
          <span className="flex items-center gap-1.5 text-[13px] font-medium leading-[1.25] tracking-nav text-nav-fg">
            OCR scanned pages
            <InfoHint>
              Read text off scanned or image-only PDF pages with the loaded
              model's vision, at upload time, so picture-only documents become
              searchable. Needs a vision model; pages with a text layer are
              unaffected.
            </InfoHint>
          </span>
          <span className="text-[12px] leading-[1.3] text-muted-foreground">
            Transcribe image-only PDF pages when attaching.
          </span>
        </div>
        <Switch
          checked={ragOcrScanned}
          onCheckedChange={setRagOcrScanned}
          aria-label="OCR scanned pages"
          className="mt-0.5"
        />
      </div>

      <div className="flex items-start justify-between gap-3">
        <div className="flex flex-col">
          <span className="flex items-center gap-1.5 text-[13px] font-medium leading-[1.25] tracking-nav text-nav-fg">
            Describe figures &amp; charts
            <InfoHint>
              Caption PDF figures, charts, tables and diagrams at upload with the
              loaded model's vision, so their content becomes searchable. Needs a
              vision model; adds vision calls for detected figures.
            </InfoHint>
          </span>
          <span className="text-[12px] leading-[1.3] text-muted-foreground">
            Read charts and diagrams when attaching.
          </span>
        </div>
        <Switch
          checked={ragCaptionFigures}
          onCheckedChange={setRagCaptionFigures}
          aria-label="Describe figures and charts"
          className="mt-0.5"
        />
      </div>
    </div>
  );
}
