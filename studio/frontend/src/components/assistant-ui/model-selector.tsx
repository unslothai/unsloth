// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"use client";

import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import { usePlatformStore } from "@/config/env";
import { cn } from "@/lib/utils";
import {
  ArrowDown01Icon,
  FolderSearchIcon,
  Logout01Icon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useCallback, useMemo, useState } from "react";
import type {
  DeletedModelRef,
  LoraModelOption,
  ModelOption,
  ModelPickTarget,
  ModelSelectorChangeMeta,
} from "./model-selector/types";
import { HubModelPicker, LoraModelPicker } from "./model-selector/pickers";
import { ModelConfigPage } from "./model-selector/model-config-page";
import type { PerModelConfig } from "@/features/chat/model-config/per-model-config";
import { savePerModelConfig } from "@/features/chat/model-config/per-model-config";
import { touchRecentModel } from "@/features/chat/model-config/recent-models";

export type {
  DeletedModelRef,
  LoraModelOption,
  ModelOption,
  ModelSelectorChangeMeta,
} from "./model-selector/types";

interface ModelSelectorProps {
  models: ModelOption[];
  loraModels?: LoraModelOption[];
  value?: string;
  defaultValue?: string;
  activeGgufVariant?: string | null;
  onValueChange?: (value: string, meta: ModelSelectorChangeMeta) => void;
  onEject?: () => void;
  onFoldersChange?: () => void;
  onPickLocalModel?: () => void | Promise<void>;
  onModelsChange?: (deletedModel?: DeletedModelRef) => void;
  deleteDisabled?: boolean;
  variant?: "outline" | "ghost" | "muted";
  size?: "sm" | "default" | "lg";
  className?: string;
  contentClassName?: string;
  open?: boolean;
  onOpenChange?: (open: boolean) => void;
  triggerDataTour?: string;
  contentDataTour?: string;
}

function ModelSelectorTrigger({
  currentModel,
  isLoaded,
  variant = "outline",
  size = "default",
  className,
  dataTour,
}: {
  currentModel?: ModelOption;
  isLoaded: boolean;
  variant?: "outline" | "ghost" | "muted";
  size?: "sm" | "default" | "lg";
  className?: string;
  dataTour?: string;
}) {
  return (
    <PopoverTrigger asChild={true}>
      <button
        type="button"
        data-tour={dataTour}
        className={cn(
          "flex min-w-0 items-center gap-2 transition-colors",
          variant === "outline" &&
          "rounded-[10px] border border-border/60 hover:bg-[#ececec] dark:hover:bg-[#2a2b2f]",
          variant === "ghost" && "rounded-[10px] hover:bg-[#ececec] dark:hover:bg-[#2a2b2f]",
          variant === "muted" && "rounded-[10px] bg-muted hover:bg-muted/80",
          size === "sm" && "h-8 px-3 text-xs",
          size === "default" && "h-9 px-3.5 text-sm",
          size === "lg" && "h-10 px-4 text-sm",
          className,
        )}
      >
        {isLoaded && (
          <span className="size-2 shrink-0 rounded-full bg-emerald-500" />
        )}
        <span className="flex min-w-0 flex-1 items-baseline gap-2">
          <span className="min-w-0 flex-1 truncate font-heading text-[16px] font-medium leading-tight text-[#232528] dark:text-white">
            {currentModel?.name ?? "Select model"}
          </span>
          {currentModel?.description && (
            <span className="shrink-0 text-xs leading-none text-muted-foreground">
              {currentModel.description}
            </span>
          )}
        </span>
        <span className="flex size-4 shrink-0 items-center justify-center">
          <HugeiconsIcon
            icon={ArrowDown01Icon}
            strokeWidth={1.75}
            className="relative top-0.5 size-3.5 text-muted-foreground"
          />
        </span>
      </button>
    </PopoverTrigger>
  );
}

type PickerTab = "hub" | "lora";

function PickerTabToggle({
  tab,
  onTabChange,
}: {
  tab: PickerTab;
  onTabChange: (tab: PickerTab) => void;
}) {
  return (
    <div
      className="menu-trigger tab-toggle relative mb-2 inline-flex h-9 w-full items-center rounded-full"
      role="radiogroup"
      aria-label="Model source"
    >
      <span
        aria-hidden="true"
        className={cn(
          "tab-toggle-pill pointer-events-none absolute inset-y-0 left-0 w-1/2 rounded-full transition-transform duration-200 ease-out",
          tab === "lora" ? "translate-x-full" : "translate-x-0",
        )}
      />
      <button
        type="button"
        role="radio"
        aria-checked={tab === "hub"}
        onClick={() => onTabChange("hub")}
        className={cn(
          "relative z-10 inline-flex h-9 flex-1 items-center justify-center rounded-full px-3 text-[12.5px] transition-colors",
          tab === "hub"
            ? "text-foreground"
            : "text-muted-foreground hover:text-foreground",
        )}
      >
        On Device
      </button>
      <button
        type="button"
        role="radio"
        aria-checked={tab === "lora"}
        onClick={() => onTabChange("lora")}
        className={cn(
          "relative z-10 inline-flex h-9 flex-1 items-center justify-center rounded-full px-3 text-[12.5px] transition-colors",
          tab === "lora"
            ? "text-foreground"
            : "text-muted-foreground hover:text-foreground",
        )}
      >
        Fine-tuned
      </button>
    </div>
  );
}

function ModelSelectorListView({
  loraModels,
  value,
  onPick,
  onEject,
  onPickLocalModel,
  tab,
  onTabChange,
}: {
  models: ModelOption[];
  loraModels: LoraModelOption[];
  value?: string;
  onPick: (target: ModelPickTarget) => void;
  onEject?: () => void;
  onFoldersChange?: () => void;
  onPickLocalModel?: () => void;
  tab: PickerTab;
  onTabChange: (tab: PickerTab) => void;
}) {
  const hasSelection = Boolean(value);
  const chatOnly = usePlatformStore((s) => s.isChatOnly());

  return (
    <>
      {chatOnly ? (
        <HubModelPicker value={value} onPick={onPick} />
      ) : (
        <>
          <PickerTabToggle tab={tab} onTabChange={onTabChange} />
          {tab === "hub" ? (
            <HubModelPicker value={value} onPick={onPick} />
          ) : (
            <LoraModelPicker
              loraModels={loraModels}
              value={value}
              onPick={onPick}
            />
          )}
        </>
      )}

      {onPickLocalModel ? (
        <div className="mt-2 border-t border-border/70 pt-2">
          <button
            type="button"
            onClick={onPickLocalModel}
            className="flex w-full items-center justify-center gap-1.5 rounded-md px-2 py-1.5 text-xs text-muted-foreground transition-colors hover:bg-muted/60"
            title="Pick a model file from disk"
          >
            <HugeiconsIcon icon={FolderSearchIcon} className="size-3.5" />
            Pick a model file from disk
          </button>
        </div>
      ) : null}
      {hasSelection && onEject ? (
        <div className="mt-2 border-t border-border/70 pt-2">
          <button
            type="button"
            onClick={onEject}
            className="flex w-full items-center justify-center gap-1.5 rounded-md px-2 py-1.5 text-xs text-destructive transition-colors hover:bg-destructive/10"
            title="Eject model"
          >
            <HugeiconsIcon icon={Logout01Icon} className="size-3.5" />
            Eject loaded model
          </button>
        </div>
      ) : null}
    </>
  );
}

export function ModelSelector({
  models,
  loraModels = [],
  value,
  defaultValue,
  activeGgufVariant,
  onValueChange,
  onEject,
  onFoldersChange,
  onPickLocalModel,
  onModelsChange,
  deleteDisabled,
  variant = "outline",
  size = "default",
  className,
  contentClassName,
  open: controlledOpen,
  onOpenChange,
  triggerDataTour,
  contentDataTour,
}: ModelSelectorProps) {
  const [uncontrolledOpen, setUncontrolledOpen] = useState(false);
  const open = controlledOpen ?? uncontrolledOpen;
  const setOpen = onOpenChange ?? setUncontrolledOpen;
  const [uncontrolled, setUncontrolled] = useState(defaultValue ?? "");

  const selected = value ?? uncontrolled;
  const isLoaded = selected !== "";

  const optionById = useMemo(() => {
    const all = new Map<string, ModelOption>();
    for (const model of models) {
      all.set(model.id, model);
    }
    for (const lora of loraModels) {
      const displayName = lora.name.includes("/")
        ? lora.name.split("/")[0].trim()
        : lora.name;
      const isLocal = lora.source === "local";
      const isTraining = lora.source === "training";
      const isExported = lora.source === "exported";
      const isMerged = lora.exportType === "merged";
      const isGguf = lora.exportType === "gguf";
      const tag = isLocal
        ? isGguf
          ? "GGUF"
          : "Local"
        : isTraining && isMerged
          ? "Full finetune"
          : isExported
            ? isMerged
              ? "Merged · Exported"
              : "LoRA · Exported"
            : "LoRA";
      all.set(lora.id, {
        ...lora,
        name: displayName,
        description: tag,
      });
    }
    return all;
  }, [loraModels, models]);

  const currentModel = useMemo(() => {
    if (!selected) return undefined;
    const found = optionById.get(selected);
    if (activeGgufVariant) {
      const desc = `GGUF · ${activeGgufVariant}`;
      return found ? { ...found, description: desc } : { id: selected, name: selected, description: desc };
    }
    return found ?? { id: selected, name: selected };
  }, [selected, optionById, activeGgufVariant]);

  const [view, setView] = useState<"list" | "config">("list");
  const [target, setTarget] = useState<ModelPickTarget | null>(null);
  const [pickerTab, setPickerTab] = useState<PickerTab>("hub");

  const resetView = useCallback(() => {
    setView("list");
    setTarget(null);
  }, []);

  const handleOpenChange = useCallback(
    (next: boolean) => {
      if (!next) {
        resetView();
        setPickerTab("hub");
      }
      setOpen(next);
    },
    [resetView, setOpen],
  );

  const handlePick = useCallback((next: ModelPickTarget) => {
    setTarget(next);
    setView("config");
  }, []);

  const handleEject = useCallback(() => {
    onEject?.();
    handleOpenChange(false);
  }, [handleOpenChange, onEject]);

  const handlePickLocalModel = useCallback(() => {
    handleOpenChange(false);
    void onPickLocalModel?.();
  }, [handleOpenChange, onPickLocalModel]);

  const handleRun = useCallback(
    (config: PerModelConfig, remember: boolean) => {
      if (!target) return;
      if (remember) {
        savePerModelConfig(target.id, target.meta.ggufVariant ?? null, config);
      }
      touchRecentModel({
        id: target.id,
        ggufVariant: target.meta.ggufVariant ?? null,
      });
      const meta: ModelSelectorChangeMeta = { ...target.meta, config };
      if (onValueChange) {
        onValueChange(target.id, meta);
      } else {
        setUncontrolled(target.id);
      }
      handleOpenChange(false);
    },
    [handleOpenChange, onValueChange, target],
  );

  return (
    <Popover open={open} onOpenChange={handleOpenChange}>
      <ModelSelectorTrigger
        currentModel={currentModel}
        isLoaded={isLoaded}
        variant={variant}
        size={size}
        className={className}
        dataTour={triggerDataTour}
      />
      <PopoverContent
        align="start"
        data-tour={contentDataTour}
        className={cn(
          "menu-soft-surface picker-scope ring-0 w-[min(440px,calc(100vw-1rem))] max-w-[calc(100vw-1rem)] min-w-0 gap-0 rounded-[20px] p-3 overflow-hidden",
          contentClassName,
        )}
      >
        {view === "list" ? (
          <div
            key="list"
            className="animate-in fade-in-0 slide-in-from-left-2 duration-150"
          >
            <ModelSelectorListView
              models={models}
              loraModels={loraModels}
              value={selected}
              onPick={handlePick}
              onEject={onEject ? handleEject : undefined}
              onFoldersChange={onFoldersChange}
              onPickLocalModel={onPickLocalModel ? handlePickLocalModel : undefined}
              tab={pickerTab}
              onTabChange={setPickerTab}
            />
          </div>
        ) : target ? (
          <div
            key={`config-${target.id}-${target.meta.ggufVariant ?? ""}`}
            className="animate-in fade-in-0 slide-in-from-right-3 duration-150"
          >
            <ModelConfigPage
              target={target}
              onBack={resetView}
              onCancel={() => handleOpenChange(false)}
              onRun={handleRun}
              onDeleted={onModelsChange}
              deleteDisabled={deleteDisabled}
            />
          </div>
        ) : null}
      </PopoverContent>
    </Popover>
  );
}

ModelSelector.Trigger = ModelSelectorTrigger;
