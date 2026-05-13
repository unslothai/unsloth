// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"use client";

import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { usePlatformStore } from "@/config/env";
import { cn } from "@/lib/utils";
import {
  ArrowDown01Icon,
  FolderSearchIcon,
  Logout01Icon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useMemo, useState } from "react";
import type {
  DeletedModelRef,
  LoraModelOption,
  ModelOption,
  ModelSelectorChangeMeta,
} from "./model-selector/types";
import { HubModelPicker, LoraModelPicker } from "./model-selector/pickers";

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
          "rounded-[10px] border border-border/60 hover:bg-[#ececec] dark:hover:bg-[#2d2e32]",
          variant === "ghost" && "rounded-[10px] hover:bg-[#ececec] dark:hover:bg-[#2d2e32]",
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
          <span className="min-w-0 flex-1 truncate font-heading text-[16px] font-medium leading-tight text-black dark:text-white">
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

function ModelSelectorContent({
  models,
  loraModels,
  value,
  onSelect,
  onEject,
  onFoldersChange,
  onPickLocalModel,
  onModelsChange,
  deleteDisabled,
  className,
  dataTour,
}: {
  models: ModelOption[];
  loraModels: LoraModelOption[];
  value?: string;
  onSelect: (id: string, meta: ModelSelectorChangeMeta) => void;
  onEject?: () => void;
  onFoldersChange?: () => void;
  onPickLocalModel?: () => void;
  onModelsChange?: (deletedModel?: DeletedModelRef) => void;
  deleteDisabled?: boolean;
  className?: string;
  dataTour?: string;
}) {
  const hasSelection = Boolean(value);
  const chatOnly = usePlatformStore((s) => s.isChatOnly());

  return (
    <PopoverContent
      align="start"
      data-tour={dataTour}
      className={cn(
        "menu-soft-surface ring-0 w-[min(440px,calc(100vw-1rem))] max-w-[calc(100vw-1rem)] min-w-0 gap-0 p-2",
        className,
      )}
    >
      {chatOnly ? (
        <HubModelPicker models={models} value={value} onSelect={onSelect} onFoldersChange={onFoldersChange} />
      ) : (
        <Tabs defaultValue="hub" className="w-full">
          <TabsList className="mb-2 w-full">
            <TabsTrigger value="hub">Hub models</TabsTrigger>
            <TabsTrigger value="lora">Fine-tuned</TabsTrigger>
          </TabsList>

          <TabsContent value="hub" className="m-0">
            <HubModelPicker models={models} value={value} onSelect={onSelect} onFoldersChange={onFoldersChange} />
          </TabsContent>

          <TabsContent value="lora" className="m-0">
            <LoraModelPicker
              loraModels={loraModels}
              value={value}
              onSelect={onSelect}
              onModelsChange={onModelsChange}
              deleteDisabled={deleteDisabled}
            />
          </TabsContent>
        </Tabs>
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
    </PopoverContent>
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
      // Strip "/ suffix" from display name (e.g. "foo_123/foo" → "foo_123")
      const displayName = lora.name.includes("/")
        ? lora.name.split("/")[0].trim()
        : lora.name;
      // Show type tag instead of base model name
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

  function handleSelect(id: string, meta: ModelSelectorChangeMeta) {
    if (onValueChange) {
      onValueChange(id, meta);
    } else {
      setUncontrolled(id);
    }
    setOpen(false);
  }

  function handleEject() {
    onEject?.();
    setOpen(false);
  }

  function handlePickLocalModel() {
    setOpen(false);
    void onPickLocalModel?.();
  }

  return (
    <Popover open={open} onOpenChange={setOpen}>
      <ModelSelectorTrigger
        currentModel={currentModel}
        isLoaded={isLoaded}
        variant={variant}
        size={size}
        className={className}
        dataTour={triggerDataTour}
      />
      <ModelSelectorContent
        models={models}
        loraModels={loraModels}
        value={selected}
        onSelect={handleSelect}
        onEject={onEject ? handleEject : undefined}
        onFoldersChange={onFoldersChange}
        onPickLocalModel={onPickLocalModel ? handlePickLocalModel : undefined}
        onModelsChange={onModelsChange}
        deleteDisabled={deleteDisabled}
        className={contentClassName}
        dataTour={contentDataTour}
      />
    </Popover>
  );
}

ModelSelector.Trigger = ModelSelectorTrigger;
ModelSelector.Content = ModelSelectorContent;
