"use client";

import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import { cn } from "@/lib/utils";
import { ArrowDown01Icon, Logout01Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { type ReactNode, useState } from "react";

export interface ModelOption {
  id: string;
  name: string;
  description?: string;
  icon?: ReactNode;
}

interface ModelSelectorProps {
  models: ModelOption[];
  value?: string;
  defaultValue?: string;
  onValueChange?: (value: string) => void;
  onEject?: () => void;
  variant?: "outline" | "ghost" | "muted";
  size?: "sm" | "default" | "lg";
  className?: string;
  contentClassName?: string;
}

// --- Composable sub-components ---

function ModelSelectorTrigger({
  currentModel,
  isLoaded,
  variant = "outline",
  size = "default",
  className,
}: {
  currentModel?: ModelOption;
  isLoaded: boolean;
  variant?: "outline" | "ghost" | "muted";
  size?: "sm" | "default" | "lg";
  className?: string;
}) {
  return (
    <PopoverTrigger asChild={true}>
      <button
        type="button"
        className={cn(
          "flex items-center gap-2 transition-colors",
          variant === "outline" &&
            "rounded-full border border-border/60 hover:bg-accent",
          variant === "ghost" && "rounded-md hover:bg-accent",
          variant === "muted" && "rounded-md bg-muted hover:bg-muted/80",
          size === "sm" && "h-8 px-3 text-xs",
          size === "default" && "h-9 px-3.5 text-sm",
          size === "lg" && "h-10 px-4 text-sm",
          className,
        )}
      >
        {isLoaded && (
          <span className="size-2 shrink-0 rounded-full bg-emerald-500" />
        )}
        <span
          className={isLoaded ? "text-foreground" : "text-muted-foreground"}
        >
          {currentModel?.name ?? "Select a model\u2026"}
        </span>
        {currentModel?.description && (
          <span className="text-muted-foreground text-xs">
            {currentModel.description}
          </span>
        )}
        <HugeiconsIcon
          icon={ArrowDown01Icon}
          className="size-3 shrink-0 text-muted-foreground"
        />
      </button>
    </PopoverTrigger>
  );
}

function ModelSelectorContent({
  models,
  value,
  onSelect,
  onEject,
  className,
}: {
  models: ModelOption[];
  value?: string;
  onSelect: (id: string) => void;
  onEject?: () => void;
  className?: string;
}) {
  return (
    <PopoverContent
      align="start"
      className={cn("w-auto min-w-[280px] gap-0 p-1", className)}
    >
      {models.map((model) => (
        <ModelSelectorItem
          key={model.id}
          model={model}
          isActive={value === model.id}
          onSelect={onSelect}
          onEject={onEject}
        />
      ))}
    </PopoverContent>
  );
}

function ModelSelectorItem({
  model,
  isActive,
  onSelect,
  onEject,
}: {
  model: ModelOption;
  isActive: boolean;
  onSelect: (id: string) => void;
  onEject?: () => void;
}) {
  return (
    <button
      type="button"
      aria-pressed={isActive}
      onClick={() => onSelect(model.id)}
      className={cn(
        "group flex w-full cursor-pointer items-center gap-2.5 rounded-md px-3 py-2 text-left text-sm transition-colors hover:bg-accent",
        isActive && "bg-accent/50",
      )}
    >
      <span
        className={cn(
          "size-2 shrink-0 rounded-full",
          isActive ? "bg-emerald-500" : "bg-transparent",
        )}
      />
      {model.icon && <span className="shrink-0">{model.icon}</span>}
      <div className="min-w-0 flex-1">
        <div className="truncate text-sm">{model.name}</div>
        {model.description && (
          <div className="truncate text-xs text-muted-foreground">
            {model.description}
          </div>
        )}
      </div>
      {isActive && onEject && (
        <button
          type="button"
          onClick={(e) => {
            e.stopPropagation();
            onEject();
          }}
          className="flex items-center gap-1 rounded px-1.5 py-0.5 text-[10px] text-muted-foreground opacity-0 transition-opacity group-hover:opacity-100 hover:bg-muted hover:text-foreground"
          title="Eject model"
        >
          <HugeiconsIcon icon={Logout01Icon} className="size-3" />
          Eject
        </button>
      )}
    </button>
  );
}

// --- Main component ---

export function ModelSelector({
  models,
  value,
  defaultValue,
  onValueChange,
  onEject,
  variant = "outline",
  size = "default",
  className,
  contentClassName,
}: ModelSelectorProps) {
  const [open, setOpen] = useState(false);
  const [uncontrolled, setUncontrolled] = useState(defaultValue ?? "");
  const selected = value ?? uncontrolled;
  const isLoaded = selected !== "";
  const currentModel = models.find((m) => m.id === selected);

  function handleSelect(id: string) {
    if (onValueChange) {
      onValueChange(id);
    } else {
      setUncontrolled(id);
    }
    setOpen(false);
  }

  function handleEject() {
    onEject?.();
    setOpen(false);
  }

  return (
    <Popover open={open} onOpenChange={setOpen}>
      <ModelSelectorTrigger
        currentModel={currentModel}
        isLoaded={isLoaded}
        variant={variant}
        size={size}
        className={className}
      />
      <ModelSelectorContent
        models={models}
        value={selected}
        onSelect={handleSelect}
        onEject={onEject ? handleEject : undefined}
        className={contentClassName}
      />
    </Popover>
  );
}

// Composable exports
ModelSelector.Trigger = ModelSelectorTrigger;
ModelSelector.Content = ModelSelectorContent;
ModelSelector.Item = ModelSelectorItem;
