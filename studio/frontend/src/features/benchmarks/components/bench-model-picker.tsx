// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0


import { ModelSelector } from "@/features/model-picker";
import type {
  LoraModelOption,
  ModelOption,
  ModelSelectorChangeMeta,
} from "@/features/model-picker/components/model-selector/types";
import { cn } from "@/lib/utils";
import type { ReactElement } from "react";

export function BenchModelPicker({
  models,
  loraModels,
  value,
  ggufVariant,
  loaded,
  locked,
  variant = "ghost",
  className,
  placeholder,
  onPick,
}: {
  models: ModelOption[];
  loraModels?: LoraModelOption[];
  value?: string;
  ggufVariant?: string | null;
  loaded?: boolean;
  locked?: boolean;
  variant?: "outline" | "ghost" | "muted";
  className?: string;
  placeholder?: string;
  onPick: (id: string, meta: ModelSelectorChangeMeta) => void;
}): ReactElement {
  return (
    <div className={cn("min-w-0", locked && "pointer-events-none opacity-60")}>
      <ModelSelector
        models={models}
        loraModels={loraModels}
        value={value ?? undefined}
        activeGgufVariant={ggufVariant ?? null}
        loaded={loaded}
        onValueChange={onPick}
        variant={variant}
        className={className}
        placeholder={placeholder}
      />
    </div>
  );
}
