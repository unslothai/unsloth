// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The model dropdown both benchmark pages share: the config-sweep page's header pill
// ("Pick a GGUF to benchmark") and the Evals page's Setup field. What picking *means*
// differs per page — sweeps records the pick for the runner to load later, Evals loads
// it right away — so the selection state stays with the caller. This fixes the shared
// look and wiring: same ModelSelector, same locked treatment, same loaded/variant markers.

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
  /** Fine-tuned / exported checkpoints, shown as their own section (Evals offers them). */
  loraModels?: LoraModelOption[];
  /** The picked model id, or undefined for the placeholder. */
  value?: string;
  /** Quant shown as the "GGUF · <quant>" suffix on the trigger. */
  ggufVariant?: string | null;
  /** Whether the pick is resident. Undefined lets ModelSelector treat any value as loaded. */
  loaded?: boolean;
  /** Sweeps grays the pill out while a run owns the model. */
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
