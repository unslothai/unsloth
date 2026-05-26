// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  deletePerModelConfig,
  hasPerModelConfig,
  modelStorageKey,
} from "@/features/chat";
import { cn } from "@/lib/utils";
import { useState } from "react";

export function PerModelConfigNotice({
  modelId,
  ggufVariant = null,
  className,
}: {
  modelId: string;
  ggufVariant?: string | null;
  className?: string;
}) {
  const [resetKey, setResetKey] = useState<string | null>(null);
  const configKey = modelStorageKey(modelId, ggufVariant);
  const remembered =
    resetKey !== configKey && hasPerModelConfig(modelId, ggufVariant);

  if (!remembered) return null;

  return (
    <div
      className={cn(
        "flex items-start gap-2.5 rounded-[10px] border border-border/50 bg-foreground/[0.02] px-3 py-2 dark:bg-white/[0.02]",
        className,
      )}
    >
      <div className="flex min-w-0 flex-1 flex-col">
        <span className="text-[12px] font-medium leading-snug tracking-tight text-foreground">
          Running with custom settings
        </span>
        <span className="text-[11px] leading-snug tracking-tight text-muted-foreground/85">
          Saved runtime overrides will apply on the next run.
        </span>
      </div>
      <button
        type="button"
        onClick={() => {
          deletePerModelConfig(modelId, ggufVariant);
          setResetKey(configKey);
        }}
        className="hub-action-btn h-6 shrink-0 px-2 text-[11px]"
      >
        Reset
      </button>
    </div>
  );
}
