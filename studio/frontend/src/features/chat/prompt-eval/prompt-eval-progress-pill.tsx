// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { cn } from "@/lib/utils";
import { XIcon } from "lucide-react";
import type { PromptEvalProgress } from "./types";

function progressLabel(p: PromptEvalProgress): string {
  if (p.phase === "loading") return `Loading ${p.currentModelName}…`;
  if (p.phase === "done") return "Prompt Eval complete";
  return `Prompt ${p.promptIdx + 1}/${p.totalPrompts} · Model ${p.modelIdx + 1}/${p.totalModels} · ${p.currentModelName}`;
}

export function PromptEvalProgressPill({
  progress,
  onCancel,
}: {
  progress: PromptEvalProgress;
  onCancel: () => void;
}) {
  const done = progress.phase === "done";

  return (
    <div
      className={cn(
        "flex h-[34px] items-center gap-2 rounded-[8px] bg-primary/10 px-3 text-xs font-medium text-primary",
      )}
    >
      {!done && (
        <span className="size-1.5 shrink-0 rounded-full bg-primary animate-pulse" />
      )}
      <span className="max-w-[280px] truncate">{progressLabel(progress)}</span>
      {!done && (
        <button
          type="button"
          onClick={onCancel}
          className="ml-1 flex size-4 shrink-0 items-center justify-center rounded-full opacity-60 transition-opacity hover:opacity-100"
          aria-label="Cancel Prompt Eval"
        >
          <XIcon className="size-3" />
        </button>
      )}
    </div>
  );
}
