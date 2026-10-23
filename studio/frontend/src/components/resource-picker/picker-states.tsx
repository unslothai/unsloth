// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useT } from "@/i18n";
import { cn } from "@/lib/utils";
import { Refresh01Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { PICKER_OPTION_FOCUS_VISIBLE_CLASS } from "./picker-focus";

export function RetryButton({ onRetry }: { onRetry: () => void }) {
  const t = useT();
  return (
    <button
      type="button"
      onClick={onRetry}
      className={cn(
        "mt-1 inline-flex items-center gap-1.5 rounded-full px-3 py-1 text-ui-11 font-medium text-foreground transition-colors bg-[rgb(0_0_0_/_calc(0.06*var(--contrast-wash-gain,1)))] hover:bg-[rgb(0_0_0_/_calc(0.09*var(--contrast-wash-gain,1)))] dark:bg-[rgb(255_255_255_/_calc(0.06*var(--contrast-wash-gain,1)))] dark:hover:bg-[rgb(255_255_255_/_calc(0.1*var(--contrast-wash-gain,1)))]",
        PICKER_OPTION_FOCUS_VISIBLE_CLASS,
      )}
    >
      <HugeiconsIcon icon={Refresh01Icon} strokeWidth={1.75} className="size-3" />
      {t("picker.retry")}
    </button>
  );
}

export function PickerSearchError({
  title,
  detail,
  onRetry,
  compact = false,
}: {
  title: string;
  detail: string;
  onRetry: () => void;
  compact?: boolean;
}) {
  return (
    <div
      role="alert"
      className={
        compact
          ? "flex items-start justify-between gap-3 border-t border-border/60 px-2.5 py-2"
          : "flex flex-col items-center gap-1.5 px-4 py-8 text-center"
      }
    >
      <div className={cn("min-w-0", compact && "text-left")}>
        <p className="text-ui-12p5 font-medium text-foreground">{title}</p>
        <p className="text-ui-11 leading-snug text-muted-foreground">
          {detail}
        </p>
      </div>
      <RetryButton onRetry={onRetry} />
    </div>
  );
}
