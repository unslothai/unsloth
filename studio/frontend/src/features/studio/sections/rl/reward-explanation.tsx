// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import type { RewardExplanation } from "@/features/training";
import { type TranslationKey, useT } from "@/i18n";
import { InformationCircleIcon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import type { ReactElement } from "react";

type Row = [TranslationKey, string | number | null | undefined];

export function RewardExplanationHint({
  explanation,
}: {
  explanation: RewardExplanation;
}): ReactElement {
  const t = useT();
  const e = explanation;
  const normalized = Boolean(e.normalizers?.length);
  const rows: Row[] = [
    ["rl.rewardEditor.referenceColumn", e.comparison_column],
    ["rl.rewardExplanation.response", e.response],
    ["rl.rewardExplanation.reference", e.reference],
    [
      "rl.rewardExplanation.normalizedResponse",
      normalized ? e.normalized_response : undefined,
    ],
    [
      "rl.rewardExplanation.normalizedReference",
      normalized ? e.normalized_reference : undefined,
    ],
    ["rl.rewardEditor.pattern", e.pattern],
    ["rl.rewardExplanation.missingKeys", e.missing_keys?.join(", ")],
    ["rl.rewardExplanation.characters", e.characters],
    ["rl.rewardEditor.maxChars", e.limit],
    [
      e.error_kind === "absolute"
        ? "rl.rewardExplanation.absoluteError"
        : "rl.rewardExplanation.relativeError",
      e.error,
    ],
    ["rl.rewardEditor.within", e.tolerance],
  ];
  return (
    <Tooltip>
      <TooltipTrigger asChild={true}>
        <button
          type="button"
          aria-label={t("rl.rewardExplanation.title")}
          className="cursor-help text-muted-foreground hover:text-foreground"
        >
          <HugeiconsIcon icon={InformationCircleIcon} className="size-3.5" />
        </button>
      </TooltipTrigger>
      <TooltipContent
        variant="rich"
        className="max-h-[min(28rem,var(--radix-tooltip-content-available-height))] w-80 max-w-[calc(100vw-1rem)] overflow-y-auto text-start text-xs"
      >
        <p className="mb-2">
          {t(`rl.rewardExplanation.reasons.${e.reason}` as TranslationKey)}
        </p>
        <dl className="grid grid-cols-[auto_minmax(0,1fr)] gap-x-3 gap-y-1">
          {rows
            .filter(([, value]) => value !== undefined && value !== null)
            .map(([label, value]) => (
              <div key={label} className="contents">
                <dt className="text-muted-foreground">{t(label)}</dt>
                <dd
                  dir="auto"
                  className="whitespace-pre-wrap break-words font-mono"
                >
                  {value === "" ? t("rl.rewardExplanation.empty") : value}
                </dd>
              </div>
            ))}
        </dl>
        {e.reference_fallback && (
          <p className="mt-2 text-muted-foreground">
            {t("rl.rewardExplanation.referenceFallback")}
          </p>
        )}
      </TooltipContent>
    </Tooltip>
  );
}
