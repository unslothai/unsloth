// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { formatScore, type RewardExplanation } from "@/features/training";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { useT } from "@/i18n";
import { cn } from "@/lib/utils";
import { InformationCircleIcon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";

export function RewardScoreBadge({ score }: { score: number }) {
  return (
    <Badge
      variant="secondary"
      dir="ltr"
      className={cn(
        "h-5 min-w-10 rounded-none border-0 px-1.5 py-0 text-sm font-semibold leading-none tabular-nums text-foreground",
        score > 0
          ? "bg-primary/10"
          : score < 0
            ? "bg-destructive/10"
            : "bg-muted",
      )}
    >
      {formatScore(score)}
    </Badge>
  );
}

export function RewardTestResult({
  score,
  error,
  total = false,
  explanation,
}: {
  score?: number;
  error: string | null;
  total?: boolean;
  explanation?: RewardExplanation;
}) {
  const t = useT();
  if (error) {
    return (
      <p role="alert" className="text-xs text-destructive">
        {error}
      </p>
    );
  }
  if (score === undefined) return null;
  return (
    <div className="flex items-center justify-between gap-4 border-t border-border/60 pt-4">
      <span className="flex items-center gap-1 text-xs text-muted-foreground">
        {t(total ? "rl.rewards.totalScore" : "rl.rewardEditor.scoreLabel")}
        {explanation && <RewardScoreExplanation explanation={explanation} />}
      </span>
      <output
        aria-live="polite"
        aria-label={t(
          total ? "rl.rewards.tryTotal" : "rl.rewardEditor.result",
          {
            score: formatScore(score),
            total: formatScore(score),
          },
        )}
      >
        <RewardScoreBadge score={score} />
      </output>
    </div>
  );
}

export function RewardScoreExplanation({
  explanation,
}: { explanation: RewardExplanation }) {
  const t = useT();
  const rows: [string, string | number | undefined | null][] = [
    [t("rl.rewardEditor.referenceColumn"), explanation.comparison_column],
    [t("rl.rewardEditor.responsePart"), explanation.response],
    [t("rl.rewardEditor.referencePart"), explanation.reference],
    [
      t("rl.rewardEditor.normalization"),
      explanation.normalizers?.length
        ? explanation.normalizers
            .map((step) => t(`rl.rewardEditor.normalizers.${step}`))
            .join(" → ")
        : undefined,
    ],
    [
      t("rl.rewardExplanation.normalizedResponse"),
      explanation.normalizers?.length
        ? explanation.normalized_response
        : undefined,
    ],
    [
      t("rl.rewardExplanation.normalizedReference"),
      explanation.normalizers?.length
        ? explanation.normalized_reference
        : undefined,
    ],
    [t("rl.rewardEditor.pattern"), explanation.pattern],
    [
      t("rl.rewardEditor.matchMode"),
      explanation.mode && t(`rl.rewardEditor.${explanation.mode}`),
    ],
    [
      t("rl.rewardEditor.jsonType"),
      explanation.json_type &&
        t(`rl.rewardEditor.jsonTypes.${explanation.json_type}`),
    ],
    [
      t("rl.rewardEditor.requiredKeys"),
      explanation.required_keys?.length
        ? explanation.required_keys.join(", ")
        : undefined,
    ],
    [
      t("rl.rewardExplanation.missingKeys"),
      explanation.missing_keys?.join(", "),
    ],
    [t("rl.rewardExplanation.characters"), explanation.characters],
    [t("rl.rewardEditor.maxCharacters"), explanation.limit],
    [
      t(
        explanation.error_kind === "absolute"
          ? "rl.rewardExplanation.absoluteError"
          : "rl.rewardExplanation.relativeError",
      ),
      explanation.error,
    ],
    [t("rl.rewardEditor.within"), explanation.tolerance],
  ];
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <Button
          type="button"
          variant="ghost"
          size="icon-xs"
          aria-label={t("rl.rewardExplanation.title")}
          className="cursor-help text-muted-foreground"
        >
          <HugeiconsIcon icon={InformationCircleIcon} className="size-3.5" />
        </Button>
      </TooltipTrigger>
      <TooltipContent
        variant="rich"
        side="top"
        align="end"
        sideOffset={8}
        className="w-96 max-w-[calc(100vw-1rem)] max-h-[min(28rem,var(--radix-tooltip-content-available-height))] overflow-y-auto text-start motion-reduce:animate-none"
      >
        <div className="flex min-w-0 flex-col gap-3 text-xs">
          <p>{t(`rl.rewardExplanation.reasons.${explanation.reason}`)}</p>
          <dl className="grid min-w-0 grid-cols-1 gap-x-5 gap-y-1 sm:grid-cols-[minmax(0,1fr)_minmax(0,2fr)]">
            {rows
              .filter(([, value]) => value !== undefined && value !== null)
              .map(([label, value]) => (
                <div key={label} className="contents">
                  <dt className="text-muted-foreground sm:py-1">{label}</dt>
                  <dd
                    dir="auto"
                    className="mb-2 min-w-0 whitespace-pre-wrap break-words font-mono leading-relaxed sm:mb-0 sm:py-1"
                  >
                    {value === "" ? t("rl.rewardExplanation.empty") : value}
                  </dd>
                </div>
              ))}
          </dl>
          {explanation.reference_fallback && (
            <p className="text-muted-foreground">
              {t("rl.rewardExplanation.referenceFallback")}
            </p>
          )}
        </div>
      </TooltipContent>
    </Tooltip>
  );
}
