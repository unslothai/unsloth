// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { NewBadge } from "@/components/new-badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { type RlSampleGroup, getRlSamples } from "@/features/training";
import { useT } from "@/i18n";
import { cn } from "@/lib/utils";
import { ArrowLeft01Icon, ArrowRight01Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { type ReactElement, useEffect, useRef, useState } from "react";
import { RL_PANEL_CLASS } from "../rl/rl-panel";
import { keepNewest, shownIndex } from "./rl-sample-groups";

const POLL_MS = 5000;

const SCORE_CHIP_CLASS =
  "inline-flex h-5 items-center rounded-md px-1.5 font-mono text-ui-11";
const SAMPLE_LABEL_CLASS = "text-xs font-medium text-muted-foreground";

function scoreTone(value: number | null): string {
  if (value === null || value === 0) {
    return "bg-muted text-muted-foreground";
  }
  return value > 0
    ? "bg-emerald-100 text-emerald-700 dark:bg-emerald-950/50 dark:text-emerald-300"
    : "bg-rose-100 text-rose-700 dark:bg-rose-950/50 dark:text-rose-300";
}

function formatScore(value: number | null): string {
  if (value === null) {
    return "–";
  }
  return `${value > 0 ? "+" : ""}${Number(value.toFixed(2))}`;
}

function SampleItem({
  item,
  best,
}: {
  item: RlSampleGroup["items"][number];
  best: boolean;
}): ReactElement {
  const t = useT();
  const [open, setOpen] = useState(false);
  return (
    <div
      className={cn(
        "flex min-w-0 flex-col gap-3 rounded-xl border p-4",
        best ? "border-control-accent/50" : "border-border/70",
      )}
    >
      <div className="flex flex-wrap items-center gap-1.5">
        <span
          className={cn(
            SCORE_CHIP_CLASS,
            "font-medium",
            scoreTone(item.total),
          )}
        >
          {t("rl.samples.total")} {formatScore(item.total)}
        </span>
        {Object.entries(item.rewards).map(([name, value]) => (
          <span
            key={name}
            className={cn(
              SCORE_CHIP_CLASS,
              scoreTone(value),
            )}
          >
            {name} {formatScore(value)}
          </span>
        ))}
      </div>
      <pre
        className={cn(
          "whitespace-pre-wrap break-words font-mono text-ui-11p5 leading-ui-15 text-foreground",
          !open && "line-clamp-6",
        )}
      >
        {item.completion || t("rl.samples.empty")}
      </pre>
      <Button
        type="button"
        size="xs"
        variant="ghost"
        className="-ml-2.5 self-start text-muted-foreground"
        onClick={() => setOpen(!open)}
      >
        {open ? t("rl.samples.showLess") : t("rl.samples.showAll")}
      </Button>
    </div>
  );
}

export function RlSamplesCard({
  jobId,
  isTraining,
}: {
  jobId: string | null;
  isTraining: boolean;
}): ReactElement | null {
  const t = useT();
  const [groups, setGroups] = useState<RlSampleGroup[]>([]);
  const [selectedSeq, setSelectedSeq] = useState<number | null>(null);
  const [loaded, setLoaded] = useState(false);
  const [pollError, setPollError] = useState<string | null>(null);
  const lastSeq = useRef(0);
  // Read by the poll, which outlives renders.
  const selectedSeqRef = useRef<number | null>(null);
  selectedSeqRef.current = selectedSeq;

  useEffect(() => {
    setGroups([]);
    setSelectedSeq(null);
    setLoaded(false);
    setPollError(null);
    lastSeq.current = 0;
    if (!jobId) {
      return;
    }
    let stopped = false;
    const poll = async () => {
      try {
        const fresh = await getRlSamples(jobId, lastSeq.current);
        if (stopped) return;
        setLoaded(true);
        setPollError(null);
        if (fresh.length > 0) {
          lastSeq.current = fresh[fresh.length - 1].seq;
          setGroups((prev) => keepNewest(prev, fresh, selectedSeqRef.current));
        }
      } catch (err) {
        // Shown, and retried on the next tick while training.
        if (!stopped) {
          setLoaded(true);
          setPollError(err instanceof Error ? err.message : String(err));
        }
      }
    };
    poll();
    if (!isTraining) {
      return () => {
        stopped = true;
      };
    }
    const timer = window.setInterval(poll, POLL_MS);
    return () => {
      stopped = true;
      window.clearInterval(timer);
    };
  }, [jobId, isTraining]);

  const errorLine = pollError && (
    <p className="text-ui-11p5 text-destructive">
      {t("rl.samples.pollError", { error: pollError })}
    </p>
  );
  if (groups.length === 0) {
    return (
      <Card size="sm">
        <CardHeader>
          <CardTitle className="flex items-center gap-1.5 text-sm">
            {t("rl.samples.title")}
            <NewBadge />
          </CardTitle>
          <p className="text-ui-11p5 text-muted-foreground/85">
            {t("rl.samples.description")}
          </p>
        </CardHeader>
        <CardContent className="flex flex-col gap-2">
          {errorLine}
          <p className="text-ui-11p5 text-muted-foreground">
            {!loaded
              ? t("rl.samples.loadingAnswers")
              : isTraining
                ? t("rl.samples.waitingFirst")
                : t("rl.samples.noneKept")}
          </p>
        </CardContent>
      </Card>
    );
  }
  const shown = shownIndex(groups, selectedSeq);
  const group = groups[shown];
  const best = Math.max(...group.items.map((i) => i.total));

  return (
    <Card size="sm">
      <CardHeader>
        <CardTitle className="flex items-center justify-between gap-2 text-sm">
          <span className="flex items-center gap-1.5">
            {t("rl.samples.title")}
            <NewBadge />
          </span>
          <span className="flex items-center gap-1 text-xs font-normal text-muted-foreground">
            <Button
              type="button"
              size="icon-xs"
              variant="ghost"
              aria-label={t("rl.samples.older")}
              disabled={shown === 0}
              onClick={() => setSelectedSeq(groups[shown - 1].seq)}
            >
              <HugeiconsIcon icon={ArrowLeft01Icon} className="size-3.5" />
            </Button>
            {group.step === null
              ? t("rl.samples.batch", { n: group.seq })
              : t("rl.samples.step", { step: group.step })}
            <Button
              type="button"
              size="icon-xs"
              variant="ghost"
              aria-label={t("rl.samples.newer")}
              disabled={shown === groups.length - 1}
              onClick={() =>
                setSelectedSeq(
                  shown + 1 === groups.length - 1
                    ? null
                    : groups[shown + 1].seq,
                )
              }
            >
              <HugeiconsIcon icon={ArrowRight01Icon} className="size-3.5" />
            </Button>
          </span>
        </CardTitle>
        <p className="text-ui-11p5 text-muted-foreground/85">
          {t("rl.samples.description")}
        </p>
      </CardHeader>
      <CardContent className="flex flex-col gap-4">
        {errorLine}
        <div
          className={cn(
            RL_PANEL_CLASS,
            "grid gap-4 text-ui-11p5 leading-ui-15 md:grid-cols-[minmax(0,1fr)_minmax(0,14rem)]",
          )}
        >
          <div className="flex min-w-0 flex-col gap-1">
            <p className={SAMPLE_LABEL_CLASS}>{t("rl.samples.prompt")}</p>
            <p className="line-clamp-4 whitespace-pre-wrap break-words text-foreground">
              {group.prompt}
            </p>
          </div>
          {group.answer !== null && (
            <div className="flex min-w-0 flex-col gap-1">
              <p className={SAMPLE_LABEL_CLASS}>{t("rl.samples.answer")}</p>
              <p className="line-clamp-4 whitespace-pre-wrap break-words font-mono text-foreground">
                {group.answer}
              </p>
            </div>
          )}
        </div>
        <div className="grid gap-4 lg:grid-cols-2">
          {group.items.map((item, i) => (
            <SampleItem
              key={`${group.seq}-${i}`}
              item={item}
              best={group.items.length > 1 && item.total === best}
            />
          ))}
        </div>
      </CardContent>
    </Card>
  );
}
