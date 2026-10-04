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

const POLL_MS = 5000;
const KEEP_GROUPS = 12;

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
        "flex flex-col gap-2 rounded-xl border p-3",
        best
          ? "border-emerald-300/70 dark:border-emerald-800/70"
          : "border-border/70",
      )}
    >
      <div className="flex flex-wrap items-center gap-1.5">
        <span
          className={cn(
            "rounded-md px-1.5 py-0.5 font-mono text-ui-11 font-medium",
            scoreTone(item.total),
          )}
        >
          {t("rl.samples.total")} {formatScore(item.total)}
        </span>
        {Object.entries(item.rewards).map(([name, value]) => (
          <span
            key={name}
            className={cn(
              "rounded-md px-1.5 py-0.5 font-mono text-ui-10",
              scoreTone(value),
            )}
          >
            {name} {formatScore(value)}
          </span>
        ))}
      </div>
      <pre
        className={cn(
          "whitespace-pre-wrap break-words font-mono text-ui-11p5 text-foreground/90",
          !open && "line-clamp-6",
        )}
      >
        {item.completion || t("rl.samples.empty")}
      </pre>
      <button
        type="button"
        className="self-start text-ui-11 text-muted-foreground hover:text-foreground"
        onClick={() => setOpen(!open)}
      >
        {open ? t("rl.samples.showLess") : t("rl.samples.showAll")}
      </button>
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
  const [index, setIndex] = useState<number | null>(null);
  const lastSeq = useRef(0);

  useEffect(() => {
    setGroups([]);
    setIndex(null);
    lastSeq.current = 0;
    if (!jobId) {
      return;
    }
    let stopped = false;
    const poll = async () => {
      try {
        const fresh = await getRlSamples(jobId, lastSeq.current);
        if (!stopped && fresh.length > 0) {
          lastSeq.current = fresh[fresh.length - 1].seq;
          setGroups((prev) => [...prev, ...fresh].slice(-KEEP_GROUPS));
        }
      } catch {
        // A missed poll is retried on the next tick.
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

  if (groups.length === 0) {
    return null;
  }
  // null follows the newest group as it arrives.
  const shown = index === null ? groups.length - 1 : index;
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
              onClick={() => setIndex(shown - 1)}
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
                setIndex(shown + 1 === groups.length - 1 ? null : shown + 1)
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
      <CardContent className="flex flex-col gap-3">
        <div className="grid gap-2 rounded-xl bg-muted/40 p-3 text-ui-11p5 md:grid-cols-[minmax(0,1fr)_minmax(0,14rem)]">
          <div className="min-w-0">
            <p className="text-ui-10 uppercase tracking-[0.05em] text-muted-foreground/70">
              {t("rl.samples.prompt")}
            </p>
            <p className="line-clamp-4 whitespace-pre-wrap break-words text-foreground/90">
              {group.prompt}
            </p>
          </div>
          {group.answer !== null && (
            <div className="min-w-0">
              <p className="text-ui-10 uppercase tracking-[0.05em] text-muted-foreground/70">
                {t("rl.samples.answer")}
              </p>
              <p className="line-clamp-4 whitespace-pre-wrap break-words font-mono text-foreground/90">
                {group.answer}
              </p>
            </div>
          )}
        </div>
        <div className="grid gap-3 lg:grid-cols-2">
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
