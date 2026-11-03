// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useEffect, useState } from "react";

import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";
import { Button } from "@/components/ui/button";
import {
  TOKEN_LIMIT_CHANGED_EVENT,
  type TokenLimitPoolStatus,
  type TokenLimitStatus,
  fetchTokenLimitStatus,
  overrideTokenLimit,
} from "@/features/profile";
import { useT } from "@/i18n";
import { toast } from "@/lib/toast";
import { cn } from "@/lib/utils";
import { CHAT_HISTORY_UPDATED_EVENT } from "../api/chat-api";

const REFRESH_MS = 60_000;
// Saves land in bursts while a reply streams; one refetch after they settle is enough.
const HISTORY_SETTLE_MS = 1_500;

const PERIOD_KEYS = {
  day: "settings.profile.tokenLimit.periodNameDay",
  week: "settings.profile.tokenLimit.periodNameWeek",
  month: "settings.profile.tokenLimit.periodNameMonth",
} as const;

function pickPool(status: TokenLimitStatus): TokenLimitPoolStatus | null {
  const shown = status.pools.filter((pool) => pool.warning);
  if (shown.length === 0) return null;
  return shown.reduce((worst, pool) => {
    if (pool.reached !== worst.reached) return pool.reached ? pool : worst;
    return pool.used / pool.limit > worst.used / worst.limit ? pool : worst;
  });
}

export function TokenLimitNotice({ active }: { active: boolean }) {
  const t = useT();
  const [status, setStatus] = useState<TokenLimitStatus | null>(null);
  const [open, setOpen] = useState(false);
  const [busy, setBusy] = useState(false);
  const [selectedPool, setSelectedPool] = useState<string | null>(null);

  useEffect(() => {
    if (!active) return;
    let cancelled = false;
    let settleTimer: ReturnType<typeof setTimeout> | undefined;
    const refresh = () => {
      void fetchTokenLimitStatus("")
        .then((next) => {
          if (!cancelled) setStatus(next);
        })
        .catch(() => {});
    };
    const onHistoryUpdated = () => {
      clearTimeout(settleTimer);
      settleTimer = setTimeout(refresh, HISTORY_SETTLE_MS);
    };
    const onChanged = (event: Event) => {
      const next = (event as CustomEvent<TokenLimitStatus>).detail;
      if (next) setStatus(next);
    };
    refresh();
    const interval = setInterval(refresh, REFRESH_MS);
    window.addEventListener(CHAT_HISTORY_UPDATED_EVENT, onHistoryUpdated);
    window.addEventListener(TOKEN_LIMIT_CHANGED_EVENT, onChanged);
    return () => {
      cancelled = true;
      clearTimeout(settleTimer);
      clearInterval(interval);
      window.removeEventListener(CHAT_HISTORY_UPDATED_EVENT, onHistoryUpdated);
      window.removeEventListener(TOKEN_LIMIT_CHANGED_EVENT, onChanged);
    };
  }, [active]);

  const warningPool = status ? pickPool(status) : null;
  const pool =
    status?.pools.find((candidate) => candidate.pool === selectedPool) ??
    warningPool;
  if (!status || !warningPool || !pool) return null;

  const period = t(PERIOD_KEYS[status.settings.period]);
  const percent = Math.min(100, Math.floor((pool.used / pool.limit) * 100));
  const resetsAt = new Date(status.resetsAt).toLocaleString(undefined, {
    dateStyle: "medium",
    timeStyle: "short",
  });
  const canOverride = pool.reached && pool.override === null;

  const allowOne = async () => {
    setBusy(true);
    try {
      setStatus(
        await overrideTokenLimit(
          pool.pool,
          t("settings.profile.tokenLimit.overrideError"),
        ),
      );
      toast.success(t("settings.profile.tokenLimit.overrideGranted"));
      setOpen(false);
      setSelectedPool(null);
    } catch (error) {
      toast.error(
        error instanceof Error
          ? error.message
          : t("settings.profile.tokenLimit.overrideError"),
      );
    } finally {
      setBusy(false);
    }
  };

  return (
    <>
      <Button
        type="button"
        variant="ghost"
        size="sm"
        data-token-limit-notice=""
        onClick={() => setOpen(true)}
        className={cn(
          "h-[var(--studio-chat-control-height,34px)] shrink-0 px-2 text-ui-13p5",
          pool.reached
            ? "text-destructive"
            : "text-amber-600 dark:text-amber-400",
        )}
      >
        {pool.reached
          ? t("settings.profile.tokenLimit.chipReached")
          : t("settings.profile.tokenLimit.chipWarning", { percent })}
      </Button>
      <AlertDialog
        open={open}
        onOpenChange={(next) => {
          if (!busy) {
            setOpen(next);
            if (!next) setSelectedPool(null);
          }
        }}
      >
        <AlertDialogContent>
          {status.pools.length > 1 ? (
            <div className="flex gap-2">
              {status.pools.map((candidate) => (
                <Button
                  key={candidate.pool}
                  type="button"
                  size="sm"
                  variant={
                    candidate.pool === pool.pool ? "secondary" : "outline"
                  }
                  aria-pressed={candidate.pool === pool.pool}
                  disabled={busy}
                  onClick={() => setSelectedPool(candidate.pool)}
                >
                  {t(
                    candidate.pool === "local"
                      ? "settings.profile.tokenLimit.poolLocal"
                      : "settings.profile.tokenLimit.poolApi",
                  )}
                </Button>
              ))}
            </div>
          ) : null}
          <AlertDialogHeader>
            <AlertDialogTitle>
              {canOverride
                ? t("settings.profile.tokenLimit.overrideTitle")
                : t("settings.profile.tokenLimit.title")}
            </AlertDialogTitle>
            <AlertDialogDescription>
              {canOverride
                ? t("settings.profile.tokenLimit.overrideBody", { period })
                : pool.reached
                  ? t("settings.profile.tokenLimit.reachedBody", {
                      period,
                      time: resetsAt,
                    })
                  : t("settings.profile.tokenLimit.warningBody", {
                      percent,
                      period,
                      time: resetsAt,
                    })}
            </AlertDialogDescription>
          </AlertDialogHeader>
          <p className="text-sm text-muted-foreground">
            {t("settings.profile.tokenLimit.usage", {
              used: pool.used.toLocaleString(),
              limit: pool.limit.toLocaleString(),
            })}
          </p>
          <AlertDialogFooter>
            <AlertDialogCancel disabled={busy}>
              {t("settings.profile.tokenLimit.close")}
            </AlertDialogCancel>
            {canOverride ? (
              <AlertDialogAction
                disabled={busy}
                onClick={(event) => {
                  event.preventDefault();
                  void allowOne();
                }}
              >
                {t("settings.profile.tokenLimit.overrideConfirm")}
              </AlertDialogAction>
            ) : null}
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </>
  );
}
