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
  type TokenLimitStatus,
  fetchTokenLimitStatus,
  overrideTokenLimit,
} from "@/features/profile";
import { useT } from "@/i18n";
import { toast } from "@/lib/toast";
import { cn } from "@/lib/utils";
import { CHAT_HISTORY_UPDATED_EVENT } from "../api/chat-api";

const REFRESH_MS = 60_000;
const HISTORY_SETTLE_MS = 1_500;

const PERIOD_KEYS = {
  day: "settings.profile.tokenLimit.periodNameDay",
  week: "settings.profile.tokenLimit.periodNameWeek",
  month: "settings.profile.tokenLimit.periodNameMonth",
} as const;

export function TokenLimitNotice({ active }: { active: boolean }) {
  const t = useT();
  const [status, setStatus] = useState<TokenLimitStatus | null>(null);
  const [open, setOpen] = useState(false);
  const [busy, setBusy] = useState(false);

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

  if (!status?.warning) return null;

  const { used, reached } = status;
  const { limit } = status.settings;
  const period = t(PERIOD_KEYS[status.settings.period]);
  const percent = Math.min(100, Math.floor((used / limit) * 100));
  const resetsAt = new Date(status.resetsAt).toLocaleString(undefined, {
    dateStyle: "medium",
    timeStyle: "short",
  });
  const canOverride = reached && status.override === null;

  const allowOne = async () => {
    setBusy(true);
    try {
      setStatus(
        await overrideTokenLimit(t("settings.profile.tokenLimit.overrideError")),
      );
      toast.success(t("settings.profile.tokenLimit.overrideGranted"));
      setOpen(false);
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
          reached
            ? "text-destructive"
            : "text-amber-600 dark:text-amber-400",
        )}
      >
        {reached
          ? t("settings.profile.tokenLimit.chipReached")
          : t("settings.profile.tokenLimit.chipWarning", { percent })}
      </Button>
      <AlertDialog
        open={open}
        onOpenChange={(next) => {
          if (!busy) setOpen(next);
        }}
      >
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>
              {canOverride
                ? t("settings.profile.tokenLimit.overrideTitle")
                : t("settings.profile.tokenLimit.title")}
            </AlertDialogTitle>
            <AlertDialogDescription>
              {canOverride
                ? t("settings.profile.tokenLimit.overrideBody", { period })
                : reached
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
              used: used.toLocaleString(),
              limit: limit.toLocaleString(),
            })}
          </p>
          <AlertDialogFooter>
            <AlertDialogCancel disabled={busy}>
              {t("common.close")}
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
