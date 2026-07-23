// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import {
  Delete02Icon,
  Copy01Icon,
  MoreHorizontalIcon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useT } from "@/i18n";
import { copyToClipboard } from "@/lib/copy-to-clipboard";
import type { ApiKey } from "../api/api-keys";

type SettingsT = ReturnType<typeof useT>;

function relative(iso: string | null, t: SettingsT): string {
  if (!iso) return t("settings.apiKeys.relativeNever");
  const diff = Date.now() - new Date(iso).getTime();
  const days = Math.floor(diff / 86400000);
  if (days < 1) {
    const hours = Math.floor(diff / 3600000);
    if (hours < 1) return t("settings.apiKeys.relativeJustNow");
    return t("settings.apiKeys.relativeHoursAgo", { count: hours });
  }
  if (days < 30) return t("settings.apiKeys.relativeDaysAgo", { count: days });
  if (days < 365) {
    return t("settings.apiKeys.relativeMonthsAgo", {
      count: Math.floor(days / 30),
    });
  }
  return t("settings.apiKeys.relativeYearsAgo", {
    count: Math.floor(days / 365),
  });
}

function expiresText(iso: string | null, t: SettingsT): string {
  if (!iso) return t("settings.apiKeys.relativeNever");
  const diff = new Date(iso).getTime() - Date.now();
  if (diff < 0) return t("settings.apiKeys.expired");
  const days = Math.floor(diff / 86400000);
  if (days < 1) return t("settings.apiKeys.today");
  return t("settings.apiKeys.inDays", { count: days });
}

export function ApiKeyRow({
  apiKey,
  onRevoke,
}: {
  apiKey: ApiKey;
  onRevoke: (key: ApiKey) => void;
}) {
  const t = useT();
  const prefix = `sk-unsloth-${apiKey.key_prefix}…`;
  return (
    <div className="group flex items-center gap-3 border-b border-border/60 px-1 py-3 last:border-b-0 transition-colors hover:bg-accent/40">
      <span
        className="size-1.5 shrink-0 rounded-full bg-emerald-500"
        aria-hidden="true"
      />
      <div className="flex min-w-0 flex-1 flex-col gap-0.5">
        <div className="flex min-w-0 items-center justify-between gap-3">
          <span className="truncate text-sm font-medium text-foreground" title={apiKey.name}>
            {apiKey.name}
          </span>
          <code className="shrink-0 font-mono text-[0.6875rem] text-muted-foreground">
            {prefix}
          </code>
        </div>
        <div className="flex flex-wrap gap-x-1.5 text-[0.6875rem] text-muted-foreground">
          <span>
            {t("settings.apiKeys.created", {
              value: relative(apiKey.created_at, t),
            })}
          </span>
          <span>·</span>
          <span>
            {t("settings.apiKeys.used", {
              value: relative(apiKey.last_used_at, t),
            })}
          </span>
          <span>·</span>
          <span>
            {t("settings.apiKeys.expires", {
              value: expiresText(apiKey.expires_at, t),
            })}
          </span>
        </div>
      </div>
      <DropdownMenu>
        <DropdownMenuTrigger asChild>
          <Button
            variant="ghost"
            size="sm"
            className="size-7 p-0 opacity-0 transition-opacity group-hover:opacity-100 data-[state=open]:opacity-100 max-sm:!opacity-100 max-sm:size-9"
            aria-label={t("settings.apiKeys.actionsFor", { name: apiKey.name })}
          >
            <HugeiconsIcon icon={MoreHorizontalIcon} className="size-4" />
          </Button>
        </DropdownMenuTrigger>
        <DropdownMenuContent align="end">
          <DropdownMenuItem onClick={async () => { await copyToClipboard(prefix); }}>
            <HugeiconsIcon icon={Copy01Icon} className="size-3.5 mr-2" />
            {t("settings.apiKeys.copyPrefix")}
          </DropdownMenuItem>
          <DropdownMenuItem
            onClick={() => onRevoke(apiKey)}
            className="text-destructive focus:text-destructive"
          >
            <HugeiconsIcon icon={Delete02Icon} className="size-3.5 mr-2" />
            {t("settings.apiKeys.revokeToken")}
          </DropdownMenuItem>
        </DropdownMenuContent>
      </DropdownMenu>
    </div>
  );
}
