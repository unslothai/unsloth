// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { type Locale, formatRelativeTime, useLocale, useT } from "@/i18n";
import { copyToClipboard } from "@/lib/copy-to-clipboard";
import { toast } from "@/lib/toast";
import { cn } from "@/lib/utils";
import {
  Delete02Icon,
  Refresh01Icon,
  MoreHorizontalIcon,
} from "@hugeicons/core-free-icons";
import { Copy01Icon } from "@/lib/copy-icon";
import { HugeiconsIcon } from "@hugeicons/react";
import type { ApiKey } from "../api/api-keys";

type SettingsT = ReturnType<typeof useT>;

export function relative(
  iso: string | null,
  t: SettingsT,
  locale: Locale,
): string {
  if (!iso) return t("settings.apiKeys.relativeNever");
  const diff = Date.now() - new Date(iso).getTime();
  const days = Math.floor(diff / 86400000);
  if (days < 1) {
    const hours = Math.floor(diff / 3600000);
    if (hours < 1) {
      const minutes = Math.floor(diff / 60000);
      if (minutes < 1) return t("settings.apiKeys.relativeJustNow");
      return formatRelativeTime(locale, -minutes, "minute");
    }
    return formatRelativeTime(locale, -hours, "hour");
  }
  if (days < 30) return formatRelativeTime(locale, -days, "day");
  if (days < 365) {
    const months = Math.floor(days / 30);
    return formatRelativeTime(locale, -months, "month");
  }
  const years = Math.floor(days / 365);
  return formatRelativeTime(locale, -years, "year");
}

export function isExpired(iso: string | null): boolean {
  return iso !== null && new Date(iso).getTime() <= Date.now();
}

export function expiresText(
  iso: string | null,
  t: SettingsT,
  locale: Locale,
): string {
  if (!iso) return t("settings.apiKeys.relativeNever");
  const diff = new Date(iso).getTime() - Date.now();
  if (diff < 0) return t("settings.apiKeys.expired");
  if (diff < 86400000) return t("settings.apiKeys.today");
  // ceil: a fresh 7-day token still has 7 days, not "6 days" from 6.99
  return formatRelativeTime(locale, Math.ceil(diff / 86400000), "day");
}

export const TOKEN_TABLE =
  "grid grid-cols-[minmax(0,1fr)_auto_auto_auto_auto_auto] gap-x-6 max-sm:grid-cols-[minmax(0,1fr)_auto_auto_auto] max-sm:gap-x-4";
const TOKEN_ROW = "col-span-full grid grid-cols-subgrid items-center px-3";
const TOKEN_CELL =
  "whitespace-nowrap text-ui-12 text-muted-foreground first-letter:uppercase";

export function ApiKeyTableHeader() {
  const t = useT();
  return (
    <div
      className={cn(
        TOKEN_ROW,
        "pt-3 pb-1.5 text-ui-11 font-medium text-muted-foreground",
      )}
      aria-hidden="true"
    >
      <span className="pl-4.5">{t("settings.apiKeys.columnName")}</span>
      <span className="max-sm:hidden">{t("settings.apiKeys.columnToken")}</span>
      <span className="max-sm:hidden">
        {t("settings.apiKeys.columnCreated")}
      </span>
      <span>{t("settings.apiKeys.columnLastUsed")}</span>
      <span>{t("settings.apiKeys.columnExpires")}</span>
      <span className="w-7 max-sm:w-9" />
    </div>
  );
}

export function ApiKeyRow({
  apiKey,
  onRevoke,
  onRegenerate,
}: {
  apiKey: ApiKey;
  onRevoke: (key: ApiKey) => void;
  onRegenerate: (key: ApiKey) => void;
}) {
  const t = useT();
  const locale = useLocale();
  const prefix = `sk-unsloth-${apiKey.key_prefix}`;
  const expired = isExpired(apiKey.expires_at);
  return (
    <div
      className={cn(
        TOKEN_ROW,
        "group rounded-lg py-2.5 transition-colors hover:bg-muted dark:hover:bg-[rgb(255_255_255_/_calc(0.06*var(--contrast-wash-gain,1)))]",
      )}
    >
      <div className="flex min-w-0 items-center gap-3">
        <span
          className={cn(
            "size-1.5 shrink-0 rounded-full",
            expired ? "bg-muted-foreground/40" : "bg-emerald-500",
          )}
          aria-hidden="true"
        />
        <span
          className="truncate text-sm font-medium text-foreground"
          title={apiKey.name}
        >
          {apiKey.name}
        </span>
      </div>
      <code className="font-mono text-ui-11 text-muted-foreground max-sm:hidden">
        {prefix}…
      </code>
      <span className={cn(TOKEN_CELL, "max-sm:hidden")}>
        {relative(apiKey.created_at, t, locale)}
      </span>
      <span className={TOKEN_CELL}>
        {relative(apiKey.last_used_at, t, locale)}
      </span>
      <span className={cn(TOKEN_CELL, expired && "text-destructive")}>
        {expired
          ? t("settings.apiKeys.expiredStatus")
          : expiresText(apiKey.expires_at, t, locale)}
      </span>
      <DropdownMenu>
        <DropdownMenuTrigger asChild={true}>
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
          <DropdownMenuItem
            onClick={async () => {
              if (await copyToClipboard(prefix)) {
                toast.success(t("settings.apiKeys.copied"));
              } else {
                toast.error(t("settings.apiKeys.copyFailed"));
              }
            }}
          >
            <HugeiconsIcon icon={Copy01Icon} className="size-3.5 mr-2" />
            {t("settings.apiKeys.copyPrefix")}
          </DropdownMenuItem>
          <DropdownMenuItem onClick={() => onRegenerate(apiKey)}>
            <HugeiconsIcon icon={Refresh01Icon} className="size-3.5 mr-2" />
            {t("settings.apiKeys.regenerateToken")}
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
