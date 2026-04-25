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
import { copyToClipboard } from "@/lib/copy-to-clipboard";
import type { ApiKey } from "../api/api-keys";
import { useI18n } from "@/features/i18n";

function relative(iso: string | null, locale: string): string {
  if (!iso) return locale === "zh-CN" ? "从未" : "never";
  const diff = Date.now() - new Date(iso).getTime();
  const days = Math.floor(diff / 86400000);
  if (days < 1) {
    const hours = Math.floor(diff / 3600000);
    if (hours < 1) return locale === "zh-CN" ? "刚刚" : "just now";
    return locale === "zh-CN" ? `${hours} 小时前` : `${hours}h ago`;
  }
  if (days < 30) return locale === "zh-CN" ? `${days} 天前` : `${days}d ago`;
  if (days < 365)
    return locale === "zh-CN"
      ? `${Math.floor(days / 30)} 个月前`
      : `${Math.floor(days / 30)}mo ago`;
  return locale === "zh-CN"
    ? `${Math.floor(days / 365)} 年前`
    : `${Math.floor(days / 365)}y ago`;
}

function expiresText(iso: string | null, locale: string): string {
  if (!iso) return locale === "zh-CN" ? "永不过期" : "never";
  const diff = new Date(iso).getTime() - Date.now();
  if (diff < 0) return locale === "zh-CN" ? "已过期" : "expired";
  const days = Math.floor(diff / 86400000);
  if (days < 1) return locale === "zh-CN" ? "今天" : "today";
  return locale === "zh-CN" ? `${days} 天后` : `in ${days}d`;
}

export function ApiKeyRow({
  apiKey,
  onRevoke,
}: {
  apiKey: ApiKey;
  onRevoke: (key: ApiKey) => void;
}) {
  const { t, locale } = useI18n();
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
          <code className="shrink-0 font-mono text-[11px] text-muted-foreground">
            {prefix}
          </code>
        </div>
        <div className="flex flex-wrap gap-x-1.5 text-[11px] text-muted-foreground">
          <span>{t("settings.apiKeys.created")} {relative(apiKey.created_at, locale)}</span>
          <span>·</span>
          <span>{t("settings.apiKeys.used")} {relative(apiKey.last_used_at, locale)}</span>
          <span>·</span>
          <span>{t("settings.apiKeys.expires")} {expiresText(apiKey.expires_at, locale)}</span>
        </div>
      </div>
      <DropdownMenu>
        <DropdownMenuTrigger asChild>
          <Button
            variant="ghost"
            size="sm"
            className="size-7 p-0 opacity-0 transition-opacity group-hover:opacity-100 data-[state=open]:opacity-100 max-sm:!opacity-100 max-sm:size-9"
            aria-label={`${t("settings.apiKeys.actionsFor")} ${apiKey.name}`}
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
            {t("settings.apiKeys.revokeKey")}
          </DropdownMenuItem>
        </DropdownMenuContent>
      </DropdownMenu>
    </div>
  );
}
