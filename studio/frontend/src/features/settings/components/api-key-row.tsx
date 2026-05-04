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

function relative(iso: string | null): string {
  if (!iso) return "never";
  const diff = Date.now() - new Date(iso).getTime();
  const days = Math.floor(diff / 86400000);
  if (days < 1) {
    const hours = Math.floor(diff / 3600000);
    if (hours < 1) return "just now";
    return `${hours}h ago`;
  }
  if (days < 30) return `${days}d ago`;
  if (days < 365) return `${Math.floor(days / 30)}mo ago`;
  return `${Math.floor(days / 365)}y ago`;
}

function expiresText(iso: string | null): string {
  if (!iso) return "never";
  const diff = new Date(iso).getTime() - Date.now();
  if (diff < 0) return "expired";
  const days = Math.floor(diff / 86400000);
  if (days < 1) return "today";
  return `in ${days}d`;
}

export function ApiKeyRow({
  apiKey,
  onRevoke,
}: {
  apiKey: ApiKey;
  onRevoke: (key: ApiKey) => void;
}) {
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
          <span>Created {relative(apiKey.created_at)}</span>
          <span>·</span>
          <span>Used {relative(apiKey.last_used_at)}</span>
          <span>·</span>
          <span>Expires {expiresText(apiKey.expires_at)}</span>
        </div>
      </div>
      <DropdownMenu>
        <DropdownMenuTrigger asChild>
          <Button
            variant="ghost"
            size="sm"
            className="size-7 p-0 opacity-0 transition-opacity group-hover:opacity-100 data-[state=open]:opacity-100 max-sm:!opacity-100 max-sm:size-9"
            aria-label={`Actions for ${apiKey.name}`}
          >
            <HugeiconsIcon icon={MoreHorizontalIcon} className="size-4" />
          </Button>
        </DropdownMenuTrigger>
        <DropdownMenuContent align="end">
          <DropdownMenuItem onClick={async () => { await copyToClipboard(prefix); }}>
            <HugeiconsIcon icon={Copy01Icon} className="size-3.5 mr-2" />
            Copy prefix
          </DropdownMenuItem>
          <DropdownMenuItem
            onClick={() => onRevoke(apiKey)}
            className="text-destructive focus:text-destructive"
          >
            <HugeiconsIcon icon={Delete02Icon} className="size-3.5 mr-2" />
            Revoke token
          </DropdownMenuItem>
        </DropdownMenuContent>
      </DropdownMenu>
    </div>
  );
}
