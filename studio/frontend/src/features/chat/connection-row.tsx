// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import { Spinner } from "@/components/ui/spinner";
import { Switch } from "@/components/ui/switch";
import { TrashIcon } from "@/lib/hugeicons-derived";
import { cn } from "@/lib/utils";
import { Edit03Icon, Wifi02Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import type { ReactNode } from "react";
import { ApiProviderLogo } from "./api-provider-logo";
import type {
  ConnectionStatus,
  ConnectionStatusTone,
} from "./connection-status";
import type { ExternalProviderConfig } from "./external-providers";

const STATUS_DOT_CLASS: Record<ConnectionStatusTone, string | null> = {
  ok: "bg-emerald-500",
  warn: "bg-amber-500",
  error: "bg-destructive",
  muted: null,
};

const STATUS_TEXT_CLASS: Record<ConnectionStatusTone, string> = {
  ok: "text-foreground/80",
  warn: "text-amber-700 dark:text-amber-400",
  error: "text-destructive",
  muted: "text-muted-foreground",
};

export function ConnectionStatusLine({
  status,
  className,
}: {
  status: ConnectionStatus;
  className?: string;
}) {
  const dot = STATUS_DOT_CLASS[status.tone];
  return (
    <div
      className={cn(
        "flex min-w-0 items-center gap-1.5 text-xs leading-snug",
        className,
      )}
      title={
        (status.full ?? status.detail)
          ? `${status.label}. ${status.full ?? status.detail}`
          : status.label
      }
    >
      {dot ? (
        <span
          aria-hidden={true}
          className={cn("size-1.5 shrink-0 rounded-full", dot)}
        />
      ) : null}
      <span
        className={cn("shrink-0 font-medium", STATUS_TEXT_CLASS[status.tone])}
      >
        {status.label}
      </span>
      {status.detail ? (
        <span className="truncate text-muted-foreground">{status.detail}</span>
      ) : null}
    </div>
  );
}

export function ConnectionRow({
  provider,
  enabled,
  status,
  source,
  modelSummary,
  badges,
  busy,
  checking,
  onEdit,
  onCheck,
  onDelete,
  onEnabledChange,
}: {
  provider: ExternalProviderConfig;
  enabled: boolean;
  status: ConnectionStatus;
  /** Provider label and endpoint, when they say more than the name does. */
  source: string;
  modelSummary: string;
  badges?: ReactNode;
  busy: boolean;
  checking: boolean;
  onEdit: () => void;
  onCheck: () => void;
  onDelete: () => void;
  onEnabledChange: (enabled: boolean) => void;
}) {
  const readOnly = provider.sharedByOwner === true;
  // One description line, as other settings rows have: the fix when something is wrong, else the models.
  const healthy = status.tone === "ok" || status.tone === "muted";
  const models = source ? `${source} · ${modelSummary}` : modelSummary;
  const summary = (
    <>
      <ApiProviderLogo
        providerType={provider.providerType}
        className="size-5"
        title={provider.name}
      />
      <div className="flex min-w-0 flex-1 flex-col gap-0.5">
        <div className="flex min-w-0 items-center gap-2">
          <span className="truncate text-sm font-medium text-foreground">
            {provider.name}
          </span>
          {badges}
        </div>
        <ConnectionStatusLine
          status={
            healthy
              ? { ...status, detail: models, full: provider.models.join(", ") }
              : status
          }
        />
      </div>
    </>
  );
  return (
    <div
      data-connection-id={provider.id}
      data-connection-enabled={enabled}
      className="group grid min-w-0 grid-cols-[minmax(0,1fr)_auto] items-center gap-x-6 gap-y-2 py-3 max-sm:grid-cols-1"
    >
      {readOnly ? (
        <div
          className={cn(
            "flex min-w-0 items-center gap-2.5",
            !enabled && "opacity-60",
          )}
        >
          {summary}
        </div>
      ) : (
        <button
          type="button"
          disabled={busy}
          onClick={onEdit}
          aria-label={`Edit ${provider.name}`}
          className={cn(
            "flex min-w-0 items-center gap-2.5 rounded-md text-left focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring",
            !enabled && "opacity-60",
          )}
        >
          {summary}
        </button>
      )}
      <div className="flex shrink-0 items-center justify-end gap-0.5 text-muted-foreground">
        {enabled ? (
          <Button
            type="button"
            size="icon-sm"
            variant="ghost"
            className="size-7 hover:text-foreground"
            disabled={busy || checking}
            aria-busy={checking}
            onClick={onCheck}
            title="Check connection"
            aria-label={`Check ${provider.name}`}
          >
            {checking ? (
              <Spinner className="size-4" />
            ) : (
              <HugeiconsIcon icon={Wifi02Icon} className="size-4" />
            )}
          </Button>
        ) : null}
        {readOnly ? null : (
          <>
            <Button
              type="button"
              size="icon-sm"
              variant="ghost"
              className="size-7 hover:text-foreground"
              disabled={busy}
              onClick={onEdit}
              title="Edit connection"
              aria-label={`Edit ${provider.name} settings`}
            >
              <HugeiconsIcon icon={Edit03Icon} className="size-4" />
            </Button>
            <Button
              type="button"
              size="icon-sm"
              variant="ghost"
              className="size-7 hover:text-destructive"
              disabled={busy}
              onClick={onDelete}
              title="Delete connection"
              aria-label={`Delete ${provider.name}`}
            >
              <HugeiconsIcon icon={TrashIcon} className="size-4" />
            </Button>
            <Switch
              className="ml-1.5"
              checked={enabled}
              disabled={busy}
              onCheckedChange={onEnabledChange}
              aria-label={`${enabled ? "Turn off" : "Turn on"} ${provider.name}`}
              title={
                enabled ? "Turn this connection off" : "Turn this connection on"
              }
            />
          </>
        )}
      </div>
    </div>
  );
}
