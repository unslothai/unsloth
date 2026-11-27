// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { SegmentedPill } from "@/components/segmented-control";
import { Button } from "@/components/ui/button";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { useT } from "@/i18n";
import { copyToClipboard } from "@/lib/copy-to-clipboard";
import { Tick02Icon } from "@/lib/tick-icon";
import { cn } from "@/lib/utils";
import { Copy01Icon } from "@/lib/copy-icon";
import { HugeiconsIcon } from "@hugeicons/react";
import { useEffect, useId, useRef, useState } from "react";
import type { ApiKeyAccess } from "../api/api-keys";

const ACCESS_OPTIONS: ApiKeyAccess[] = ["full", "inference"];

function selectToken(input: HTMLInputElement | null) {
  if (!input) return;
  input.focus();
  input.select();
  // select() scrolls to the end; keep the sk-unsloth- start in view.
  input.scrollLeft = 0;
}

export function KeyRevealCard({
  rawKey,
  onDone,
  access,
  onAccessChange,
}: {
  rawKey: string;
  onDone: () => void;
  access?: ApiKeyAccess;
  onAccessChange?: (access: ApiKeyAccess) => Promise<void>;
}) {
  const t = useT();
  const pillId = useId();
  const [saving, setSaving] = useState(false);
  const pickAccess = async (option: ApiKeyAccess) => {
    if (!onAccessChange || saving || option === access) return;
    setSaving(true);
    try {
      await onAccessChange(option);
    } finally {
      setSaving(false);
    }
  };
  const [copied, setCopied] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  // Preselected so Ctrl+C works where the copy button cannot.
  useEffect(() => {
    selectToken(inputRef.current);
  }, []);

  const handleCopy = async () => {
    if (await copyToClipboard(rawKey)) {
      setCopied(true);
      setTimeout(() => setCopied(false), 1800);
    } else {
      selectToken(inputRef.current);
    }
  };

  return (
    <div className="flex min-w-0 flex-col gap-3 rounded-xl border border-border/60 bg-muted/20 p-4 dark:border-transparent dark:bg-[rgb(255_255_255_/_calc(0.06*var(--contrast-wash-gain,1)))]">
      <div className="flex items-center gap-1.5">
        <HugeiconsIcon
          icon={Tick02Icon}
          className="size-3.5 text-emerald-600 dark:text-emerald-500"
        />
        <span className="text-sm font-medium text-foreground">
          {t("settings.apiKeys.newTokenCreated")}
        </span>
      </div>
      <div className="flex h-control w-full items-center gap-3 rounded-full border border-border bg-background pr-2 pl-4 font-mono text-sm dark:border-transparent dark:bg-[rgb(255_255_255_/_calc(0.06*var(--contrast-wash-gain,1)))]">
        <input
          ref={inputRef}
          type="text"
          readOnly={true}
          value={rawKey}
          onFocus={(event) => selectToken(event.currentTarget)}
          aria-label={t("settings.apiKeys.newTokenCreated")}
          className="min-w-0 flex-1 bg-transparent text-foreground outline-none"
          data-reload-snapshot-sensitive={true}
        />
        <Button
          type="button"
          variant="ghost"
          size="icon-xs"
          onClick={handleCopy}
          className={cn(
            "shrink-0",
            copied &&
              "text-emerald-600 hover:text-emerald-600 dark:text-emerald-500",
          )}
          aria-label={
            copied
              ? t("settings.apiKeys.accessTokenCopied")
              : t("settings.apiKeys.copyAccessToken")
          }
        >
          <HugeiconsIcon
            icon={copied ? Tick02Icon : Copy01Icon}
            className="size-4"
          />
        </Button>
      </div>
      <div className="flex items-center justify-between gap-3">
        <p className="text-ui-11 text-muted-foreground">
          {t("settings.apiKeys.copyNow")}
        </p>
        <div className="flex items-center gap-2">
          {access && onAccessChange ? (
            <div className="hub-tab-toggle inline-flex h-control items-center rounded-full">
              {ACCESS_OPTIONS.map((option) => {
                const active = access === option;
                return (
                  <Tooltip key={option}>
                    <TooltipTrigger asChild={true}>
                      <button
                        type="button"
                        onClick={() => void pickAccess(option)}
                        aria-pressed={active}
                        disabled={saving}
                        className={cn(
                          "relative inline-flex h-control items-center rounded-full px-3.5 text-ui-12 font-medium transition-colors focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring disabled:opacity-60",
                          active
                            ? "text-foreground"
                            : "text-muted-foreground hover:text-foreground",
                        )}
                      >
                        {active && <SegmentedPill group={`${pillId}-access`} />}
                        {option === "full"
                          ? t("settings.apiKeys.accessFull")
                          : t("settings.apiKeys.accessInference")}
                      </button>
                    </TooltipTrigger>
                    <TooltipContent side="bottom" sideOffset={6}>
                      {option === "full"
                        ? t("settings.apiKeys.accessFullHint")
                        : t("settings.apiKeys.accessInferenceHint")}
                    </TooltipContent>
                  </Tooltip>
                );
              })}
            </div>
          ) : null}
          <Button
            type="button"
            variant="dark"
            size="sm"
            onClick={onDone}
            className="focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
          >
            {t("common.done")}
          </Button>
        </div>
      </div>
    </div>
  );
}
