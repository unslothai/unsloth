// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { useT } from "@/i18n";
import { copyToClipboard } from "@/lib/copy-to-clipboard";
import { toast } from "@/lib/toast";
import { Tick02Icon } from "@/lib/tick-icon";
import { cn } from "@/lib/utils";
import { Copy01Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useEffect, useRef, useState } from "react";

export function KeyRevealCard({
  rawKey,
  onDone,
}: {
  rawKey: string;
  onDone: () => void;
}) {
  const t = useT();
  const [copied, setCopied] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    const input = inputRef.current;
    if (!input) return;
    input.focus({ preventScroll: true });
    input.select();
  }, []);

  const handleCopy = async () => {
    if (await copyToClipboard(rawKey)) {
      setCopied(true);
      setTimeout(() => setCopied(false), 1800);
      return;
    }
    inputRef.current?.focus({ preventScroll: true });
    inputRef.current?.select();
    toast.error(t("settings.apiKeys.copyAccessTokenFailed"));
  };

  return (
    <div className="flex flex-col gap-2 rounded-lg border border-emerald-500/30 bg-emerald-500/5 p-3">
      <div className="flex items-center gap-1.5">
        <HugeiconsIcon
          icon={Tick02Icon}
          className="size-3.5 text-emerald-600 dark:text-emerald-500"
        />
        <span className="text-xs font-medium text-emerald-700 dark:text-emerald-500">
          {t("settings.apiKeys.newTokenCreated")}
        </span>
      </div>
      <div
        className={cn(
          "flex w-full items-stretch gap-2 rounded-md border border-border bg-muted/40 p-2 transition-colors",
          copied && "border-emerald-500/40 bg-emerald-500/10",
        )}
      >
        <Input
          ref={inputRef}
          readOnly
          value={rawKey}
          onFocus={(event) => event.currentTarget.select()}
          className="h-auto min-h-9 flex-1 rounded-md border-0 bg-transparent px-2 py-1.5 font-mono text-sm shadow-none focus-visible:ring-0 dark:bg-transparent"
          data-reload-snapshot-sensitive
          aria-label={t("settings.apiKeys.copyAccessToken")}
        />
        <Button
          type="button"
          variant="outline"
          size="sm"
          onClick={() => void handleCopy()}
          className={cn(
            "shrink-0 self-center",
            copied && "border-emerald-500/40 text-emerald-700 dark:text-emerald-500",
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
      <div className="flex items-center justify-between gap-3 pt-0.5">
        <p className="text-ui-11 text-muted-foreground">
          {t("settings.apiKeys.copyNow")}
        </p>
        <Button
          type="button"
          size="sm"
          onClick={onDone}
          className="focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
        >
          {t("common.done")}
        </Button>
      </div>
    </div>
  );
}
