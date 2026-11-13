// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { SegmentedPill } from "@/components/segmented-control";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { useT } from "@/i18n";
import { cn } from "@/lib/utils";
import { PlusSignIcon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useId, useState } from "react";
import { createApiKey } from "../api/api-keys";
import { apiKeyActionMessage } from "./api-key-action-message";

const EXPIRY_PRESETS = [
  { label: "Never", value: null as number | null },
  { label: "7d", value: 7 },
  { label: "30d", value: 30 },
  { label: "90d", value: 90 },
];

export function nextDefaultTokenNumber(
  names: readonly string[],
  pattern: RegExp,
): number {
  let highest = 0;
  for (const existing of names) {
    const match = pattern.exec(existing.trim());
    if (match) highest = Math.max(highest, Number(match[1]));
  }
  return highest + 1;
}

export function CreateKeyForm({
  onCreated,
  onError,
  existingNames,
}: {
  onCreated: (rawKey: string) => void;
  /** null clears an earlier failure */
  onError: (message: string | null) => void;
  existingNames: readonly string[];
}) {
  const t = useT();
  const pillId = useId();
  const [name, setName] = useState("");
  const [expiry, setExpiry] = useState<number | null>(null);
  const [loading, setLoading] = useState(false);
  const [prefix, suffix] = t("settings.apiKeys.defaultTokenName").split("{n}");
  const escape = (text = "") => text.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  const defaultName = t("settings.apiKeys.defaultTokenName", {
    n: String(
      nextDefaultTokenNumber(
        existingNames,
        new RegExp(`^${escape(prefix)}(\\d+)${escape(suffix)}$`),
      ),
    ),
  });

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (loading) return;
    setLoading(true);
    onError(null);
    try {
      const result = await createApiKey(name.trim() || defaultName, expiry);
      onCreated(result.key);
      setName("");
    } catch (cause) {
      onError(apiKeyActionMessage(t("settings.apiKeys.createError"), cause));
    } finally {
      setLoading(false);
    }
  };

  return (
    <form onSubmit={handleSubmit} className="flex flex-col gap-2">
      <div className="flex flex-wrap items-center gap-2">
        <Input
          value={name}
          onChange={(e) => setName(e.target.value)}
          placeholder={t("settings.apiKeys.tokenNamePlaceholder")}
          className="h-control min-w-[calc(200px*var(--ui-space-scale,1))] flex-1 text-sm"
          aria-label={t("settings.apiKeys.newAccessTokenName")}
        />
        <div className="hub-tab-toggle inline-flex h-control items-center rounded-full">
          {EXPIRY_PRESETS.map((p) => {
            const active = expiry === p.value;
            return (
              <Tooltip key={p.label}>
                <TooltipTrigger asChild={true}>
                  <button
                    type="button"
                    onClick={() => setExpiry(p.value)}
                    aria-pressed={active}
                    className={cn(
                      "relative inline-flex h-control items-center rounded-full px-3.5 text-ui-12 font-medium transition-colors focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring",
                      active
                        ? "text-foreground"
                        : "text-muted-foreground hover:text-foreground",
                    )}
                  >
                    {active && <SegmentedPill group={`${pillId}-expiry`} />}
                    {p.value === null ? t("settings.apiKeys.never") : p.label}
                  </button>
                </TooltipTrigger>
                <TooltipContent side="bottom" sideOffset={6}>
                  {p.value === null
                    ? t("settings.apiKeys.neverExpiresHint")
                    : t("settings.apiKeys.expiresInDaysHint", {
                        n: String(p.value),
                      })}
                </TooltipContent>
              </Tooltip>
            );
          })}
        </div>
        <Tooltip>
          <TooltipTrigger asChild={true}>
            <Button type="submit" variant="dark" size="sm" disabled={loading}>
              {!loading && (
                <HugeiconsIcon icon={PlusSignIcon} data-icon="inline-start" />
              )}
              {loading
                ? t("settings.apiKeys.creating")
                : t("settings.apiKeys.createToken")}
            </Button>
          </TooltipTrigger>
          <TooltipContent side="bottom" sideOffset={6}>
            {t("settings.apiKeys.createTokenHint", {
              name: name.trim() || defaultName,
            })}
          </TooltipContent>
        </Tooltip>
      </div>
    </form>
  );
}
