// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Switch } from "@/components/ui/switch";
import {
  TOKEN_LIMIT_CHANGED_EVENT,
  type TokenLimitSettings,
  type TokenLimitStatus,
  fetchTokenLimitStatus,
  updateTokenLimit,
} from "@/features/profile";
import { useT } from "@/i18n";
import { toast } from "@/lib/toast";
import { useEffect, useState } from "react";
import { SettingsRow } from "./settings-row";
import { SettingsSection } from "./settings-section";

const PERIODS = [
  ["day", "settings.profile.tokenLimit.periodDay"],
  ["week", "settings.profile.tokenLimit.periodWeek"],
  ["month", "settings.profile.tokenLimit.periodMonth"],
] as const;

function parseCap(value: string): number | null {
  const cap = Number(value.replace(/[\s,._]/g, ""));
  return Number.isInteger(cap) && cap > 0 ? cap : null;
}

function browserTimezone(): string {
  try {
    return Intl.DateTimeFormat().resolvedOptions().timeZone ?? "";
  } catch {
    return "";
  }
}

export function TokenLimitSection() {
  const t = useT();
  const [status, setStatus] = useState<TokenLimitStatus | null>(null);
  const [period, setPeriod] = useState<TokenLimitSettings["period"]>("day");
  const [limit, setLimit] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);

  const load = (next: TokenLimitStatus) => {
    setStatus(next);
    setPeriod(next.settings.period);
    setLimit(String(next.settings.limit));
  };

  useEffect(() => {
    let cancelled = false;
    void fetchTokenLimitStatus(t("settings.profile.tokenLimit.loadError"))
      .then((loaded) => {
        if (!cancelled) load(loaded);
      })
      .catch((loadError) => {
        if (!cancelled) {
          setError(
            loadError instanceof Error
              ? loadError.message
              : t("settings.profile.tokenLimit.loadError"),
          );
        }
      });
    const onChanged = (event: Event) => {
      const next = (event as CustomEvent<TokenLimitStatus>).detail;
      if (next) setStatus(next);
    };
    window.addEventListener(TOKEN_LIMIT_CHANGED_EVENT, onChanged);
    return () => {
      cancelled = true;
      window.removeEventListener(TOKEN_LIMIT_CHANGED_EVENT, onChanged);
    };
  }, [t]);

  const cap = parseCap(limit);
  const busy = !status || saving;
  const enabled = status?.settings.enabled ?? false;

  // Every save carries the visible draft, or a cap typed before flipping the switch is dropped.
  const save = async (patch: Partial<TokenLimitSettings> = {}) => {
    setSaving(true);
    setError(null);
    try {
      load(
        await updateTokenLimit(
          {
            period,
            ...(cap === null ? {} : { limit: cap }),
            timezone: browserTimezone(),
            ...patch,
          },
          t("settings.profile.tokenLimit.saveError"),
        ),
      );
      toast.success(t("settings.profile.tokenLimit.saved"));
    } catch (saveError) {
      setError(
        saveError instanceof Error
          ? saveError.message
          : t("settings.profile.tokenLimit.saveError"),
      );
    } finally {
      setSaving(false);
    }
  };

  const resetsAt = status
    ? new Date(status.resetsAt).toLocaleString(undefined, {
        dateStyle: "medium",
        timeStyle: "short",
      })
    : "";

  return (
    <SettingsSection
      title={t("settings.profile.tokenLimit.title")}
      description={t("settings.profile.tokenLimit.description")}
    >
      <SettingsRow
        label={t("settings.profile.tokenLimit.enable")}
        description={t("settings.profile.tokenLimit.enableDescription")}
      >
        <Switch
          aria-label={t("settings.profile.tokenLimit.enable")}
          checked={enabled}
          // Turning it off never needs a valid cap.
          disabled={busy || (!enabled && cap === null)}
          onCheckedChange={(next) => void save({ enabled: next })}
        />
      </SettingsRow>
      {status?.settings.enabled ? (
        <SettingsRow
          label={t("settings.profile.tokenLimit.poolCombined")}
          description={t("settings.profile.tokenLimit.resets", {
            time: resetsAt,
          })}
        >
          <span
            className={
              status.reached
                ? "text-sm font-medium text-destructive"
                : "text-sm text-muted-foreground"
            }
          >
            {t("settings.profile.tokenLimit.usage", {
              used: status.used.toLocaleString(),
              limit: status.settings.limit.toLocaleString(),
            })}
          </span>
        </SettingsRow>
      ) : null}
      <SettingsRow label={t("settings.profile.tokenLimit.period")}>
        <Select
          value={period}
          onValueChange={(value) =>
            setPeriod(value as TokenLimitSettings["period"])
          }
          disabled={busy}
        >
          <SelectTrigger
            aria-label={t("settings.profile.tokenLimit.period")}
            className="w-40 data-[size=default]:h-control"
          >
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {PERIODS.map(([value, key]) => (
              <SelectItem key={value} value={value}>
                {t(key)}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </SettingsRow>
      <SettingsRow label={t("settings.profile.tokenLimit.limit")}>
        <Input
          inputMode="numeric"
          value={limit}
          onChange={(event) => setLimit(event.target.value)}
          aria-label={t("settings.profile.tokenLimit.limit")}
          aria-invalid={status ? cap === null : undefined}
          disabled={busy}
          className="w-40"
        />
      </SettingsRow>
      <div className="flex flex-wrap items-center justify-end gap-2 pb-3">
        {error ? (
          <p role="alert" className="mr-auto text-xs text-destructive">
            {error}
          </p>
        ) : null}
        <Button size="sm" disabled={busy || cap === null} onClick={() => void save()}>
          {t("common.save")}
        </Button>
      </div>
    </SettingsSection>
  );
}
