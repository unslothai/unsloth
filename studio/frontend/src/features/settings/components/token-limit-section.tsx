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
  type TokenLimitPool,
  type TokenLimitStatus,
  type TokenLimitUpdate,
  fetchTokenLimitStatus,
  updateTokenLimit,
} from "@/features/profile";
import { useT } from "@/i18n";
import { toast } from "@/lib/toast";
import { useEffect, useState } from "react";
import { SettingsRow } from "./settings-row";
import { SettingsSection } from "./settings-section";

type Draft = {
  period: TokenLimitStatus["settings"]["period"];
  mode: TokenLimitStatus["settings"]["mode"];
  method: TokenLimitStatus["settings"]["method"];
  limit: string;
  localLimit: string;
  apiLimit: string;
};

function draftFrom(status: TokenLimitStatus): Draft {
  const { settings } = status;
  return {
    period: settings.period,
    mode: settings.mode,
    method: settings.method,
    limit: String(settings.limit),
    localLimit: String(settings.localLimit),
    apiLimit: String(settings.apiLimit),
  };
}

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

const POOL_LABEL_KEYS = {
  combined: "settings.profile.tokenLimit.poolCombined",
  local: "settings.profile.tokenLimit.poolLocal",
  api: "settings.profile.tokenLimit.poolApi",
} as const satisfies Record<TokenLimitPool, string>;

export function TokenLimitSection() {
  const t = useT();
  const [status, setStatus] = useState<TokenLimitStatus | null>(null);
  const [draft, setDraft] = useState<Draft | null>(null);
  const [password, setPassword] = useState("");
  const [newPassword, setNewPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    let cancelled = false;
    void fetchTokenLimitStatus(t("settings.profile.tokenLimit.loadError"))
      .then((loaded) => {
        if (cancelled) return;
        setStatus(loaded);
        setDraft(draftFrom(loaded));
      })
      .catch((loadError) => {
        if (cancelled) return;
        setError(
          loadError instanceof Error
            ? loadError.message
            : t("settings.profile.tokenLimit.loadError"),
        );
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

  const locked = status?.settings.passwordProtected ?? false;

  const save = async (patch: TokenLimitUpdate, message?: string) => {
    setSaving(true);
    setError(null);
    try {
      const saved = await updateTokenLimit(
        {
          ...patch,
          timezone: browserTimezone(),
          ...(locked ? { password } : {}),
        },
        t("settings.profile.tokenLimit.saveError"),
      );
      setStatus(saved);
      setDraft(draftFrom(saved));
      setPassword("");
      setNewPassword("");
      toast.success(message ?? t("settings.profile.tokenLimit.saved"));
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

  const caps = draft
    ? {
        limit: parseCap(draft.limit),
        localLimit: parseCap(draft.localLimit),
        apiLimit: parseCap(draft.apiLimit),
      }
    : null;
  const capsValid =
    caps !== null &&
    (draft?.mode === "split"
      ? caps.localLimit !== null && caps.apiLimit !== null
      : caps.limit !== null);
  const needsPassword = locked && password.length === 0;
  const busy = !status || !draft || saving;

  const saveDraft = () => {
    if (!draft || !caps || !capsValid) return;
    void save({
      period: draft.period,
      mode: draft.mode,
      method: draft.method,
      ...(caps.limit !== null ? { limit: caps.limit } : {}),
      ...(caps.localLimit !== null ? { localLimit: caps.localLimit } : {}),
      ...(caps.apiLimit !== null ? { apiLimit: caps.apiLimit } : {}),
      ...(newPassword ? { newPassword } : {}),
    });
  };

  const capInput = (
    key: "limit" | "localLimit" | "apiLimit",
    label: string,
  ) => (
    <SettingsRow key={key} label={label}>
      <Input
        inputMode="numeric"
        value={draft?.[key] ?? ""}
        onChange={(event) =>
          setDraft((current) =>
            current ? { ...current, [key]: event.target.value } : current,
          )
        }
        aria-label={label}
        aria-invalid={draft ? caps?.[key] === null : undefined}
        disabled={busy}
        className="w-40"
      />
    </SettingsRow>
  );

  const selectInput = <K extends "period" | "mode" | "method">(
    key: K,
    options: readonly (readonly [Draft[K], string])[],
    description?: string,
  ) => {
    const label = t(`settings.profile.tokenLimit.${key}`);
    return (
      <SettingsRow label={label} description={description}>
        <Select
          value={draft?.[key] ?? options[0][0]}
          onValueChange={(value) =>
            setDraft((current) =>
              current ? { ...current, [key]: value } : current,
            )
          }
          disabled={busy}
        >
          <SelectTrigger
            aria-label={label}
            className={`${key === "period" ? "w-40" : "w-56"} data-[size=default]:h-control`}
          >
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {options.map(([value, text]) => (
              <SelectItem key={value} value={value}>
                {text}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </SettingsRow>
    );
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
          checked={status?.settings.enabled ?? false}
          disabled={busy || needsPassword}
          onCheckedChange={(enabled) => void save({ enabled })}
        />
      </SettingsRow>
      {status?.pools.map((pool) => (
        <SettingsRow
          key={pool.pool}
          label={t(POOL_LABEL_KEYS[pool.pool])}
          description={t("settings.profile.tokenLimit.resets", {
            time: resetsAt,
          })}
        >
          <span
            className={
              pool.reached
                ? "text-sm font-medium text-destructive"
                : "text-sm text-muted-foreground"
            }
          >
            {t("settings.profile.tokenLimit.usage", {
              used: pool.used.toLocaleString(),
              limit: pool.limit.toLocaleString(),
            })}
          </span>
        </SettingsRow>
      ))}
      {selectInput("period", [
        ["day", t("settings.profile.tokenLimit.periodDay")],
        ["week", t("settings.profile.tokenLimit.periodWeek")],
        ["month", t("settings.profile.tokenLimit.periodMonth")],
      ])}
      {selectInput(
        "mode",
        [
          ["combined", t("settings.profile.tokenLimit.modeCombined")],
          ["split", t("settings.profile.tokenLimit.modeSplit")],
        ],
        t("settings.profile.tokenLimit.modeDescription"),
      )}
      {selectInput(
        "method",
        [
          ["total", t("settings.profile.tokenLimit.methodTotal")],
          ["incremental", t("settings.profile.tokenLimit.methodIncremental")],
        ],
        t("settings.profile.tokenLimit.methodDescription"),
      )}
      {draft?.mode === "split"
        ? [
            capInput("localLimit", t("settings.profile.tokenLimit.localLimit")),
            capInput("apiLimit", t("settings.profile.tokenLimit.apiLimit")),
          ]
        : capInput("limit", t("settings.profile.tokenLimit.limit"))}
      <SettingsRow
        label={t("settings.profile.tokenLimit.newPassword")}
        description={t("settings.profile.tokenLimit.passwordDescription")}
      >
        <Input
          type="password"
          autoComplete="new-password"
          value={newPassword}
          onChange={(event) => setNewPassword(event.target.value)}
          aria-label={t("settings.profile.tokenLimit.newPassword")}
          disabled={busy}
          className="w-40"
        />
      </SettingsRow>
      {locked ? (
        <SettingsRow
          label={t("settings.profile.tokenLimit.currentPassword")}
          description={t("settings.profile.tokenLimit.passwordRequired")}
        >
          <Input
            type="password"
            autoComplete="current-password"
            value={password}
            onChange={(event) => setPassword(event.target.value)}
            aria-label={t("settings.profile.tokenLimit.currentPassword")}
            disabled={busy}
            className="w-40"
          />
        </SettingsRow>
      ) : null}
      <div className="flex flex-wrap items-center justify-end gap-2 pb-3">
        {error ? (
          <p role="alert" className="mr-auto text-xs text-destructive">
            {error}
          </p>
        ) : null}
        {locked ? (
          <Button
            variant="outline"
            size="sm"
            disabled={busy || needsPassword}
            onClick={() =>
              void save(
                { newPassword: "" },
                t("settings.profile.tokenLimit.passwordRemoved"),
              )
            }
          >
            {t("settings.profile.tokenLimit.removePassword")}
          </Button>
        ) : null}
        <Button
          size="sm"
          disabled={busy || needsPassword || !capsValid}
          onClick={saveDraft}
        >
          {t("settings.profile.tokenLimit.save")}
        </Button>
      </div>
    </SettingsSection>
  );
}
