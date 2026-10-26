// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Switch } from "@/components/ui/switch";
// Direct path, not the barrel: the barrel re-exports the page, defeating its lazy route.
import { useApiMonitorOverlayStore } from "@/features/api-monitor/overlay-store";
import { getApiMonitor } from "@/features/chat/api/chat-api";
import type { ApiMonitorResponse } from "@/features/chat/types/api";
import { useLocale, useT } from "@/i18n";
import { ActivityIcon } from "@hugeicons/core-free-icons";
import { ArrowRightIcon } from "lucide-react";
import { HugeiconsIcon } from "@hugeicons/react";
import { useNavigate } from "@tanstack/react-router";
import { type ReactElement, useEffect, useState } from "react";
import { useSettingsDialogStore } from "../stores/settings-dialog-store";
import { SettingsRow } from "./settings-row";
import { SettingsSection } from "./settings-section";

export function MonitorLink(): ReactElement {
  const navigate = useNavigate();
  const locale = useLocale();
  const t = useT();
  const [data, setData] = useState<ApiMonitorResponse | null>(null);
  const autoOpen = useApiMonitorOverlayStore((s) => s.autoOpen);
  const setAutoOpen = useApiMonitorOverlayStore((s) => s.setAutoOpen);

  useEffect(() => {
    let cancelled = false;
    void getApiMonitor()
      .then((next) => {
        if (!cancelled) setData(next);
      })
      .catch(() => undefined);
    return () => {
      cancelled = true;
    };
  }, []);

  const active = data?.active_requests ?? 0;
  const recent = data?.entries.length ?? 0;

  return (
    <SettingsSection title={t("settings.resources.liveMonitor.apiTitle")}>
      <div className="flex flex-col pt-3">
        <button
          type="button"
          onClick={() => {
            useSettingsDialogStore.getState().closeDialog();
            void navigate({ to: "/api-monitor" });
          }}
          className="flex w-full min-w-0 items-center gap-3 rounded-lg border border-border/70 bg-background px-4 py-3 text-left transition-colors hover:bg-muted/40 focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
        >
          <span className="relative flex size-8 shrink-0 items-center justify-center icon-disc">
            <HugeiconsIcon
              icon={ActivityIcon}
              strokeWidth={1.75}
              className="size-4"
            />
            {active > 0 ? (
              <span className="absolute right-1 top-1 size-2 rounded-full bg-emerald-500" />
            ) : null}
          </span>
          <span className="flex min-w-0 flex-col">
            <span className="truncate text-sm font-semibold text-foreground">
              {t("settings.resources.liveMonitor.summary")}
            </span>
            <span className="min-h-[1lh] truncate text-xs text-muted-foreground">
              {data == null
                ? null
                : t("settings.resources.liveMonitor.status", {
                    active: active.toLocaleString(locale),
                    recent: recent.toLocaleString(locale),
                    model:
                      data.active_model ??
                      t("settings.resources.liveMonitor.noModelLoaded"),
                  })}
            </span>
          </span>
          <ArrowRightIcon
            strokeWidth={1.75}
            className="ml-auto size-4 shrink-0 text-muted-foreground"
          />
        </button>

        {/* Where the panel's own "stop opening this" gets turned back on. */}
        <SettingsRow
          label={t("settings.resources.liveMonitor.autoOpen")}
          description={t("settings.resources.liveMonitor.autoOpenDescription")}
        >
          <Switch
            checked={autoOpen}
            onCheckedChange={setAutoOpen}
            aria-label={t("settings.resources.liveMonitor.autoOpen")}
          />
        </SettingsRow>
      </div>
    </SettingsSection>
  );
}
