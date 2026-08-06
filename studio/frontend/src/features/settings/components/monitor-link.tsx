


// The monitor has its own page, normally reached from the floating panel; this card is
// the way in from Settings.

import { Switch } from "@/components/ui/switch";
// Direct path, not the barrel: the barrel re-exports the page, defeating its lazy route.
import { useApiMonitorOverlayStore } from "@/features/api-monitor/overlay-store";
import { getApiMonitor } from "@/features/chat/api/chat-api";
import type { ApiMonitorResponse } from "@/features/chat/types/api";
import { useLocale, useT } from "@/i18n";
import { ActivityIcon, ArrowRight02Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useNavigate } from "@tanstack/react-router";
import { type ReactElement, useEffect, useState } from "react";
import { useSettingsDialogStore } from "../stores/settings-dialog-store";

export function MonitorLink(): ReactElement {
  const navigate = useNavigate();
  const locale = useLocale();
  const t = useT();
  const [data, setData] = useState<ApiMonitorResponse | null>(null);
  const autoOpen = useApiMonitorOverlayStore((s) => s.autoOpen);
  const setAutoOpen = useApiMonitorOverlayStore((s) => s.setAutoOpen);

  // One snapshot, not a poll: the live view is the monitor page.
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
    <div className="flex flex-col gap-2">
      <button
        type="button"
        onClick={() => {
          useSettingsDialogStore.getState().closeDialog();
          void navigate({ to: "/api-monitor" });
        }}
        className="flex w-full min-w-0 items-center gap-3 rounded-lg border border-border/70 bg-background px-4 py-3 text-left transition-colors hover:bg-muted/40 focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
      >
        <span className="relative flex size-8 shrink-0 items-center justify-center rounded-md border border-border/70 bg-muted/40">
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
          <span className="text-sm font-semibold text-foreground">
            {t("settings.resources.liveMonitor.apiTitle")}
          </span>
          <span className="truncate text-xs text-muted-foreground">
            {data == null
              ? t("settings.resources.liveMonitor.summary")
              : t("settings.resources.liveMonitor.status", {
                  active: active.toLocaleString(locale),
                  recent: recent.toLocaleString(locale),
                  model:
                    data.active_model ??
                    t("settings.resources.liveMonitor.noModelLoaded"),
                })}
          </span>
        </span>
        <HugeiconsIcon
          icon={ArrowRight02Icon}
          strokeWidth={1.75}
          className="ml-auto size-4 shrink-0 text-muted-foreground"
        />
      </button>

      {/* Where the panel's own "stop opening this" gets turned back on. */}
      <div className="flex items-center justify-between gap-3 rounded-lg px-1 py-1">
        <span className="flex min-w-0 flex-col">
          <span className="text-sm text-foreground">
            {t("settings.resources.liveMonitor.autoOpen")}
          </span>
          <span className="text-xs text-muted-foreground">
            {t("settings.resources.liveMonitor.autoOpenDescription")}
          </span>
        </span>
        <Switch
          checked={autoOpen}
          onCheckedChange={setAutoOpen}
          aria-label={t("settings.resources.liveMonitor.autoOpen")}
        />
      </div>
    </div>
  );
}
