// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";
import { Button } from "@/components/ui/button";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Switch } from "@/components/ui/switch";
import { useIsAccountOwner } from "@/features/auth";
import { useBrowserAgentStore } from "@/features/browser/agent/agent-store";
import {
  type ComputerUseCoordinates,
  type ComputerUsePermission,
  type ComputerUseStatus,
  removeComputerUseApp,
  requestComputerUsePermission,
  updateComputerUseSettings,
} from "@/features/computer-use/api";
import { refreshComputerUseStatus, useComputerUseStore } from "@/features/computer-use/store";
import { useT } from "@/i18n";
import { CursorInWindowIcon } from "@/lib/browser-use-icon";
import { ComputerIcon, KeyboardIcon, Target02Icon, ViewIcon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon, type IconSvgElement } from "@hugeicons/react";
import { type ReactNode, useEffect, useState } from "react";
import { toast } from "sonner";
import { SettingsSection } from "../components/settings-section";

const GRANTS: { kind: ComputerUsePermission; icon: IconSvgElement }[] = [
  { kind: "screen", icon: ViewIcon },
  { kind: "input", icon: KeyboardIcon },
];

/** One row of a card: what it controls on the left, the control on the right. */
function ControlRow({
  icon,
  label,
  description,
  children,
}: {
  icon: IconSvgElement;
  label: string;
  description: ReactNode;
  children: ReactNode;
}) {
  return (
    <div data-settings-label={label} className="flex flex-wrap items-center justify-end gap-x-6 gap-y-2 py-3.5">
      <div className="flex min-w-[calc(11rem*var(--ui-space-scale,1))] flex-1 basis-0 items-center gap-3.5">
        <span className="flex size-10 shrink-0 items-center justify-center rounded-xl bg-muted text-foreground">
          <HugeiconsIcon icon={icon} strokeWidth={1.75} className="size-[calc(22px*var(--ui-space-scale,1))]" />
        </span>
        <div className="flex min-w-0 flex-col gap-0.5">
          <span className="text-sm font-medium text-foreground">{label}</span>
          <span className="text-xs text-muted-foreground">{description}</span>
        </div>
      </div>
      {children}
    </div>
  );
}

const CARD_CLASS = "mt-2 flex flex-col divide-y divide-border/60 rounded-2xl border border-border/80 px-4";

/** Settings > Computer use: which apps models may control on this computer, the system grants
 *  that takes, and the apps they may use without asking. */
export function ComputerUseTab() {
  const t = useT();
  const isOwner = useIsAccountOwner();
  const status = useComputerUseStore((state) => state.status);
  const browserUse = useBrowserAgentStore((state) => state.enabled);
  const [busy, setBusy] = useState(false);
  const [confirmOpen, setConfirmOpen] = useState(false);

  useEffect(() => {
    void refreshComputerUseStatus();
    // A grant given in the system's own settings shows once the user comes back.
    const onFocus = () => void refreshComputerUseStatus();
    window.addEventListener("focus", onFocus);
    return () => window.removeEventListener("focus", onFocus);
  }, []);

  const run = async (work: () => Promise<ComputerUseStatus>) => {
    setBusy(true);
    try {
      useComputerUseStore.getState().setStatus(await work());
    } catch (error) {
      toast.error(error instanceof Error ? error.message : t("computerUse.settings.saveFailed"));
    } finally {
      setBusy(false);
    }
  };
  const on = !!status && status.available && status.enabled;
  const locked = busy || !isOwner;

  return (
    <div className="settings-page">
      <header className="flex min-w-0 flex-col gap-1">
        <h1 data-settings-label={t("computerUse.settings.title")} className="text-xl font-semibold font-heading">
          {t("computerUse.settings.title")}
        </h1>
        <p data-settings-label={t("computerUse.settings.description")} className="text-xs text-muted-foreground">
          {t("computerUse.settings.description")}
        </p>
      </header>
      <SettingsSection title={t("computerUse.settings.control")}>
        <div className={CARD_CLASS}>
          {status ? (
            <ControlRow
              icon={ComputerIcon}
              label={t("computerUse.settings.anyApp")}
              description={status.available ? t("computerUse.settings.anyAppDescription") : status.reason}
            >
              <Switch
                aria-label={t("computerUse.settings.anyApp")}
                checked={on}
                disabled={locked || !status.available}
                onCheckedChange={(next) => {
                  if (next) setConfirmOpen(true);
                  else void run(() => updateComputerUseSettings({ enabled: false }));
                }}
              />
            </ControlRow>
          ) : null}
          <ControlRow
            icon={CursorInWindowIcon}
            label={t("computerUse.settings.browser")}
            description={t("computerUse.settings.browserDescription")}
          >
            <Switch
              aria-label={t("computerUse.settings.browser")}
              checked={browserUse}
              onCheckedChange={(next) => useBrowserAgentStore.getState().setEnabled(next)}
            />
          </ControlRow>
          {on
            ? GRANTS.filter(({ kind }) => status.permissions[kind] !== null).map(({ kind, icon }) => (
                <ControlRow
                  key={kind}
                  icon={icon}
                  label={t(`computerUse.settings.${kind}`)}
                  description={t(`computerUse.settings.${kind}Description`)}
                >
                  {status.permissions[kind] ? (
                    <span className="text-sm text-muted-foreground">{t("computerUse.settings.allowed")}</span>
                  ) : (
                    <Button
                      size="sm"
                      variant="secondary"
                      disabled={locked}
                      onClick={() => void run(() => requestComputerUsePermission(kind))}
                    >
                      {t("computerUse.settings.allow")}
                    </Button>
                  )}
                </ControlRow>
              ))
            : null}
          {on ? (
            <ControlRow
              icon={Target02Icon}
              label={t("computerUse.settings.coordinates")}
              description={t("computerUse.settings.coordinatesDescription")}
            >
              <Select
                value={status.coordinates}
                disabled={locked}
                onValueChange={(value) =>
                  void run(() => updateComputerUseSettings({ coordinates: value as ComputerUseCoordinates }))
                }
              >
                <SelectTrigger className="w-44" aria-label={t("computerUse.settings.coordinates")}>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="auto">{t("computerUse.settings.coordinatesAuto")}</SelectItem>
                  <SelectItem value="pixels">{t("computerUse.settings.coordinatesPixels")}</SelectItem>
                  <SelectItem value="grid">{t("computerUse.settings.coordinatesGrid")}</SelectItem>
                </SelectContent>
              </Select>
            </ControlRow>
          ) : null}
        </div>
        {status?.available ? (
          <p className="mt-2 text-xs text-muted-foreground">
            {on && GRANTS.some(({ kind }) => status.permissions[kind] === false)
              ? t("computerUse.settings.restartNote")
              : t("computerUse.settings.stopNote")}
          </p>
        ) : null}
      </SettingsSection>
      {status?.available ? (
        <SettingsSection
          title={t("computerUse.settings.allowedApps")}
          description={t("computerUse.settings.allowedAppsDescription")}
        >
          <div className={CARD_CLASS}>
            {status.allowedApps.length === 0 ? (
              <p className="py-6 text-center text-sm text-muted-foreground">{t("computerUse.settings.noApps")}</p>
            ) : (
              status.allowedApps.map((app) => (
                <div key={app} data-settings-label={app} className="flex items-center justify-between gap-4 py-2.5">
                  <span className="min-w-0 truncate text-sm font-medium text-foreground">{app}</span>
                  <Button
                    size="sm"
                    variant="ghost"
                    disabled={locked}
                    aria-label={t("computerUse.settings.removeApp", { app })}
                    onClick={() => void run(() => removeComputerUseApp(app))}
                  >
                    {t("computerUse.settings.remove")}
                  </Button>
                </div>
              ))
            )}
          </div>
        </SettingsSection>
      ) : null}
      <AlertDialog open={confirmOpen} onOpenChange={setConfirmOpen}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>{t("computerUse.settings.confirmTitle")}</AlertDialogTitle>
            <AlertDialogDescription className="text-pretty leading-relaxed">
              {t("computerUse.settings.confirmBody")}
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel variant="muted">{t("computerUse.settings.cancel")}</AlertDialogCancel>
            <AlertDialogAction
              variant="destructive"
              onClick={() => {
                setConfirmOpen(false);
                void run(() => updateComputerUseSettings({ enabled: true }));
              }}
            >
              {t("computerUse.settings.confirm")}
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </div>
  );
}
