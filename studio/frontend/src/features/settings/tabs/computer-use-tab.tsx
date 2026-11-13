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
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Switch } from "@/components/ui/switch";
import { useIsAccountOwner } from "@/features/auth";
import { useBrowserAgentStore } from "@/features/browser/agent/agent-store";
import {
  type ComputerUseCoordinates,
  type ComputerUsePermission,
  type ComputerUseStatus,
  blockComputerUseApp,
  removeComputerUseApp,
  requestComputerUsePermission,
  resumeComputerUse,
  unblockComputerUseApp,
  updateComputerUseSettings,
} from "@/features/computer-use/api";
import { refreshComputerUseStatus, useComputerUseStore } from "@/features/computer-use/store";
import { useT } from "@/i18n";
import { CursorInWindowIcon } from "@/lib/browser-use-icon";
import { Cursor02Icon, PlusSignIcon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { type ReactNode, useEffect, useState } from "react";
import { toast } from "sonner";
import { SettingsRow } from "../components/settings-row";
import { SettingsSection } from "../components/settings-section";

const GRANTS: ComputerUsePermission[] = ["screen", "input"];
const ROW_ICON_CLASS = "me-1.5 size-7";

/** A string whose <b>…</b> part is emphasised, so each language keeps its own word order. */
function bold(text: string): ReactNode {
  return text
    .split(/<\/?b>/)
    .map((part, index) =>
      index % 2 ? (
        <strong key={part} className="font-semibold text-foreground">
          {part}
        </strong>
      ) : (
        part
      ),
    );
}

/** Apps by name, each with Remove: the bordered list of Settings > Browser's site permissions. */
function AppList({
  apps,
  disabled,
  onRemove,
  children,
}: {
  apps: string[];
  disabled: boolean;
  onRemove: (app: string) => void;
  /** The row a new name is typed into, when the list takes one. */
  children?: ReactNode;
}) {
  const t = useT();
  return (
    <div className="mt-2 flex flex-col divide-y divide-border/60 rounded-2xl border border-border/80 px-4">
      {apps.length === 0 && !children ? (
        <p className="py-3 text-sm text-muted-foreground">{t("computerUse.settings.noApps")}</p>
      ) : (
        apps.map((app) => (
          <div key={app} data-settings-label={app} className="flex items-center justify-between gap-4 py-2">
            <span className="min-w-0 truncate text-sm font-medium text-foreground">{app}</span>
            <Button
              size="sm"
              variant="ghost"
              disabled={disabled}
              aria-label={t("computerUse.settings.removeApp", { app })}
              onClick={() => onRemove(app)}
            >
              {t("computerUse.settings.remove")}
            </Button>
          </div>
        ))
      )}
      {children}
    </div>
  );
}

/** Settings > Computer use: which apps models may control on this computer, the system grants
 *  that takes, and the apps they may use without asking. */
export function ComputerUseTab() {
  const t = useT();
  const isOwner = useIsAccountOwner();
  const status = useComputerUseStore((state) => state.status);
  const browserUse = useBrowserAgentStore((state) => state.enabled);
  const [busy, setBusy] = useState(false);
  const [confirmOpen, setConfirmOpen] = useState(false);
  // The name being typed into Blocked apps, or null while that row is closed.
  const [draft, setDraft] = useState<string | null>(null);

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
  const block = () => {
    const name = draft?.trim();
    if (!name) return;
    setDraft(null);
    void run(() => blockComputerUseApp(name));
  };

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
        {status ? (
          <SettingsRow
            icon={<HugeiconsIcon icon={Cursor02Icon} strokeWidth={1.75} className={ROW_ICON_CLASS} />}
            label={t("computerUse.settings.anyApp")}
            description={status.available ? bold(t("computerUse.settings.anyAppDescription")) : status.reason}
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
          </SettingsRow>
        ) : null}
        <SettingsRow
          icon={<HugeiconsIcon icon={CursorInWindowIcon} strokeWidth={1.75} className={ROW_ICON_CLASS} />}
          label={t("computerUse.settings.browser")}
          description={t("computerUse.settings.browserDescription")}
        >
          <Switch
            aria-label={t("computerUse.settings.browser")}
            checked={browserUse}
            onCheckedChange={(next) => useBrowserAgentStore.getState().setEnabled(next)}
          />
        </SettingsRow>
        {on
          ? GRANTS.filter((kind) => status.permissions[kind] !== null).map((kind) => (
              <SettingsRow
                key={kind}
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
              </SettingsRow>
            ))
          : null}
        {on ? (
          <SettingsRow
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
              <SelectTrigger className="w-40" aria-label={t("computerUse.settings.coordinates")}>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="auto">{t("computerUse.settings.coordinatesAuto")}</SelectItem>
                <SelectItem value="pixels">{t("computerUse.settings.coordinatesPixels")}</SelectItem>
                <SelectItem value="grid">{t("computerUse.settings.coordinatesGrid")}</SelectItem>
              </SelectContent>
            </Select>
          </SettingsRow>
        ) : null}
        {on && status.stoppedChats > 0 ? (
          <SettingsRow
            label={t("computerUse.settings.stopped")}
            description={t("computerUse.settings.stoppedDescription", { count: status.stoppedChats })}
          >
            <Button size="sm" variant="secondary" disabled={locked} onClick={() => void run(resumeComputerUse)}>
              {t("computerUse.settings.allowAgain")}
            </Button>
          </SettingsRow>
        ) : null}
        {status?.available ? (
          <p className="mt-2 text-xs text-muted-foreground">
            {on && GRANTS.some((kind) => status.permissions[kind] === false)
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
          <AppList
            apps={status.allowedApps}
            disabled={locked}
            onRemove={(app) => void run(() => removeComputerUseApp(app))}
          />
        </SettingsSection>
      ) : null}
      {status?.available ? (
        <SettingsSection
          title={t("computerUse.settings.blockedApps")}
          description={t("computerUse.settings.blockedAppsDescription")}
          action={
            <Button variant="outline" size="sm" onClick={() => setDraft("")} disabled={locked || draft !== null}>
              <HugeiconsIcon icon={PlusSignIcon} strokeWidth={1.75} className="size-4" />
              {t("computerUse.settings.blockApp")}
            </Button>
          }
        >
          <AppList
            apps={status.blockedApps}
            disabled={locked}
            onRemove={(app) => void run(() => unblockComputerUseApp(app))}
          >
            {draft !== null ? (
              <form
                className="flex items-center gap-2 py-2"
                onSubmit={(event) => {
                  event.preventDefault();
                  block();
                }}
              >
                <Input
                  autoFocus={true}
                  value={draft}
                  maxLength={80}
                  aria-label={t("computerUse.settings.appName")}
                  placeholder={t("computerUse.settings.appNamePlaceholder")}
                  onChange={(event) => setDraft(event.target.value)}
                  onKeyDown={(event) => {
                    if (event.key === "Escape") setDraft(null);
                  }}
                  className="h-8 flex-1"
                />
                <Button variant="secondary" type="submit" size="sm" disabled={!draft.trim()}>
                  {t("computerUse.settings.block")}
                </Button>
                <Button type="button" variant="ghost" size="sm" onClick={() => setDraft(null)}>
                  {t("computerUse.settings.cancel")}
                </Button>
              </form>
            ) : null}
          </AppList>
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
