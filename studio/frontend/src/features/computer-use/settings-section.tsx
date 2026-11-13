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
import { SettingsRow } from "@/features/settings/components/settings-row";
import { SettingsSection } from "@/features/settings/components/settings-section";
import { useSettingsDialogStore } from "@/features/settings/stores/settings-dialog-store";
import { useT } from "@/i18n";
import { ComputerIcon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useEffect, useRef, useState } from "react";
import { toast } from "sonner";
import {
  type ComputerUseCoordinates,
  type ComputerUsePermission,
  type ComputerUseStatus,
  requestComputerUsePermission,
  updateComputerUseSettings,
} from "./api";
import { refreshComputerUseStatus, useComputerUseStore } from "./store";

const GRANTS: ComputerUsePermission[] = ["screen", "input"];

/** Settings > Security: the owner's switch for computer use, the system grants it needs, and how
 *  the model names a point. Nothing renders on a backend without computer use. */
export function ComputerUseSettings() {
  const t = useT();
  const isOwner = useIsAccountOwner();
  const status = useComputerUseStore((state) => state.status);
  const [busy, setBusy] = useState(false);
  const [confirmOpen, setConfirmOpen] = useState(false);
  const sectionRef = useRef<HTMLElement | null>(null);
  const scrollTarget = useSettingsDialogStore((s) => s.scrollTarget);
  const consumeScrollTarget = useSettingsDialogStore((s) => s.consumeScrollTarget);

  useEffect(() => {
    void refreshComputerUseStatus();
    // A grant given in the system's own settings shows once the user comes back.
    const onFocus = () => void refreshComputerUseStatus();
    window.addEventListener("focus", onFocus);
    return () => window.removeEventListener("focus", onFocus);
  }, []);

  const shown = status !== null;
  useEffect(() => {
    if (scrollTarget !== "sandbox-computer-use" || !shown) return;
    const frame = window.requestAnimationFrame(() => {
      sectionRef.current?.scrollIntoView({ block: "start", behavior: "smooth" });
      consumeScrollTarget("sandbox-computer-use");
    });
    return () => window.cancelAnimationFrame(frame);
  }, [consumeScrollTarget, scrollTarget, shown]);

  if (!status) return null;

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
  const on = status.available && status.enabled;

  return (
    <SettingsSection ref={sectionRef} title={t("computerUse.settings.title")}>
      <SettingsRow
        icon={
          <HugeiconsIcon
            icon={ComputerIcon}
            strokeWidth={1.75}
            // The size and cap-line drop of the Permissions rows above it.
            className="me-1.5 mt-[calc(0.2rem*var(--ui-font-scale,1))] size-[calc(22px*var(--ui-space-scale,1))]"
          />
        }
        iconAlign="top"
        label={t("computerUse.settings.enable")}
        description={status.available ? t("computerUse.settings.enableDescription") : status.reason}
      >
        <Switch
          aria-label={t("computerUse.settings.enable")}
          checked={on}
          disabled={busy || !isOwner || !status.available}
          onCheckedChange={(next) => {
            if (next) setConfirmOpen(true);
            else void run(() => updateComputerUseSettings({ enabled: false }));
          }}
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
                  disabled={busy || !isOwner}
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
            disabled={busy || !isOwner}
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
        </SettingsRow>
      ) : null}
      {status.available ? (
        <p className="mt-2 text-xs text-muted-foreground">
          {on && GRANTS.some((kind) => status.permissions[kind] === false)
            ? t("computerUse.settings.restartNote")
            : t("computerUse.settings.stopNote")}
        </p>
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
    </SettingsSection>
  );
}
