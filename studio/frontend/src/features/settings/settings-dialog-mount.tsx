// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  LazyImportBoundary,
  LazyImportFailure,
} from "@/components/lazy-import-boundary";
import { useT } from "@/i18n";
import {
  Suspense,
  lazy,
  useEffect,
  useLayoutEffect,
  useRef,
  useState,
} from "react";
import { isImeComposing } from "./hooks/use-shortcut";

import { useMonitorOverlayStore } from "./stores/monitor-overlay-store";
import { useSettingsDialogStore } from "./stores/settings-dialog-store";

const SettingsDialog = lazy(() =>
  import("./settings-dialog").then((module) => ({
    default: module.SettingsDialog,
  })),
);

function SettingsDialogLoading({ active }: { active: boolean }) {
  const t = useT();

  const dialogRef = useRef<HTMLDialogElement>(null);
  const closeDialog = useSettingsDialogStore((state) => state.closeDialog);
  useEffect(() => {
    if (!active) return;
    const cancel = (event: KeyboardEvent) => {
      if (event.key !== "Escape" || isImeComposing(event)) return;
      event.preventDefault();
      event.stopPropagation();
      closeDialog();
    };
    window.addEventListener("keydown", cancel, true);
    return () => window.removeEventListener("keydown", cancel, true);
  }, [active, closeDialog]);
  useLayoutEffect(() => {
    const dialog = dialogRef.current;
    if (!active || !dialog) return;
    dialog.showModal();
    dialog.focus();
    return () => dialog.close();
  }, [active]);

  if (!active) return null;
  return (
    <dialog
      ref={dialogRef}
      tabIndex={-1}
      className="fixed inset-0 z-[100] m-0 grid h-full max-h-none w-full max-w-none place-items-center border-0 bg-black/50 p-4"
      data-testid="settings-dialog-loading"
      aria-label={t("common.loading")}
    >
      <div className="rounded-xl border border-border bg-popover px-6 py-4 text-popover-foreground shadow-xl">
        {t("common.loading")}
      </div>
    </dialog>
  );
}

export function SettingsDialogMount({ active }: { active: boolean }) {
  const t = useT();

  const closeDialog = useSettingsDialogStore((state) => state.closeDialog);
  const open = useSettingsDialogStore((state) => state.open);
  const monitorOpen = useMonitorOverlayStore((state) => state.isOpen);

  const setMonitorOpen = useMonitorOverlayStore((state) => state.setIsOpen);
  const [mounted, setMounted] = useState(open || monitorOpen);

  useEffect(() => {
    if (open || monitorOpen) setMounted(true);
  }, [open, monitorOpen]);

  if (!active || !mounted) return null;
  return (
    <LazyImportBoundary
      fallback={
        open || monitorOpen ? (
          <LazyImportFailure
            message={t("settings.dialog.panelFailed")}
            reloadLabel={t("settings.dialog.panelReload")}
            dismissLabel={t("common.close")}
            onDismiss={() => {
              closeDialog();
              setMonitorOpen(false);
            }}
            testId="settings-dialog-load-failure"
            className="fixed top-1/2 left-1/2 z-[100] max-w-sm -translate-x-1/2 -translate-y-1/2 rounded-xl border border-border bg-popover p-5 text-popover-foreground shadow-xl"
          />
        ) : null
      }
    >
      <Suspense fallback={<SettingsDialogLoading active={open} />}>
        <SettingsDialog />
      </Suspense>
    </LazyImportBoundary>
  );
}
