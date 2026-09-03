// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  LazyImportBoundary,
  LazyImportFailure,
} from "@/components/lazy-import-boundary";
import { useT } from "@/i18n";
import { Suspense, lazy, useEffect, useState } from "react";
import { isImeComposing } from "./hooks/use-shortcut";
import { useSettingsDialogStore } from "./stores/settings-dialog-store";

const SettingsDialog = lazy(() =>
  import("./settings-dialog").then((module) => ({
    default: module.SettingsDialog,
  })),
);

function SettingsDialogLoading({ active }: { active: boolean }) {
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
  return null;
}

export function SettingsDialogMount() {
  const t = useT();
  const open = useSettingsDialogStore((state) => state.open);
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    if (open) setMounted(true);
  }, [open]);

  if (!mounted) return null;
  return (
    <LazyImportBoundary
      fallback={
        <LazyImportFailure
          message={t("settings.dialog.panelFailed")}
          reloadLabel={t("settings.dialog.panelReload")}
          testId="settings-dialog-load-failure"
          className="fixed top-1/2 left-1/2 z-[100] max-w-sm -translate-x-1/2 -translate-y-1/2 rounded-xl border border-border bg-popover p-5 text-popover-foreground shadow-xl"
        />
      }
    >
      <Suspense fallback={<SettingsDialogLoading active={open} />}>
        <SettingsDialog />
      </Suspense>
    </LazyImportBoundary>
  );
}
