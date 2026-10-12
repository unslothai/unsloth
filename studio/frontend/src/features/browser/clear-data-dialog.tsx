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
import { useT } from "@/i18n";
import { toast } from "@/lib/toast";
import { useBrowserHistoryStore } from "./history-store";
import { clearNativeBrowsingData, useNativeBrowser } from "./native-support";
import { useBrowserPrefsStore } from "./prefs-store";
import { clearPageCache } from "./store";

export function ClearBrowsingDataDialog({
  open,
  onOpenChange,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}) {
  const t = useT();
  const native = useNativeBrowser((state) => state.enabled);
  return (
    <AlertDialog open={open} onOpenChange={onOpenChange}>
      <AlertDialogContent>
        <AlertDialogHeader>
          <AlertDialogTitle>{t("browser.clearData.title")}</AlertDialogTitle>
          <AlertDialogDescription>
            {t(native ? "browser.native.clearDataDescription" : "browser.clearData.description")}
          </AlertDialogDescription>
        </AlertDialogHeader>
        <AlertDialogFooter>
          <AlertDialogCancel>{t("browser.clearData.cancel")}</AlertDialogCancel>
          <AlertDialogAction
            onClick={() => {
              const history = useBrowserHistoryStore.getState();
              history.clearHistory();
              history.clearDownloads();
              clearPageCache();
              // Suggestions come from history, so the ones taken off come back with it gone.
              useBrowserPrefsStore.getState().restoreSuggestions();
              clearNativeBrowsingData().then(
                () => toast.success(t("browser.clearData.done")),
                () => toast.error(t("browser.native.clearDataFailed")),
              );
            }}
          >
            {t("browser.clearData.confirm")}
          </AlertDialogAction>
        </AlertDialogFooter>
      </AlertDialogContent>
    </AlertDialog>
  );
}
