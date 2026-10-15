// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useIsAccountOwner } from "@/features/auth";
import { useT } from "@/i18n";
import { toast } from "@/lib/toast";
import { useEffect } from "react";
import { claimHubSourceNotice } from "../api/hub-settings";
import { useSettingsDialogStore } from "../stores/settings-dialog-store";

export function useHubSourceNotice(): void {
  const t = useT();
  const isOwner = useIsAccountOwner();
  const openDialog = useSettingsDialogStore((state) => state.openDialog);

  useEffect(() => {
    if (!isOwner) return;
    // Not cancelled on cleanup: the grant is spent once claimed.
    void claimHubSourceNotice().then((granted) => {
      if (!granted) return;
      toast.info(t("settings.general.hub.autoSourceTitle"), {
        description: t("settings.general.hub.autoSourceDescription"),
        duration: Number.POSITIVE_INFINITY,
        action: {
          label: t("settings.general.hub.autoSourceAction"),
          onClick: () => openDialog("general", { scrollTarget: "general-hub" }),
        },
      });
    });
  }, [isOwner, openDialog, t]);
}
