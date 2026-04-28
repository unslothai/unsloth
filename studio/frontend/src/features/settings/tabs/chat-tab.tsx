// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import {
  clearAllChats,
  countAllChats,
} from "@/features/chat/utils/clear-all-chats";
import { downloadChatExport } from "@/features/chat/utils/export-chat-history";
import { Delete02Icon, Download02Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useEffect, useState } from "react";
import { useI18n } from "@/features/i18n";
import { SettingsRow } from "../components/settings-row";
import { SettingsSection } from "../components/settings-section";

export function ChatTab() {
  const { t } = useI18n();
  const [confirmOpen, setConfirmOpen] = useState(false);
  const [count, setCount] = useState<number | null>(null);
  const [exporting, setExporting] = useState(false);
  const [clearing, setClearing] = useState(false);

  useEffect(() => {
    void countAllChats().then(setCount);
  }, []);

  const handleExport = async () => {
    setExporting(true);
    try {
      await downloadChatExport();
    } finally {
      setExporting(false);
    }
  };

  const handleClear = async () => {
    setClearing(true);
    try {
      await clearAllChats();
      setCount(0);
      setConfirmOpen(false);
    } finally {
      setClearing(false);
    }
  };

  return (
    <div className="flex flex-col gap-6">
      <header className="flex flex-col gap-1">
        <h1 className="text-lg font-semibold font-heading">
          {t("settings.chat.title")}
        </h1>
        <p className="text-xs text-muted-foreground">
          {t("settings.chat.subtitle")}
        </p>
      </header>

      <SettingsSection title={t("settings.chat.data")}>
        <SettingsRow
          label={t("settings.chat.export.label")}
          description={t("settings.chat.export.description")}
        >
          <Button
            variant="outline"
            size="sm"
            onClick={handleExport}
            disabled={exporting || count === 0}
          >
            <HugeiconsIcon icon={Download02Icon} className="size-3.5 mr-1.5" />
            {exporting ? t("settings.chat.export.exporting") : t("settings.chat.export.cta")}
          </Button>
        </SettingsRow>

        <SettingsRow
          destructive
          label={t("settings.chat.clear.label")}
          description={
            count === null
              ? t("settings.chat.clear.descLoading")
              : count === 0
                ? t("settings.chat.clear.descEmpty")
                : t("settings.chat.clear.descCount")
                    .replace("{count}", String(count))
                    .replace("{suffix}", count === 1 ? "" : "s")
          }
        >
          <Button
            variant="outline"
            size="sm"
            onClick={() => setConfirmOpen(true)}
            disabled={count === 0}
            className="text-destructive hover:text-destructive hover:border-destructive/60"
          >
            <HugeiconsIcon icon={Delete02Icon} className="size-3.5 mr-1.5" />
            {t("settings.chat.clear.cta")}
          </Button>
        </SettingsRow>
      </SettingsSection>

      <Dialog open={confirmOpen} onOpenChange={setConfirmOpen}>
        <DialogContent className="max-w-md">
          <DialogHeader>
            <DialogTitle>
              {t("settings.chat.clear.confirmTitle")
                .replace("{count}", String(count ?? 0))
                .replace("{suffix}", count === 1 ? "" : "s")}
            </DialogTitle>
            <DialogDescription>
              {t("settings.chat.clear.confirmDescription")}
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button variant="outline" onClick={() => setConfirmOpen(false)}>
              {t("settings.chat.clear.cancel")}
            </Button>
            <Button
              onClick={handleClear}
              disabled={clearing}
              className="bg-destructive hover:bg-destructive/90 text-destructive-foreground"
            >
              {clearing
                ? t("settings.chat.clear.clearing")
                : t("settings.chat.clear.confirmCta")
                    .replace("{count}", String(count ?? 0))
                    .replace("{suffix}", count === 1 ? "" : "s")}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}
