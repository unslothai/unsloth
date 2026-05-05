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
import { SettingsRow } from "../components/settings-row";
import { SettingsSection } from "../components/settings-section";

export function ChatTab() {
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
        <h1 className="text-lg font-semibold font-heading">Chat</h1>
        <p className="text-xs text-muted-foreground">
          Manage your chat history stored on this device.
        </p>
      </header>

      <SettingsSection title="Data">
        <SettingsRow
          label="Export chat history"
          description="Download all chats and messages as a JSON file."
        >
          <Button
            variant="outline"
            size="sm"
            onClick={handleExport}
            disabled={exporting || count === 0}
          >
            <HugeiconsIcon icon={Download02Icon} className="size-3.5 mr-1.5" />
            {exporting ? "Exporting…" : "Export"}
          </Button>
        </SettingsRow>

        <SettingsRow
          destructive
          label="Clear all chats"
          description={
            count === null
              ? "Permanently delete every chat on this device."
              : count === 0
                ? "No chats to clear."
                : `Permanently delete all ${count} chat${count === 1 ? "" : "s"} on this device.`
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
            Clear chats
          </Button>
        </SettingsRow>
      </SettingsSection>

      <Dialog open={confirmOpen} onOpenChange={setConfirmOpen}>
        <DialogContent className="max-w-md">
          <DialogHeader>
            <DialogTitle>
              Clear {count ?? 0} chat{count === 1 ? "" : "s"}?
            </DialogTitle>
            <DialogDescription>
              This permanently deletes every chat and message stored on this device. This cannot be undone.
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button variant="outline" onClick={() => setConfirmOpen(false)}>
              Cancel
            </Button>
            <Button
              onClick={handleClear}
              disabled={clearing}
              className="bg-destructive hover:bg-destructive/90 text-destructive-foreground"
            >
              {clearing ? "Clearing…" : `Clear ${count ?? 0} chat${count === 1 ? "" : "s"}`}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}
