// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { DropdownMenuItem, DropdownMenuSeparator } from "@/components/ui/dropdown-menu";
import { useT } from "@/i18n";
import {
  COMBINED_EXPORT_FORMATS_LIST,
  type ConvExportFormat,
  EXPORT_FORMATS_LIST,
  exportBulkConversationsMerged,
  exportBulkConversationsSeparate,
} from "../prompt-storage/prompt-storage-dialog";

export function BulkExportItems({
  onExport,
}: {
  onExport: (format: ConvExportFormat, merged: boolean) => void;
}) {
  const t = useT();
  return (
    <>
      {COMBINED_EXPORT_FORMATS_LIST.map(({ fmt, label }) => (
        <DropdownMenuItem key={`merged-${fmt}`} onSelect={() => onExport(fmt, true)}>
          {label} {t("settings.chat.exportCombinedSuffix")}
        </DropdownMenuItem>
      ))}
      <DropdownMenuSeparator className="mx-3" />
      {EXPORT_FORMATS_LIST.map(({ fmt, label }) => (
        <DropdownMenuItem key={`separate-${fmt}`} onSelect={() => onExport(fmt, false)}>
          {label} {t("settings.chat.exportPerChatSuffix")}
        </DropdownMenuItem>
      ))}
    </>
  );
}

export async function exportThreads(
  threadIds: string[],
  format: ConvExportFormat,
  merged: boolean,
  name: string,
): Promise<void> {
  const stem = name.replace(/[^\p{L}\p{N}_-]+/gu, "_").slice(0, 48);
  const basename = `${stem || "chats"}-${new Date().toISOString().slice(0, 10)}`;
  if (merged || threadIds.length === 1) {
    await exportBulkConversationsMerged(threadIds, format, basename);
  } else {
    await exportBulkConversationsSeparate(threadIds, format, basename);
  }
}
