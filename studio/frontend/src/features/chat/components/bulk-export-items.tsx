// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  DropdownMenuGroup,
  DropdownMenuLabel,
  DropdownMenuSeparator,
} from "@/components/ui/dropdown-menu";
import { useT } from "@/i18n";
import type { ReactNode } from "react";
import {
  COMBINED_EXPORT_FORMATS_LIST,
  type ConvExportFormat,
  EXPORT_FORMATS_LIST,
  exportBulkConversationsMerged,
  exportBulkConversationsSeparate,
} from "../prompt-storage/prompt-storage-dialog";
import { ExportFormatItems } from "./export-format-items";

export function BulkExportItems({
  onExport,
  combinedLead,
  subContentClassName,
}: {
  onExport: (format: ConvExportFormat, merged: boolean) => void;
  combinedLead?: ReactNode;
  subContentClassName?: string;
}) {
  const t = useT();
  const options = (list: typeof EXPORT_FORMATS_LIST) =>
    list.map(({ fmt, label }) => ({ key: fmt, label }));
  return (
    <>
      <DropdownMenuGroup>
        <DropdownMenuLabel className="pb-1 pt-2 text-ui-11 font-medium">
          {t("projectsPage.combined")}
        </DropdownMenuLabel>
        {combinedLead}
        <ExportFormatItems
          options={options(COMBINED_EXPORT_FORMATS_LIST)}
          onSelect={(fmt) => onExport(fmt, true)}
          subContentClassName={subContentClassName}
        />
      </DropdownMenuGroup>
      <DropdownMenuSeparator className="mx-3" />
      <DropdownMenuGroup>
        <DropdownMenuLabel className="pb-1 pt-2 text-ui-11 font-medium">
          {t("projectsPage.perChat")}
        </DropdownMenuLabel>
        <ExportFormatItems
          options={options(EXPORT_FORMATS_LIST)}
          onSelect={(fmt) => onExport(fmt, false)}
          subContentClassName={subContentClassName}
        />
      </DropdownMenuGroup>
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
