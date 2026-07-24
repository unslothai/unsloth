// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import { Delete02Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useT } from "@/i18n";

export function SidebarBulkSelectionBar({
  count,
  onDelete,
  onClear,
}: {
  count: number;
  onDelete: () => void;
  onClear: () => void;
}) {
  const t = useT();
  if (count <= 0) return null;

  return (
    <div className="mb-1 flex items-center justify-between gap-2 rounded-full border border-border/60 bg-muted/40 px-2.5 py-1">
      <button
        type="button"
        onClick={onClear}
        className="min-w-0 truncate text-xs font-medium text-muted-foreground transition-colors hover:text-foreground"
      >
        {t("shell.selection.selectedCount", { count })}
      </button>
      <Button
        type="button"
        size="sm"
        variant="destructive"
        className="h-6 shrink-0 rounded-full px-2.5 text-xs"
        onClick={onDelete}
      >
        <HugeiconsIcon icon={Delete02Icon} strokeWidth={1.75} className="size-3.5" />
        {t("common.delete")}
      </Button>
    </div>
  );
}
