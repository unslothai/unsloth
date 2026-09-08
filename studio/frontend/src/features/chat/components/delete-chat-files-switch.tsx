// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Switch } from "@/components/ui/switch";
import { useT } from "@/i18n";

/** The file half of a chat delete, shown wherever one is confirmed. "Always delete files" makes the
 *  delete destructive beyond the chat itself, so every confirmation has to say so and let the user
 *  turn it off for this one. */
export function DeleteChatFilesSwitch({
  id,
  checked,
  onCheckedChange,
  description,
}: {
  id: string;
  checked: boolean;
  onCheckedChange: (next: boolean) => void;
  /** For a delete covering more than one chat, which reads differently. */
  description?: string;
}) {
  const t = useT();
  const label = t("shell.selection.deleteFilesLabel");
  return (
    <div className="flex items-start justify-between gap-4 rounded-md border border-border/60 bg-muted/35 px-3 py-2.5">
      <label htmlFor={id} className="min-w-0 space-y-1">
        <span className="block text-sm font-medium text-foreground">
          {label}
        </span>
        <span className="block break-words text-xs leading-5 text-muted-foreground">
          {description ?? t("shell.selection.deleteChatFilesDescription")}
        </span>
      </label>
      <Switch
        id={id}
        checked={checked}
        onCheckedChange={onCheckedChange}
        aria-label={label}
      />
    </div>
  );
}
