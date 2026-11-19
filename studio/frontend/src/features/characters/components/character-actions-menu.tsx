// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { ConfirmDeleteDialog } from "@/components/confirm-delete-dialog";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { useT } from "@/i18n";
import { TrashIcon } from "@/lib/hugeicons-derived";
import { Copy01Icon, Download01Icon, Edit03Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useState, type ReactElement } from "react";
import { deleteCharacterAction, duplicateCharacterAction, exportCharacterFile } from "../actions";
import type { Character } from "../types";

type CharacterActionsMenuProps = {
  character: Character;
  /** The button that opens the menu. */
  trigger: ReactElement;
  align?: "start" | "end";
  onEdit?: () => void;
  onDuplicated?: (copy: Character) => void;
  onDeleted?: () => void;
};

/** Edit, duplicate, export and delete for one character, with the delete confirmation. */
export function CharacterActionsMenu({
  character,
  trigger,
  align = "end",
  onEdit,
  onDuplicated,
  onDeleted,
}: CharacterActionsMenuProps) {
  const t = useT();
  const [deleting, setDeleting] = useState(false);
  return (
    <>
      <DropdownMenu>
        <DropdownMenuTrigger asChild={true}>{trigger}</DropdownMenuTrigger>
        <DropdownMenuContent align={align} className="min-w-52">
          {onEdit ? (
            <DropdownMenuItem onSelect={onEdit}>
              <HugeiconsIcon icon={Edit03Icon} strokeWidth={1.75} />
              {t("characters.actions.edit")}
            </DropdownMenuItem>
          ) : null}
          <DropdownMenuItem
            onSelect={() =>
              void duplicateCharacterAction(character, t).then((copy) => {
                if (copy) onDuplicated?.(copy);
              })
            }
          >
            <HugeiconsIcon icon={Copy01Icon} strokeWidth={1.75} />
            {t("common.duplicate")}
          </DropdownMenuItem>
          <DropdownMenuItem onSelect={() => void exportCharacterFile(character, "png", t)}>
            <HugeiconsIcon icon={Download01Icon} strokeWidth={1.75} />
            {t("characters.actions.exportPng")}
          </DropdownMenuItem>
          <DropdownMenuItem onSelect={() => void exportCharacterFile(character, "json", t)}>
            <HugeiconsIcon icon={Download01Icon} strokeWidth={1.75} />
            {t("characters.actions.exportJson")}
          </DropdownMenuItem>
          <DropdownMenuSeparator />
          <DropdownMenuItem variant="destructive" onSelect={() => setDeleting(true)}>
            <HugeiconsIcon icon={TrashIcon} strokeWidth={1.75} />
            {t("common.delete")}
          </DropdownMenuItem>
        </DropdownMenuContent>
      </DropdownMenu>
      <ConfirmDeleteDialog
        open={deleting}
        title={t("characters.deleteTitle", { name: character.name })}
        description={t("characters.deleteBody", { name: character.name })}
        confirmLabel={t("common.delete")}
        onOpenChange={setDeleting}
        onConfirm={() => {
          setDeleting(false);
          void deleteCharacterAction(character, t).then((ok) => {
            if (ok) onDeleted?.();
          });
        }}
      />
    </>
  );
}
