// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { useT } from "@/i18n";
import { TrashIcon } from "@/lib/hugeicons-derived";
import { cn } from "@/lib/utils";
import {
  Copy01Icon,
  Download01Icon,
  Edit03Icon,
  MoreHorizontalIcon,
  StarIcon,
  Tick02Icon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import type { Character } from "../types";
import { Portrait } from "./portrait";

type CharacterCardProps = {
  character: Character;
  selecting: boolean;
  selected: boolean;
  onOpen: () => void;
  onChat: () => void;
  onToggleSelect: () => void;
  onFavorite: () => void;
  onEdit: () => void;
  onExport: (format: "png" | "json") => void;
  onDuplicate: () => void;
  onDelete: () => void;
};

/** One character in the library grid: portrait first, name and a line of context on top of it. */
export function CharacterCard({
  character,
  selecting,
  selected,
  onOpen,
  onChat,
  onToggleSelect,
  onFavorite,
  onEdit,
  onExport,
  onDuplicate,
  onDelete,
}: CharacterCardProps) {
  const t = useT();
  const subtitle =
    character.chatCount === 0
      ? t("characters.noChatsYet")
      : character.chatCount === 1
        ? t("characters.chatCountOne")
        : t("characters.chatCountMany", { count: character.chatCount });
  return (
    <div
      className={cn(
        "group/char relative isolate overflow-hidden rounded-[22px] bg-muted shadow-[0_1px_0_rgb(0_0_0/0.04)] ring-1 ring-black/[0.04] transition-[transform,box-shadow] duration-200 dark:ring-white/[0.06]",
        "hover:-translate-y-0.5 hover:shadow-[0_14px_32px_-18px_rgb(0_0_0/0.45)]",
        selected && "ring-2 ring-primary dark:ring-primary",
      )}
      data-character-card={character.id}
    >
      <button
        type="button"
        className="block w-full cursor-pointer text-left focus-visible:outline-none"
        onClick={selecting ? onToggleSelect : onOpen}
        aria-pressed={selecting ? selected : undefined}
        aria-label={character.name}
      >
        <Portrait
          id={character.id}
          name={character.name}
          hasImage={character.hasAvatar}
          rev={character.avatarRev}
          variant="card"
          className="aspect-[2/3] w-full"
        />
        <span className="pointer-events-none absolute inset-x-0 bottom-0 h-2/5 bg-gradient-to-t from-black/75 via-black/30 to-transparent" />
        <span className="pointer-events-none absolute inset-x-0 bottom-0 flex flex-col gap-0.5 px-3.5 pb-3 text-white">
          <span className="line-clamp-2 font-heading text-ui-16 font-semibold leading-tight [text-shadow:0_1px_2px_rgb(0_0_0/0.35)]">
            {character.name}
          </span>
          <span className="text-ui-12 text-white/75">{subtitle}</span>
        </span>
      </button>

      {selecting ? (
        <span
          aria-hidden={true}
          className={cn(
            "pointer-events-none absolute left-2.5 top-2.5 flex size-6 items-center justify-center rounded-full border-2 border-white/90 bg-black/35 text-white shadow-sm backdrop-blur-[2px]",
            selected && "border-primary bg-primary",
          )}
        >
          {selected ? <HugeiconsIcon icon={Tick02Icon} strokeWidth={2.4} className="size-3.5" /> : null}
        </span>
      ) : (
        <>
          <button
            type="button"
            onClick={onFavorite}
            aria-label={character.favorite ? t("characters.unfavorite") : t("characters.favorite")}
            aria-pressed={character.favorite}
            className={cn(
              "absolute left-2 top-2 flex size-8 cursor-pointer items-center justify-center rounded-full text-white transition-opacity focus-visible:opacity-100 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-white/80",
              character.favorite
                ? "opacity-100"
                : "bg-black/35 opacity-0 backdrop-blur-[2px] group-hover/char:opacity-100",
            )}
          >
            <HugeiconsIcon
              icon={StarIcon}
              strokeWidth={1.8}
              className={cn("size-4", character.favorite && "fill-amber-300 text-amber-300 drop-shadow")}
            />
          </button>
          <div className="absolute right-2 top-2 flex items-center gap-1 opacity-0 transition-opacity focus-within:opacity-100 group-hover/char:opacity-100 has-[[aria-expanded=true]]:opacity-100">
            <DropdownMenu>
              <DropdownMenuTrigger asChild={true}>
                <button
                  type="button"
                  aria-label={t("characters.moreActions", { name: character.name })}
                  className="flex size-8 cursor-pointer items-center justify-center rounded-full bg-black/30 text-white backdrop-blur-[2px] hover:bg-black/45 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-white/80"
                >
                  <HugeiconsIcon icon={MoreHorizontalIcon} strokeWidth={1.8} className="size-4" />
                </button>
              </DropdownMenuTrigger>
              <DropdownMenuContent align="end" className="min-w-48">
                <DropdownMenuItem onSelect={onEdit}>
                  <HugeiconsIcon icon={Edit03Icon} strokeWidth={1.75} />
                  {t("characters.actions.edit")}
                </DropdownMenuItem>
                <DropdownMenuItem onSelect={onDuplicate}>
                  <HugeiconsIcon icon={Copy01Icon} strokeWidth={1.75} />
                  {t("characters.actions.duplicate")}
                </DropdownMenuItem>
                <DropdownMenuItem onSelect={() => onExport("png")}>
                  <HugeiconsIcon icon={Download01Icon} strokeWidth={1.75} />
                  {t("characters.actions.exportPng")}
                </DropdownMenuItem>
                <DropdownMenuItem onSelect={() => onExport("json")}>
                  <HugeiconsIcon icon={Download01Icon} strokeWidth={1.75} />
                  {t("characters.actions.exportJson")}
                </DropdownMenuItem>
                <DropdownMenuSeparator />
                <DropdownMenuItem variant="destructive" onSelect={onDelete}>
                  <HugeiconsIcon icon={TrashIcon} strokeWidth={1.75} />
                  {t("characters.actions.delete")}
                </DropdownMenuItem>
              </DropdownMenuContent>
            </DropdownMenu>
          </div>
          <Button
            type="button"
            size="sm"
            variant="primary"
            onClick={onChat}
            className="absolute bottom-3 right-3 rounded-full opacity-0 shadow-md transition-opacity focus-visible:opacity-100 group-hover/char:opacity-100"
          >
            {t("characters.chatAction")}
          </Button>
        </>
      )}
    </div>
  );
}
