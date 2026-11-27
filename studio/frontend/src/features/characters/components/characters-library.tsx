// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Skeleton } from "@/components/ui/skeleton";
import { useNativeFileDrop } from "@/features/native-intents";
import { useT } from "@/i18n";
import { cn } from "@/lib/utils";
import { PlusSignIcon, Search01Icon, StarIcon, Upload01Icon, UserGroupIcon, WinkingIcon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useNavigate } from "@tanstack/react-router";
import { useEffect, useMemo, useState, type ReactNode } from "react";
import { CARD_EXTENSIONS, importCards, pickCardFiles, toggleFavorite, useOpenCharacterChat } from "../actions";
import {
  filterCharacters,
  groupTags,
  liveSelection,
  readSort,
  sortCharacters,
  writeSort,
  type SortKey,
} from "../library-filters";
import { useCharactersStore } from "../store";
import type { Character } from "../types";
import { CharacterCard } from "./character-card";
import { CharactersPageHeader } from "./characters-page-header";
import { LoadFailed } from "./load-failed";
import { GroupSelectBar } from "./group-select-bar";

const TAG_CHIPS = 10;

export function CharactersLibrary() {
  const t = useT();
  const navigate = useNavigate();
  const status = useCharactersStore((s) => s.status);
  const characters = useCharactersStore((s) => s.characters);
  const order = useCharactersStore((s) => s.order);
  const refresh = useCharactersStore((s) => s.refresh);
  const [query, setQuery] = useState("");
  const [favoritesOnly, setFavoritesOnly] = useState(false);
  const [tag, setTag] = useState<string | null>(null);
  const [sort, setSort] = useState<SortKey>(readSort);
  const [selecting, setSelecting] = useState(false);
  const [picked, setPicked] = useState<string[]>([]);

  useEffect(() => {
    void refresh().catch(() => undefined);
  }, [refresh]);

  const list = useMemo(() => order.map((id) => characters[id]).filter(Boolean), [order, characters]);
  const tagChips = useMemo(() => groupTags(list).slice(0, TAG_CHIPS), [list]);
  const shown = useMemo(
    () => sortCharacters(filterCharacters(list, { query, favoritesOnly, tag }), sort),
    [list, query, favoritesOnly, tag, sort],
  );
  const selected = useMemo(() => liveSelection(picked, characters), [picked, characters]);

  // useNativeFileDrop also catches drops in Unsloth Desktop.
  const drop = useNativeFileDrop({
    onFiles: (files) => void importCards(files, t),
    accept: CARD_EXTENSIONS,
    multiple: true,
  });

  const { open: startChat, starting } = useOpenCharacterChat();

  const newCharacter = () => void navigate({ to: "/characters/$characterId/edit", params: { characterId: "new" } });
  const loading = status === "idle" || status === "loading";

  return (
    <div className="relative min-h-[60vh]" ref={drop.ref} {...drop.dragHandlers}>
      <CharactersPageHeader
        description={t("characters.description")}
        actions={
          <div className="flex w-full min-w-0 flex-wrap items-center justify-end gap-2 sm:w-auto sm:flex-1 sm:flex-nowrap">
            <div className="relative min-w-0 flex-1 max-sm:order-last max-sm:basis-full sm:max-w-xs">
              <span className="pointer-events-none absolute left-3.5 top-1/2 -translate-y-1/2 text-muted-foreground">
                <HugeiconsIcon icon={Search01Icon} strokeWidth={1.75} className="size-4" />
              </span>
              <Input
                value={query}
                onChange={(event) => setQuery(event.target.value)}
                placeholder={t("characters.search")}
                aria-label={t("characters.search")}
                className="h-control w-full rounded-full border-none bg-muted pl-10 pr-4 shadow-none dark:bg-card"
              />
            </div>
            <Button variant="outline" onClick={() => pickCardFiles(t)} title={t("characters.importHint")}>
              <HugeiconsIcon icon={Upload01Icon} strokeWidth={1.75} className="size-icon" />
              {t("characters.import")}
            </Button>
            <Button variant="dark" onClick={newCharacter}>
              <HugeiconsIcon icon={PlusSignIcon} strokeWidth={1.75} className="size-icon" />
              {t("characters.newCharacter")}
            </Button>
          </div>
        }
      />

      {list.length > 0 ? (
        <div className="mt-6 flex items-center gap-2">
          <div className="-mx-1 flex min-w-0 flex-1 items-center gap-2 overflow-x-auto px-1 py-0.5 [mask-image:linear-gradient(to_right,black_calc(100%-2rem),transparent)] [scrollbar-width:none]">
            <FilterChip
              active={!favoritesOnly && !tag}
              onClick={() => {
                setFavoritesOnly(false);
                setTag(null);
              }}
            >
              {t("characters.allTags")}
            </FilterChip>
            <FilterChip active={favoritesOnly} onClick={() => setFavoritesOnly((value) => !value)}>
              <HugeiconsIcon icon={StarIcon} strokeWidth={1.8} className={cn("size-3.5", favoritesOnly && "fill-current")} />
              {t("characters.favorites")}
            </FilterChip>
            {tagChips.map(({ key, label }) => (
              <FilterChip key={key} active={tag === key} onClick={() => setTag(tag === key ? null : key)}>
                {label}
              </FilterChip>
            ))}
          </div>
          <div className="flex shrink-0 items-center gap-2">
            <Select
              value={sort}
              onValueChange={(value) => {
                const next = value as SortKey;
                setSort(next);
                writeSort(next);
              }}
            >
              <SelectTrigger size="sm" className="h-8 w-auto gap-2 rounded-full" aria-label={t("characters.sort.label")}>
                <SelectValue />
              </SelectTrigger>
              <SelectContent align="end">
                <SelectItem value="recent">{t("characters.sort.recent")}</SelectItem>
                <SelectItem value="name">{t("characters.sort.name")}</SelectItem>
                <SelectItem value="newest">{t("characters.sort.newest")}</SelectItem>
              </SelectContent>
            </Select>
            <Button
              variant={selecting ? "muted" : "ghost"}
              size="sm"
              className="rounded-full"
              onClick={() => {
                setSelecting((value) => !value);
                setPicked([]);
              }}
            >
              <HugeiconsIcon icon={UserGroupIcon} strokeWidth={1.75} className="size-icon" />
              {selecting ? t("common.cancel") : t("characters.select")}
            </Button>
          </div>
        </div>
      ) : null}

      {loading && list.length === 0 ? (
        <div className="mt-6 grid grid-cols-[repeat(auto-fill,minmax(168px,1fr))] gap-4 max-sm:grid-cols-2 max-sm:gap-3">
          {Array.from({ length: 6 }).map((_, index) => (
            <Skeleton key={index} className="aspect-[2/3] w-full rounded-[22px]" />
          ))}
        </div>
      ) : status === "error" && list.length === 0 ? (
        <LoadFailed onRetry={() => void refresh().catch(() => undefined)} />
      ) : list.length === 0 ? (
        <EmptyLibrary onImport={() => pickCardFiles(t)} onCreate={newCharacter} />
      ) : shown.length === 0 ? (
        <p className="mt-16 text-center text-ui-15 text-muted-foreground">{t("characters.noMatches")}</p>
      ) : (
        <div className="mt-5 grid grid-cols-[repeat(auto-fill,minmax(168px,1fr))] gap-4 max-sm:grid-cols-2 max-sm:gap-3" data-testid="character-grid">
          {shown.map((character) => (
            <CharacterCard
              key={character.id}
              character={character}
              selecting={selecting}
              selected={selected.includes(character.id)}
              onOpen={() => void navigate({ to: "/characters/$characterId", params: { characterId: character.id } })}
              onChat={() => void startChat([character.id])}
              onToggleSelect={() =>
                setPicked((current) =>
                  current.includes(character.id)
                    ? current.filter((id) => id !== character.id)
                    : [...current, character.id],
                )
              }
              onFavorite={() => void toggleFavorite(character)}
              onEdit={() => void navigate({ to: "/characters/$characterId/edit", params: { characterId: character.id } })}
            />
          ))}
        </div>
      )}

      {selecting ? (
        <GroupSelectBar
          selected={selected.map((id) => characters[id]).filter((c): c is Character => Boolean(c))}
          starting={starting}
          onStart={() => void startChat(selected)}
        />
      ) : null}

      {drop.dragging ? (
        <div className="pointer-events-none fixed inset-0 z-50 flex items-center justify-center bg-background/70 backdrop-blur-sm">
          <div className="flex flex-col items-center gap-3 rounded-3xl border-2 border-dashed border-primary/60 bg-popover px-14 py-10 text-center shadow-xl">
            <HugeiconsIcon icon={Upload01Icon} strokeWidth={1.6} className="size-8 text-primary" />
            <p className="text-ui-17 font-semibold text-foreground">{t("characters.dropToImport")}</p>
            <p className="text-ui-13 text-muted-foreground">{t("characters.importHint")}</p>
          </div>
        </div>
      ) : null}
    </div>
  );
}

function FilterChip({ active, onClick, children }: { active: boolean; onClick: () => void; children: ReactNode }) {
  return (
    <button
      type="button"
      onClick={onClick}
      aria-pressed={active}
      className={cn(
        "inline-flex h-8 shrink-0 cursor-pointer items-center gap-1.5 whitespace-nowrap rounded-full px-3.5 text-ui-13 font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring",
        active ? "bg-foreground text-background" : "bg-muted text-muted-foreground hover:text-foreground dark:bg-card",
      )}
    >
      {children}
    </button>
  );
}

function EmptyLibrary({ onImport, onCreate }: { onImport: () => void; onCreate: () => void }) {
  const t = useT();
  return (
    <div className="mt-14 flex flex-col items-center gap-4 rounded-[28px] bg-muted/60 px-6 py-16 text-center dark:bg-card">
      <span className="flex size-16 items-center justify-center rounded-2xl bg-background text-foreground shadow-sm">
        <HugeiconsIcon icon={WinkingIcon} strokeWidth={1.5} className="size-8" />
      </span>
      <div className="flex flex-col gap-1.5">
        <h2 className="text-ui-20 font-semibold text-foreground">{t("characters.emptyTitle")}</h2>
        <p className="max-w-md text-ui-15 text-muted-foreground">{t("characters.emptyBody")}</p>
      </div>
      <div className="flex flex-wrap justify-center gap-2">
        <Button variant="outline" onClick={onImport}>
          <HugeiconsIcon icon={Upload01Icon} strokeWidth={1.75} className="size-icon" />
          {t("characters.import")}
        </Button>
        <Button variant="dark" onClick={onCreate}>
          <HugeiconsIcon icon={PlusSignIcon} strokeWidth={1.75} className="size-icon" />
          {t("characters.newCharacter")}
        </Button>
      </div>
    </div>
  );
}
