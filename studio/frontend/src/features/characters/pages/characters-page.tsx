// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useAppShellReadySignal } from "@/components/app-readiness";
import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Skeleton } from "@/components/ui/skeleton";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { useNativeFileDrop } from "@/features/native-intents";
import { useEffectiveProfile } from "@/features/profile";
import { useT } from "@/i18n";
import { TrashIcon } from "@/lib/hugeicons-derived";
import { toast } from "@/lib/toast";
import { cn } from "@/lib/utils";
import {
  Edit03Icon,
  WinkingIcon,
  MoreHorizontalIcon,
  PlusSignIcon,
  Search01Icon,
  StarIcon,
  Upload01Icon,
  UserGroupIcon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useNavigate, useSearch } from "@tanstack/react-router";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  deleteCharacterAction,
  duplicateCharacterAction,
  exportCharacterFile,
  importCards,
  startCharacterChat,
  toggleFavorite,
} from "../actions";
import { deletePersona, updatePersona } from "../api";
import { CharacterCard } from "../components/character-card";
import { ConfirmDialog } from "../components/confirm-dialog";
import { PersonaDialog } from "../components/persona-dialog";
import { Portrait } from "../components/portrait";
import { useCharactersStore } from "../store";
import type { Character, Persona } from "../types";

type SortKey = "recent" | "name" | "newest";
const CARD_ACCEPT = ".png,.json,.charx,image/png,application/json";
const CARD_EXTENSIONS = ".png,.json,.charx";
const SORT_KEY = "unsloth_characters_sort";

function readSort(): SortKey {
  try {
    const value = window.localStorage.getItem(SORT_KEY);
    return value === "name" || value === "newest" ? value : "recent";
  } catch {
    return "recent";
  }
}

function errorText(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

export function CharactersPage() {
  const navigate = useNavigate();
  const search = useSearch({ from: "/characters" });
  const tab = search.view === "personas" ? "personas" : "characters";
  useAppShellReadySignal();

  return (
    <main className="mx-auto w-full max-w-6xl 4xl:max-w-7xl px-6 pb-16 pt-8 font-heading max-sm:px-4 sm:px-10">
      <Tabs
        value={tab}
        onValueChange={(value) =>
          void navigate({ to: "/characters", search: value === "personas" ? { view: "personas" } : {}, replace: true })
        }
      >
        {tab === "characters" ? <CharactersLibrary openImport={search.import === true} /> : <PersonasList />}
      </Tabs>
    </main>
  );
}

function PageTabs() {
  const t = useT();
  return (
    <TabsList className="h-9">
      <TabsTrigger value="characters" className="px-4">
        {t("characters.tabs.characters")}
      </TabsTrigger>
      <TabsTrigger value="personas" className="px-4">
        {t("characters.tabs.personas")}
      </TabsTrigger>
    </TabsList>
  );
}

function CharactersLibrary({ openImport }: { openImport: boolean }) {
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
  const [selected, setSelected] = useState<string[]>([]);
  const [deleting, setDeleting] = useState<Character | null>(null);
  const [starting, setStarting] = useState(false);
  const fileRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    void refresh().catch(() => undefined);
  }, [refresh]);
  useEffect(() => {
    if (!openImport) return;
    fileRef.current?.click();
    void navigate({ to: "/characters", search: {}, replace: true });
  }, [openImport, navigate]);

  const list = useMemo(() => order.map((id) => characters[id]).filter(Boolean), [order, characters]);
  // Cards spell the same tag differently ("Fantasy", "fantasy"), so tags group by lowercase and each chip
  // shows the most common spelling.
  const tagCounts = useMemo(() => {
    const groups = new Map<string, { count: number; spellings: Map<string, number> }>();
    for (const character of list) {
      for (const value of character.tags) {
        const key = value.toLowerCase();
        const group = groups.get(key) ?? { count: 0, spellings: new Map<string, number>() };
        group.count += 1;
        group.spellings.set(value, (group.spellings.get(value) ?? 0) + 1);
        groups.set(key, group);
      }
    }
    return [...groups.entries()]
      .map(([key, group]) => ({
        key,
        count: group.count,
        label: [...group.spellings.entries()].sort((a, b) => b[1] - a[1])[0][0],
      }))
      .sort((a, b) => b.count - a.count || a.label.localeCompare(b.label));
  }, [list]);
  const shown = useMemo(() => {
    const needle = query.trim().toLowerCase();
    const filtered = list.filter(
      (character) =>
        (!favoritesOnly || character.favorite) &&
        (!tag || character.tags.some((value) => value.toLowerCase() === tag)) &&
        (!needle ||
          character.name.toLowerCase().includes(needle) ||
          character.tags.some((value) => value.toLowerCase().includes(needle)) ||
          character.creator.toLowerCase().includes(needle)),
    );
    if (sort === "name") return [...filtered].sort((a, b) => a.name.localeCompare(b.name));
    if (sort === "newest") return [...filtered].sort((a, b) => b.createdAt - a.createdAt);
    return filtered;
  }, [list, query, favoritesOnly, tag, sort]);

  // Drag-and-drop anywhere on the page, the desktop app included.
  const drop = useNativeFileDrop({
    onFiles: (files) => runImportRef.current(files),
    accept: CARD_EXTENSIONS,
    multiple: true,
  });
  const runImportRef = useRef<(files: File[]) => Promise<void>>(async () => {});

  const runImport = useCallback(
    async (files: File[]) => {
      if (files.length === 0) return;
      const imported = await importCards(files, t);
      if (imported.length > 1) toast.success(t("characters.importedMany", { count: imported.length }));
    },
    [t],
  );

  runImportRef.current = runImport;

  const startChat = useCallback(
    async (ids: string[]) => {
      if (starting) return;
      setStarting(true);
      try {
        const threadId = await startCharacterChat(ids);
        void navigate({ to: "/chat", search: { thread: threadId } });
      } catch (error) {
        toast.error(t("characters.chat.startFailed"), { description: errorText(error) });
      } finally {
        setStarting(false);
      }
    },
    [navigate, starting, t],
  );

  const loading = status === "idle" || status === "loading";
  const tagChips = tagCounts.slice(0, 10);

  return (
    <div className="relative min-h-[60vh]" ref={drop.ref} {...drop.dragHandlers}>
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div className="flex items-center gap-4">
          <h1 className="text-ui-30 font-semibold leading-[1.04] tracking-[-0.028em] text-foreground sm:text-ui-34">
            {t("characters.title")}
          </h1>
          <PageTabs />
        </div>
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
          <Button variant="outline" onClick={() => fileRef.current?.click()} title={t("characters.importHint")}>
            <HugeiconsIcon icon={Upload01Icon} strokeWidth={1.75} className="size-icon" />
            {t("characters.import")}
          </Button>
          <Button variant="dark" onClick={() => void navigate({ to: "/characters/$characterId/edit", params: { characterId: "new" } })}>
            <HugeiconsIcon icon={PlusSignIcon} strokeWidth={1.75} className="size-icon" />
            {t("characters.newCharacter")}
          </Button>
          <input
            ref={fileRef}
            type="file"
            multiple={true}
            accept={CARD_ACCEPT}
            className="hidden"
            data-testid="character-import-input"
            onChange={(event) => {
              const files = [...(event.target.files ?? [])];
              event.target.value = "";
              void runImport(files);
            }}
          />
        </div>
      </div>
      <p className="mt-4 max-w-2xl text-ui-15 text-muted-foreground">{t("characters.description")}</p>

      {list.length > 0 ? (
        <div className="mt-6 flex items-center gap-2">
          <div className="-mx-1 flex min-w-0 flex-1 items-center gap-2 overflow-x-auto px-1 py-0.5 [mask-image:linear-gradient(to_right,black_calc(100%-2rem),transparent)] [scrollbar-width:none]">
          <FilterChip active={!favoritesOnly && !tag} onClick={() => { setFavoritesOnly(false); setTag(null); }}>
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
                try {
                  window.localStorage.setItem(SORT_KEY, next);
                } catch {
                  // the order is a convenience
                }
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
                setSelected([]);
              }}
            >
              <HugeiconsIcon icon={UserGroupIcon} strokeWidth={1.75} className="size-icon" />
              {selecting ? t("characters.cancelSelection") : t("characters.select")}
            </Button>
          </div>
        </div>
      ) : null}

      {loading && list.length === 0 ? (
        <div className="mt-6 grid grid-cols-[repeat(auto-fill,minmax(168px,1fr))] gap-4">
          {Array.from({ length: 6 }).map((_, index) => (
            <Skeleton key={index} className="aspect-[2/3] w-full rounded-[22px]" />
          ))}
        </div>
      ) : status === "error" && list.length === 0 ? (
        <div className="mt-16 flex flex-col items-center gap-3 text-center text-muted-foreground">
          <p>{t("characters.loadFailed")}</p>
          <Button variant="outline" onClick={() => void refresh().catch(() => undefined)}>
            {t("characters.retry")}
          </Button>
        </div>
      ) : list.length === 0 ? (
        <EmptyLibrary onImport={() => fileRef.current?.click()} />
      ) : shown.length === 0 ? (
        <p className="mt-16 text-center text-ui-15 text-muted-foreground">{t("characters.noMatches")}</p>
      ) : (
        <div className="mt-5 grid grid-cols-[repeat(auto-fill,minmax(168px,1fr))] gap-4" data-testid="character-grid">
          {shown.map((character) => (
            <CharacterCard
              key={character.id}
              character={character}
              selecting={selecting}
              selected={selected.includes(character.id)}
              onOpen={() => void navigate({ to: "/characters/$characterId", params: { characterId: character.id } })}
              onChat={() => void startChat([character.id])}
              onToggleSelect={() =>
                setSelected((current) =>
                  current.includes(character.id)
                    ? current.filter((id) => id !== character.id)
                    : [...current, character.id],
                )
              }
              onFavorite={() => void toggleFavorite(character)}
              onEdit={() => void navigate({ to: "/characters/$characterId/edit", params: { characterId: character.id } })}
              onExport={(format) => void exportCharacterFile(character, format, t)}
              onDuplicate={() => void duplicateCharacterAction(character, t)}
              onDelete={() => setDeleting(character)}
            />
          ))}
        </div>
      )}

      {selecting ? (
        <div className="pointer-events-none sticky bottom-6 z-20 mt-8 flex justify-center">
          <div className="pointer-events-auto flex items-center gap-3 rounded-full bg-popover py-2 pl-5 pr-2 shadow-[0_12px_32px_-12px_rgb(0_0_0/0.35)] ring-1 ring-border">
            <span className="text-ui-14 text-muted-foreground">
              {selected.length < 2
                ? t("characters.groupNeedsTwo")
                : t("characters.selectedCount", { count: selected.length })}
            </span>
            <div className="flex -space-x-2">
              {selected.slice(0, 5).map((id) => {
                const character = characters[id];
                return character ? (
                  <Portrait
                    key={id}
                    id={id}
                    name={character.name}
                    hasImage={character.hasAvatar}
                    rev={character.avatarRev}
                    className="size-7 rounded-full text-[0.8rem] ring-2 ring-popover"
                  />
                ) : null;
              })}
            </div>
            <Button
              variant="primary"
              className="rounded-full"
              disabled={selected.length < 2 || starting}
              onClick={() => void startChat(selected)}
            >
              <HugeiconsIcon icon={UserGroupIcon} strokeWidth={1.75} className="size-icon" />
              {t("characters.startGroupChat")}
            </Button>
          </div>
        </div>
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

      <ConfirmDialog
        open={deleting !== null}
        title={t("characters.deleteTitle", { name: deleting?.name ?? "" })}
        body={t("characters.deleteBody", { name: deleting?.name ?? "" })}
        confirm={t("characters.delete")}
        cancel={t("characters.cancel")}
        onClose={() => setDeleting(null)}
        onConfirm={() => {
          const target = deleting;
          setDeleting(null);
          if (target) void deleteCharacterAction(target, t);
        }}
      />
    </div>
  );
}

function FilterChip({ active, onClick, children }: { active: boolean; onClick: () => void; children: React.ReactNode }) {
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

function EmptyLibrary({ onImport }: { onImport: () => void }) {
  const t = useT();
  const navigate = useNavigate();
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
        <Button variant="dark" onClick={() => void navigate({ to: "/characters/$characterId/edit", params: { characterId: "new" } })}>
          <HugeiconsIcon icon={PlusSignIcon} strokeWidth={1.75} className="size-icon" />
          {t("characters.newCharacter")}
        </Button>
      </div>
    </div>
  );
}

function PersonasList() {
  const t = useT();
  const personas = useCharactersStore((s) => s.personas);
  const status = useCharactersStore((s) => s.personasStatus);
  const refreshPersonas = useCharactersStore((s) => s.refreshPersonas);
  const profile = useEffectiveProfile();
  const [editing, setEditing] = useState<Persona | "new" | null>(null);
  const [deleting, setDeleting] = useState<Persona | null>(null);

  useEffect(() => {
    void refreshPersonas().catch(() => undefined);
  }, [refreshPersonas]);

  async function makeDefault(persona: Persona) {
    try {
      useCharactersStore.getState().upsertPersona(await updatePersona(persona.id, { isDefault: true }));
      await refreshPersonas();
    } catch (error) {
      toast.error(errorText(error));
    }
  }

  async function remove(persona: Persona) {
    try {
      await deletePersona(persona.id);
      useCharactersStore.getState().removePersona(persona.id);
      await refreshPersonas();
    } catch (error) {
      toast.error(errorText(error));
    }
  }

  return (
    <div>
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div className="flex items-center gap-4">
          <h1 className="text-ui-30 font-semibold leading-[1.04] tracking-[-0.028em] text-foreground sm:text-ui-34">
            {t("characters.title")}
          </h1>
          <PageTabs />
        </div>
        <Button variant="dark" onClick={() => setEditing("new")}>
          <HugeiconsIcon icon={PlusSignIcon} strokeWidth={1.75} className="size-icon" />
          {t("characters.personas.new")}
        </Button>
      </div>
      <p className="mt-4 max-w-2xl text-ui-15 text-muted-foreground">{t("characters.personas.description")}</p>

      {status !== "ready" && personas.length === 0 ? (
        <div className="mt-8 flex flex-col gap-3">
          {Array.from({ length: 2 }).map((_, index) => (
            <Skeleton key={index} className="h-20 w-full rounded-2xl" />
          ))}
        </div>
      ) : personas.length === 0 ? (
        <div className="mt-8 flex flex-col items-start gap-3 rounded-2xl bg-muted/60 px-6 py-6 dark:bg-card">
          <p className="text-ui-15 text-muted-foreground">
            {t("characters.personas.empty", { name: profile.addressName || profile.displayTitle })}
          </p>
          <Button variant="outline" onClick={() => setEditing("new")}>
            {t("characters.personas.emptyAction")}
          </Button>
        </div>
      ) : (
        <ul className="mt-8 flex flex-col gap-2" data-testid="persona-list">
          {personas.map((persona) => (
            <li
              key={persona.id}
              className="group/persona flex items-center gap-4 rounded-2xl px-4 py-3 transition-colors hover:bg-muted/70 dark:hover:bg-card"
            >
              <Portrait
                kind="persona"
                id={persona.id}
                name={persona.name}
                hasImage={persona.hasAvatar}
                rev={persona.avatarRev}
                className="size-12 rounded-2xl text-[1.2rem]"
              />
              <button type="button" className="min-w-0 flex-1 cursor-pointer text-left" onClick={() => setEditing(persona)}>
                <span className="flex items-center gap-2">
                  <span className="truncate text-ui-16 font-semibold text-foreground">{persona.name}</span>
                  {persona.isDefault ? (
                    <span className="rounded-full bg-primary/12 px-2 py-0.5 text-ui-11 font-semibold text-primary">
                      {t("characters.personas.default")}
                    </span>
                  ) : null}
                </span>
                <span className="line-clamp-1 text-ui-14 text-muted-foreground">{persona.description}</span>
              </button>
              <DropdownMenu>
                <DropdownMenuTrigger asChild={true}>
                  <Button variant="ghost" size="icon-sm" aria-label={t("characters.moreActions", { name: persona.name })}>
                    <HugeiconsIcon icon={MoreHorizontalIcon} strokeWidth={1.75} className="size-icon" />
                  </Button>
                </DropdownMenuTrigger>
                <DropdownMenuContent align="end">
                  <DropdownMenuItem onSelect={() => setEditing(persona)}>
                    <HugeiconsIcon icon={Edit03Icon} strokeWidth={1.75} />
                    {t("characters.personas.edit")}
                  </DropdownMenuItem>
                  {!persona.isDefault ? (
                    <DropdownMenuItem onSelect={() => void makeDefault(persona)}>
                      <HugeiconsIcon icon={StarIcon} strokeWidth={1.75} />
                      {t("characters.personas.makeDefault")}
                    </DropdownMenuItem>
                  ) : null}
                  <DropdownMenuSeparator />
                  <DropdownMenuItem variant="destructive" onSelect={() => setDeleting(persona)}>
                    <HugeiconsIcon icon={TrashIcon} strokeWidth={1.75} />
                    {t("characters.personas.delete")}
                  </DropdownMenuItem>
                </DropdownMenuContent>
              </DropdownMenu>
            </li>
          ))}
        </ul>
      )}

      <PersonaDialog
        open={editing !== null}
        persona={editing === "new" ? null : editing}
        onClose={() => setEditing(null)}
      />
      <ConfirmDialog
        open={deleting !== null}
        title={t("characters.personas.deleteTitle", { name: deleting?.name ?? "" })}
        body={t("characters.personas.deleteBody")}
        confirm={t("characters.delete")}
        cancel={t("characters.cancel")}
        onClose={() => setDeleting(null)}
        onConfirm={() => {
          const target = deleting;
          setDeleting(null);
          if (target) void remove(target);
        }}
      />
    </div>
  );
}
