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
import { Skeleton } from "@/components/ui/skeleton";
import { useLocale, useT } from "@/i18n";
import { TrashIcon } from "@/lib/hugeicons-derived";
import { toast } from "@/lib/toast";
import { cn } from "@/lib/utils";
import {
  ArrowLeft01Icon,
  BubbleChatIcon,
  Copy01Icon,
  Download01Icon,
  Edit03Icon,
  MoreHorizontalIcon,
  StarIcon,
  UserGroupIcon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { Link, useNavigate } from "@tanstack/react-router";
import { useEffect, useMemo, useState } from "react";
import {
  deleteCharacterAction,
  duplicateCharacterAction,
  exportCharacterFile,
  startCharacterChat,
  toggleFavorite,
} from "../actions";
import { getCharacter, listCharacterChats } from "../api";
import { ConfirmDialog } from "../components/confirm-dialog";
import { Portrait } from "../components/portrait";
import { viewPicture } from "../images";
import { RoleplayText } from "../components/roleplay-text";
import { applyMacros } from "../prompt/macros";
import { useCharactersStore } from "../store";
import { relativeTime } from "../time";
import type { CharacterChatSummary } from "../types";

function errorText(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

export function CharacterProfilePage({ characterId }: { characterId: string }) {
  const t = useT();
  const locale = useLocale();
  const navigate = useNavigate();
  const character = useCharactersStore((s) => s.characters[characterId]);
  const characters = useCharactersStore((s) => s.characters);
  const chatIndex = useCharactersStore((s) => s.chatIndex);
  const [missing, setMissing] = useState(false);
  const [chats, setChats] = useState<CharacterChatSummary[] | null>(null);
  const [deleting, setDeleting] = useState(false);
  const [starting, setStarting] = useState(false);
  const [notesOpen, setNotesOpen] = useState(false);
  useAppShellReadySignal();

  useEffect(() => {
    let live = true;
    getCharacter(characterId)
      .then((loaded) => live && useCharactersStore.getState().upsert(loaded))
      .catch(() => live && setMissing(true));
    listCharacterChats(characterId)
      .then((list) => live && setChats(list))
      .catch(() => live && setChats([]));
    void useCharactersStore.getState().refreshIndex().catch(() => undefined);
    if (useCharactersStore.getState().status === "idle") void useCharactersStore.getState().refresh().catch(() => undefined);
    return () => {
      live = false;
    };
  }, [characterId]);

  const greeting = useMemo(
    () => (character ? applyMacros(character.firstMes, { char: character.name, user: "{{user}}" }).trim() : ""),
    [character],
  );

  async function chat() {
    if (!character || starting) return;
    setStarting(true);
    try {
      const threadId = await startCharacterChat([character.id]);
      void navigate({ to: "/chat", search: { thread: threadId } });
    } catch (error) {
      toast.error(t("characters.chat.startFailed"), { description: errorText(error) });
    } finally {
      setStarting(false);
    }
  }

  if (missing && !character) {
    return (
      <main className="mx-auto flex w-full max-w-5xl flex-col items-start gap-4 px-6 pt-10 font-heading sm:px-10">
        <BackLink />
        <p className="text-ui-16 text-muted-foreground">{t("characters.notFound")}</p>
      </main>
    );
  }
  if (!character) {
    return (
      <main className="mx-auto w-full max-w-5xl px-6 pt-10 font-heading sm:px-10">
        <div className="grid gap-10 md:grid-cols-[minmax(0,320px)_1fr]">
          <Skeleton className="aspect-[2/3] w-full rounded-[28px]" />
          <div className="flex flex-col gap-4">
            <Skeleton className="h-10 w-64 rounded-xl" />
            <Skeleton className="h-4 w-48 rounded-lg" />
            <Skeleton className="h-28 w-full rounded-2xl" />
          </div>
        </div>
      </main>
    );
  }

  const greetingsCount = [character.firstMes, ...character.alternateGreetings].filter((g) => g.trim()).length;
  const loreCount = character.characterBook?.entries.length ?? 0;
  const latest = chats?.find((entry) => !entry.archived) ?? null;
  const meta = [
    character.creator ? t("characters.profile.by", { creator: character.creator }) : null,
    character.characterVersion ? t("characters.profile.version", { version: character.characterVersion }) : null,
  ].filter(Boolean);

  return (
    <main className="mx-auto w-full max-w-5xl 4xl:max-w-6xl px-6 pb-16 pt-8 font-heading max-sm:px-4 sm:px-10">
      <BackLink />
      <div className="mt-5 grid gap-8 md:grid-cols-[minmax(0,300px)_1fr] lg:gap-12">
        <div className="md:sticky md:top-8 md:self-start">
          <Portrait
            id={character.id}
            name={character.name}
            hasImage={character.hasAvatar}
            rev={character.avatarRev}
            variant="card"
            label={t("characters.actions.viewPicture")}
            className="aspect-[2/3] w-full max-w-75 rounded-[28px] shadow-[0_24px_48px_-28px_rgb(0_0_0/0.55)] max-md:mx-auto max-md:max-w-60"
            onClick={character.hasAvatar ? () => viewPicture("character", character.id, character.name, character.avatarRev) : undefined}
          />
        </div>

        <div className="flex min-w-0 flex-col gap-7">
          <header className="flex flex-col gap-3">
            <div className="flex items-start gap-3">
              <h1 className="min-w-0 flex-1 text-ui-34 font-semibold leading-[1.05] tracking-[-0.028em] text-foreground">
                {character.name}
              </h1>
              <Button
                variant="ghost"
                size="icon"
                className="mt-1 rounded-full"
                aria-pressed={character.favorite}
                aria-label={character.favorite ? t("characters.unfavorite") : t("characters.favorite")}
                onClick={() => void toggleFavorite(character)}
              >
                <HugeiconsIcon
                  icon={StarIcon}
                  strokeWidth={1.75}
                  className={cn("size-5", character.favorite && "fill-amber-400 text-amber-400")}
                />
              </Button>
            </div>
            {meta.length || character.tags.length ? (
              <div className="flex flex-wrap items-center gap-2 text-ui-14 text-muted-foreground">
                {meta.length ? <span>{meta.join(" · ")}</span> : null}
                {character.tags.map((tag) => (
                  <span key={tag} className="rounded-full bg-muted px-2.5 py-0.5 text-ui-12 font-medium dark:bg-card">
                    {tag}
                  </span>
                ))}
              </div>
            ) : null}
            <div className="mt-1 flex flex-wrap items-center gap-2">
              <Button variant="primary" className="rounded-full px-5" disabled={starting} onClick={() => void chat()}>
                <HugeiconsIcon icon={BubbleChatIcon} strokeWidth={1.75} className="size-icon" />
                {t("characters.actions.chat")}
              </Button>
              {latest ? (
                <Button
                  variant="outline"
                  className="rounded-full"
                  onClick={() => void navigate({ to: "/chat", search: { thread: latest.threadId } })}
                >
                  {t("characters.actions.continue")}
                </Button>
              ) : null}
              <Button
                variant="outline"
                className="rounded-full"
                onClick={() => void navigate({ to: "/characters/$characterId/edit", params: { characterId: character.id } })}
              >
                <HugeiconsIcon icon={Edit03Icon} strokeWidth={1.75} className="size-icon" />
                {t("characters.actions.edit")}
              </Button>
              <DropdownMenu>
                <DropdownMenuTrigger asChild={true}>
                  <Button variant="ghost" size="icon" className="rounded-full" aria-label={t("characters.moreActions", { name: character.name })}>
                    <HugeiconsIcon icon={MoreHorizontalIcon} strokeWidth={1.75} className="size-icon" />
                  </Button>
                </DropdownMenuTrigger>
                <DropdownMenuContent align="start" className="min-w-52">
                  <DropdownMenuItem onSelect={() => void exportCharacterFile(character, "png", t)}>
                    <HugeiconsIcon icon={Download01Icon} strokeWidth={1.75} />
                    {t("characters.actions.exportPng")}
                  </DropdownMenuItem>
                  <DropdownMenuItem onSelect={() => void exportCharacterFile(character, "json", t)}>
                    <HugeiconsIcon icon={Download01Icon} strokeWidth={1.75} />
                    {t("characters.actions.exportJson")}
                  </DropdownMenuItem>
                  <DropdownMenuItem
                    onSelect={() =>
                      void duplicateCharacterAction(character, t).then(
                        (copy) => copy && navigate({ to: "/characters/$characterId", params: { characterId: copy.id } }),
                      )
                    }
                  >
                    <HugeiconsIcon icon={Copy01Icon} strokeWidth={1.75} />
                    {t("characters.actions.duplicate")}
                  </DropdownMenuItem>
                  <DropdownMenuSeparator />
                  <DropdownMenuItem variant="destructive" onSelect={() => setDeleting(true)}>
                    <HugeiconsIcon icon={TrashIcon} strokeWidth={1.75} />
                    {t("characters.actions.delete")}
                  </DropdownMenuItem>
                </DropdownMenuContent>
              </DropdownMenu>
            </div>
          </header>

          {greeting ? (
            <section className="relative rounded-[22px] bg-muted/70 px-5 py-4 dark:bg-card">
              <p className="line-clamp-6 whitespace-pre-wrap font-sans text-ui-15 leading-relaxed text-foreground/90 [font-weight:410]">
                <RoleplayText text={greeting} />
              </p>
              <p className="mt-2 text-ui-12 text-muted-foreground">
                {greetingsCount === 1
                  ? t("characters.profile.greetingsOne")
                  : t("characters.profile.greetingsMany", { count: greetingsCount })}
                {loreCount
                  ? ` · ${loreCount === 1 ? t("characters.profile.loreOne") : t("characters.profile.loreMany", { count: loreCount })}`
                  : ""}
              </p>
            </section>
          ) : null}

          <section className="flex flex-col gap-2">
            <h2 className="text-ui-13 font-semibold uppercase tracking-[0.06em] text-muted-foreground">
              {t("characters.profile.creatorNotes")}
            </h2>
            {character.creatorNotes.trim() ? (
              <>
                <p
                  className={cn(
                    "whitespace-pre-wrap font-sans text-ui-15 leading-relaxed text-foreground/85",
                    !notesOpen && "line-clamp-5",
                  )}
                >
                  {character.creatorNotes.trim()}
                </p>
                {character.creatorNotes.length > 320 ? (
                  <button
                    type="button"
                    className="self-start text-ui-13 font-medium text-muted-foreground underline-offset-4 hover:text-foreground hover:underline"
                    onClick={() => setNotesOpen((open) => !open)}
                  >
                    {notesOpen ? t("characters.profile.showLess") : t("characters.profile.showMore")}
                  </button>
                ) : null}
              </>
            ) : (
              <p className="text-ui-14 text-muted-foreground">{t("characters.profile.noNotes")}</p>
            )}
          </section>

          <section className="flex flex-col gap-2">
            <h2 className="text-ui-13 font-semibold uppercase tracking-[0.06em] text-muted-foreground">
              {t("characters.profile.chats")}
            </h2>
            {chats === null ? (
              <Skeleton className="h-14 w-full rounded-2xl" />
            ) : chats.length === 0 ? (
              <p className="text-ui-14 text-muted-foreground">{t("characters.profile.noChats", { name: character.name })}</p>
            ) : (
              <ul className="flex flex-col" data-testid="character-chats">
                {chats.map((entry) => {
                  const others = (chatIndex[entry.threadId] ?? [])
                    .filter((id) => id !== character.id)
                    .map((id) => characters[id]?.name)
                    .filter(Boolean);
                  return (
                    <li key={entry.threadId}>
                      <Link
                        to="/chat"
                        search={{ thread: entry.threadId }}
                        className="flex items-center gap-3 rounded-xl px-3 py-2.5 transition-colors hover:bg-muted/70 dark:hover:bg-card"
                      >
                        <span className="flex size-8 shrink-0 items-center justify-center rounded-lg bg-muted text-muted-foreground dark:bg-card">
                          <HugeiconsIcon
                            icon={entry.castSize > 1 ? UserGroupIcon : BubbleChatIcon}
                            strokeWidth={1.75}
                            className="size-4"
                          />
                        </span>
                        <span className="min-w-0 flex-1">
                          <span className="block truncate text-ui-15 text-foreground">{entry.title}</span>
                          {others.length ? (
                            <span className="block truncate text-ui-12 text-muted-foreground">
                              {t("characters.profile.groupWith", { names: others.join(", ") })}
                            </span>
                          ) : null}
                        </span>
                        <span className="shrink-0 text-ui-12 text-muted-foreground">
                          {relativeTime(locale, entry.updatedAt)}
                        </span>
                      </Link>
                    </li>
                  );
                })}
              </ul>
            )}
          </section>
        </div>
      </div>

      <ConfirmDialog
        open={deleting}
        title={t("characters.deleteTitle", { name: character.name })}
        body={t("characters.deleteBody", { name: character.name })}
        confirm={t("characters.delete")}
        cancel={t("characters.cancel")}
        onClose={() => setDeleting(false)}
        onConfirm={() => {
          setDeleting(false);
          void deleteCharacterAction(character, t).then((ok) => {
            if (ok) void navigate({ to: "/characters" });
          });
        }}
      />
    </main>
  );
}

function BackLink() {
  const t = useT();
  return (
    <Link
      to="/characters"
      className="inline-flex items-center gap-1.5 rounded-full text-ui-14 font-medium text-muted-foreground transition-colors hover:text-foreground"
    >
      <HugeiconsIcon icon={ArrowLeft01Icon} strokeWidth={1.75} className="size-4" />
      {t("characters.allCharacters")}
    </Link>
  );
}
