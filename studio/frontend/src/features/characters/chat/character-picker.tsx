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
import { Input } from "@/components/ui/input";
import { Switch } from "@/components/ui/switch";
import { useT } from "@/i18n";
import { cn } from "@/lib/utils";
import { PlusSignIcon, Search01Icon, Tick02Icon, Upload01Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useNavigate } from "@tanstack/react-router";
import { useEffect, useMemo, useState } from "react";
import { Portrait } from "../components/portrait";
import { useCharactersStore } from "../store";

type PickerProps = {
  open: boolean;
  /** start: open a new chat with the pick; add: add the pick to this chat's cast. */
  mode: "start" | "add";
  exclude?: string[];
  onClose: () => void;
  onPick: (ids: string[]) => void;
};

export function CharacterPickerDialog({ open, mode, exclude = [], onClose, onPick }: PickerProps) {
  const t = useT();
  const navigate = useNavigate();
  const characters = useCharactersStore((s) => s.characters);
  const order = useCharactersStore((s) => s.order);
  const status = useCharactersStore((s) => s.status);
  const [query, setQuery] = useState("");
  const [several, setSeveral] = useState(mode === "add");
  const [picked, setPicked] = useState<string[]>([]);

  useEffect(() => {
    if (!open) return;
    setQuery("");
    setPicked([]);
    setSeveral(mode === "add");
    if (useCharactersStore.getState().status !== "ready") void useCharactersStore.getState().refresh().catch(() => undefined);
  }, [open, mode]);

  const list = useMemo(() => {
    const needle = query.trim().toLowerCase();
    return order
      .map((id) => characters[id])
      .filter((c) => c && !exclude.includes(c.id))
      .filter((c) => !needle || c.name.toLowerCase().includes(needle) || c.tags.some((tag) => tag.toLowerCase().includes(needle)));
  }, [order, characters, exclude, query]);

  function choose(id: string) {
    if (!several) {
      onPick([id]);
      return;
    }
    setPicked((current) => (current.includes(id) ? current.filter((x) => x !== id) : [...current, id]));
  }

  const empty = status === "ready" && order.length === 0;
  return (
    <Dialog open={open} onOpenChange={(next) => !next && onClose()}>
      <DialogContent className="flex max-h-[min(720px,88dvh)] flex-col gap-4 sm:max-w-xl">
        <DialogHeader>
          <DialogTitle>{mode === "add" ? t("characters.picker.add") : t("characters.picker.title")}</DialogTitle>
          {mode === "start" ? <DialogDescription>{t("characters.description")}</DialogDescription> : null}
        </DialogHeader>
        {empty ? (
          <p className="py-6 text-center text-ui-15 text-muted-foreground">{t("characters.picker.empty")}</p>
        ) : (
          <>
            <div className="relative">
              <span className="pointer-events-none absolute left-3.5 top-1/2 -translate-y-1/2 text-muted-foreground">
                <HugeiconsIcon icon={Search01Icon} strokeWidth={1.75} className="size-4" />
              </span>
              <Input
                value={query}
                autoFocus={true}
                onChange={(event) => setQuery(event.target.value)}
                placeholder={t("characters.picker.search")}
                aria-label={t("characters.picker.search")}
                className="h-control rounded-full border-none bg-muted pl-10 shadow-none dark:bg-card"
              />
            </div>
            {mode === "start" ? (
              <label className="flex items-center justify-between gap-3 px-1 text-ui-13 text-muted-foreground">
                {t("characters.picker.group")}
                <Switch checked={several} onCheckedChange={(next) => { setSeveral(next); setPicked([]); }} />
              </label>
            ) : null}
            <ul className="-mx-2 min-h-0 flex-1 overflow-y-auto px-2" data-testid="character-picker-list">
              {list.length === 0 ? (
                <li className="py-6 text-center text-ui-14 text-muted-foreground">{t("characters.picker.noMatches")}</li>
              ) : (
                list.map((character) => {
                  const selected = picked.includes(character.id);
                  return (
                    <li key={character.id}>
                      <button
                        type="button"
                        onClick={() => choose(character.id)}
                        aria-pressed={several ? selected : undefined}
                        className={cn(
                          "flex w-full cursor-pointer items-center gap-3 rounded-2xl px-2 py-2 text-left transition-colors hover:bg-muted dark:hover:bg-card",
                          selected && "bg-primary/10 hover:bg-primary/15",
                        )}
                      >
                        <Portrait
                          id={character.id}
                          name={character.name}
                          hasImage={character.hasAvatar}
                          rev={character.avatarRev}
                          className="size-11 rounded-[14px] text-[0.9rem]"
                        />
                        <span className="min-w-0 flex-1">
                          <span className="block truncate text-ui-15 font-semibold text-foreground">{character.name}</span>
                          <span className="block truncate text-ui-13 text-muted-foreground">
                            {character.tags.slice(0, 4).join(" · ") || character.creator}
                          </span>
                        </span>
                        {several ? (
                          <span
                            className={cn(
                              "flex size-5 shrink-0 items-center justify-center rounded-full border-2 border-border",
                              selected && "border-primary bg-primary text-primary-foreground",
                            )}
                          >
                            {selected ? <HugeiconsIcon icon={Tick02Icon} strokeWidth={2.4} className="size-3" /> : null}
                          </span>
                        ) : null}
                      </button>
                    </li>
                  );
                })
              )}
            </ul>
          </>
        )}
        <DialogFooter className="flex-row flex-wrap items-center gap-2 sm:justify-between">
          <div className="flex gap-1">
            <Button
              variant="ghost"
              size="sm"
              onClick={() => {
                onClose();
                void navigate({ to: "/characters", search: { import: true } });
              }}
            >
              <HugeiconsIcon icon={Upload01Icon} strokeWidth={1.75} className="size-icon" />
              {t("characters.picker.import")}
            </Button>
            <Button
              variant="ghost"
              size="sm"
              onClick={() => {
                onClose();
                void navigate({ to: "/characters/$characterId/edit", params: { characterId: "new" } });
              }}
            >
              <HugeiconsIcon icon={PlusSignIcon} strokeWidth={1.75} className="size-icon" />
              {t("characters.picker.create")}
            </Button>
          </div>
          {several ? (
            <Button
              variant="dark"
              disabled={picked.length < (mode === "add" ? 1 : 2)}
              onClick={() => onPick(picked)}
            >
              {mode === "add" ? t("characters.picker.add") : t("characters.picker.startGroup", { count: picked.length })}
            </Button>
          ) : null}
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
