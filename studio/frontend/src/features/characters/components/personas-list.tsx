// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { ConfirmDeleteDialog } from "@/components/confirm-delete-dialog";
import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Skeleton } from "@/components/ui/skeleton";
import { useEffectiveProfile } from "@/features/profile";
import { useT } from "@/i18n";
import { TrashIcon } from "@/lib/hugeicons-derived";
import { toast } from "@/lib/toast";
import { Edit03Icon, MoreHorizontalIcon, PlusSignIcon, StarIcon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useEffect, useState } from "react";
import { deletePersona, errorText, updatePersona } from "../api";
import { useCharactersStore } from "../store";
import type { Persona } from "../types";
import { CharactersPageHeader, LoadFailed } from "./characters-page-header";
import { PersonaDialog } from "./persona-dialog";
import { Portrait } from "./portrait";

export function PersonasList() {
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

  const loading = status === "idle" || status === "loading";
  return (
    <div>
      <CharactersPageHeader
        description={t("characters.personas.description")}
        actions={
          <Button variant="dark" onClick={() => setEditing("new")}>
            <HugeiconsIcon icon={PlusSignIcon} strokeWidth={1.75} className="size-icon" />
            {t("characters.personas.new")}
          </Button>
        }
      />

      {loading && personas.length === 0 ? (
        <div className="mt-8 flex flex-col gap-3">
          {Array.from({ length: 2 }).map((_, index) => (
            <Skeleton key={index} className="h-20 w-full rounded-2xl" />
          ))}
        </div>
      ) : status === "error" && personas.length === 0 ? (
        <LoadFailed onRetry={() => void refreshPersonas().catch(() => undefined)} />
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
                    {t("common.delete")}
                  </DropdownMenuItem>
                </DropdownMenuContent>
              </DropdownMenu>
            </li>
          ))}
        </ul>
      )}

      <PersonaDialog open={editing !== null} persona={editing === "new" ? null : editing} onClose={() => setEditing(null)} />
      <ConfirmDeleteDialog
        open={deleting !== null}
        title={t("characters.personas.deleteTitle", { name: deleting?.name ?? "" })}
        description={t("characters.personas.deleteBody")}
        confirmLabel={t("common.delete")}
        onOpenChange={(open) => !open && setDeleting(null)}
        onConfirm={() => {
          const target = deleting;
          setDeleting(null);
          if (target) void remove(target);
        }}
      />
    </div>
  );
}
