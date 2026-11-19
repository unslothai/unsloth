// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useAppShellReadySignal } from "@/components/app-readiness";
import { Tabs } from "@/components/ui/tabs";
import { useNavigate, useSearch } from "@tanstack/react-router";
import { CharactersLibrary } from "../components/character-library";
import { PersonasList } from "../components/personas-list";

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
        {tab === "characters" ? <CharactersLibrary /> : <PersonasList />}
      </Tabs>
    </main>
  );
}
