// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import { TabsList, TabsTrigger } from "@/components/ui/tabs";
import { useT } from "@/i18n";
import type { ReactNode } from "react";

/** Title, the Characters/Personas tabs, and the current tab's actions on the right. */
export function CharactersPageHeader({ actions, description }: { actions: ReactNode; description: string }) {
  const t = useT();
  return (
    <>
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div className="flex items-center gap-4">
          <h1 className="text-ui-30 font-semibold leading-[1.04] tracking-[-0.028em] text-foreground sm:text-ui-34">
            {t("characters.title")}
          </h1>
          <TabsList className="h-9">
            <TabsTrigger value="characters" className="px-4">
              {t("characters.tabs.characters")}
            </TabsTrigger>
            <TabsTrigger value="personas" className="px-4">
              {t("characters.tabs.personas")}
            </TabsTrigger>
          </TabsList>
        </div>
        {actions}
      </div>
      <p className="mt-4 max-w-2xl text-ui-15 text-muted-foreground">{description}</p>
    </>
  );
}

export function LoadFailed({ onRetry }: { onRetry: () => void }) {
  const t = useT();
  return (
    <div className="mt-16 flex flex-col items-center gap-3 text-center text-muted-foreground">
      <p>{t("characters.loadFailed")}</p>
      <Button variant="outline" onClick={onRetry}>
        {t("characters.retry")}
      </Button>
    </div>
  );
}
