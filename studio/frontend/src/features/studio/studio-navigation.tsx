// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { TabsList, TabsTrigger } from "@/components/ui/tabs";
import { useT } from "@/i18n";
import { cn } from "@/lib/utils";
import type { ReactElement } from "react";
import type { TrainSubTab } from "./use-studio-navigation";

export function TrainSubNav({
  value,
  isTrainingRunning,
  showTrainingView,
}: {
  value: TrainSubTab;
  isTrainingRunning: boolean;
  showTrainingView: boolean;
}): ReactElement {
  const t = useT();
  const items: ReadonlyArray<{
    value: TrainSubTab;
    label: string;
    disabled: boolean;
  }> = [
    {
      value: "configure",
      label: t("studio.tabs.configure"),
      disabled: isTrainingRunning,
    },
    {
      value: "current-run",
      label: t("studio.tabs.currentRun"),
      disabled: !showTrainingView,
    },
    { value: "history", label: t("studio.tabs.history"), disabled: false },
  ];
  return (
    <TabsList
      unstyled={true}
      className="flex items-center gap-6 text-ui-13 tracking-nav"
    >
      {items.map((item) => {
        const active = value === item.value;
        return (
          <TabsTrigger
            key={item.value}
            value={item.value}
            disabled={item.disabled}
            indicatorClassName="hidden"
            className={cn(
              "relative h-9 flex-none select-none rounded-none border-0 px-0 py-0 text-ui-13 transition-colors disabled:cursor-not-allowed disabled:opacity-40",
              "after:pointer-events-none after:absolute after:inset-x-0 after:bottom-[-1px] after:h-[2px] after:rounded-full after:bg-foreground after:transition-opacity",
              active
                ? "font-semibold text-foreground after:opacity-100"
                : "text-muted-foreground hover:text-foreground after:opacity-0",
            )}
          >
            {item.label}
          </TabsTrigger>
        );
      })}
    </TabsList>
  );
}
