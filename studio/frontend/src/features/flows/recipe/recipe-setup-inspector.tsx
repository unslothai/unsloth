// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import { HugeiconsIcon } from "@hugeicons/react";
import { Suspense, lazy } from "react";
import { Cancel01Icon, Delete02Icon } from "../editor/icon-data";
import { NodeIconTile } from "../editor/node-card";
import { useDocStore } from "../stores/doc-store";
import { useUiStore } from "../stores/ui-store";
import { setupLook } from "./recipe-kinds";
import { recipeSetup, setupId } from "./recipe-setup";

const SetupForm = lazy(() =>
  import("./recipe-forms").then((m) => ({ default: m.SetupForm })),
);

const HINT: Record<string, string> = {
  seed: "The rows every generated row starts from. A dataset wired into the Recipe Dataset step's Seed input replaces it.",
  // biome-ignore lint/style/useNamingConvention: recipe config kinds
  model_provider: "Where the recipe's models run. Exporting or sharing the flow drops keys.",
  // biome-ignore lint/style/useNamingConvention: recipe config kinds
  model_config: "A model and its generation settings; AI columns pick it by name.",
  // biome-ignore lint/style/useNamingConvention: recipe config kinds
  tool_config: "Tools an AI column may call. Local (stdio) tool servers only run from the Data Recipes page.",
};

/** The inspector for a recipe group's seed, provider connection, model preset or tool profile card. */
export function SetupInspector() {
  const target = useUiStore((s) => s.recipeSetup);
  const formKey = useUiStore((s) => s.recipeSetupKey);
  const graph = useDocStore((s) => s.graph);
  const group = target ? graph.groups[target.groupId] : undefined;
  const card =
    target && group
      ? recipeSetup(graph, group).cards.find(
          (c) => c.id === setupId(target.groupId, target.kind, target.name),
        )
      : undefined;
  const seedType = (
    group?.config?.seed_config as { source?: { seed_type?: string } } | undefined
  )?.source?.seed_type;
  if (!target || !card) return null;
  const { icon, tone } = setupLook(card.kind, seedType);
  const close = () => useUiStore.getState().setRecipeSetup(null);
  return (
    <aside
      className="flex h-full w-85 shrink-0 flex-col border-border border-l bg-background text-foreground"
      data-testid="flows-inspector"
    >
      <header className="flex items-start gap-2.5 border-border/60 border-b px-4 py-3">
        <NodeIconTile icon={icon} tone={tone} />
        <div className="min-w-0 flex-1">
          <p className="truncate font-semibold text-sm">{card.title}</p>
          <p className="mt-0.5 text-ui-11 leading-snug text-muted-foreground">
            {HINT[card.kind]}
          </p>
        </div>
        <Button
          type="button"
          variant="ghost"
          size="icon-xs"
          aria-label="Close inspector"
          onClick={close}
          className="text-muted-foreground"
        >
          <HugeiconsIcon icon={Cancel01Icon} size={15} />
        </Button>
      </header>
      <div className="flex flex-wrap gap-1.5 border-border/60 border-b px-4 py-2">
        {seedType === "github_repo" && card.kind === "seed" && (
          <Button
            type="button"
            variant="outline"
            size="xs"
            onClick={() =>
              useUiStore
                .getState()
                .setRecipeEditor({ groupId: target.groupId, easy: true })
            }
            data-testid="recipe-github-easy"
          >
            GitHub crawler
          </Button>
        )}
        <Button
          type="button"
          variant="ghost"
          size="icon-xs"
          aria-label="Remove from the recipe"
          title="Remove from the recipe"
          onClick={() =>
            void import("./recipe-blocks").then((m) => m.removeSetup(target))
          }
          className="ml-auto text-muted-foreground hover:bg-destructive/10 hover:text-destructive"
        >
          <HugeiconsIcon icon={Delete02Icon} size={14} />
        </Button>
      </div>
      <div className="flex-1 overflow-y-auto px-4 py-3">
        <Suspense fallback={null}>
          <SetupForm key={formKey} target={target} />
        </Suspense>
      </div>
    </aside>
  );
}
