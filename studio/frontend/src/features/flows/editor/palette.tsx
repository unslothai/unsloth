// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { CollapsibleSectionTriggerButton } from "@/features/recipe-studio/dialogs/shared/collapsible-section-trigger";
import { HugeiconsIcon } from "@hugeicons/react";
import { memo, useDeferredValue, useMemo, useState } from "react";
import { useCatalogStore } from "../stores/catalog-store";
import { useUiStore } from "../stores/ui-store";
import type { NodeSpec } from "../types";
import { Cancel01Icon, Search01Icon } from "./icon-data";
import { nodeIcon } from "./icons";
import { NodeIconTile } from "./node-card";
import { CATEGORY_TONES } from "./node-tones";
import { NODE_DRAG_MIME } from "./node-factory";

/** Recipe Studio's block sheet row (BlockSheetButton), at palette density. */
function PaletteItem({
  spec,
  color,
  onAdd,
}: { spec: NodeSpec; color: string; onAdd: (type: string) => void }) {
  return (
    <button
      type="button"
      draggable={true}
      onDragStart={(e) => {
        e.dataTransfer.setData(NODE_DRAG_MIME, spec.type);
        e.dataTransfer.effectAllowed = "move";
      }}
      onClick={() => onAdd(spec.type)}
      title={spec.description}
      data-testid={`palette-${spec.type}`}
      className="flex w-full cursor-grab items-center gap-2.5 border-l-2 border-transparent bg-background px-3 py-1.5 text-left transition hover:border-border/60 hover:bg-muted/35 active:cursor-grabbing"
    >
      <NodeIconTile
        icon={nodeIcon(spec.icon)}
        tone={CATEGORY_TONES[spec.category]}
        accent={color}
      />
      <div className="min-w-0 flex-1">
        <div className="flex items-center gap-2">
          <p className="truncate text-sm font-semibold text-foreground">
            {spec.title}
          </p>
          {spec.lane === "gpu" && (
            <Badge variant="outline" className="rounded-full text-ui-10">
              GPU
            </Badge>
          )}
        </div>
        <p className="truncate text-ui-11 text-muted-foreground">
          {spec.description}
        </p>
      </div>
    </button>
  );
}

function PaletteImpl({ onAdd }: { onAdd: (type: string) => void }) {
  const catalog = useCatalogStore((s) => s.catalog);
  const [query, setQuery] = useState("");
  const deferred = useDeferredValue(query.trim().toLowerCase());
  const [collapsed, setCollapsed] = useState<Record<string, boolean>>({});

  const groups = useMemo(() => {
    if (!catalog) return [];
    return catalog.categories
      .map((category) => ({
        category,
        specs: catalog.nodes.filter(
          (spec) =>
            spec.category === category.id &&
            (!deferred ||
              spec.title.toLowerCase().includes(deferred) ||
              spec.description.toLowerCase().includes(deferred) ||
              spec.type.includes(deferred)),
        ),
      }))
      .filter((group) => group.specs.length > 0);
  }, [catalog, deferred]);

  return (
    <div
      className="flex h-full w-64 shrink-0 flex-col border-border border-r bg-background"
      onKeyDown={(e) => {
        if (e.key === "Escape") useUiStore.getState().togglePalette();
      }}
    >
      <div className="flex items-center justify-between pt-2 pr-1.5 pl-4">
        <span className="font-semibold text-sm text-foreground">Nodes</span>
        <Button
          type="button"
          variant="ghost"
          size="icon-sm"
          aria-label="Hide nodes"
          title="Hide nodes (Esc)"
          onClick={() => useUiStore.getState().togglePalette()}
          className="text-muted-foreground"
        >
          <HugeiconsIcon icon={Cancel01Icon} size={15} />
        </Button>
      </div>
      <div className="relative border-border border-b px-3 pt-1 pb-2">
        <HugeiconsIcon
          icon={Search01Icon}
          className="pointer-events-none absolute top-1/2 left-5.5 size-3.5 -translate-y-1/2 text-muted-foreground"
        />
        <Input
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Search nodes"
          className="h-8 pl-8"
          data-testid="flows-palette-search"
        />
      </div>
      <div className="flex-1 overflow-y-auto px-1.5 py-2">
        {groups.map(({ category, specs }) => {
          const isCollapsed = !deferred && collapsed[category.id];
          return (
            <div key={category.id} className="mb-2">
              <CollapsibleSectionTriggerButton
                label={category.title}
                open={!isCollapsed}
                summary={String(specs.length)}
                className="px-3 py-1.5"
                onClick={() =>
                  setCollapsed((c) => ({
                    ...c,
                    [category.id]: !c[category.id],
                  }))
                }
              />
              {!isCollapsed &&
                specs.map((spec) => (
                  <PaletteItem
                    key={spec.type}
                    spec={spec}
                    color={category.color}
                    onAdd={onAdd}
                  />
                ))}
            </div>
          );
        })}
        {deferred && groups.length === 0 && (
          <p className="px-3 py-2 text-xs text-muted-foreground">
            No matching nodes.
          </p>
        )}
      </div>
      <p className="border-border border-t px-3 py-2 text-ui-11 leading-snug text-muted-foreground">
        Drag onto the canvas, or double-click the canvas to search.
      </p>
    </div>
  );
}

export const Palette = memo(PaletteImpl);
