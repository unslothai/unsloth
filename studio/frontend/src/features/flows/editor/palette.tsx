// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Input } from "@/components/ui/input";
import { HugeiconsIcon } from "@hugeicons/react";
import { memo, useDeferredValue, useMemo, useState } from "react";
import { useCatalogStore } from "../stores/catalog-store";
import type { NodeSpec } from "../types";
import { Search01Icon } from "./icon-data";
import { nodeIcon } from "./icons";
import { NODE_DRAG_MIME } from "./node-factory";

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
      className="group flex w-full cursor-grab items-center gap-2 rounded-md px-2 py-1.5 text-left text-[12.5px] text-foreground hover:bg-muted active:cursor-grabbing"
    >
      <HugeiconsIcon
        icon={nodeIcon(spec.icon)}
        size={14}
        strokeWidth={1.8}
        style={{ color }}
      />
      <span className="flex-1 truncate">{spec.title}</span>
      {spec.lane === "gpu" && (
        <span className="text-[9px] uppercase text-muted-foreground/70 group-hover:text-muted-foreground">
          gpu
        </span>
      )}
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
    <div className="flex h-full w-58 shrink-0 flex-col border-border border-r bg-background">
      <div className="relative border-border border-b px-2 py-2">
        <HugeiconsIcon
          icon={Search01Icon}
          size={14}
          className="pointer-events-none absolute top-1/2 left-4.5 -translate-y-1/2 text-muted-foreground"
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
            <div key={category.id} className="mb-1.5">
              <button
                type="button"
                className="flex w-full items-center gap-2 px-2 py-1 text-[10.5px] font-semibold uppercase tracking-wider text-muted-foreground hover:text-foreground"
                onClick={() =>
                  setCollapsed((c) => ({
                    ...c,
                    [category.id]: !c[category.id],
                  }))
                }
              >
                <span
                  className="h-2 w-2 rounded-full"
                  style={{ background: category.color }}
                />
                <span className="flex-1 text-left">{category.title}</span>
                <span className="text-muted-foreground/70">{isCollapsed ? "+" : "−"}</span>
              </button>
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
      </div>
      <p className="border-border border-t px-3 py-2 text-[10.5px] leading-snug text-muted-foreground/70">
        Drag onto the canvas, or double-click the canvas to search.
      </p>
    </div>
  );
}

export const Palette = memo(PaletteImpl);
