// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { HugeiconsIcon } from "@hugeicons/react";
import { memo, useDeferredValue, useMemo, useState } from "react";
import { useCatalogStore } from "../stores/catalog-store";
import type { NodeSpec } from "../types";
import { Search01Icon } from "./icon-data";
import { nodeIcon } from "./icons";
import { NODE_DRAG_MIME } from "./node-factory";
import { REROUTE_TYPE } from "./reroutes";

const REROUTE_HINT =
  "Steer a wire by hand: drop it on a wire to split it, or wire into it.";

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
      className="group flex w-full cursor-grab items-center gap-2 rounded-md px-2 py-1.5 text-left text-[12.5px] text-zinc-300 hover:bg-white/5 hover:text-white active:cursor-grabbing"
    >
      <HugeiconsIcon
        icon={nodeIcon(spec.icon)}
        size={14}
        strokeWidth={1.8}
        style={{ color }}
      />
      <span className="flex-1 truncate">{spec.title}</span>
      {spec.lane === "gpu" && (
        <span className="text-[9px] uppercase text-zinc-600 group-hover:text-zinc-400">
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
    <div className="flex h-full w-58 shrink-0 flex-col border-white/10 border-r bg-zinc-950/80">
      <div className="flex items-center gap-2 border-white/10 border-b px-3 py-2">
        <HugeiconsIcon
          icon={Search01Icon}
          size={14}
          className="text-zinc-500"
        />
        <input
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Search nodes"
          className="w-full bg-transparent text-sm text-zinc-200 outline-none placeholder:text-zinc-600"
          data-testid="flows-palette-search"
        />
      </div>
      <div className="flex-1 overflow-y-auto px-1.5 py-2">
        {(!deferred || "reroute wire".includes(deferred)) && (
          <button
            type="button"
            draggable={true}
            onDragStart={(e) => {
              e.dataTransfer.setData(NODE_DRAG_MIME, REROUTE_TYPE);
              e.dataTransfer.effectAllowed = "move";
            }}
            onClick={() => onAdd(REROUTE_TYPE)}
            title={REROUTE_HINT}
            data-testid="palette-reroute"
            className="mb-1.5 flex w-full cursor-grab items-center gap-2 rounded-md px-2 py-1.5 text-left text-[12.5px] text-zinc-300 hover:bg-white/5 hover:text-white active:cursor-grabbing"
          >
            <span className="flex w-3.5 justify-center">
              <span className="h-2 w-3.5 rounded-full border border-zinc-400" />
            </span>
            <span className="flex-1 truncate">Reroute</span>
          </button>
        )}
        {groups.map(({ category, specs }) => {
          const isCollapsed = !deferred && collapsed[category.id];
          return (
            <div key={category.id} className="mb-1.5">
              <button
                type="button"
                className="flex w-full items-center gap-2 px-2 py-1 text-[10.5px] font-semibold uppercase tracking-wider text-zinc-500 hover:text-zinc-300"
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
                <span className="text-zinc-600">{isCollapsed ? "+" : "−"}</span>
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
      <p className="border-white/10 border-t px-3 py-2 text-[10.5px] leading-snug text-zinc-600">
        Drag onto the canvas, or double-click the canvas to search.
      </p>
    </div>
  );
}

export const Palette = memo(PaletteImpl);
