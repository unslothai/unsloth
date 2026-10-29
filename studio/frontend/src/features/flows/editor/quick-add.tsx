// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { HugeiconsIcon } from "@hugeicons/react";
import { Command } from "cmdk";
import { useEffect, useMemo, useRef } from "react";
import { useCatalogStore } from "../stores/catalog-store";
import { useDocStore } from "../stores/doc-store";
import { useUiStore } from "../stores/ui-store";
import type { NodeSpec } from "../types";
import { compatibleSpecs } from "./connection-rules";
import { nodeIcon } from "./icons";
import { addNodeConnected } from "./node-factory";
import { REROUTE_TYPE, rerouteCanFeed, rerouteTakes } from "./reroutes";

export function QuickAdd() {
  const quickAdd = useUiStore((s) => s.quickAdd);
  const setQuickAdd = useUiStore((s) => s.setQuickAdd);
  const catalog = useCatalogStore((s) => s.catalog);
  const categories = useCatalogStore((s) => s.categories);
  const ref = useRef<HTMLDivElement>(null);

  const options = useMemo((): { spec: NodeSpec; port?: string }[] => {
    if (!catalog || !quickAdd) return [];
    if (!quickAdd.from) return catalog.nodes.map((spec) => ({ spec }));
    const { from } = quickAdd;
    const graph = useDocStore.getState().graph;
    // Off a reroute, check against the whole chain, not just its first input's type.
    const matches =
      from.kind === "data" && graph.reroutes[from.nodeId]
        ? catalog.nodes.flatMap((spec) => {
            const port =
              from.side === "out"
                ? spec.inputs.find((p) => rerouteCanFeed(graph, from.nodeId, p))
                : spec.outputs.find((p) => rerouteTakes(graph, from.nodeId, p));
            return port ? [{ spec, port: port.id }] : [];
          })
        : compatibleSpecs(catalog.nodes, from);
    const exactType = quickAdd.from.port?.type;
    return matches.sort((a, b) => {
      const score = (m: { spec: NodeSpec; port: string }) => {
        const ports =
          quickAdd.from?.side === "out" ? m.spec.inputs : m.spec.outputs;
        return ports.find((p) => p.id === m.port)?.type === exactType ? 0 : 1;
      };
      return score(a) - score(b);
    });
  }, [catalog, quickAdd]);

  useEffect(() => {
    if (!quickAdd) return;
    const onDown = (event: PointerEvent) => {
      if (ref.current && !ref.current.contains(event.target as Node))
        setQuickAdd(null);
    };
    window.addEventListener("pointerdown", onDown, true);
    return () => window.removeEventListener("pointerdown", onDown, true);
  }, [quickAdd, setQuickAdd]);

  if (!quickAdd) return null;
  const left = Math.min(quickAdd.screen.x, window.innerWidth - 340);
  const top = Math.min(quickAdd.screen.y, window.innerHeight - 380);

  return (
    <div
      ref={ref}
      className="fixed z-50 w-80 overflow-hidden rounded-xl border border-white/10 bg-zinc-900/98 text-zinc-100 shadow-2xl shadow-black/60"
      style={{ left, top }}
      data-testid="flows-quick-add"
    >
      <Command
        loop={true}
        onKeyDown={(e) => {
          if (e.key === "Escape") {
            e.stopPropagation();
            setQuickAdd(null);
          }
        }}
      >
        <Command.Input
          // biome-ignore lint/a11y/noAutofocus: the popover exists to type into
          autoFocus={true}
          placeholder={
            quickAdd.from ? "Add a connected node…" : "Search nodes…"
          }
          className="w-full border-white/10 border-b bg-transparent px-3 py-2.5 text-sm outline-none placeholder:text-zinc-500"
        />
        <Command.List className="max-h-80 overflow-y-auto p-1">
          <Command.Empty className="px-3 py-6 text-center text-xs text-zinc-500">
            No matching nodes
          </Command.Empty>
          {options.map(({ spec, port }) => (
            <Command.Item
              key={spec.type}
              value={`${spec.title} ${spec.type} ${spec.category} ${spec.description}`}
              onSelect={() => {
                addNodeConnected(
                  spec.type,
                  [quickAdd.flow.x, quickAdd.flow.y],
                  quickAdd.from,
                  port,
                );
                setQuickAdd(null);
              }}
              className="flex cursor-pointer items-center gap-2 rounded-md px-2 py-1.5 text-[13px] aria-selected:bg-white/10"
            >
              <HugeiconsIcon
                icon={nodeIcon(spec.icon)}
                size={15}
                strokeWidth={1.8}
                style={{ color: categories[spec.category]?.color }}
              />
              <span className="flex-1 truncate">{spec.title}</span>
              <span className="text-[10px] text-zinc-500">
                {categories[spec.category]?.title ?? spec.category}
              </span>
            </Command.Item>
          ))}
          {quickAdd.from?.kind !== "flow" && (
            <Command.Item
              value="Reroute wire waypoint"
              onSelect={() => {
                const { flow, from } = quickAdd;
                // Centre the reroute where the wire was dropped.
                const x = from?.side === "in" ? flow.x + 260 : flow.x;
                const y = from ? flow.y + 20 : flow.y;
                addNodeConnected(REROUTE_TYPE, [x - 130, y - 20], from, "in");
                setQuickAdd(null);
              }}
              className="flex cursor-pointer items-center gap-2 rounded-md px-2 py-1.5 text-[13px] aria-selected:bg-white/10"
              data-testid="flows-quick-add-reroute"
            >
              <span className="flex w-[15px] justify-center">
                <span className="h-2 w-3.5 rounded-full border border-zinc-400" />
              </span>
              <span className="flex-1 truncate">Reroute</span>
              <span className="text-[10px] text-zinc-500">Wire</span>
            </Command.Item>
          )}
        </Command.List>
      </Command>
    </div>
  );
}
