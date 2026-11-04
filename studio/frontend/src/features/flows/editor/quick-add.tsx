// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  Command,
  CommandEmpty,
  CommandInput,
  CommandItem,
  CommandList,
  CommandShortcut,
} from "@/components/ui/command";
import { useEffect, useMemo, useRef } from "react";
import { COLUMN_BLOCKS, SEED_BLOCKS, SETUP_BLOCKS } from "../recipe/recipe-kinds";
import { useCatalogStore } from "../stores/catalog-store";
import { useDocStore } from "../stores/doc-store";
import { useUiStore } from "../stores/ui-store";
import type { NodeSpec } from "../types";
import { compatibleSpecs } from "./connection-rules";
import { nodeIcon } from "./icons";
import { NodeIconTile } from "./node-card";
import { addNodeConnected } from "./node-factory";
import { snapPoint } from "./placement";
import { CATEGORY_TONES } from "./node-tones";
import { REROUTE_TYPE, rerouteCanFeed, rerouteTakes } from "./reroutes";

export function QuickAdd() {
  const quickAdd = useUiStore((s) => s.quickAdd);
  const setQuickAdd = useUiStore((s) => s.setQuickAdd);
  const catalog = useCatalogStore((s) => s.catalog);
  const categories = useCatalogStore((s) => s.categories);
  const ref = useRef<HTMLDivElement>(null);

  const options = useMemo((): { spec: NodeSpec; port?: string }[] => {
    if (!catalog || !quickAdd) return [];
    // Design-only blocks (a recipe's columns) are added inside their own editor, not here.
    const addable = catalog.nodes.filter((spec) => !spec.design);
    if (!quickAdd.from) return addable.map((spec) => ({ spec }));
    const { from } = quickAdd;
    const graph = useDocStore.getState().graph;
    // Off a reroute, check against the whole chain, not just its first input's type.
    const matches =
      from.kind === "data" && graph.reroutes[from.nodeId]
        ? addable.flatMap((spec) => {
            const port =
              from.side === "out"
                ? spec.inputs.find((p) => rerouteCanFeed(graph, from.nodeId, p))
                : spec.outputs.find((p) => rerouteTakes(graph, from.nodeId, p));
            return port ? [{ spec, port: port.id }] : [];
          })
        : compatibleSpecs(addable, from);
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
  if (quickAdd.recipe) return <RecipeQuickAdd groupId={quickAdd.recipe} />;
  const left = Math.min(quickAdd.screen.x, window.innerWidth - 340);
  const top = Math.min(quickAdd.screen.y, window.innerHeight - 380);

  return (
    <div
      ref={ref}
      className="fixed z-50 w-80 overflow-hidden rounded-xl border border-border bg-popover text-popover-foreground shadow-[var(--popup-shadow)]"
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
        <CommandInput
          // biome-ignore lint/a11y/noAutofocus: the popover exists to type into
          autoFocus={true}
          placeholder={
            quickAdd.from ? "Add a connected node…" : "Search nodes…"
          }
        />
        <CommandList className="max-h-80">
          <CommandEmpty className="text-xs text-muted-foreground">
            No matching nodes
          </CommandEmpty>
          {options.map(({ spec, port }) => (
            <CommandItem
              key={spec.type}
              value={`${spec.title} ${spec.type} ${spec.category} ${spec.description}`}
              onSelect={() => {
                addNodeConnected(
                  spec.type,
                  snapPoint(
                    [quickAdd.flow.x, quickAdd.flow.y],
                    useUiStore.getState().snapToGrid,
                  ),
                  quickAdd.from,
                  port,
                );
                setQuickAdd(null);
              }}
              className="cursor-pointer px-2 py-1.5"
            >
              <NodeIconTile
                icon={nodeIcon(spec.icon)}
                tone={CATEGORY_TONES[spec.category]}
                accent={categories[spec.category]?.color}
              />
              <span className="flex-1 truncate">{spec.title}</span>
              <CommandShortcut className="tracking-normal">
                {categories[spec.category]?.title ?? spec.category}
              </CommandShortcut>
            </CommandItem>
          ))}
          {quickAdd.from?.kind !== "flow" && (
            <CommandItem
              value="Reroute wire waypoint"
              onSelect={() => {
                const { flow, from } = quickAdd;
                // Centre the reroute where the wire was dropped.
                const x = from?.side === "in" ? flow.x + 260 : flow.x;
                const y = from ? flow.y + 20 : flow.y;
                addNodeConnected(REROUTE_TYPE, [x - 130, y - 20], from, "in");
                setQuickAdd(null);
              }}
              className="cursor-pointer px-2 py-1.5"
              data-testid="flows-quick-add-reroute"
            >
              <span className="flex size-7 shrink-0 items-center justify-center">
                <span className="h-2 w-3.5 rounded-full border border-muted-foreground" />
              </span>
              <span className="flex-1 truncate">Reroute</span>
              <CommandShortcut className="tracking-normal">Wire</CommandShortcut>
            </CommandItem>
          )}
        </CommandList>
      </Command>
    </div>
  );
}

/** Quick add over a recipe group: the columns, seeds and setup blocks Recipe Studio's sheet offers. */
function RecipeQuickAdd({ groupId }: { groupId: string }) {
  const quickAdd = useUiStore((s) => s.quickAdd);
  const setQuickAdd = useUiStore((s) => s.setQuickAdd);
  const hasSeed = useDocStore(
    (s) => !!s.graph.groups[groupId]?.config?.seed_config,
  );
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const onDown = (event: PointerEvent) => {
      if (ref.current && !ref.current.contains(event.target as Node))
        setQuickAdd(null);
    };
    window.addEventListener("pointerdown", onDown, true);
    return () => window.removeEventListener("pointerdown", onDown, true);
  }, [setQuickAdd]);
  if (!quickAdd) return null;
  const sections = [
    { title: "Column", blocks: COLUMN_BLOCKS },
    { title: "Seed data", blocks: hasSeed ? [] : SEED_BLOCKS },
    { title: "Setup", blocks: SETUP_BLOCKS },
  ];
  return (
    <div
      ref={ref}
      className="fixed z-50 w-80 overflow-hidden rounded-xl border border-border bg-popover text-popover-foreground shadow-[var(--popup-shadow)]"
      style={{
        left: Math.min(quickAdd.screen.x, window.innerWidth - 340),
        top: Math.min(quickAdd.screen.y, window.innerHeight - 380),
      }}
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
        <CommandInput
          // biome-ignore lint/a11y/noAutofocus: the popover exists to type into
          autoFocus={true}
          placeholder="Add to this recipe…"
        />
        <CommandList className="max-h-80">
          <CommandEmpty className="text-xs text-muted-foreground">
            Nothing matches
          </CommandEmpty>
          {sections.map((section) =>
            section.blocks.map((block) => (
              <CommandItem
                key={`${block.kind}:${block.type}`}
                value={`${block.label} ${section.title}`}
                onSelect={() => {
                  setQuickAdd(null);
                  void import("../recipe/recipe-blocks").then((m) =>
                    m.addRecipeBlock(groupId, block.kind, block.type),
                  );
                }}
                className="cursor-pointer px-2 py-1.5"
                data-testid={`recipe-add-${block.type}`}
              >
                <NodeIconTile icon={block.icon} tone={block.tone} />
                <span className="flex-1 truncate">{block.label}</span>
                <CommandShortcut className="tracking-normal">
                  {section.title}
                </CommandShortcut>
              </CommandItem>
            )),
          )}
        </CommandList>
      </Command>
    </div>
  );
}
