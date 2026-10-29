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
import { useCatalogStore } from "../stores/catalog-store";
import { useUiStore } from "../stores/ui-store";
import type { NodeSpec } from "../types";
import { compatibleSpecs } from "./connection-rules";
import { nodeIcon } from "./icons";
import { NodeIconTile } from "./node-card";
import { addNodeConnected } from "./node-factory";
import { snapPoint } from "./placement";
import { CATEGORY_TONES } from "./node-tones";

export function QuickAdd() {
  const quickAdd = useUiStore((s) => s.quickAdd);
  const setQuickAdd = useUiStore((s) => s.setQuickAdd);
  const catalog = useCatalogStore((s) => s.catalog);
  const categories = useCatalogStore((s) => s.categories);
  const ref = useRef<HTMLDivElement>(null);

  const options = useMemo((): { spec: NodeSpec; port?: string }[] => {
    if (!catalog || !quickAdd) return [];
    if (!quickAdd.from) return catalog.nodes.map((spec) => ({ spec }));
    const matches = compatibleSpecs(catalog.nodes, quickAdd.from);
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
        </CommandList>
      </Command>
    </div>
  );
}
