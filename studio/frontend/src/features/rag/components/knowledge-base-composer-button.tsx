// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  ChevronDownIcon,
  XIcon,
} from "lucide-react";
import { Tick02Icon } from "@/lib/tick-icon";
import { HugeiconsIcon } from "@hugeicons/react";
import { FileDatabaseIcon } from "@hugeicons/core-free-icons";
import { useCallback, useEffect, useLayoutEffect, useRef, useState } from "react";

import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { useIsAccountOwner } from "@/features/auth";
import { useRagToolDisabled } from "@/features/chat/hooks/use-rag-tool-disabled";
import { useChatRuntimeStore } from "@/features/chat/stores/chat-runtime-store";

import {
  listKnowledgeBases,
  subscribeKnowledgeBasesChanged,
} from "../api/rag-api";
import type { KnowledgeBase } from "../types/rag";
import { EmbeddingModelMenuChip, EmbeddingModelMenuList } from "./embedding-model-menu-picker";
import { KnowledgeBaseDialog } from "./knowledge-base-dialog";

// Dims but stays interactive (so retrieval can be turned off) while the model cannot run it.
export function KnowledgeBaseComposerButton({
  side = "bottom",
}: {
  side?: "top" | "bottom";
} = {}) {
  const ragEnabled = useChatRuntimeStore((s) => s.ragEnabled);
  const setRagEnabled = useChatRuntimeStore((s) => s.setRagEnabled);
  const ragDisabled = useRagToolDisabled();
  const ragSource = useChatRuntimeStore((s) => s.ragSource);
  const setRagSource = useChatRuntimeStore((s) => s.setRagSource);
  // The embedding model is a server setting only the owner can change.
  const isOwner = useIsAccountOwner();

  const [kbs, setKbs] = useState<KnowledgeBase[]>([]);
  const [kbsLoaded, setKbsLoaded] = useState(false);
  const [menuOpen, setMenuOpen] = useState(false);
  // The embedding list replaces the source list in the same menu.
  const [view, setView] = useState<"source" | "embedding">("source");
  const contentRef = useRef<HTMLDivElement>(null);
  const shownViewRef = useRef(view);
  // A swap unmounts the focused item while the menu stays open, which strands focus outside it and
  // kills arrow keys. Focus the new view's first item, as Radix does on open.
  useLayoutEffect(() => {
    if (shownViewRef.current === view) return;
    shownViewRef.current = view;
    contentRef.current
      ?.querySelector<HTMLElement>('[role^="menuitem"]:not([data-disabled])')
      ?.focus();
  }, [view]);
  const [dialogOpen, setDialogOpen] = useState(false);

  // Refreshes overlap; an older answer landing last would restore a deleted KB.
  const latestRefreshRef = useRef(0);
  const refresh = useCallback(async () => {
    const request = ++latestRefreshRef.current;
    try {
      const rows = await listKnowledgeBases();
      if (request === latestRefreshRef.current) setKbs(rows);
    } catch {
      // Keep prior state on failure.
    } finally {
      if (request === latestRefreshRef.current) setKbsLoaded(true);
    }
  }, []);

  useEffect(() => {
    void refresh();
    return subscribeKnowledgeBasesChanged(() => void refresh());
  }, [refresh]);

  // Gate on kbsLoaded, not kbs.length: deleting the last KB empties the list and would leave a
  // ghost kb_id selected.
  useEffect(() => {
    if (
      kbsLoaded &&
      ragSource.type === "kb" &&
      !kbs.some((kb) => kb.id === ragSource.kbId)
    ) {
      setRagSource({ type: "thread" });
    }
  }, [kbs, kbsLoaded, ragSource, setRagSource]);

  if (!ragEnabled) return null;

  return (
    <>
      <DropdownMenu
        open={menuOpen}
        onOpenChange={(open) => {
          setMenuOpen(open);
          if (open) void refresh();
          else setView("source");
        }}
      >
        <DropdownMenuTrigger asChild={true}>
          <button
            type="button"
            className="composer-pill-btn"
            data-pill-label="RAG"
            data-active={ragDisabled ? "false" : "true"}
            aria-label="Retrieval source"
          >
            <span
              role="button"
              aria-label="Turn off retrieval"
              tabIndex={-1}
              onPointerDown={(e) => {
                if (e.currentTarget.closest('[data-pill-compact="true"]')) return;
                e.stopPropagation();
              }}
              onClick={(e) => {
                if (e.currentTarget.closest('[data-pill-compact="true"]')) return;
                e.stopPropagation();
                setRagEnabled(false);
              }}
              className="composer-pill-glyph cursor-pointer"
            >
              <HugeiconsIcon
                icon={FileDatabaseIcon}
                strokeWidth={2}
                className="size-[calc(15px*var(--ui-space-scale,1))]"
              />
              <XIcon className="composer-pill-x" />
            </span>
            <span>RAG</span>
            <ChevronDownIcon strokeWidth={1.5} className="composer-pill-caret size-[calc(15px*var(--ui-space-scale,1))]" />
          </button>
        </DropdownMenuTrigger>
        <DropdownMenuContent
          ref={contentRef}
          side={side}
          align="start"
          sideOffset={2}
          avoidCollisions={true}
          className={
            isOwner
              ? "unsloth-plus-menu mcp-menu w-[calc(320px*var(--ui-space-scale,1))]"
              : "unsloth-plus-menu mcp-menu w-[calc(232px*var(--ui-space-scale,1))]"
          }
        >
          {isOwner && view === "embedding" ? (
            <EmbeddingModelMenuList onBack={() => setView("source")} />
          ) : (
            <>
            <DropdownMenuLabel className="flex items-center justify-between gap-3">
              <span className="shrink-0">Retrieve from</span>
              {isOwner ? <EmbeddingModelMenuChip onOpen={() => setView("embedding")} /> : null}
            </DropdownMenuLabel>
            <DropdownMenuItem
              onSelect={() => setRagSource({ type: "thread" })}
              className={
                ragSource.type === "thread"
                  ? "relative text-primary font-medium"
                  : "relative"
              }
            >
              <span className="truncate">This thread's documents</span>
              {ragSource.type === "thread" ? (
                <HugeiconsIcon
                  icon={Tick02Icon}
                  strokeWidth={2}
                  className="ml-auto"
                />
              ) : null}
            </DropdownMenuItem>
            {kbs.length > 0 ? <DropdownMenuSeparator /> : null}
            {kbs.map((kb) => {
              const selected =
                ragSource.type === "kb" && ragSource.kbId === kb.id;
              return (
                <DropdownMenuItem
                  key={kb.id}
                  onSelect={() => setRagSource({ type: "kb", kbId: kb.id })}
                  className={
                    selected ? "relative text-primary font-medium" : "relative"
                  }
                >
                  <span className="truncate">{kb.name}</span>
                  {selected ? (
                    <HugeiconsIcon
                      icon={Tick02Icon}
                      strokeWidth={2}
                      className="ml-auto"
                    />
                  ) : null}
                </DropdownMenuItem>
              );
            })}
            <DropdownMenuSeparator />
            <DropdownMenuItem
              onSelect={() => {
                setMenuOpen(false);
                setDialogOpen(true);
              }}
            >
              Manage knowledge bases…
            </DropdownMenuItem>
            </>
          )}
        </DropdownMenuContent>
      </DropdownMenu>
      <KnowledgeBaseDialog
        open={dialogOpen}
        onOpenChange={setDialogOpen}
      />
    </>
  );
}
