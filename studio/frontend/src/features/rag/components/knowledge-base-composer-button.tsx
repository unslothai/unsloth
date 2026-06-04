// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { CheckIcon, LibraryBigIcon } from "lucide-react";
import { type FC, useCallback, useEffect, useState } from "react";

import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { useRagToolAvailable } from "@/features/chat/hooks/use-rag-tool-available";
import { useChatRuntimeStore } from "@/features/chat/stores/chat-runtime-store";

import { listKnowledgeBases } from "../api/rag-api";
import type { KnowledgeBase } from "../types/rag";
import { KnowledgeBaseDialog } from "./knowledge-base-dialog";

// Matches the Thinking/MCP pill chevron so the affordance reads the same.
const ArrowDownStandardIcon: FC<{ className?: string }> = ({ className }) => (
  <svg
    className={className}
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth={1.5}
    strokeLinecap="round"
    strokeLinejoin="round"
    xmlns="http://www.w3.org/2000/svg"
    aria-hidden={true}
  >
    <path d="M5.99977 9.00005L11.9998 15L17.9998 9" />
  </svg>
);

/**
 * Composer dropdown that picks the retrieval source (this thread's documents or
 * a saved knowledge base) and links out to "Manage knowledge bases". Mirrors the
 * MCP composer button so the two read as siblings. Only rendered when retrieval
 * is on and the loaded model can run search_knowledge_base.
 */
export function KnowledgeBaseComposerButton({
  side = "bottom",
}: {
  side?: "top" | "bottom";
} = {}) {
  const ragEnabled = useChatRuntimeStore((s) => s.ragEnabled);
  const ragAvailable = useRagToolAvailable();
  const ragSource = useChatRuntimeStore((s) => s.ragSource);
  const setRagSource = useChatRuntimeStore((s) => s.setRagSource);

  const [kbs, setKbs] = useState<KnowledgeBase[]>([]);
  const [kbsLoaded, setKbsLoaded] = useState(false);
  const [menuOpen, setMenuOpen] = useState(false);
  const [dialogOpen, setDialogOpen] = useState(false);

  const refresh = useCallback(async () => {
    try {
      const rows = await listKnowledgeBases();
      setKbs(rows);
    } catch {
      // Keep prior state if the list call fails.
    } finally {
      setKbsLoaded(true);
    }
  }, []);

  // Load on mount and whenever the menu opens, so newly created KBs show up.
  useEffect(() => {
    void refresh();
  }, [refresh]);

  // If the selected KB was deleted, fall back to thread source so we never send
  // a stale kb_id. Gate on kbsLoaded (not kbs.length): deleting the *last* KB
  // empties the list, and a length>0 guard would then skip the reset and leave
  // the source stuck on a ghost KB (no Add Files, retrieval misses).
  useEffect(() => {
    if (
      kbsLoaded &&
      ragSource.type === "kb" &&
      !kbs.some((kb) => kb.id === ragSource.kbId)
    ) {
      setRagSource({ type: "thread" });
    }
  }, [kbs, kbsLoaded, ragSource, setRagSource]);

  // Mirrors MCP: this pill only exists while the feature is on (RAG enabled and
  // the loaded model can run search_knowledge_base), so when shown it always
  // reads as active. RAG is toggled on/off from the composer "+" menu.
  if (!ragEnabled || !ragAvailable) return null;

  return (
    <>
      <DropdownMenu
        open={menuOpen}
        onOpenChange={(open) => {
          setMenuOpen(open);
          if (open) void refresh();
        }}
      >
        <DropdownMenuTrigger asChild={true}>
          <button
            type="button"
            className="composer-pill-btn"
            data-active="true"
            aria-label="Retrieval source"
          >
            <LibraryBigIcon className="size-[15px]" />
            <span>RAG</span>
            <ArrowDownStandardIcon className="composer-pill-caret size-[15px]" />
          </button>
        </DropdownMenuTrigger>
        <DropdownMenuContent
          side={side}
          align="start"
          sideOffset={2}
          avoidCollisions={true}
          className="unsloth-plus-menu mcp-menu w-[232px]"
        >
          <DropdownMenuLabel>Retrieve from</DropdownMenuLabel>
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
              <CheckIcon className="ml-auto" />
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
                {selected ? <CheckIcon className="ml-auto" /> : null}
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
        </DropdownMenuContent>
      </DropdownMenu>
      <KnowledgeBaseDialog
        open={dialogOpen}
        onOpenChange={(next) => {
          setDialogOpen(next);
          // Resync after creating / editing / deleting KBs.
          if (!next) void refresh();
        }}
      />
    </>
  );
}
