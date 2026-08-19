// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { XIcon } from "lucide-react";
import { Tick02Icon } from "@/lib/tick-icon";
import { HugeiconsIcon } from "@hugeicons/react";
import { FileDatabaseIcon } from "@hugeicons/core-free-icons";
import { type FC, useCallback, useEffect, useState } from "react";

import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { useRagToolDisabled } from "@/features/chat/hooks/use-rag-tool-disabled";
import { useChatRuntimeStore } from "@/features/chat/stores/chat-runtime-store";

import { listKnowledgeBases } from "../api/rag-api";
import type { KnowledgeBase } from "../types/rag";
import { KnowledgeBaseDialog } from "./knowledge-base-dialog";

// Matches the Thinking/MCP pill chevron.
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

// Picks the retrieval source. Shown whenever retrieval is on; dims but stays
// interactive (so it can be turned off) while the loaded model can't run it.
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

  const [kbs, setKbs] = useState<KnowledgeBase[]>([]);
  const [kbsLoaded, setKbsLoaded] = useState(false);
  const [menuOpen, setMenuOpen] = useState(false);
  const [dialogOpen, setDialogOpen] = useState(false);

  const refresh = useCallback(async () => {
    try {
      const rows = await listKnowledgeBases();
      setKbs(rows);
    } catch {
      // Keep prior state on failure.
    } finally {
      setKbsLoaded(true);
    }
  }, []);

  // Load on mount so newly created KBs show up.
  useEffect(() => {
    void refresh();
  }, [refresh]);

  // If the selected KB was deleted, fall back to thread source so we never send a
  // stale kb_id. Gate on kbsLoaded, not kbs.length: deleting the last KB empties the
  // list, so a length>0 guard would skip the reset and stick on a ghost KB.
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
            {/* Icon doubles as an off switch: hover swaps to an X; clicking it
                turns RAG off without opening the menu. In compact icon-only
                mode the glyph is the whole button, so clicks fall through to
                the trigger and open the menu instead. */}
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
                className="size-[15px]"
              />
              <XIcon className="composer-pill-x" />
            </span>
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
        </DropdownMenuContent>
      </DropdownMenu>
      <KnowledgeBaseDialog
        open={dialogOpen}
        onOpenChange={(next) => {
          setDialogOpen(next);
          if (!next) void refresh();
        }}
      />
    </>
  );
}
