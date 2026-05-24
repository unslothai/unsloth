// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Separator } from "@/components/ui/separator";
import type { KnowledgeBase } from "@/features/rag/api/rag-api";
import { KBDetailPanel } from "@/features/rag/components/kb-detail-panel";
import { KBList } from "@/features/rag/components/kb-list";
import { RagDefaultsSection } from "@/features/rag/components/rag-defaults-section";
import { ThreadIndexList } from "@/features/rag/components/thread-index-list";
import { useState } from "react";

export function KnowledgeBasesTab() {
  const [selected, setSelected] = useState<KnowledgeBase | null>(null);

  return (
    <div className="flex h-full min-h-0 flex-col gap-4">
      <div>
        <h2 className="text-lg font-semibold">Knowledge bases</h2>
        <p className="text-sm text-muted-foreground">
          Create reusable document collections and pick one per chat thread to
          ground answers in your own files.
        </p>
      </div>
      <Separator />
      <div className="flex min-h-0 flex-1 gap-4">
        <div className="w-[220px] shrink-0">
          <KBList selectedId={selected?.id ?? null} onSelect={setSelected} />
        </div>
        <Separator orientation="vertical" />
        <div className="min-w-0 flex-1">
          {selected ? (
            <KBDetailPanel kb={selected} />
          ) : (
            <div className="flex h-full items-center justify-center text-sm text-muted-foreground">
              Select a knowledge base, or create a new one to get started.
            </div>
          )}
        </div>
      </div>
      <Separator />
      <ThreadIndexList />
      <Separator />
      <RagDefaultsSection />
    </div>
  );
}
