// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { FileTextIcon, XIcon } from "lucide-react";
import type { FC } from "react";

import { cn } from "@/lib/utils";
import type { PendingDoc } from "../hooks/use-thread-doc-uploads";

interface PendingDocChipsProps {
  docs: PendingDoc[];
  onRemove: (id: string) => void;
}

export const PendingDocChips: FC<PendingDocChipsProps> = ({ docs, onRemove }) => {
  if (docs.length === 0) return null;
  return (
    <div className="mb-2 flex w-full flex-row flex-wrap items-center gap-2 px-1.5 pt-0.5 pb-1">
      {docs.map((doc) => {
        const statusLabel =
          doc.status === "uploading"
            ? "Uploading…"
            : doc.status === "ingesting"
              ? "Indexing…"
              : doc.status === "error"
                ? (doc.errorMessage ?? "Failed")
                : "Ready";
        const statusClass =
          doc.status === "error"
            ? "text-destructive"
            : doc.status === "ready"
              ? "text-muted-foreground"
              : "text-muted-foreground italic";
        return (
          <div
            key={doc.id}
            className="flex items-center gap-2 rounded-lg border border-foreground/20 bg-muted px-3 py-1.5 text-xs"
          >
            <FileTextIcon className="size-3.5 text-muted-foreground" />
            <div className="flex flex-col">
              <span className="max-w-48 truncate">{doc.file.name}</span>
              <span className={cn("text-[10px] leading-tight", statusClass)}>
                {statusLabel}
              </span>
            </div>
            <button
              type="button"
              className="text-muted-foreground hover:text-destructive"
              onClick={() => onRemove(doc.id)}
              aria-label="Remove document"
            >
              <XIcon className="size-3.5" />
            </button>
          </div>
        );
      })}
    </div>
  );
};
