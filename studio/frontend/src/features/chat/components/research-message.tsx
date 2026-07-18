// SPDX-License-Identifier: AGPL-3.0-only

import type { Citation } from "@/components/assistant-ui/citation-utils";
import { DocumentSourcesGroup } from "@/components/assistant-ui/rag-sources";
import {
  type SourceData,
  SourcesGroup,
} from "@/components/assistant-ui/sources";
import { MarkdownPreview } from "@/components/markdown/markdown-preview";
import { Button } from "@/components/ui/button";
import { Spinner } from "@/components/ui/spinner";
import { cn } from "@/lib/utils";
import { useAuiState } from "@assistant-ui/react";
import { Check, Telescope, TriangleAlert } from "lucide-react";
import { type ReactElement, useEffect } from "react";
import {
  ensureResearchRunFollowed,
  ingestResearchUpdate,
  useResearchRunStore,
} from "../stores/research-run-store";
import type { ResearchMessageMetadata } from "../types/research";
import { researchStatusLabel } from "./research-activity-panel";

export function ResearchMessage(): ReactElement {
  const metadata = useAuiState(
    ({ message }) =>
      (message.metadata as { custom?: ResearchMessageMetadata } | undefined)
        ?.custom ?? {},
  );
  const fallbackText = useAuiState(({ message }) =>
    message.parts
      .filter((part) => part.type === "text")
      .map((part) => part.text)
      .join("\n"),
  );
  const runId = metadata.researchRunId ?? metadata.researchRun?.id ?? "";
  const session = useResearchRunStore((state) => state.sessions[runId]);
  const openPanel = useResearchRunStore((state) => state.openPanel);
  const initialRun = metadata.researchRun;

  useEffect(() => {
    if (!runId) {
      return;
    }
    if (initialRun) {
      ingestResearchUpdate(initialRun);
    }
    if (!session?.following) {
      ensureResearchRunFollowed(runId, initialRun);
    }
  }, [runId, initialRun, session?.following]);

  const run = session?.run ?? metadata.researchRun;
  if (!run) {
    if (fallbackText.trim()) {
      return (
        <MarkdownPreview
          markdown={fallbackText}
          className="max-h-none overflow-visible border-0 bg-transparent p-0 text-[15.5px]"
        />
      );
    }
    return (
      <div className="flex items-center gap-2 text-sm text-muted-foreground">
        <Spinner /> Loading research…
      </div>
    );
  }

  if (run.status === "completed" && run.report) {
    const sources: SourceData[] = run.sources.map((source) => ({
      id: String(source.id ?? source.url),
      url: source.url,
      title: source.title || source.url,
      description: source.snippet ?? undefined,
    }));
    const documentSources: Citation[] = (run.documentSources ?? []).map(
      (source, index) => ({
        id: source.chunkId ?? String(source.id ?? index),
        filename: source.filename,
        page: source.page,
        score: source.score,
        text: source.snippet ?? "",
        documentId: source.documentId,
        chunkId: source.chunkId,
      }),
    );
    const documentCount = new Set(
      documentSources.map((source) => source.documentId ?? source.filename),
    ).size;
    const sourceCount = sources.length + documentCount;
    return (
      <div className="min-w-0">
        <button
          type="button"
          onClick={() => openPanel(run.id)}
          className="mb-3 flex items-center gap-2 rounded-full text-sm text-muted-foreground transition-colors hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
        >
          <span className="flex size-5 items-center justify-center rounded-full bg-primary/10 text-primary">
            <Check className="size-3" />
          </span>
          <span>Deep research completed · {sourceCount} sources</span>
          <span className="text-primary">View activity</span>
        </button>
        <MarkdownPreview
          markdown={run.report}
          className="max-h-none overflow-visible border-0 bg-transparent p-0 text-[15.5px]"
        />
        <SourcesGroup sources={sources} />
        <DocumentSourcesGroup sources={documentSources} />
      </div>
    );
  }

  const failed = run.status === "failed";
  const cancelled = run.status === "cancelled";
  const needsApproval = run.status === "awaiting_approval";
  return (
    <div
      className={cn(
        "rounded-[22px] border border-border/70 bg-card/65 p-4",
        needsApproval && "border-amber-500/25 bg-amber-500/[0.035]",
        failed && "border-destructive/25 bg-destructive/[0.025]",
      )}
    >
      <div className="flex items-start gap-3">
        <span
          className={cn(
            "mt-0.5 flex size-8 shrink-0 items-center justify-center rounded-[12px] bg-primary/10 text-primary",
            failed && "bg-destructive/10 text-destructive",
          )}
        >
          {failed ? (
            <TriangleAlert className="size-4" />
          ) : cancelled ? (
            <Telescope className="size-4" />
          ) : (
            <Spinner className="size-4" />
          )}
        </span>
        <div className="min-w-0 flex-1">
          <p className="font-heading text-sm font-medium">
            {failed
              ? "Research could not be completed"
              : cancelled
                ? "Research stopped"
                : needsApproval
                  ? "Your research plan is ready"
                  : researchStatusLabel(run.status)}
          </p>
          <p className="mt-1 text-[12.5px] leading-relaxed text-muted-foreground">
            {session?.error
              ? session.error
              : failed
                ? run.error
                : needsApproval
                  ? "Review the approach before the agent starts gathering evidence."
                  : cancelled
                    ? "The activity gathered so far is still available."
                    : (run.plan?.title ?? "Building a rigorous research plan…")}
          </p>
          <Button
            size="sm"
            variant={needsApproval ? "default" : "outline"}
            className="mt-3"
            onClick={() => openPanel(run.id)}
          >
            {needsApproval ? "Review plan" : "View activity"}
          </Button>
        </div>
      </div>
    </div>
  );
}
