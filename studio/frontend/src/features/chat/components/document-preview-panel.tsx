// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"use client";

import { MarkdownPreview } from "@/components/markdown/markdown-preview";
import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogTitle } from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
  SheetTrigger,
} from "@/components/ui/sheet";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { cn } from "@/lib/utils";
import {
  ChevronLeftIcon,
  ChevronRightIcon,
  CopyIcon,
  FileTextIcon,
  ImageIcon,
  InfoIcon,
  ListIcon,
  SearchIcon,
} from "lucide-react";
import {
  type PropsWithChildren,
  type ReactElement,
  useEffect,
  useId,
  useMemo,
  useRef,
  useState,
} from "react";
import { toast } from "sonner";
import type { ExtractedDocument, ExtractedFigure } from "../types";
import {
  documentFigureImageDataUrl,
  documentImageReferenceLabel,
} from "../utils/document-extraction";

const FIGURE_ROW_HEIGHT = 104;
const FIGURE_LIST_MAX_HEIGHT = 360;
const FIGURE_OVERSCAN = 4;
const SOURCE_LINE_HEIGHT = 22;
const SOURCE_VIEW_MIN_HEIGHT = 416;
const SOURCE_VIEW_MAX_HEIGHT = 520;
const SOURCE_OVERSCAN = 8;

type TocItem = {
  id: string;
  depth: number;
  text: string;
  line: number;
};

type SearchResult = {
  line: number;
  text: string;
};

type FigureLightboxState = {
  url: string;
  label: string;
  caption: string;
};

export type DocumentSheetNavigation = {
  currentIndex: number;
  totalCount: number;
  onNavigate: (direction: -1 | 1) => void;
};

function formatTokens(tokens: number | undefined): string {
  if (typeof tokens !== "number") return "";
  if (tokens < 1000) return `${tokens}`;
  return `${(tokens / 1000).toFixed(1)}k`;
}

function formatBytes(bytes: number | undefined): string {
  if (typeof bytes !== "number") return "";
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function cleanHeading(value: string): string {
  return value
    .replace(/!\[[^\]]*]\([^)]*\)/g, "")
    .replace(/\[([^\]]+)]\([^)]*\)/g, "$1")
    .replace(/[*_`>#-]/g, "")
    .replace(/\s+/g, " ")
    .trim();
}

function buildToc(markdown: string, idPrefix: string): TocItem[] {
  return markdown
    .split(/\r?\n/)
    .map((line, index) => {
      const match = /^(#{1,6})\s+(.+?)\s*#*\s*$/.exec(line);
      if (!match) return null;
      const text = cleanHeading(match[2] ?? "");
      if (!text) return null;
      return {
        id: `${idPrefix}-toc-${index}`,
        depth: match[1]?.length ?? 1,
        text,
        line: index + 1,
      };
    })
    .filter((item): item is TocItem => item !== null);
}

function findSearchResults(markdown: string, query: string): SearchResult[] {
  const q = query.trim().toLowerCase();
  if (!q) return [];
  return markdown
    .split(/\r?\n/)
    .map((line, index) =>
      line.toLowerCase().includes(q) ? { line: index + 1, text: line } : null,
    )
    .filter((item): item is SearchResult => item !== null)
    .slice(0, 50);
}

function highlightText(text: string, query: string): ReactElement | string {
  const q = query.trim();
  if (!q) return text || " ";
  const lower = text.toLowerCase();
  const needle = q.toLowerCase();
  const parts: ReactElement[] = [];
  let cursor = 0;
  let matchIndex = lower.indexOf(needle, cursor);
  let key = 0;
  while (matchIndex !== -1) {
    if (matchIndex > cursor) {
      parts.push(<span key={key++}>{text.slice(cursor, matchIndex)}</span>);
    }
    parts.push(
      <mark
        key={key++}
        className="rounded-sm bg-primary/20 px-0.5 text-foreground"
      >
        {text.slice(matchIndex, matchIndex + needle.length)}
      </mark>,
    );
    cursor = matchIndex + needle.length;
    matchIndex = lower.indexOf(needle, cursor);
  }
  if (cursor < text.length) {
    parts.push(<span key={key++}>{text.slice(cursor)}</span>);
  }
  return <>{parts.length > 0 ? parts : " "}</>;
}

function figureStatus(figure: ExtractedFigure): string {
  if (figure.caption) return figure.caption;
  if (figure.error) return `Caption failed: ${figure.error}`;
  if (figure.image_base64) {
    return figure.kind === "page"
      ? "Full page image attached for visual inspection"
      : "Image attached for visual inspection";
  }
  return "No caption produced";
}

function ProvenanceField({
  label,
  value,
}: {
  label: string;
  value: string | number | null | undefined;
}): ReactElement | null {
  if (value === null || value === undefined || value === "") return null;
  return (
    <div className="grid grid-cols-[5.75rem_minmax(0,1fr)] gap-2 text-xs">
      <dt className="text-muted-foreground">{label}</dt>
      <dd className="min-w-0 break-words font-medium text-foreground">
        {value}
      </dd>
    </div>
  );
}

function FigureRow({
  figure,
  index,
  sentImageIndexes,
  onSelectFigure,
}: {
  figure: ExtractedFigure;
  index: number;
  sentImageIndexes: ReadonlySet<number>;
  onSelectFigure: (figure: FigureLightboxState) => void;
}): ReactElement {
  const imageUrl = useMemo(
    () => documentFigureImageDataUrl(figure),
    [figure.image_base64, figure.image_mime],
  );
  const label = documentImageReferenceLabel(index);
  return (
    <div className="grid h-[104px] gap-2 overflow-hidden border-b px-3 py-2 text-xs last:border-b-0 sm:grid-cols-[8rem_minmax(0,1fr)]">
      <div className="flex min-w-0 items-start gap-2 font-medium">
        {imageUrl ? (
          <button
            type="button"
            onClick={() =>
              onSelectFigure({
                url: imageUrl,
                label,
                caption: figureStatus(figure),
              })
            }
            className="group relative h-16 w-20 shrink-0 cursor-zoom-in overflow-hidden rounded-md border bg-background transition hover:border-primary/60 focus:outline-none focus-visible:ring-2 focus-visible:ring-ring"
            aria-label={`Open ${label} at full size`}
          >
            <img
              src={imageUrl}
              alt={figure.caption || label}
              className="h-full w-full object-cover transition group-hover:scale-105"
              loading="lazy"
            />
          </button>
        ) : (
          <span className="flex h-16 w-20 shrink-0 items-center justify-center rounded-md border bg-background">
            <ImageIcon
              className="size-4 text-muted-foreground"
              aria-hidden="true"
            />
          </span>
        )}
        <span className="pt-0.5">{label}</span>
      </div>
      <div className="min-w-0">
        <div className="flex min-w-0 flex-wrap gap-x-2 gap-y-0.5 text-[11px] text-muted-foreground">
          <span>
            {figure.page == null ? "Page unknown" : `Page ${figure.page}`}
          </span>
          <span>{figure.kind === "page" ? "page image" : "figure"}</span>
          {sentImageIndexes.has(index) ? (
            <span className="font-medium text-primary">sent visual</span>
          ) : null}
        </div>
        <p className="mt-1 line-clamp-3 whitespace-pre-wrap break-words text-foreground">
          {figureStatus(figure)}
        </p>
      </div>
    </div>
  );
}

function VirtualizedFigureList({
  figures,
  sentImageIndexes,
  onSelectFigure,
}: {
  figures: ExtractedFigure[];
  sentImageIndexes: ReadonlySet<number>;
  onSelectFigure: (figure: FigureLightboxState) => void;
}): ReactElement {
  const [scrollTop, setScrollTop] = useState(0);
  const height = Math.min(
    FIGURE_LIST_MAX_HEIGHT,
    Math.max(FIGURE_ROW_HEIGHT, figures.length * FIGURE_ROW_HEIGHT),
  );
  const start = Math.max(
    0,
    Math.floor(scrollTop / FIGURE_ROW_HEIGHT) - FIGURE_OVERSCAN,
  );
  const visibleCount =
    Math.ceil(height / FIGURE_ROW_HEIGHT) + FIGURE_OVERSCAN * 2;
  const end = Math.min(figures.length, start + visibleCount);
  const visible = figures.slice(start, end);

  return (
    <div
      className="overflow-auto rounded-md border bg-muted/20"
      style={{ height }}
      onScroll={(event) => setScrollTop(event.currentTarget.scrollTop)}
    >
      <div
        className="relative"
        style={{ height: figures.length * FIGURE_ROW_HEIGHT }}
      >
        <div
          className="absolute inset-x-0 top-0"
          style={{ transform: `translateY(${start * FIGURE_ROW_HEIGHT}px)` }}
        >
          {visible.map((figure, offset) => {
            const index = start + offset;
            return (
              <FigureRow
                key={figure.id || index}
                figure={figure}
                index={index}
                sentImageIndexes={sentImageIndexes}
                onSelectFigure={onSelectFigure}
              />
            );
          })}
        </div>
      </div>
    </div>
  );
}

function VirtualizedSourceLines({
  lines,
  query,
  selectedLine,
  idPrefix,
  filename,
}: {
  lines: string[];
  query: string;
  selectedLine: number | null;
  idPrefix: string;
  filename: string;
}): ReactElement {
  const viewportRef = useRef<HTMLDivElement | null>(null);
  const [scrollTop, setScrollTop] = useState(0);
  const height = Math.min(
    SOURCE_VIEW_MAX_HEIGHT,
    Math.max(SOURCE_VIEW_MIN_HEIGHT, lines.length * SOURCE_LINE_HEIGHT),
  );
  const start = Math.max(
    0,
    Math.floor(scrollTop / SOURCE_LINE_HEIGHT) - SOURCE_OVERSCAN,
  );
  const visibleCount =
    Math.ceil(height / SOURCE_LINE_HEIGHT) + SOURCE_OVERSCAN * 2;
  const end = Math.min(lines.length, start + visibleCount);
  const visible = lines.slice(start, end);

  useEffect(() => {
    if (selectedLine === null) return;
    viewportRef.current?.scrollTo({
      top: Math.max(0, (selectedLine - 1) * SOURCE_LINE_HEIGHT - height / 2),
      behavior: "smooth",
    });
  }, [height, selectedLine]);

  return (
    <div
      ref={viewportRef}
      className="mt-2 min-h-[26rem] min-w-0 overflow-auto rounded-lg border bg-background p-3 font-mono text-xs leading-relaxed"
      style={{ height, maxHeight: "62dvh" }}
      role="region"
      aria-label={`Extracted source text from ${filename}`}
      tabIndex={0}
      onScroll={(event) => setScrollTop(event.currentTarget.scrollTop)}
    >
      <div
        className="relative"
        style={{ height: lines.length * SOURCE_LINE_HEIGHT }}
      >
        <div
          className="absolute inset-x-0 top-0"
          style={{ transform: `translateY(${start * SOURCE_LINE_HEIGHT}px)` }}
        >
          {visible.map((line, offset) => {
            const lineNumber = start + offset + 1;
            return (
              <div
                key={lineNumber}
                id={`${idPrefix}-line-${lineNumber}`}
                className={cn(
                  "grid h-[22px] grid-cols-[3.5rem_minmax(0,1fr)] gap-3 overflow-hidden rounded-sm px-1",
                  selectedLine === lineNumber && "bg-primary/10",
                )}
              >
                <span className="select-none text-right text-muted-foreground">
                  {lineNumber}
                </span>
                <span className="truncate whitespace-pre break-words">
                  {highlightText(line, query)}
                </span>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}

export function DocumentPreviewPanel({
  document: extractedDocument,
  filename,
  sizeBytes,
  extractedAt,
  sentImageIndexes = new Set<number>(),
}: {
  document: ExtractedDocument;
  filename: string;
  sizeBytes?: number;
  extractedAt?: number;
  sentImageIndexes?: ReadonlySet<number>;
}): ReactElement {
  const [activeTab, setActiveTab] = useState("preview");
  const [query, setQuery] = useState("");
  const [selectedLine, setSelectedLine] = useState<number | null>(null);
  const [lightboxFigure, setLightboxFigure] =
    useState<FigureLightboxState | null>(null);
  const idPrefix = useId().replace(/:/g, "");
  const markdown = extractedDocument.markdown.trim();
  const sourceText =
    markdown ||
    `No extractable text. ${extractedDocument.figures.length} image reference${
      extractedDocument.figures.length === 1 ? "" : "s"
    } detected.`;
  const sourceLines = useMemo(() => sourceText.split(/\r?\n/), [sourceText]);
  const toc = useMemo(() => buildToc(markdown, idPrefix), [markdown, idPrefix]);
  const searchResults = useMemo(
    () => findSearchResults(sourceText, query),
    [sourceText, query],
  );

  const copyMarkdown = (): void => {
    void navigator.clipboard
      .writeText(extractedDocument.markdown)
      .then(() => toast.success("Markdown copied"))
      .catch(() => toast.error("Copy failed"));
  };

  const openLine = (line: number): void => {
    setSelectedLine(line);
    setActiveTab("source");
  };

  const visualCount = sentImageIndexes.size;
  const extractedAtLabel = extractedAt
    ? new Date(extractedAt).toLocaleString()
    : null;

  return (
    <div className="grid min-h-0 flex-1 gap-4 lg:grid-cols-[14rem_minmax(0,1fr)]">
      <aside className="min-h-0 min-w-0 space-y-3 overflow-auto rounded-lg border bg-muted/20 p-3">
        <div className="space-y-2">
          <div className="flex items-center gap-2 text-xs font-semibold">
            <InfoIcon className="size-3.5 text-muted-foreground" />
            Provenance
          </div>
          <dl className="space-y-1.5">
            <ProvenanceField label="File" value={filename} />
            <ProvenanceField label="Size" value={formatBytes(sizeBytes)} />
            <ProvenanceField label="Extracted" value={extractedAtLabel} />
            <ProvenanceField
              label="Pages"
              value={extractedDocument.page_count}
            />
            <ProvenanceField
              label="Tokens"
              value={formatTokens(extractedDocument.tokens_est)}
            />
            <ProvenanceField
              label="Figures"
              value={extractedDocument.figures.length}
            />
            <ProvenanceField
              label="Visuals"
              value={visualCount > 0 ? `${visualCount} sent` : "text only"}
            />
            <ProvenanceField label="VLM" value={extractedDocument.vlm_model} />
            <ProvenanceField
              label="Backend"
              value={extractedDocument.vlm_source}
            />
          </dl>
        </div>

        {extractedDocument.warnings.length > 0 ? (
          <div className="rounded-md border border-amber-400/40 bg-amber-50/70 px-3 py-2 text-xs text-amber-900 dark:bg-amber-950/30 dark:text-amber-100">
            <div className="mb-1 font-medium">Warnings</div>
            <ul className="list-disc space-y-1 pl-4">
              {extractedDocument.warnings.map((warning, index) => (
                <li key={`${warning}-${index}`}>{warning}</li>
              ))}
            </ul>
          </div>
        ) : null}

        <div className="space-y-2">
          <label
            htmlFor={`${idPrefix}-search`}
            className="flex items-center gap-2 text-xs font-semibold"
          >
            <SearchIcon className="size-3.5 text-muted-foreground" />
            Search
          </label>
          <Input
            id={`${idPrefix}-search`}
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="Find in document"
            className="h-8 text-xs"
          />
          {query.trim() ? (
            <div className="max-h-40 space-y-1 overflow-auto">
              {searchResults.length > 0 ? (
                searchResults.map((result) => (
                  <button
                    key={`${result.line}-${result.text}`}
                    type="button"
                    className="block w-full rounded-md px-2 py-1 text-left text-xs hover:bg-accent focus:outline-none focus-visible:ring-2 focus-visible:ring-ring"
                    onClick={() => openLine(result.line)}
                  >
                    <span className="mr-2 text-muted-foreground">
                      {result.line}
                    </span>
                    <span className="line-clamp-2">
                      {highlightText(result.text, query)}
                    </span>
                  </button>
                ))
              ) : (
                <p className="px-2 text-xs text-muted-foreground">No matches</p>
              )}
            </div>
          ) : null}
        </div>

        <div className="space-y-2">
          <div className="flex items-center gap-2 text-xs font-semibold">
            <ListIcon className="size-3.5 text-muted-foreground" />
            Table of Contents
          </div>
          {toc.length > 0 ? (
            <div className="max-h-56 space-y-0.5 overflow-auto">
              {toc.map((item) => (
                <button
                  key={item.id}
                  type="button"
                  className="block w-full rounded-md py-1 pr-2 text-left text-xs hover:bg-accent focus:outline-none focus-visible:ring-2 focus-visible:ring-ring"
                  style={{
                    paddingLeft: `${Math.min(item.depth - 1, 4) * 10 + 8}px`,
                  }}
                  onClick={() => openLine(item.line)}
                >
                  <span className="line-clamp-2">{item.text}</span>
                </button>
              ))}
            </div>
          ) : (
            <p className="text-xs text-muted-foreground">No headings found</p>
          )}
        </div>
      </aside>

      <Tabs
        value={activeTab}
        onValueChange={setActiveTab}
        className="min-h-0 min-w-0"
      >
        <div className="flex flex-wrap items-center justify-between gap-2">
          <TabsList className="h-9">
            <TabsTrigger value="preview">Preview</TabsTrigger>
            <TabsTrigger value="source">Source</TabsTrigger>
            <TabsTrigger value="figures">
              Figures
              {extractedDocument.figures.length > 0
                ? ` (${extractedDocument.figures.length})`
                : ""}
            </TabsTrigger>
          </TabsList>
          <Button
            type="button"
            size="sm"
            variant="secondary"
            onClick={copyMarkdown}
          >
            <CopyIcon className="size-4" aria-hidden="true" />
            Copy Markdown
          </Button>
        </div>

        <TabsContent value="preview" className="min-h-0">
          <div className="mt-2 min-h-[26rem] min-w-0 overflow-hidden rounded-lg border bg-background">
            <MarkdownPreview
              markdown={markdown || "_No extractable text._"}
              className="max-h-[62dvh] min-h-[26rem] rounded-none border-0 bg-background p-4 text-sm leading-6"
            />
          </div>
        </TabsContent>

        <TabsContent value="source" className="min-h-0">
          <VirtualizedSourceLines
            lines={sourceLines}
            query={query}
            selectedLine={activeTab === "source" ? selectedLine : null}
            idPrefix={idPrefix}
            filename={filename}
          />
        </TabsContent>

        <TabsContent value="figures" className="min-h-0">
          <div className="mt-2 min-h-[26rem] min-w-0 rounded-lg border bg-background p-3">
            {extractedDocument.figures.length > 0 ? (
              <VirtualizedFigureList
                figures={extractedDocument.figures}
                sentImageIndexes={sentImageIndexes}
                onSelectFigure={setLightboxFigure}
              />
            ) : (
              <div className="flex min-h-48 flex-col items-center justify-center gap-2 text-center text-sm text-muted-foreground">
                <FileTextIcon className="size-8" aria-hidden="true" />
                No image references were extracted.
              </div>
            )}
          </div>
        </TabsContent>
      </Tabs>

      <Dialog
        open={lightboxFigure !== null}
        onOpenChange={(open) => {
          if (!open) setLightboxFigure(null);
        }}
      >
        <DialogContent
          className="flex max-h-[92vh] !w-[min(1200px,94vw)] !max-w-none flex-col gap-3 rounded-2xl bg-background/95 p-4 backdrop-blur"
          showCloseButton={true}
        >
          <DialogTitle className="pr-10 text-sm font-medium">
            {lightboxFigure?.label ?? "Figure"}
          </DialogTitle>
          {lightboxFigure ? (
            <div className="flex min-h-0 flex-1 items-center justify-center overflow-auto rounded-lg bg-black/40 p-2">
              <img
                src={lightboxFigure.url}
                alt={lightboxFigure.caption || lightboxFigure.label}
                className="max-h-[78vh] max-w-full object-contain"
              />
            </div>
          ) : null}
          {lightboxFigure?.caption ? (
            <p className="max-h-32 overflow-auto whitespace-pre-wrap text-xs text-muted-foreground">
              {lightboxFigure.caption}
            </p>
          ) : null}
        </DialogContent>
      </Dialog>
    </div>
  );
}

export function DocumentPreviewSheet({
  document: extractedDocument,
  filename,
  sizeBytes,
  extractedAt,
  sentImageIndexes,
  navigation,
  open,
  onOpenChange,
  children,
}: PropsWithChildren<{
  document: ExtractedDocument;
  filename: string;
  sizeBytes?: number;
  extractedAt?: number;
  sentImageIndexes?: ReadonlySet<number>;
  navigation?: DocumentSheetNavigation;
  open?: boolean;
  onOpenChange?: (open: boolean) => void;
}>): ReactElement {
  const showNav = navigation !== undefined && navigation.totalCount > 1;
  return (
    <Sheet open={open} onOpenChange={onOpenChange}>
      <SheetTrigger asChild={true}>{children}</SheetTrigger>
      <SheetContent
        side="right"
        className="flex !w-[min(1100px,94vw)] !max-w-none flex-col p-0 data-[side=right]:!w-[min(1100px,94vw)] data-[side=right]:sm:!max-w-none"
      >
        <SheetHeader className="border-b pr-14">
          <SheetTitle className="flex min-w-0 items-center gap-2">
            <FileTextIcon
              className="size-4 shrink-0 text-muted-foreground"
              aria-hidden="true"
            />
            <span className="truncate">{filename}</span>
            {showNav && navigation ? (
              <span className="ml-auto flex shrink-0 items-center gap-1 pl-2 text-xs font-normal text-muted-foreground">
                <button
                  type="button"
                  onClick={() => navigation.onNavigate(-1)}
                  className="inline-flex size-7 items-center justify-center rounded-full border border-border/70 bg-background text-muted-foreground shadow-sm transition-colors hover:bg-accent hover:text-foreground focus:outline-none focus-visible:ring-2 focus-visible:ring-ring dark:bg-card"
                  aria-label="Previous attached document"
                  title="Previous document"
                >
                  <ChevronLeftIcon className="size-3.5" aria-hidden="true" />
                </button>
                <span className="select-none px-1 tabular-nums">
                  {navigation.currentIndex + 1} / {navigation.totalCount}
                </span>
                <button
                  type="button"
                  onClick={() => navigation.onNavigate(1)}
                  className="inline-flex size-7 items-center justify-center rounded-full border border-border/70 bg-background text-muted-foreground shadow-sm transition-colors hover:bg-accent hover:text-foreground focus:outline-none focus-visible:ring-2 focus-visible:ring-ring dark:bg-card"
                  aria-label="Next attached document"
                  title="Next document"
                >
                  <ChevronRightIcon className="size-3.5" aria-hidden="true" />
                </button>
              </span>
            ) : null}
          </SheetTitle>
          <SheetDescription>
            {extractedDocument.page_count} page
            {extractedDocument.page_count === 1 ? "" : "s"} -{" "}
            {formatTokens(extractedDocument.tokens_est)} tokens -{" "}
            {extractedDocument.figures.length} figure
            {extractedDocument.figures.length === 1 ? "" : "s"}
          </SheetDescription>
        </SheetHeader>
        <div className="min-h-0 flex-1 overflow-hidden p-4">
          <DocumentPreviewPanel
            document={extractedDocument}
            filename={filename}
            sizeBytes={sizeBytes}
            extractedAt={extractedAt}
            sentImageIndexes={sentImageIndexes}
          />
        </div>
      </SheetContent>
    </Sheet>
  );
}
