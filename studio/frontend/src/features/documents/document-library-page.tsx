import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import {
  Empty,
  EmptyDescription,
  EmptyHeader,
  EmptyMedia,
  EmptyTitle,
} from "@/components/ui/empty";
import { Input } from "@/components/ui/input";
import {
  Pagination,
  PaginationContent,
  PaginationItem,
} from "@/components/ui/pagination";
import { Progress } from "@/components/ui/progress";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Spinner } from "@/components/ui/spinner";
import { Tabs, TabsContent } from "@/components/ui/tabs";
import { PageHeading } from "@/features/hub/components/page-heading";
import { HubTopBar } from "@/features/hub/catalog/hub-top-bar";
import {
  type AllModelsView,
  HubListHeader,
} from "@/features/hub/catalog/models-table";
import {
  deleteGenericDocument,
  downloadDatasetDocument,
  ingestGenericDocuments,
  inspectDocumentUploads,
  updateGenericDocument,
  type PlatformDocument,
  hasControlCharacters,
  type PlatformUploadInspection,
} from "@/integrations/platform-backend";
import { cn } from "@/lib/utils";
import { toast } from "@/lib/toast";
import {
  Ban,
  CircleAlert,
  ChevronLeft,
  ChevronRight,
  Database,
  Download,
  Eye,
  FileArchive,
  FileCheck2,
  FileText,
  ImageIcon,
  MoreHorizontal,
  Pencil,
  Play,
  RefreshCw,
  RotateCcw,
  Search,
  ShieldCheck,
  Square,
  Trash2,
  UploadCloud,
  X,
} from "lucide-react";
import { useDeferredValue, useEffect, useMemo, useRef, useState } from "react";
import {
  DocumentAssetDialog,
  DocumentInlinePreview,
} from "./document-asset-dialog";
import { useDocumentLibrary } from "./use-document-library";

type DocumentTab = "dataset" | "generic";

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  const units = ["KB", "MB", "GB"];
  let value = bytes / 1024;
  let unit = units[0];
  for (let index = 1; value >= 1024 && index < units.length; index += 1) {
    value /= 1024;
    unit = units[index];
  }
  return `${value.toFixed(value >= 10 ? 0 : 1)} ${unit}`;
}

const STATUS_COPY = {
  pending: { label: "Bekliyor", className: "bg-muted text-muted-foreground" },
  running: {
    label: "İşleniyor",
    className: "bg-blue-500/10 text-blue-700 dark:text-blue-300",
  },
  completed: {
    label: "Hazır",
    className: "bg-emerald-500/10 text-emerald-700 dark:text-emerald-300",
  },
  cancelled: {
    label: "Durduruldu",
    className: "bg-amber-500/10 text-amber-700 dark:text-amber-300",
  },
  failed: {
    label: "Başarısız",
    className: "bg-destructive/10 text-destructive",
  },
} as const;

function visiblePageNumbers(page: number, totalPages: number): number[] {
  const first = Math.max(1, Math.min(page - 2, totalPages - 4));
  const last = Math.min(totalPages, first + 4);
  return Array.from({ length: last - first + 1 }, (_, index) => first + index);
}

function DocumentStatPill({
  icon: Icon,
  label,
  value,
}: {
  icon: typeof Database;
  label: string;
  value: number;
}) {
  return (
    <span className="hub-stat-pill">
      <Icon className="size-3.5" strokeWidth={1.75} />
      <span className="hub-stat-pill-value">
        {value.toLocaleString("tr-TR")}
      </span>
      <span>{label}</span>
    </span>
  );
}

function DocumentScopeToggle({
  value,
  onChange,
}: {
  value: DocumentTab;
  onChange: (value: DocumentTab) => void;
}) {
  return (
    <div
      className="hub-menu-trigger hub-tab-toggle relative inline-flex h-9 w-full shrink-0 items-center rounded-full lg:w-[280px]"
      role="radiogroup"
      aria-label="Belge kapsamı"
    >
      <span
        aria-hidden="true"
        className={cn(
          "hub-tab-toggle-pill pointer-events-none absolute inset-y-0 left-0 w-1/2 rounded-full transition-transform duration-200 ease-out",
          value === "generic" ? "translate-x-full" : "translate-x-0",
        )}
      />
      <button
        type="button"
        role="radio"
        aria-checked={value === "dataset"}
        onClick={() => onChange("dataset")}
        className={cn(
          "relative z-10 inline-flex h-9 flex-1 items-center justify-center rounded-full px-3 text-ui-12p5 transition-colors",
          value === "dataset"
            ? "text-foreground"
            : "text-muted-foreground hover:text-foreground",
        )}
      >
        Dataset belgeleri
      </button>
      <button
        type="button"
        role="radio"
        aria-checked={value === "generic"}
        onClick={() => onChange("generic")}
        className={cn(
          "relative z-10 inline-flex h-9 flex-1 items-center justify-center rounded-full px-3 text-ui-12p5 transition-colors",
          value === "generic"
            ? "text-foreground"
            : "text-muted-foreground hover:text-foreground",
        )}
      >
        Genel belgeler
      </button>
    </div>
  );
}

function formatDocumentDate(value: string | null): string {
  if (!value) return "Tarih yok";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "Tarih yok";
  return new Intl.DateTimeFormat("tr-TR", {
    dateStyle: "medium",
    timeStyle: "short",
  }).format(date);
}

function downloadName(document: PlatformDocument): string {
  const sanitized = Array.from(
    document.name.replace(/[\\/:*?"<>|]/g, "_"),
    (character) => (hasControlCharacters(character) ? "_" : character),
  ).join("");
  return sanitized || "document";
}

function DocumentSummaryCard({
  document,
  mode,
  focused,
  checked,
  onActivate,
  onCheckedChange,
}: {
  document: PlatformDocument;
  mode: "two" | "split";
  focused: boolean;
  checked: boolean;
  onActivate: () => void;
  onCheckedChange: (checked: boolean) => void;
}) {
  const status = STATUS_COPY[document.status];
  const active = document.status === "pending" || document.status === "running";
  if (mode === "split") {
    return (
      <div
        className="group/row flex h-14 w-full items-center gap-2.5 rounded-[12px] px-2.5 text-left outline-none transition-colors hover:bg-foreground/[0.04] data-[selected]:bg-foreground/[0.07] focus-within:ring-1 focus-within:ring-ring focus-within:ring-inset dark:hover:bg-white/[0.05] dark:data-[selected]:bg-white/[0.08]"
        data-selected={focused || undefined}
      >
        <Checkbox
          checked={checked}
          onCheckedChange={(value) => onCheckedChange(value === true)}
          aria-label={`${document.name} seç`}
          className="shrink-0"
        />
        <button
          type="button"
          aria-label={`${document.name} ayrıntılarını aç`}
          onClick={onActivate}
          className="flex min-w-0 flex-1 items-center gap-2.5 rounded-[9px] text-left outline-none"
        >
          <span className="flex size-8 shrink-0 items-center justify-center rounded-[9px] bg-foreground/[0.055]">
            <FileText className="size-3.5" strokeWidth={1.75} />
          </span>
          <span className="flex min-w-0 flex-1 flex-col">
            <span className="flex min-w-0 items-center gap-1.5">
              <span className="truncate text-ui-12p5 font-semibold leading-ui-16 text-foreground">
                {document.name}
              </span>
              <span
                aria-label={status.label}
                className={cn(
                  "size-1.5 shrink-0 rounded-full",
                  document.status === "completed"
                    ? "bg-status-success"
                    : document.status === "failed"
                      ? "bg-status-danger"
                      : document.status === "cancelled"
                        ? "bg-status-warning"
                        : "bg-blue-500",
                )}
              />
            </span>
            <span className="mt-0.5 truncate text-ui-10p5 leading-ui-14 text-muted-foreground/80">
              {document.suffix.toUpperCase() || "DOSYA"} ·{" "}
              {formatBytes(document.sizeBytes)}
            </span>
          </span>
          <span className="flex shrink-0 flex-col items-end gap-0.5 text-ui-10p5 tabular-nums text-muted-foreground/70">
            <span>{Math.round(document.progress * 100)}%</span>
            <span>{document.chunkCount.toLocaleString("tr-TR")} parça</span>
          </span>
        </button>
      </div>
    );
  }

  return (
    <article
      className="hub-result-row hub-result-card group/row relative flex h-[78px] min-w-0 items-center gap-3 px-4"
      data-selected={focused || undefined}
    >
      <Checkbox
        checked={checked}
        onCheckedChange={(value) => onCheckedChange(value === true)}
        aria-label={`${document.name} seç`}
        className="relative z-[2] shrink-0"
      />
      <button
        type="button"
        aria-label={`${document.name} ayrıntılarını aç`}
        onClick={onActivate}
        className="absolute inset-0 z-0 cursor-pointer rounded-[inherit] outline-none focus-visible:ring-1 focus-visible:ring-ring focus-visible:ring-inset"
      />
      <div className="pointer-events-none relative z-[1] flex min-w-0 flex-1 items-center gap-3">
        <div className="flex size-9 shrink-0 items-center justify-center rounded-[12px] bg-foreground/[0.055]">
          <FileText className="size-4" strokeWidth={1.75} />
        </div>
        <div className="min-w-0 flex-1">
          <div className="flex min-w-0 items-center gap-2">
            <p className="truncate text-sm font-semibold">{document.name}</p>
            <span className="size-1.5 shrink-0 rounded-full bg-primary/70" />
          </div>
          <p className="mt-0.5 truncate text-xs text-muted-foreground">
            {document.suffix.toUpperCase() || "DOSYA"} ·{" "}
            {formatBytes(document.sizeBytes)}
          </p>
        </div>
        <div className="flex w-32 min-w-0 items-center gap-2">
          <Badge className={cn("shrink-0 gap-1 text-[10px]", status.className)}>
            {active ? <Spinner className="size-2.5" /> : null}
            {status.label}
          </Badge>
          <Progress
            value={document.progress * 100}
            className="h-1 min-w-8 flex-1"
          />
        </div>
        <span className="w-8 shrink-0 text-right text-ui-11 tabular-nums text-muted-foreground">
          {Math.round(document.progress * 100)}%
        </span>
      </div>
    </article>
  );
}

const DOCUMENT_LIST_COLUMNS = {
  select: "flex w-8 shrink-0 items-center",
  document: "flex min-w-0 flex-[2.4] items-center gap-3",
  status: "hidden w-[108px] shrink-0 sm:flex",
  progress: "hidden min-w-0 flex-1 items-center gap-2 md:flex",
  content: "hidden w-[116px] shrink-0 lg:block",
  actions: "flex w-[88px] shrink-0 items-center justify-end gap-0.5",
} as const;

function DocumentCompactHeader({
  allChecked,
  onAllCheckedChange,
}: {
  allChecked: boolean;
  onAllCheckedChange: (checked: boolean) => void;
}) {
  return (
    <div
      role="row"
      className="flex w-full items-center gap-3 px-4 pb-2 text-ui-11 font-medium text-muted-foreground/55"
    >
      <span className={DOCUMENT_LIST_COLUMNS.select}>
        <Checkbox
          checked={allChecked}
          onCheckedChange={(checked) => onAllCheckedChange(checked === true)}
          aria-label="Bu sayfadaki belgeleri seç"
        />
      </span>
      <span role="columnheader" className={DOCUMENT_LIST_COLUMNS.document}>
        Belge
      </span>
      <span role="columnheader" className={DOCUMENT_LIST_COLUMNS.status}>
        Durum
      </span>
      <span role="columnheader" className={DOCUMENT_LIST_COLUMNS.progress}>
        İlerleme
      </span>
      <span role="columnheader" className={DOCUMENT_LIST_COLUMNS.content}>
        İçerik
      </span>
      <span role="columnheader" className={DOCUMENT_LIST_COLUMNS.actions}>
        İşlemler
      </span>
    </div>
  );
}

function DocumentCompactRow({
  document,
  checked,
  renameValue,
  renaming,
  onCheckedChange,
  onActivate,
  onRenameValueChange,
  onCancelRename,
  onSaveRename,
  onBeginRename,
  onPreview,
  onMedia,
  onDownload,
  onProcess,
  onStop,
  onDelete,
}: {
  document: PlatformDocument;
  checked: boolean;
  renameValue: string;
  renaming: boolean;
  onCheckedChange: (checked: boolean) => void;
  onActivate: () => void;
  onRenameValueChange: (value: string) => void;
  onCancelRename: () => void;
  onSaveRename: () => void;
  onBeginRename: () => void;
  onPreview: () => void;
  onMedia: () => void;
  onDownload: () => void;
  onProcess: () => void;
  onStop: () => void;
  onDelete: () => void;
}) {
  const status = STATUS_COPY[document.status];
  const active = document.status === "pending" || document.status === "running";
  const canRetry =
    document.status === "failed" || document.status === "cancelled";

  return (
    <div
      role="row"
      className="hub-result-row group/row relative flex h-16 w-full items-center gap-3 px-4"
      data-selected={checked || undefined}
    >
      <div className={DOCUMENT_LIST_COLUMNS.select}>
        <Checkbox
          checked={checked}
          onCheckedChange={(value) => onCheckedChange(value === true)}
          aria-label={`${document.name} seç`}
          className="relative z-[2]"
        />
      </div>
      <div className={DOCUMENT_LIST_COLUMNS.document}>
        <div className="flex size-9 shrink-0 items-center justify-center rounded-[12px] bg-foreground/[0.055]">
          <FileText className="size-4" strokeWidth={1.75} />
        </div>
        <div className="min-w-0 flex-1">
          {renaming ? (
            <div className="relative z-[2] flex min-w-0 items-center gap-1">
              <Input
                value={renameValue}
                onChange={(event) => onRenameValueChange(event.target.value)}
                className="field-soft h-8 min-w-0 border-0 shadow-none"
                autoFocus
                onKeyDown={(event) => {
                  if (event.key === "Enter" && renameValue.trim())
                    onSaveRename();
                  if (event.key === "Escape") onCancelRename();
                }}
              />
              <Button size="icon-xs" variant="ghost" onClick={onCancelRename}>
                <X />
              </Button>
            </div>
          ) : (
            <button
              type="button"
              aria-label={document.name}
              onClick={onActivate}
              className="relative z-[1] block max-w-full truncate text-left text-ui-13p5 font-semibold leading-ui-17 text-foreground outline-none hover:underline"
            >
              {document.name}
            </button>
          )}
          <p className="mt-0.5 truncate text-ui-11p5 leading-ui-15 text-muted-foreground/80">
            {document.suffix.toUpperCase() || "DOSYA"} ·{" "}
            {formatBytes(document.sizeBytes)}
          </p>
        </div>
      </div>
      <div className={DOCUMENT_LIST_COLUMNS.status}>
        <Badge className={cn("gap-1 text-[11px]", status.className)}>
          {active ? (
            <Spinner className="size-3" />
          ) : document.status === "completed" ? (
            <FileCheck2 className="size-3" />
          ) : document.status === "failed" ? (
            <Ban className="size-3" />
          ) : null}
          {status.label}
        </Badge>
      </div>
      <div className={DOCUMENT_LIST_COLUMNS.progress}>
        <Progress
          value={document.progress * 100}
          className="h-1 min-w-10 flex-1"
        />
        <span className="w-8 shrink-0 text-right text-ui-11 tabular-nums text-muted-foreground">
          {Math.round(document.progress * 100)}%
        </span>
      </div>
      <div
        className={cn(
          DOCUMENT_LIST_COLUMNS.content,
          "text-ui-11p5 tabular-nums text-muted-foreground",
        )}
      >
        <p>{document.chunkCount.toLocaleString("tr-TR")} parça</p>
        <p className="text-muted-foreground/65">
          {document.tokenCount.toLocaleString("tr-TR")} token
        </p>
      </div>
      <div className={DOCUMENT_LIST_COLUMNS.actions}>
        <Button
          size="icon-xs"
          variant="ghost"
          title="Önizle"
          onClick={onPreview}
        >
          <Eye />
        </Button>
        {active ? (
          <Button
            size="icon-xs"
            variant="ghost"
            title="Durdur"
            onClick={onStop}
          >
            <Square />
          </Button>
        ) : (
          <Button
            size="icon-xs"
            variant="ghost"
            title={canRetry ? "Yeniden işle" : "İşle"}
            onClick={onProcess}
          >
            {canRetry ? <RotateCcw /> : <Play />}
          </Button>
        )}
        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <Button
              size="icon-xs"
              variant="ghost"
              title="Diğer işlemler"
              aria-label={`${document.name} için diğer işlemler`}
            >
              <MoreHorizontal />
            </Button>
          </DropdownMenuTrigger>
          <DropdownMenuContent align="end" className="w-48">
            <DropdownMenuItem onSelect={onMedia}>
              <ImageIcon /> Medyayı görüntüle
            </DropdownMenuItem>
            <DropdownMenuItem onSelect={onDownload}>
              <Download /> İndir
            </DropdownMenuItem>
            <DropdownMenuItem onSelect={onBeginRename}>
              <Pencil /> Yeniden adlandır
            </DropdownMenuItem>
            <DropdownMenuSeparator />
            <DropdownMenuItem
              className="text-destructive focus:text-destructive"
              onSelect={onDelete}
            >
              <Trash2 /> Sil
            </DropdownMenuItem>
          </DropdownMenuContent>
        </DropdownMenu>
      </div>
    </div>
  );
}

function DocumentDetailsPanel({
  document,
  mutating,
  renameValue,
  renaming,
  onRenameValueChange,
  onBeginRename,
  onCancelRename,
  onSaveRename,
  onPreview,
  onMedia,
  onDownload,
  onProcess,
  onStop,
  onDelete,
  onBack,
}: {
  document: PlatformDocument;
  mutating: boolean;
  renameValue: string;
  renaming: boolean;
  onRenameValueChange: (value: string) => void;
  onBeginRename: () => void;
  onCancelRename: () => void;
  onSaveRename: () => void;
  onPreview: () => void;
  onMedia: () => void;
  onDownload: () => void;
  onProcess: () => void;
  onStop: () => void;
  onDelete: () => void;
  onBack?: () => void;
}) {
  const status = STATUS_COPY[document.status];
  const active = document.status === "pending" || document.status === "running";
  const canRetry =
    document.status === "failed" || document.status === "cancelled";
  return (
    <aside
      aria-label={`${document.name} detayları`}
      className="relative flex min-h-0 flex-1 flex-col"
    >
      <div
        data-hub-scroll="true"
        className="mr-2 min-h-0 flex-1 overflow-x-hidden overflow-y-auto [overflow-anchor:none] [scrollbar-gutter:stable] [scrollbar-width:thin]"
      >
        {onBack ? (
          <div className="hub-detail-bar sticky top-0 z-20 lg:hidden">
            <div className="mx-auto w-full max-w-[var(--hub-measure-compact)] px-5 py-3">
              <button
                type="button"
                onClick={onBack}
                className="-ml-1.5 inline-flex h-8 cursor-pointer items-center gap-1.5 rounded-full pl-1.5 pr-2.5 text-ui-12p5 font-medium text-muted-foreground transition-colors hover:bg-foreground/[0.05] hover:text-foreground dark:hover:bg-white/[0.06]"
              >
                <ChevronLeft className="size-3.5" strokeWidth={1.75} />
                Belgelere dön
              </button>
            </div>
          </div>
        ) : null}
        <div className="mx-auto flex w-full max-w-[var(--hub-measure-compact)] min-w-0 flex-col gap-4 px-5 pb-20 lg:pt-4">
          <div className="flex min-w-0 flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
            <div className="flex min-w-0 items-start gap-3">
              <div className="flex size-12 shrink-0 items-center justify-center rounded-2xl bg-foreground/[0.055]">
                <FileText className="size-5" strokeWidth={1.75} />
              </div>
              <div className="min-w-0">
                {renaming ? (
                  <div className="flex min-w-0 items-center gap-1.5">
                    <Input
                      value={renameValue}
                      onChange={(event) =>
                        onRenameValueChange(event.target.value)
                      }
                      className="field-soft h-9 min-w-0 border-0 shadow-none"
                      autoFocus
                      onKeyDown={(event) => {
                        if (event.key === "Enter" && renameValue.trim())
                          onSaveRename();
                        if (event.key === "Escape") onCancelRename();
                      }}
                    />
                    <Button
                      size="xs"
                      onClick={onSaveRename}
                      disabled={!renameValue.trim()}
                    >
                      Kaydet
                    </Button>
                    <Button
                      size="icon-xs"
                      variant="ghost"
                      onClick={onCancelRename}
                    >
                      <X />
                    </Button>
                  </div>
                ) : (
                  <h2 className="truncate text-xl font-semibold tracking-tight">
                    {document.name}
                  </h2>
                )}
                <div className="mt-1.5 flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
                  <Badge className={cn("gap-1 text-[11px]", status.className)}>
                    {active ? (
                      <Spinner className="size-3" />
                    ) : document.status === "completed" ? (
                      <FileCheck2 className="size-3" />
                    ) : null}
                    {status.label}
                  </Badge>
                  <span>{document.suffix.toUpperCase() || "DOSYA"}</span>
                  <span>·</span>
                  <span>{formatBytes(document.sizeBytes)}</span>
                </div>
              </div>
            </div>
            <div className="flex shrink-0 items-center gap-1">
              <Button
                size="icon-sm"
                variant="ghost"
                title="Yeniden adlandır"
                aria-label="Belgeyi yeniden adlandır"
                onClick={onBeginRename}
              >
                <Pencil />
              </Button>
              <Button
                size="icon-sm"
                variant="ghost"
                title="İndir"
                aria-label="Belgeyi indir"
                onClick={onDownload}
              >
                <Download />
              </Button>
              <DropdownMenu>
                <DropdownMenuTrigger asChild>
                  <Button
                    size="icon-sm"
                    variant="ghost"
                    aria-label="Belge seçenekleri"
                  >
                    <MoreHorizontal />
                  </Button>
                </DropdownMenuTrigger>
                <DropdownMenuContent align="end">
                  <DropdownMenuItem onSelect={onMedia}>
                    <ImageIcon /> Medyayı görüntüle
                  </DropdownMenuItem>
                  <DropdownMenuItem onSelect={onPreview}>
                    <Eye /> Ayrı önizleme aç
                  </DropdownMenuItem>
                  <DropdownMenuSeparator />
                  <DropdownMenuItem
                    className="text-destructive focus:text-destructive"
                    onSelect={onDelete}
                  >
                    <Trash2 /> Sil
                  </DropdownMenuItem>
                </DropdownMenuContent>
              </DropdownMenu>
            </div>
          </div>

          <div className="field-soft rounded-2xl p-4">
            <div className="flex items-center justify-between gap-3 text-xs">
              <span className="truncate text-muted-foreground">
                {document.progressMessage ||
                  (document.status === "completed"
                    ? "İşleme tamamlandı"
                    : "İşleme durumu")}
              </span>
              <span className="tabular-nums">
                {Math.round(document.progress * 100)}%
              </span>
            </div>
            <Progress value={document.progress * 100} className="mt-2 h-1.5" />
            <div className="mt-3 flex flex-wrap gap-2">
              {active ? (
                <Button
                  size="xs"
                  variant="outline"
                  disabled={mutating}
                  onClick={onStop}
                >
                  <Square /> Durdur
                </Button>
              ) : (
                <Button
                  size="xs"
                  variant="outline"
                  disabled={mutating}
                  onClick={onProcess}
                >
                  {canRetry ? <RotateCcw /> : <Play />}
                  {canRetry ? "Yeniden işle" : "İşle"}
                </Button>
              )}
              <Button size="xs" variant="outline" onClick={onPreview}>
                <Eye /> Önizle
              </Button>
            </div>
          </div>

          <dl className="grid grid-cols-2 gap-2 sm:grid-cols-4">
            {[
              ["Parça", document.chunkCount.toLocaleString("tr-TR")],
              ["Token", document.tokenCount.toLocaleString("tr-TR")],
              ["Kaynak", document.sourceType || "—"],
              ["Ayrıştırıcı", document.parserId || "—"],
            ].map(([label, value]) => (
              <div key={label} className="rounded-xl bg-foreground/[0.035] p-3">
                <dt className="text-[11px] text-muted-foreground">{label}</dt>
                <dd
                  className="mt-1 truncate text-sm font-medium tabular-nums"
                  title={value}
                >
                  {value}
                </dd>
              </div>
            ))}
          </dl>

          <div className="flex flex-wrap gap-x-5 gap-y-1 border-y py-3 text-xs text-muted-foreground">
            <span>Oluşturma: {formatDocumentDate(document.createdAt)}</span>
            <span>Güncelleme: {formatDocumentDate(document.updatedAt)}</span>
            {document.pipelineName ? (
              <span>Pipeline: {document.pipelineName}</span>
            ) : null}
          </div>

          <section>
            <div className="mb-3 flex items-center justify-between gap-3">
              <div>
                <h3 className="text-sm font-semibold">Belge içeriği</h3>
                <p className="mt-0.5 text-xs text-muted-foreground">
                  Güvenli backend önizlemesi
                </p>
              </div>
              <Button size="xs" variant="ghost" onClick={onPreview}>
                Genişlet
              </Button>
            </div>
            <DocumentInlinePreview document={document} />
          </section>
        </div>
      </div>
    </aside>
  );
}

function GenericDocumentsPanel() {
  const inputRef = useRef<HTMLInputElement>(null);
  const requestRef = useRef<AbortController | null>(null);
  const [inspections, setInspections] = useState<PlatformUploadInspection[]>(
    [],
  );
  const [documentId, setDocumentId] = useState("");
  const [name, setName] = useState("");
  const [busy, setBusy] = useState(false);
  const [confirmDelete, setConfirmDelete] = useState(false);

  useEffect(() => () => requestRef.current?.abort(), []);

  const controllerForOperation = () => {
    requestRef.current?.abort();
    const controller = new AbortController();
    requestRef.current = controller;
    return controller;
  };

  const inspect = async (files: File[]) => {
    if (files.length === 0) return;
    const controller = controllerForOperation();
    setBusy(true);
    try {
      setInspections(await inspectDocumentUploads(files, controller.signal));
      toast.success(`${files.length} dosya güvenli biçimde incelendi.`);
    } catch (error) {
      toast.error("Dosya incelemesi tamamlanamadı", {
        description: error instanceof Error ? error.message : String(error),
      });
    } finally {
      if (!controller.signal.aborted) setBusy(false);
      if (requestRef.current === controller) requestRef.current = null;
      if (inputRef.current) inputRef.current.value = "";
    }
  };

  const runGeneric = async (
    operation: (id: string, signal: AbortSignal) => Promise<unknown>,
    success: string,
  ) => {
    const id = documentId.trim();
    if (!id) return;
    const controller = controllerForOperation();
    setBusy(true);
    try {
      await operation(id, controller.signal);
      toast.success(success);
    } catch (error) {
      if (!controller.signal.aborted) {
        toast.error("İşlem tamamlanamadı", {
          description: error instanceof Error ? error.message : String(error),
        });
      }
    } finally {
      if (!controller.signal.aborted) setBusy(false);
      if (requestRef.current === controller) requestRef.current = null;
    }
  };

  return (
    <div className="grid gap-3 xl:grid-cols-[minmax(0,1.2fr)_minmax(360px,.8fr)]">
      <section className="rounded-2xl bg-foreground/[0.025] p-4">
        <div className="flex items-start justify-between gap-4">
          <div>
            <h2 className="text-lg font-semibold">Bağımsız dosya inceleme</h2>
            <p className="mt-1 text-sm text-muted-foreground">
              Dosyayı kalıcı bir belgeye dönüştürmeden ayrıştırılabilir
              bilgilerini denetleyin.
            </p>
          </div>
          <Badge variant="secondary" className="rounded-full">
            Geçici · saklanmaz
          </Badge>
        </div>
        <button
          type="button"
          className="field-soft mt-4 flex min-h-32 w-full flex-col items-center justify-center gap-2 rounded-2xl border-0 p-4 transition-colors hover:bg-foreground/[0.07] active:scale-[0.995]"
          onClick={() => inputRef.current?.click()}
          disabled={busy}
        >
          {busy ? (
            <Spinner className="size-6" />
          ) : (
            <FileArchive className="size-8 text-muted-foreground" />
          )}
          <span className="font-medium">İncelemek için dosya seçin</span>
          <span className="text-xs text-muted-foreground">
            İçerik tarayıcı depolamasına yazılmaz.
          </span>
        </button>
        <input
          ref={inputRef}
          className="hidden"
          type="file"
          multiple
          onChange={(event) =>
            void inspect(Array.from(event.target.files ?? []))
          }
        />
        {inspections.length > 0 ? (
          <div className="mt-4 grid gap-2 sm:grid-cols-2">
            {inspections.map((item, index) => (
              <div
                key={`${String(item.name)}-${index}`}
                className="rounded-xl bg-background/70 p-3"
              >
                <p className="truncate font-medium">
                  {String(item.name ?? `Dosya ${index + 1}`)}
                </p>
                <p className="mt-1 text-xs text-muted-foreground">
                  {String(item.type ?? item.suffix ?? "Tür bilinmiyor")} ·{" "}
                  {typeof item.size === "number"
                    ? formatBytes(item.size)
                    : "Boyut bilinmiyor"}
                </p>
              </div>
            ))}
          </div>
        ) : null}
      </section>

      <section className="rounded-2xl bg-foreground/[0.025] p-4">
        <div className="flex items-center gap-2">
          <ShieldCheck className="size-5" />
          <h2 className="text-lg font-semibold">Belge kimliğiyle yönetim</h2>
        </div>
        <p className="mt-1 text-sm text-muted-foreground">
          Genel liste aktif hybrid sürümünde dataset bağlamı olmadan
          çalışamıyor; tekil GET ise sahiplik kontrolü yapmıyor. İkisi de
          güvenlik nedeniyle runtime-disabled. Kimliği bilinen belgelerde
          sahiplik denetimli güncelleme, ingestion ve silme işlemleri
          kullanılabilir.
        </p>
        <div className="mt-4 space-y-3">
          <Input
            className="field-soft border-0 shadow-none"
            value={documentId}
            onChange={(event) => setDocumentId(event.target.value)}
            placeholder="Belge kimliği"
            aria-label="Genel belge kimliği"
          />
          <Input
            className="field-soft border-0 shadow-none"
            value={name}
            onChange={(event) => setName(event.target.value)}
            placeholder="Yeni belge adı"
            aria-label="Genel belge adı"
          />
          <div className="rounded-xl bg-background/60 p-3">
            <div className="flex flex-wrap gap-2">
              <Button
                size="sm"
                variant="outline"
                disabled={busy || !documentId.trim() || !name.trim()}
                onClick={() =>
                  void runGeneric(
                    (id, signal) =>
                      updateGenericDocument(id, { name: name.trim() }, signal),
                    "Belge güncellendi.",
                  )
                }
              >
                <Pencil /> Kaydet
              </Button>
              <Button
                size="sm"
                variant="outline"
                disabled={busy || !documentId.trim()}
                onClick={() =>
                  void runGeneric(
                    (id, signal) => ingestGenericDocuments([id], "1", signal),
                    "Ingestion başlatıldı.",
                  )
                }
              >
                <Play /> Ingest
              </Button>
              <Button
                size="sm"
                variant="destructive"
                disabled={busy || !documentId.trim()}
                onClick={() => setConfirmDelete(true)}
              >
                <Trash2 /> Sil
              </Button>
            </div>
          </div>
        </div>
      </section>

      <AlertDialog open={confirmDelete} onOpenChange={setConfirmDelete}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Genel belge silinsin mi?</AlertDialogTitle>
            <AlertDialogDescription>
              Belge kimliğiyle eşleşen kayıt kalıcı olarak silinecek. Backend
              sahiplik denetimi başarısız olursa işlem reddedilir.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Vazgeç</AlertDialogCancel>
            <AlertDialogAction
              variant="destructive"
              onClick={() =>
                void runGeneric(
                  (id, signal) => deleteGenericDocument(id, signal),
                  "Belge silindi.",
                ).then(() => setConfirmDelete(false))
              }
            >
              Kalıcı olarak sil
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </div>
  );
}

export function DocumentLibraryPage() {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const library = useDocumentLibrary();
  const [activeTab, setActiveTab] = useState<DocumentTab>("dataset");
  const [layoutMode, setLayoutMode] = useState<AllModelsView>("split");
  const [focusedDocumentId, setFocusedDocumentId] = useState<string | null>(
    null,
  );
  const [autoParse, setAutoParse] = useState(true);
  const [dragging, setDragging] = useState(false);
  const [query, setQuery] = useState("");
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [deleteIds, setDeleteIds] = useState<string[]>([]);
  const [assetDocument, setAssetDocument] = useState<PlatformDocument | null>(
    null,
  );
  const [assetMode, setAssetMode] = useState<"preview" | "media">("preview");
  const [assetOpen, setAssetOpen] = useState(false);
  const [renameId, setRenameId] = useState<string | null>(null);
  const [renameValue, setRenameValue] = useState("");
  const deferredQuery = useDeferredValue(query);
  const setLibraryKeywords = library.setKeywords;

  useEffect(() => {
    setLibraryKeywords(deferredQuery);
  }, [deferredQuery, setLibraryKeywords]);

  const pageNumbers = useMemo(
    () => visiblePageNumbers(library.page, library.totalPages),
    [library.page, library.totalPages],
  );

  const submitFiles = async (files: File[]) => {
    if (files.length === 0) return;
    try {
      const summary = await library.upload(files, autoParse);
      for (const failure of summary.rejected)
        toast.error(`${failure.file.name} yüklenmedi`, {
          description: failure.message,
        });
      if (summary.accepted.length > 0) {
        toast.success(
          `${summary.accepted.length} dosya yüklendi${autoParse ? " ve işleme alındı" : ""}.`,
        );
      }
      if (summary.partialFailure)
        toast.warning("Bazı dosyalar yüklenemedi", {
          description: summary.partialFailure,
        });
    } catch (error) {
      toast.error("Yükleme tamamlanamadı", {
        description: error instanceof Error ? error.message : String(error),
      });
    } finally {
      if (fileInputRef.current) fileInputRef.current.value = "";
    }
  };

  const run = async (action: () => Promise<unknown>, success: string) => {
    try {
      await action();
      toast.success(success);
    } catch (error) {
      toast.error("İşlem tamamlanamadı", {
        description: error instanceof Error ? error.message : String(error),
      });
    }
  };

  const openAsset = (document: PlatformDocument, mode: "preview" | "media") => {
    setAssetDocument(document);
    setAssetMode(mode);
    setAssetOpen(true);
  };

  const changePage = (nextPage: number) => {
    library.setPage(nextPage);
    setSelected(new Set());
  };

  const download = async (document: PlatformDocument) => {
    try {
      const asset = await downloadDatasetDocument(
        document.datasetId,
        document.id,
      );
      const url = URL.createObjectURL(asset.blob);
      const anchor = window.document.createElement("a");
      anchor.href = url;
      anchor.download = downloadName(document);
      anchor.rel = "noopener";
      anchor.click();
      URL.revokeObjectURL(url);
    } catch (error) {
      toast.error("Belge indirilemedi", {
        description: error instanceof Error ? error.message : String(error),
      });
    }
  };

  const selectedIds = [...selected];
  const firstVisibleDocument =
    library.totalDocuments === 0
      ? 0
      : (library.page - 1) * library.pageSize + 1;
  const lastVisibleDocument = Math.min(
    library.page * library.pageSize,
    library.totalDocuments,
  );
  const activeDocumentCount = library.documents.filter(
    (document) =>
      document.status === "pending" || document.status === "running",
  ).length;
  const focusedDocument =
    library.documents.find((document) => document.id === focusedDocumentId) ??
    library.documents[0] ??
    null;

  const focusDocument = (documentId: string, revealDetails = false) => {
    setFocusedDocumentId(documentId);
    if (revealDetails) setLayoutMode("split");
  };

  const setDocumentChecked = (documentId: string, checked: boolean) => {
    setSelected((current) => {
      const next = new Set(current);
      if (checked) next.add(documentId);
      else next.delete(documentId);
      return next;
    });
  };

  const setAllVisibleChecked = (checked: boolean) => {
    setSelected((current) => {
      const next = new Set(current);
      for (const document of library.documents) {
        if (checked) next.add(document.id);
        else next.delete(document.id);
      }
      return next;
    });
  };

  const allVisibleChecked =
    library.documents.length > 0 &&
    library.documents.every((document) => selected.has(document.id));

  const bulkActions =
    selectedIds.length > 0 ? (
      <DropdownMenu>
        <DropdownMenuTrigger asChild>
          <Button
            size="xs"
            variant="outline"
            className="rounded-full px-2.5"
            disabled={library.mutating}
          >
            <FileCheck2 /> {selectedIds.length} seçili
          </Button>
        </DropdownMenuTrigger>
        <DropdownMenuContent align="end" className="w-44">
          <DropdownMenuItem
            onSelect={() =>
              void run(
                () => library.parse(selectedIds),
                "Belgeler işleme alındı.",
              )
            }
          >
            <Play /> İşle
          </DropdownMenuItem>
          <DropdownMenuItem
            onSelect={() =>
              void run(() => library.stop(selectedIds), "İşlem durduruldu.")
            }
          >
            <Square /> Durdur
          </DropdownMenuItem>
          <DropdownMenuSeparator />
          <DropdownMenuItem
            className="text-destructive focus:text-destructive"
            onSelect={() => setDeleteIds(selectedIds)}
          >
            <Trash2 /> Sil
          </DropdownMenuItem>
        </DropdownMenuContent>
      </DropdownMenu>
    ) : undefined;

  const datasetHeaderActions = (
    <div className="flex min-w-0 items-center gap-1.5">
      <Select
        value={library.datasetId}
        onValueChange={(value) => {
          library.setDatasetId(value);
          setQuery("");
          setSelected(new Set());
        }}
        disabled={library.loadingDatasets || library.datasets.length === 0}
      >
        <SelectTrigger
          size="sm"
          aria-label="Dataset değiştir"
          title="Dataset değiştir"
          iconClassName="hidden"
          className="field-soft size-8 justify-center rounded-full border-0 p-0 shadow-none"
        >
          <Database className="size-3.5" strokeWidth={1.75} />
          <span className="sr-only">
            <SelectValue placeholder="Dataset seçin" />
          </span>
        </SelectTrigger>
        <SelectContent align="end">
          {library.datasets.map((dataset) => (
            <SelectItem key={dataset.id} value={dataset.id}>
              {dataset.name}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
      <Button
        size="icon-xs"
        variant="ghost"
        className="field-soft rounded-full"
        title="Belgeleri yenile"
        aria-label="Belgeleri yenile"
        onClick={() => void library.refresh()}
        disabled={!library.datasetId || library.loadingDocuments}
      >
        <RefreshCw className={cn(library.loadingDocuments && "animate-spin")} />
      </Button>
      {bulkActions}
    </div>
  );

  return (
    <main className="hub-page flex h-full min-h-0 min-w-0 flex-1 flex-col overflow-hidden bg-background">
      <Tabs
        value={activeTab}
        onValueChange={(value) => {
          if (value === "dataset" || value === "generic") setActiveTab(value);
        }}
        className="flex min-h-0 flex-1 flex-col gap-0"
      >
        <HubTopBar>
          <header className="font-heading flex flex-col gap-3 sm:flex-row sm:flex-wrap sm:items-center sm:justify-between">
            <PageHeading
              title="Belgeler"
              subtitle="Dataset kaynaklarını yükleyin, işleyin ve tek bir yerden yönetin."
            />
            <div className="flex min-w-0 flex-wrap items-center gap-1.5 sm:justify-end">
              <DocumentStatPill
                icon={Database}
                label="Dataset"
                value={library.datasets.length}
              />
              <DocumentStatPill
                icon={FileText}
                label="Belge"
                value={library.totalDocuments}
              />
              <DocumentStatPill
                icon={Play}
                label="İşleniyor"
                value={activeDocumentCount}
              />
            </div>
          </header>

          <div className="flex min-w-0 flex-col gap-2 lg:flex-row lg:items-center">
            <DocumentScopeToggle value={activeTab} onChange={setActiveTab} />

            {activeTab === "dataset" ? (
              <>
                <div className="relative min-w-0 flex-1 lg:min-w-[220px]">
                  <Search className="pointer-events-none absolute left-3.5 top-1/2 size-4 -translate-y-1/2 text-muted-foreground" />
                  <Input
                    type="search"
                    name="document-search"
                    autoComplete="off"
                    spellCheck={false}
                    value={query}
                    onChange={(event) => {
                      setQuery(event.target.value);
                      setSelected(new Set());
                    }}
                    placeholder="Belgelerde ara"
                    disabled={!library.datasetId}
                    className="field-soft h-9 rounded-full border-0 pl-10 pr-4 text-sm shadow-none focus-visible:ring-0"
                  />
                </div>

                <button
                  type="button"
                  role="checkbox"
                  aria-checked={autoParse}
                  title="Yüklemeden sonra otomatik işle"
                  onClick={() => setAutoParse((current) => !current)}
                  className="field-soft inline-flex h-9 shrink-0 items-center gap-2 rounded-full px-3 text-ui-12p5 text-muted-foreground transition-colors hover:text-foreground"
                >
                  <span
                    className={cn(
                      "flex size-5 items-center justify-center rounded-full transition-colors",
                      autoParse
                        ? "bg-status-success/15 text-status-success"
                        : "bg-foreground/[0.06]",
                    )}
                  >
                    <FileCheck2 className="size-3" strokeWidth={1.9} />
                  </span>
                  Otomatik işle
                  <span
                    aria-hidden="true"
                    className={cn(
                      "size-1.5 rounded-full",
                      autoParse
                        ? "bg-status-success"
                        : "bg-muted-foreground/35",
                    )}
                  />
                </button>
                <Button
                  size="sm"
                  className="h-9 shrink-0 rounded-full px-3.5"
                  onClick={() => fileInputRef.current?.click()}
                  disabled={!library.datasetId || library.mutating}
                >
                  {library.mutating ? (
                    <Spinner className="size-4" />
                  ) : (
                    <UploadCloud />
                  )}
                  Dosya seç
                </Button>
                <input
                  ref={fileInputRef}
                  className="hidden"
                  type="file"
                  multiple
                  accept=".pdf,.txt,.text,.md,.doc,.docx,.xls,.xlsx,.csv,.ppt,.pptx,.html,.htm,.json,.jsonl,.png,.jpg,.jpeg,.gif,.webp,.bmp,.tif,.tiff"
                  onChange={(event) =>
                    void submitFiles(Array.from(event.target.files ?? []))
                  }
                />
              </>
            ) : null}
          </div>
        </HubTopBar>

        <TabsContent
          value="dataset"
          className="relative mt-0 min-h-0 flex-1 overflow-hidden"
        >
          <section
            className="relative flex h-full min-h-0 flex-col overflow-hidden"
            onDragEnter={(event) => {
              event.preventDefault();
              setDragging(true);
            }}
            onDragOver={(event) => event.preventDefault()}
            onDragLeave={(event) => {
              if (event.currentTarget === event.target) setDragging(false);
            }}
            onDrop={(event) => {
              event.preventDefault();
              setDragging(false);
              void submitFiles(Array.from(event.dataTransfer.files));
            }}
          >
            {dragging ? (
              <div className="absolute inset-0 z-30 flex items-center justify-center rounded-2xl border border-dashed border-primary/40 bg-background/90 backdrop-blur-sm">
                <div className="text-center">
                  <UploadCloud className="mx-auto size-7" />
                  <p className="mt-2 font-medium">
                    Belgeleri yüklemek için bırakın
                  </p>
                  <p className="mt-1 text-xs text-muted-foreground">
                    Dosya başına en fazla 128 MB
                  </p>
                </div>
              </div>
            ) : null}
            <div
              className={cn(
                "relative flex min-h-0 min-w-0 flex-1 basis-0 flex-col",
                layoutMode === "split" &&
                  "lg:mx-auto lg:w-full lg:max-w-[var(--hub-measure)] lg:flex-row",
              )}
            >
              <div
                className={cn(
                  "flex min-h-0 flex-1 flex-col",
                  layoutMode === "split" &&
                    "lg:w-[clamp(460px,32%,620px)] lg:max-w-[44%] lg:flex-none lg:shrink-0 lg:border-r lg:border-border/60",
                )}
              >
                <div className="relative flex h-full min-h-0 flex-1 flex-col overflow-hidden">
                  <div
                    aria-hidden="true"
                    className="hub-scroll-fade pointer-events-none absolute inset-x-0 top-0 z-10 h-7"
                  />
                  <div
                    data-hub-scroll="true"
                    className={cn(
                      "absolute inset-0 min-h-0 overflow-x-hidden overflow-y-auto pb-6 pt-0 [overflow-anchor:none] [scrollbar-width:thin]",
                      layoutMode === "split"
                        ? "[scrollbar-gutter:stable]"
                        : "[scrollbar-gutter:stable_both-edges]",
                    )}
                  >
                    <div
                      className={
                        layoutMode === "split"
                          ? "mx-auto w-full max-w-[var(--hub-measure)] pl-5 pr-2 sm:pl-8"
                          : "mx-auto w-full max-w-[var(--hub-measure)] px-5 sm:px-8"
                      }
                    >
                      <div className="flex flex-col gap-3 pt-6">
                        {library.error ? (
                          <div
                            className={cn(
                              "flex items-start gap-3 rounded-xl border p-3",
                              library.error.kind === "permission"
                                ? "border-amber-500/30 bg-amber-500/5"
                                : "border-destructive/30 bg-destructive/5",
                            )}
                          >
                            {library.error.kind === "permission" ? (
                              <ShieldCheck className="mt-0.5 size-5 text-amber-600" />
                            ) : (
                              <CircleAlert className="mt-0.5 size-5 text-destructive" />
                            )}
                            <div className="flex-1">
                              <p className="font-medium">
                                {library.error.kind === "permission"
                                  ? "Bu dataset için yetkiniz yok"
                                  : library.error.kind === "timeout"
                                    ? "İstek zaman aşımına uğradı"
                                    : "Belge verileri alınamadı"}
                              </p>
                              <p className="mt-1 text-sm text-muted-foreground">
                                {library.error.message}
                              </p>
                            </div>
                            <Button
                              size="sm"
                              variant="outline"
                              onClick={() => void library.refresh()}
                            >
                              Yeniden dene
                            </Button>
                          </div>
                        ) : null}
                        <HubListHeader
                          title={
                            library.selectedDataset?.name ?? "Dataset seçin"
                          }
                          subtitle={
                            selectedIds.length > 0
                              ? `${selectedIds.length} belge seçili`
                              : library.datasetId
                                ? `${library.totalDocuments.toLocaleString("tr-TR")} belge · aktif dataset`
                                : "Belge listesini görüntülemek için dataset seçin"
                          }
                          count={library.totalDocuments}
                          view={layoutMode}
                          onViewChange={setLayoutMode}
                          actions={datasetHeaderActions}
                        />
                        {layoutMode === "grid" &&
                        library.documents.length > 0 ? (
                          <DocumentCompactHeader
                            allChecked={allVisibleChecked}
                            onAllCheckedChange={setAllVisibleChecked}
                          />
                        ) : null}
                      </div>

                      {library.loadingDocuments ? (
                        <div className="flex min-h-64 items-center justify-center">
                          <Spinner className="size-5" />
                        </div>
                      ) : !library.datasetId ? (
                        <Empty>
                          <EmptyHeader>
                            <EmptyMedia variant="icon">
                              <Database />
                            </EmptyMedia>
                            <EmptyTitle>Önce bir dataset oluşturun</EmptyTitle>
                            <EmptyDescription>
                              Belge yüklemek için erişebildiğiniz bir dataset
                              bulunmalı.
                            </EmptyDescription>
                          </EmptyHeader>
                        </Empty>
                      ) : library.documents.length === 0 ? (
                        <Empty className="min-h-64">
                          <EmptyHeader>
                            <EmptyMedia variant="icon">
                              <FileText />
                            </EmptyMedia>
                            <EmptyTitle>
                              {query ? "Eşleşen belge yok" : "Henüz belge yok"}
                            </EmptyTitle>
                            <EmptyDescription>
                              {query
                                ? "Arama ifadenizi değiştirin."
                                : "Yukarıdaki alana dosya bırakarak başlayın."}
                            </EmptyDescription>
                          </EmptyHeader>
                        </Empty>
                      ) : layoutMode === "two" ? (
                        <div
                          className="grid grid-cols-1 gap-x-3 gap-y-2 md:grid-cols-2"
                          role="list"
                          aria-label="Belgeler iki sütun görünümü"
                        >
                          {library.documents.map((document) => (
                            <DocumentSummaryCard
                              key={document.id}
                              document={document}
                              mode="two"
                              focused={focusedDocument?.id === document.id}
                              checked={selected.has(document.id)}
                              onActivate={() =>
                                focusDocument(document.id, true)
                              }
                              onCheckedChange={(checked) =>
                                setDocumentChecked(document.id, checked)
                              }
                            />
                          ))}
                        </div>
                      ) : layoutMode === "split" ? (
                        <div
                          className="space-y-2"
                          role="list"
                          aria-label="Belgeler"
                        >
                          {library.documents.map((document) => (
                            <DocumentSummaryCard
                              key={document.id}
                              document={document}
                              mode="split"
                              focused={focusedDocument?.id === document.id}
                              checked={selected.has(document.id)}
                              onActivate={() => focusDocument(document.id)}
                              onCheckedChange={(checked) =>
                                setDocumentChecked(document.id, checked)
                              }
                            />
                          ))}
                        </div>
                      ) : (
                        <div
                          role="table"
                          aria-label="Belgeler kompakt görünümü"
                          className="space-y-2"
                        >
                          {library.documents.map((document) => (
                            <DocumentCompactRow
                              key={document.id}
                              document={document}
                              checked={selected.has(document.id)}
                              renameValue={renameValue}
                              renaming={renameId === document.id}
                              onCheckedChange={(checked) =>
                                setDocumentChecked(document.id, checked)
                              }
                              onActivate={() =>
                                focusDocument(document.id, true)
                              }
                              onRenameValueChange={setRenameValue}
                              onCancelRename={() => setRenameId(null)}
                              onSaveRename={() =>
                                void run(async () => {
                                  await library.rename(
                                    document.id,
                                    renameValue.trim(),
                                  );
                                  setRenameId(null);
                                }, "Belge yeniden adlandırıldı.")
                              }
                              onBeginRename={() => {
                                setRenameId(document.id);
                                setRenameValue(document.name);
                              }}
                              onPreview={() => openAsset(document, "preview")}
                              onMedia={() => openAsset(document, "media")}
                              onDownload={() => void download(document)}
                              onProcess={() =>
                                void run(
                                  () => library.parse([document.id]),
                                  "Belge işleme alındı.",
                                )
                              }
                              onStop={() =>
                                void run(
                                  () => library.stop([document.id]),
                                  "İşlem durduruldu.",
                                )
                              }
                              onDelete={() => setDeleteIds([document.id])}
                            />
                          ))}
                        </div>
                      )}

                      {library.datasetId && library.totalDocuments > 0 ? (
                        <footer className="mt-3 flex flex-col gap-2 border-t px-1 py-3 sm:flex-row sm:items-center sm:justify-between">
                          <div className="flex items-center gap-2 text-xs tabular-nums text-muted-foreground">
                            <span>
                              {firstVisibleDocument}–{lastVisibleDocument} /{" "}
                              {library.totalDocuments.toLocaleString("tr-TR")}
                            </span>
                            <Select
                              value={String(library.pageSize)}
                              onValueChange={(value) => {
                                library.setPageSize(Number(value));
                                setSelected(new Set());
                              }}
                            >
                              <SelectTrigger
                                className="field-soft h-7 w-[72px] rounded-full border-0 px-2 text-xs shadow-none"
                                aria-label="Sayfa başına belge"
                              >
                                <SelectValue />
                              </SelectTrigger>
                              <SelectContent align="start">
                                {[10, 20, 50].map((size) => (
                                  <SelectItem key={size} value={String(size)}>
                                    {size}
                                  </SelectItem>
                                ))}
                              </SelectContent>
                            </Select>
                          </div>
                          <Pagination className="mx-0 w-auto justify-start sm:justify-end">
                            <PaginationContent>
                              <PaginationItem>
                                <Button
                                  size="icon-xs"
                                  variant="ghost"
                                  aria-label="Önceki sayfa"
                                  disabled={
                                    library.page <= 1 ||
                                    library.loadingDocuments
                                  }
                                  onClick={() => changePage(library.page - 1)}
                                >
                                  <ChevronLeft />
                                </Button>
                              </PaginationItem>
                              {pageNumbers.map((pageNumber) => (
                                <PaginationItem key={pageNumber}>
                                  <Button
                                    size="icon-xs"
                                    variant={
                                      pageNumber === library.page
                                        ? "outline"
                                        : "ghost"
                                    }
                                    aria-current={
                                      pageNumber === library.page
                                        ? "page"
                                        : undefined
                                    }
                                    onClick={() => changePage(pageNumber)}
                                  >
                                    {pageNumber}
                                  </Button>
                                </PaginationItem>
                              ))}
                              <PaginationItem>
                                <Button
                                  size="icon-xs"
                                  variant="ghost"
                                  aria-label="Sonraki sayfa"
                                  disabled={
                                    library.page >= library.totalPages ||
                                    library.loadingDocuments
                                  }
                                  onClick={() => changePage(library.page + 1)}
                                >
                                  <ChevronRight />
                                </Button>
                              </PaginationItem>
                            </PaginationContent>
                          </Pagination>
                        </footer>
                      ) : null}
                    </div>
                  </div>
                </div>
              </div>

              {layoutMode === "split" && focusedDocument ? (
                <div className="hub-canvas z-20 flex min-h-0 flex-col max-lg:absolute max-lg:inset-0 lg:relative lg:min-w-0 lg:flex-1">
                  <DocumentDetailsPanel
                    document={focusedDocument}
                    mutating={library.mutating}
                    renameValue={renameValue}
                    renaming={renameId === focusedDocument.id}
                    onRenameValueChange={setRenameValue}
                    onBeginRename={() => {
                      setRenameId(focusedDocument.id);
                      setRenameValue(focusedDocument.name);
                    }}
                    onCancelRename={() => setRenameId(null)}
                    onSaveRename={() =>
                      void run(async () => {
                        await library.rename(
                          focusedDocument.id,
                          renameValue.trim(),
                        );
                        setRenameId(null);
                      }, "Belge yeniden adlandırıldı.")
                    }
                    onPreview={() => openAsset(focusedDocument, "preview")}
                    onMedia={() => openAsset(focusedDocument, "media")}
                    onDownload={() => void download(focusedDocument)}
                    onProcess={() =>
                      void run(
                        () => library.parse([focusedDocument.id]),
                        "Belge işleme alındı.",
                      )
                    }
                    onStop={() =>
                      void run(
                        () => library.stop([focusedDocument.id]),
                        "İşlem durduruldu.",
                      )
                    }
                    onDelete={() => setDeleteIds([focusedDocument.id])}
                    onBack={() => setLayoutMode("two")}
                  />
                </div>
              ) : layoutMode === "split" ? (
                <div className="hidden min-h-0 flex-1 items-center justify-center px-6 text-center text-ui-13 text-muted-foreground lg:flex">
                  Ayrıntılarını görmek için bir belge seçin.
                </div>
              ) : null}
            </div>
          </section>
        </TabsContent>

        <TabsContent
          value="generic"
          className="mt-0 min-h-0 flex-1 overflow-y-auto [scrollbar-gutter:stable_both-edges]"
        >
          <div className="mx-auto w-full max-w-[var(--hub-measure)] px-5 pb-8 pt-6 sm:px-8">
            <GenericDocumentsPanel />
          </div>
        </TabsContent>
      </Tabs>

      <DocumentAssetDialog
        document={assetDocument}
        mode={assetMode}
        open={assetOpen}
        onOpenChange={setAssetOpen}
      />
      <AlertDialog
        open={deleteIds.length > 0}
        onOpenChange={(open) => {
          if (!open) setDeleteIds([]);
        }}
      >
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>
              {deleteIds.length} belge silinsin mi?
            </AlertDialogTitle>
            <AlertDialogDescription>
              Seçilen belgeler ve erişilebilir içerikleri kalıcı olarak
              kaldırılacak. Bu işlem geri alınamaz.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Vazgeç</AlertDialogCancel>
            <AlertDialogAction
              variant="destructive"
              onClick={() =>
                void run(async () => {
                  await library.remove(deleteIds);
                  setSelected(new Set());
                  setDeleteIds([]);
                }, "Belgeler silindi.")
              }
            >
              Kalıcı olarak sil
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </main>
  );
}
