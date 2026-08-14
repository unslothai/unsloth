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
import {
  Card,
  CardAction,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Checkbox } from "@/components/ui/checkbox";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import {
  Empty,
  EmptyDescription,
  EmptyHeader,
  EmptyMedia,
  EmptyTitle,
} from "@/components/ui/empty";
import { Input } from "@/components/ui/input";
import { InfoHint } from "@/components/ui/info-hint";
import { Progress } from "@/components/ui/progress";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Slider } from "@/components/ui/slider";
import { Skeleton } from "@/components/ui/skeleton";
import { Spinner } from "@/components/ui/spinner";
import { Switch } from "@/components/ui/switch";
import { Textarea } from "@/components/ui/textarea";
import {
  chunkPreviewDocument,
  type PlatformChunk,
  type PlatformChunkDraft,
  type PlatformDocument,
  type PlatformStructureTemplate,
} from "@/integrations/platform-backend";
import { cn } from "@/lib/utils";
import { toast } from "@/lib/toast";
import { useVirtualizer } from "@tanstack/react-virtual";
import {
  Activity,
  Binary,
  Blocks,
  CheckCircle2,
  ChevronLeft,
  ChevronRight,
  CircleAlert,
  Database,
  Eye,
  FileText,
  FileSearch,
  GitFork,
  Layers3,
  MessageCircleQuestion,
  Pencil,
  PauseCircle,
  Plus,
  RefreshCw,
  Search,
  ShieldCheck,
  SlidersHorizontal,
  Sparkles,
  Target,
  Trash2,
  Zap,
} from "lucide-react";
import { useEffect, useRef, useState } from "react";
import {
  type DatasetQualityError,
  useDatasetQualityWorkspace,
} from "./use-dataset-quality-workspace";

export type DatasetQualityMode = "chunks" | "retrieval";

interface DatasetQualityWorkspaceProps {
  mode: DatasetQualityMode;
  datasetId: string;
  datasetName: string;
  documents: PlatformDocument[];
  preferredDocumentId?: string | null;
  onPreview: (document: PlatformDocument, pageNumber: number | null) => void;
}

function ErrorNotice({
  error,
  retry,
}: {
  error: DatasetQualityError;
  retry: () => void;
}) {
  const permission = error.kind === "permission";
  return (
    <div
      className={cn(
        "flex items-start gap-3 rounded-2xl border p-4",
        permission
          ? "border-amber-500/30 bg-amber-500/5"
          : "border-destructive/30 bg-destructive/5",
      )}
    >
      {permission ? (
        <ShieldCheck className="mt-0.5 size-5 text-amber-600" />
      ) : (
        <CircleAlert className="mt-0.5 size-5 text-destructive" />
      )}
      <div className="min-w-0 flex-1">
        <p className="font-medium">
          {permission
            ? "Bu içerik için yetkiniz yok"
            : error.kind === "timeout"
              ? "İstek zaman aşımına uğradı"
              : "RAG kalite verileri alınamadı"}
        </p>
        <p className="mt-1 text-sm text-muted-foreground">{error.message}</p>
      </div>
      <Button size="sm" variant="outline" onClick={retry}>
        Yeniden dene
      </Button>
    </div>
  );
}

function LiveBadge({ label = "Canlı" }: { label?: string }) {
  return (
    <Badge
      variant="outline"
      className="gap-2 border-emerald-500/20 bg-emerald-500/[0.07] px-2.5 text-emerald-700 shadow-none dark:text-emerald-300"
    >
      <span className="relative flex size-1.5" aria-hidden="true">
        <span className="absolute inline-flex size-full animate-ping rounded-full bg-emerald-500 opacity-50 motion-reduce:animate-none" />
        <span className="relative inline-flex size-1.5 rounded-full bg-emerald-500" />
      </span>
      {label}
    </Badge>
  );
}

function QualityLoading({ label }: { label: string }) {
  return (
    <div
      className="grid min-h-80 content-start gap-3 px-1 py-5"
      aria-label={label}
      aria-busy="true"
    >
      {[0, 1, 2].map((row) => (
        <div
          key={row}
          className="rounded-2xl bg-muted/25 p-4 ring-1 ring-foreground/[0.05]"
        >
          <div className="flex items-center gap-2">
            <Skeleton className="size-4 rounded-md" />
            <Skeleton className="h-5 w-16 rounded-lg" />
            <Skeleton className="h-5 w-24 rounded-lg" />
          </div>
          <Skeleton className="mt-4 h-3 w-full" />
          <Skeleton className="mt-2 h-3 w-[82%]" />
          <Skeleton className="mt-2 h-3 w-[58%]" />
        </div>
      ))}
    </div>
  );
}

function MetricPill({
  icon: Icon,
  label,
  value,
  tone = "neutral",
}: {
  icon: typeof Activity;
  label: string;
  value: string;
  tone?: "neutral" | "success" | "warning";
}) {
  return (
    <div
      className={cn(
        "flex min-w-0 items-center gap-2.5 rounded-2xl px-3 py-2 ring-1 ring-inset",
        tone === "success"
          ? "bg-emerald-500/[0.07] text-emerald-800 ring-emerald-500/15 dark:text-emerald-200"
          : tone === "warning"
            ? "bg-amber-500/[0.07] text-amber-800 ring-amber-500/15 dark:text-amber-200"
            : "bg-muted/35 text-foreground ring-foreground/[0.05]",
      )}
    >
      <Icon className="size-4 shrink-0 opacity-70" aria-hidden="true" />
      <div className="min-w-0">
        <p className="truncate text-[10px] font-medium tracking-wide text-current/65">
          {label}
        </p>
        <p className="truncate text-sm font-semibold tabular-nums">{value}</p>
      </div>
    </div>
  );
}

function scoreLabel(score: number | null): string {
  if (score === null) return "Skor yok";
  if (score >= 0.75) return "Güçlü eşleşme";
  if (score >= 0.45) return "Orta eşleşme";
  return "Düşük eşleşme";
}

function ScoreBadge({ score }: { score: number | null }) {
  if (score === null) return <Badge variant="outline">Skor yok</Badge>;
  return (
    <Badge
      variant="outline"
      className={cn(
        "gap-1.5 px-2.5 font-semibold tabular-nums shadow-none",
        score >= 0.75
          ? "border-emerald-500/30 bg-emerald-500/10 text-emerald-700 dark:text-emerald-300"
          : score >= 0.45
            ? "border-amber-500/30 bg-amber-500/10 text-amber-700 dark:text-amber-300"
            : "border-border bg-muted/50 text-muted-foreground",
      )}
    >
      <span
        className={cn(
          "size-1.5 rounded-full",
          score >= 0.75
            ? "bg-emerald-500"
            : score >= 0.45
              ? "bg-amber-500"
              : "bg-muted-foreground/50",
        )}
        aria-hidden="true"
      />
      {(score * 100).toFixed(1)}%
    </Badge>
  );
}

function ScoreBreakdown({ chunk }: { chunk: PlatformChunk }) {
  const scores = [
    ["Vektör", chunk.scores.vectorSimilarity],
    ["Terim", chunk.scores.termSimilarity],
    ["Rerank", chunk.scores.rerankScore],
  ] as const;
  const componentScores = scores.filter((entry) => entry[1] !== null);
  const visibleScores =
    componentScores.length > 0
      ? componentScores
      : chunk.normalizedScore === null
        ? []
        : [["Genel", chunk.normalizedScore] as const];
  if (visibleScores.length === 0) return null;

  return (
    <div className="grid gap-2 rounded-xl bg-muted/25 p-3 sm:grid-cols-3">
      {visibleScores.map(([label, value]) => (
        <div key={label} className="min-w-0">
          <div className="mb-1.5 flex items-center justify-between gap-2 text-[11px]">
            <span className="text-muted-foreground">{label}</span>
            <span className="font-mono font-medium tabular-nums">
              {((value ?? 0) * 100).toFixed(0)}
              <span className="text-muted-foreground">%</span>
            </span>
          </div>
          <Progress
            value={(value ?? 0) * 100}
            className="h-1.5"
            indicatorClassName="bg-foreground/65"
            aria-label={`${label} skoru`}
          />
        </div>
      ))}
    </div>
  );
}

function tags(value: string): string[] {
  return value
    .split(",")
    .map((entry) => entry.trim())
    .filter((entry, index, rows) => entry && rows.indexOf(entry) === index);
}

function ChunkEditor({
  open,
  chunk,
  saving,
  onOpenChange,
  onSubmit,
}: {
  open: boolean;
  chunk: PlatformChunk | null;
  saving: boolean;
  onOpenChange: (open: boolean) => void;
  onSubmit: (draft: PlatformChunkDraft) => Promise<void>;
}) {
  const [content, setContent] = useState("");
  const [keywords, setKeywords] = useState("");
  const [questions, setQuestions] = useState("");
  const [enabled, setEnabled] = useState(true);

  useEffect(() => {
    if (!open) return;
    setContent(chunk?.content ?? "");
    setKeywords(chunk?.importantKeywords.join(", ") ?? "");
    setQuestions(chunk?.questions.join(", ") ?? "");
    setEnabled(chunk?.enabled ?? true);
  }, [chunk, open]);

  const valid = content.trim().length > 0 && content.length <= 100_000;
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-2xl">
        <DialogHeader>
          <DialogTitle>{chunk ? "Chunk düzenle" : "Yeni chunk"}</DialogTitle>
          <DialogDescription>
            İçerik değişiklikleri retrieval sonuçlarını doğrudan etkiler.
          </DialogDescription>
        </DialogHeader>
        <div className="space-y-4">
          <label className="block space-y-1.5 text-sm font-medium">
            İçerik
            <Textarea
              value={content}
              onChange={(event) => setContent(event.target.value)}
              rows={10}
              maxLength={100_000}
              aria-label="Chunk içeriği"
            />
            <span className="block text-right text-xs font-normal text-muted-foreground">
              {content.length.toLocaleString("tr-TR")} / 100.000
            </span>
          </label>
          <label className="block space-y-1.5 text-sm font-medium">
            Önemli anahtar kelimeler
            <Input
              value={keywords}
              onChange={(event) => setKeywords(event.target.value)}
              placeholder="sözleşme, iade, ödeme"
            />
          </label>
          <label className="block space-y-1.5 text-sm font-medium">
            Örnek sorular
            <Input
              value={questions}
              onChange={(event) => setQuestions(event.target.value)}
              placeholder="İade süresi nedir?, Ödeme ne zaman yapılır?"
            />
          </label>
          {chunk ? (
            <div className="flex items-center justify-between rounded-xl border p-3">
              <div>
                <p className="text-sm font-medium">Retrieval içinde kullan</p>
                <p className="text-xs text-muted-foreground">
                  Kapalı chunk sonuçlara dahil edilmez.
                </p>
              </div>
              <Switch
                checked={enabled}
                onCheckedChange={setEnabled}
                aria-label="Chunk etkin"
              />
            </div>
          ) : null}
        </div>
        <DialogFooter>
          <Button variant="outline" onClick={() => onOpenChange(false)}>
            Vazgeç
          </Button>
          <Button
            disabled={!valid || saving}
            onClick={() =>
              void onSubmit({
                content: content.trim(),
                importantKeywords: tags(keywords),
                questions: tags(questions),
                enabled,
              })
            }
          >
            {saving ? <Spinner /> : null}
            {chunk ? "Değişiklikleri kaydet" : "Chunk oluştur"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

function StructurePanel({
  template,
  templates,
  loading,
  keywords,
  onKeywordsChange,
  onSearch,
  onTemplateChange,
  onDelete,
}: {
  template: PlatformStructureTemplate | null;
  templates: PlatformStructureTemplate[];
  loading: boolean;
  keywords: string;
  onKeywordsChange: (value: string) => void;
  onSearch: () => void;
  onTemplateChange: (value: string) => void;
  onDelete: (template: PlatformStructureTemplate) => void;
}) {
  return (
    <section className="rounded-3xl bg-card p-4 ring-1 ring-foreground/10">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div className="flex items-center gap-3">
          <div className="flex size-9 items-center justify-center rounded-xl bg-muted text-muted-foreground">
            <GitFork className="size-4" />
          </div>
          <div>
            <h3 className="font-heading text-sm font-semibold">Yapı grafiği</h3>
            <p className="mt-0.5 text-xs text-muted-foreground">
              Varlık ve ilişkilerin bağlantı haritası
            </p>
          </div>
        </div>
        {template ? <LiveBadge label="Hazır" /> : null}
      </div>
      <div className="mt-4 flex flex-col gap-2 sm:flex-row">
        <Select value={template?.id ?? ""} onValueChange={onTemplateChange}>
          <SelectTrigger className="sm:w-56" aria-label="Yapı grafiği şablonu">
            <SelectValue placeholder="Şablon seçin" />
          </SelectTrigger>
          <SelectContent>
            {templates.map((entry) => (
              <SelectItem key={entry.id} value={entry.id}>
                {entry.name} · {entry.kind}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
        <div className="flex min-w-0 flex-1 gap-2">
          <Input
            value={keywords}
            onChange={(event) => onKeywordsChange(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Enter") onSearch();
            }}
            placeholder="Grafikte varlık ara"
            aria-label="Yapı grafiğinde ara"
          />
          <Button
            size="icon"
            variant="outline"
            onClick={onSearch}
            disabled={loading}
            aria-label="Yapı grafiğinde ara"
          >
            {loading ? <Spinner /> : <Search />}
          </Button>
        </div>
      </div>
      {loading ? (
        <div className="grid min-h-36 content-center gap-3 py-4">
          <Skeleton className="mx-auto h-3 w-2/3" />
          <Skeleton className="mx-auto h-20 w-full" />
        </div>
      ) : !template ? (
        <div className="mt-4 rounded-2xl border border-dashed border-foreground/10 bg-muted/15 p-6 text-center">
          <GitFork className="mx-auto size-5 text-muted-foreground" />
          <p className="mt-2 text-sm font-medium">Henüz grafik yok</p>
          <p className="mt-1 text-xs text-muted-foreground">
            Bu belge için yapı grafiği üretilmemiş.
          </p>
        </div>
      ) : (
        <div className="mt-4 animate-in fade-in slide-in-from-bottom-1 duration-300">
          <div className="mb-3 grid grid-cols-2 gap-2">
            <MetricPill
              icon={Database}
              label="Varlık"
              value={template.entities.length.toLocaleString("tr-TR")}
            />
            <MetricPill
              icon={GitFork}
              label="İlişki"
              value={template.relations.length.toLocaleString("tr-TR")}
            />
          </div>
          <div className="grid gap-3">
            <div className="rounded-xl bg-muted/25 p-3">
              <p className="text-xs font-medium text-muted-foreground">
                Öne çıkan varlıklar
              </p>
              <div className="mt-2 max-h-48 space-y-2 overflow-auto">
                {template.entities.map((entity) => (
                  <div
                    key={entity.id}
                    className="rounded-lg bg-background p-2.5"
                  >
                    <p className="text-sm font-medium">{entity.name}</p>
                    {entity.description ? (
                      <p className="mt-1 line-clamp-2 text-xs text-muted-foreground">
                        {entity.description}
                      </p>
                    ) : null}
                  </div>
                ))}
              </div>
            </div>
            <div className="rounded-xl bg-muted/25 p-3">
              <p className="text-xs font-medium text-muted-foreground">
                Bağlantılar
              </p>
              <div className="mt-2 max-h-48 space-y-2 overflow-auto">
                {template.relations.map((relation) => (
                  <div
                    key={relation.id}
                    className="rounded-lg bg-background p-2.5"
                  >
                    <p className="text-sm font-medium">
                      {relation.source || "?"} → {relation.target || "?"}
                    </p>
                    {relation.description ? (
                      <p className="mt-1 line-clamp-2 text-xs text-muted-foreground">
                        {relation.description}
                      </p>
                    ) : null}
                  </div>
                ))}
              </div>
            </div>
          </div>
          <Button
            size="xs"
            variant="ghost"
            className="mt-3 w-full text-destructive hover:bg-destructive/5 hover:text-destructive"
            onClick={() => onDelete(template)}
          >
            <Trash2 /> Grafiği sil
          </Button>
        </div>
      )}
    </section>
  );
}

function ChunkWorkspace({
  datasetId,
  documentId,
  documents,
  onDocumentChange,
  onPreview,
}: {
  datasetId: string;
  documentId: string;
  documents: PlatformDocument[];
  onDocumentChange: (value: string) => void;
  onPreview: (document: PlatformDocument, pageNumber: number | null) => void;
}) {
  const workspace = useDatasetQualityWorkspace(datasetId, documentId);
  const scrollRef = useRef<HTMLDivElement>(null);
  const [query, setQuery] = useState("");
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [editorOpen, setEditorOpen] = useState(false);
  const [editingChunk, setEditingChunk] = useState<PlatformChunk | null>(null);
  const [deleteChunkIds, setDeleteChunkIds] = useState<string[]>([]);
  const [structureTemplateId, setStructureTemplateId] = useState("");
  const [deleteTemplate, setDeleteTemplate] =
    useState<PlatformStructureTemplate | null>(null);

  // TanStack Virtual owns its imperative measurement callbacks; they are not
  // passed into compiler-memoized children.
  // eslint-disable-next-line react-hooks/incompatible-library
  const rowVirtualizer = useVirtualizer({
    count: workspace.chunks.length,
    getScrollElement: () => scrollRef.current,
    estimateSize: () => 172,
    initialRect: { width: 0, height: 530 },
    overscan: 6,
  });
  const pages = Math.max(
    1,
    Math.ceil(workspace.chunkTotal / workspace.chunkPageSize),
  );
  const enabledOnPage = workspace.chunks.filter(
    (chunk) => chunk.enabled,
  ).length;
  const disabledOnPage = workspace.chunks.length - enabledOnPage;
  const activeTemplate =
    workspace.structureGraph.templates.find(
      (template) => template.id === structureTemplateId,
    ) ??
    workspace.structureGraph.templates[0] ??
    null;

  useEffect(() => {
    if (
      structureTemplateId &&
      workspace.structureGraph.templates.some(
        (template) => template.id === structureTemplateId,
      )
    )
      return;
    setStructureTemplateId(workspace.structureGraph.templates[0]?.id ?? "");
  }, [structureTemplateId, workspace.structureGraph.templates]);

  const currentDocument =
    documents.find((document) => document.id === documentId) ?? null;

  const run = async (
    operation: () => Promise<unknown>,
    message: string,
  ): Promise<boolean> => {
    try {
      await operation();
      toast.success(message);
      setSelected(new Set());
      return true;
    } catch (error) {
      toast.error("Chunk işlemi tamamlanamadı", {
        description: error instanceof Error ? error.message : String(error),
      });
      return false;
    }
  };

  if (!documentId || !currentDocument) {
    return (
      <Empty className="min-h-96">
        <EmptyHeader>
          <EmptyMedia variant="icon">
            <Blocks />
          </EmptyMedia>
          <EmptyTitle>Chunk görüntülemek için belge seçin</EmptyTitle>
          <EmptyDescription>
            Bu sayfadaki dataset belgelerinden biri seçilebilir olmalı.
          </EmptyDescription>
        </EmptyHeader>
      </Empty>
    );
  }

  return (
    <div className="grid gap-5 xl:grid-cols-[minmax(0,1.6fr)_minmax(300px,0.75fr)]">
      <Card
        size="sm"
        className="min-w-0 gap-0 rounded-3xl border-0 bg-card py-0 ring-1 ring-foreground/10"
      >
        <CardHeader className="border-b border-foreground/[0.06] px-4 py-4 sm:px-5">
          <div className="flex min-w-0 items-center gap-3">
            <div className="flex size-10 shrink-0 items-center justify-center rounded-2xl bg-foreground text-background shadow-sm">
              <Layers3 className="size-4" aria-hidden="true" />
            </div>
            <div className="min-w-0">
              <CardTitle className="font-heading text-base font-semibold tracking-tight">
                Chunk çalışma alanı
              </CardTitle>
              <CardDescription className="mt-0.5 truncate text-xs">
                {currentDocument.name}
              </CardDescription>
            </div>
          </div>
          <CardAction className="flex items-center gap-2">
            <Button
              size="sm"
              variant="ghost"
              onClick={() => void workspace.refreshChunks()}
              disabled={workspace.loadingChunks}
            >
              <RefreshCw
                className={cn(workspace.loadingChunks && "animate-spin")}
              />
              <span className="hidden sm:inline">Yenile</span>
            </Button>
            <Button
              size="sm"
              onClick={() => {
                setEditingChunk(null);
                setEditorOpen(true);
              }}
            >
              <Plus /> Yeni chunk
            </Button>
          </CardAction>
        </CardHeader>

        <CardContent className="px-4 py-4 sm:px-5">
          <div className="grid gap-2 sm:grid-cols-3">
            <MetricPill
              icon={Layers3}
              label="Toplam chunk"
              value={workspace.chunkTotal.toLocaleString("tr-TR")}
            />
            <MetricPill
              icon={CheckCircle2}
              label="Bu sayfada etkin"
              value={enabledOnPage.toLocaleString("tr-TR")}
              tone="success"
            />
            <MetricPill
              icon={PauseCircle}
              label="Bu sayfada kapalı"
              value={disabledOnPage.toLocaleString("tr-TR")}
              tone={disabledOnPage > 0 ? "warning" : "neutral"}
            />
          </div>

          <div className="mt-3 grid gap-2 rounded-2xl bg-muted/25 p-2.5 md:grid-cols-[minmax(180px,1fr)_minmax(180px,1fr)_140px]">
            <Select value={documentId} onValueChange={onDocumentChange}>
              <SelectTrigger
                className="bg-background shadow-none"
                aria-label="Chunk belgesi"
              >
                <FileText className="size-4 text-muted-foreground" />
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {documents.map((document) => (
                  <SelectItem key={document.id} value={document.id}>
                    {document.name}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            <div className="flex gap-2">
              <Input
                className="bg-background"
                value={query}
                onChange={(event) => setQuery(event.target.value)}
                onKeyDown={(event) => {
                  if (event.key === "Enter") workspace.setChunkKeywords(query);
                }}
                placeholder="İçerik veya anahtar kelime ara"
                aria-label="Chunk ara"
              />
              <Button
                size="icon"
                variant="outline"
                className="bg-background"
                aria-label="Chunk aramasını çalıştır"
                onClick={() => workspace.setChunkKeywords(query)}
              >
                <Search />
              </Button>
            </div>
            <Select
              value={workspace.chunkAvailability}
              onValueChange={(value) =>
                workspace.setChunkAvailability(
                  value as "all" | "enabled" | "disabled",
                )
              }
            >
              <SelectTrigger
                className="bg-background shadow-none"
                aria-label="Chunk durumu"
              >
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">Tüm durumlar</SelectItem>
                <SelectItem value="enabled">Etkin</SelectItem>
                <SelectItem value="disabled">Kapalı</SelectItem>
              </SelectContent>
            </Select>
          </div>
          {selected.size > 0 ? (
            <div className="mt-3 flex animate-in flex-wrap items-center gap-2 rounded-2xl bg-foreground px-3 py-2.5 text-background fade-in slide-in-from-top-1 duration-200">
              <span className="mr-auto flex items-center gap-2 text-sm font-medium">
                <CheckCircle2 className="size-4" />
                {selected.size} chunk seçili
              </span>
              <Button
                size="xs"
                variant="outline"
                className="border-background/20 bg-background/10 text-background hover:bg-background/20 hover:text-background"
                onClick={() =>
                  void run(
                    () => workspace.setChunksEnabled([...selected], true),
                    "Chunk'lar etkinleştirildi.",
                  )
                }
              >
                Etkinleştir
              </Button>
              <Button
                size="xs"
                variant="outline"
                className="border-background/20 bg-background/10 text-background hover:bg-background/20 hover:text-background"
                onClick={() =>
                  void run(
                    () => workspace.setChunksEnabled([...selected], false),
                    "Chunk'lar kapatıldı.",
                  )
                }
              >
                Kapat
              </Button>
              <Button
                size="xs"
                variant="ghost"
                className="text-background/80 hover:bg-background/10 hover:text-background"
                onClick={() => setDeleteChunkIds([...selected])}
              >
                <Trash2 /> Sil
              </Button>
            </div>
          ) : null}

          {workspace.chunkError ? (
            <div className="mt-4">
              <ErrorNotice
                error={workspace.chunkError}
                retry={() => void workspace.refreshChunks()}
              />
            </div>
          ) : workspace.loadingChunks ? (
            <QualityLoading label="Chunk verileri yükleniyor" />
          ) : workspace.chunks.length === 0 ? (
            <Empty className="mt-4 min-h-80 rounded-2xl bg-muted/15">
              <EmptyHeader>
                <EmptyMedia variant="icon">
                  <Blocks />
                </EmptyMedia>
                <EmptyTitle>
                  {workspace.chunkKeywords
                    ? "Eşleşen chunk yok"
                    : "Henüz chunk yok"}
                </EmptyTitle>
                <EmptyDescription>
                  {workspace.chunkKeywords
                    ? "Arama ifadesini veya durum filtresini değiştirin."
                    : "Belgeyi işleyin veya elle yeni bir chunk oluşturun."}
                </EmptyDescription>
              </EmptyHeader>
            </Empty>
          ) : (
            <div
              ref={scrollRef}
              className="mt-4 h-[530px] overflow-auto rounded-2xl bg-muted/20 p-1 [scrollbar-gutter:stable]"
              aria-label="Virtualized chunk listesi"
            >
              <div
                className="relative w-full"
                style={{ height: rowVirtualizer.getTotalSize() }}
              >
                {rowVirtualizer.getVirtualItems().map((virtualRow) => {
                  const chunk = workspace.chunks[virtualRow.index];
                  if (!chunk) return null;
                  return (
                    <article
                      key={chunk.id}
                      ref={rowVirtualizer.measureElement}
                      data-index={virtualRow.index}
                      className="absolute left-0 top-0 w-full p-1.5"
                      style={{ transform: `translateY(${virtualRow.start}px)` }}
                    >
                      <div
                        className={cn(
                          "group/chunk relative overflow-hidden rounded-2xl bg-background p-3.5 ring-1 ring-foreground/[0.07] transition-[transform,box-shadow,opacity] duration-200 hover:-translate-y-0.5 hover:shadow-sm",
                          !chunk.enabled && "bg-muted/40 opacity-75",
                        )}
                      >
                        <span
                          className={cn(
                            "absolute inset-y-3 left-0 w-0.5 rounded-r-full",
                            chunk.enabled
                              ? "bg-emerald-500"
                              : "bg-muted-foreground/30",
                          )}
                          aria-hidden="true"
                        />
                        <div className="flex items-start gap-3">
                          <Checkbox
                            checked={selected.has(chunk.id)}
                            onCheckedChange={(checked) => {
                              setSelected((current) => {
                                const next = new Set(current);
                                if (checked === true) next.add(chunk.id);
                                else next.delete(chunk.id);
                                return next;
                              });
                            }}
                            aria-label={`${chunk.id} chunk seç`}
                          />
                          <div className="min-w-0 flex-1">
                            <div className="flex flex-wrap items-center gap-2">
                              <Badge
                                variant="outline"
                                className={cn(
                                  "gap-1.5 shadow-none",
                                  chunk.enabled
                                    ? "border-emerald-500/20 bg-emerald-500/[0.07] text-emerald-700 dark:text-emerald-300"
                                    : "bg-muted/50 text-muted-foreground",
                                )}
                              >
                                <span
                                  className={cn(
                                    "size-1.5 rounded-full",
                                    chunk.enabled
                                      ? "bg-emerald-500"
                                      : "bg-muted-foreground/40",
                                  )}
                                />
                                {chunk.enabled ? "Etkin" : "Kapalı"}
                              </Badge>
                              {chunk.pageNumber ? (
                                <Badge variant="outline">
                                  Sayfa {chunk.pageNumber}
                                </Badge>
                              ) : null}
                              {chunk.importantKeywords
                                .slice(0, 3)
                                .map((keyword) => (
                                  <Badge key={keyword} variant="outline">
                                    {keyword}
                                  </Badge>
                                ))}
                              {chunk.importantKeywords.length > 3 ? (
                                <span className="text-[11px] text-muted-foreground">
                                  +{chunk.importantKeywords.length - 3}
                                </span>
                              ) : null}
                            </div>
                            <p className="mt-3 line-clamp-4 whitespace-pre-wrap text-sm leading-6 text-foreground/90">
                              {chunk.content || "İçerik yok"}
                            </p>
                            <div className="mt-3 flex min-w-0 items-center gap-3 border-t border-foreground/[0.05] pt-2.5 text-[11px] text-muted-foreground">
                              {chunk.questions.length > 0 ? (
                                <span className="flex min-w-0 items-center gap-1.5">
                                  <MessageCircleQuestion className="size-3.5 shrink-0" />
                                  <span className="truncate">
                                    {chunk.questions[0]}
                                  </span>
                                </span>
                              ) : (
                                <span className="flex items-center gap-1.5">
                                  <Layers3 className="size-3.5" /> Parça
                                </span>
                              )}
                              <span className="ml-auto max-w-28 truncate font-mono">
                                {chunk.id}
                              </span>
                            </div>
                          </div>
                          <div className="flex shrink-0 items-center gap-1 transition-opacity sm:opacity-55 sm:group-hover/chunk:opacity-100 sm:group-focus-within/chunk:opacity-100">
                            <Button
                              size="icon-xs"
                              variant="ghost"
                              aria-label="Chunk kaynağını aç"
                              onClick={() =>
                                onPreview(currentDocument, chunk.pageNumber)
                              }
                            >
                              <Eye />
                            </Button>
                            <Button
                              size="icon-xs"
                              variant="ghost"
                              aria-label="Chunk düzenle"
                              onClick={() => {
                                void workspace
                                  .loadChunk(chunk.id)
                                  .then((detail) => {
                                    setEditingChunk(detail);
                                    setEditorOpen(true);
                                  })
                                  .catch((error) =>
                                    toast.error("Chunk ayrıntısı alınamadı", {
                                      description:
                                        error instanceof Error
                                          ? error.message
                                          : String(error),
                                    }),
                                  );
                              }}
                            >
                              <Pencil />
                            </Button>
                            <Button
                              size="icon-xs"
                              variant="ghost"
                              className="text-destructive hover:text-destructive"
                              aria-label="Chunk sil"
                              onClick={() => setDeleteChunkIds([chunk.id])}
                            >
                              <Trash2 />
                            </Button>
                          </div>
                        </div>
                      </div>
                    </article>
                  );
                })}
              </div>
            </div>
          )}

          <footer className="mt-3 flex items-center justify-between border-t border-foreground/[0.06] pt-3">
            <Select
              value={String(workspace.chunkPageSize)}
              onValueChange={(value) =>
                workspace.setChunkPageSize(Number(value))
              }
            >
              <SelectTrigger className="w-28" aria-label="Sayfa başına chunk">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {[25, 50, 100, 200].map((size) => (
                  <SelectItem key={size} value={String(size)}>
                    {size} / sayfa
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            <div className="flex items-center gap-1">
              <Button
                size="icon-xs"
                variant="ghost"
                disabled={workspace.chunkPage <= 1}
                onClick={() => workspace.setChunkPage(workspace.chunkPage - 1)}
                aria-label="Önceki chunk sayfası"
              >
                <ChevronLeft />
              </Button>
              <span className="min-w-16 text-center text-xs tabular-nums text-muted-foreground">
                {workspace.chunkPage} / {pages}
              </span>
              <Button
                size="icon-xs"
                variant="ghost"
                disabled={workspace.chunkPage >= pages}
                onClick={() => workspace.setChunkPage(workspace.chunkPage + 1)}
                aria-label="Sonraki chunk sayfası"
              >
                <ChevronRight />
              </Button>
            </div>
          </footer>
        </CardContent>
      </Card>

      <div className="min-w-0">
        {workspace.structureError ? (
          <ErrorNotice
            error={workspace.structureError}
            retry={() => void workspace.refreshStructure()}
          />
        ) : (
          <StructurePanel
            template={activeTemplate}
            templates={workspace.structureGraph.templates}
            loading={workspace.loadingStructure}
            keywords={workspace.structureKeywords}
            onKeywordsChange={workspace.setStructureKeywords}
            onSearch={() => void workspace.refreshStructure()}
            onTemplateChange={setStructureTemplateId}
            onDelete={setDeleteTemplate}
          />
        )}
      </div>

      <ChunkEditor
        open={editorOpen}
        chunk={editingChunk}
        saving={workspace.mutating}
        onOpenChange={setEditorOpen}
        onSubmit={async (draft) => {
          const succeeded = await run(
            () =>
              editingChunk
                ? workspace.updateChunk(editingChunk.id, draft)
                : workspace.createChunk(draft),
            editingChunk ? "Chunk güncellendi." : "Chunk oluşturuldu.",
          );
          if (succeeded) setEditorOpen(false);
        }}
      />
      <AlertDialog
        open={deleteChunkIds.length > 0}
        onOpenChange={(open) => !open && setDeleteChunkIds([])}
      >
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>
              {deleteChunkIds.length} chunk kalıcı olarak silinsin mi?
            </AlertDialogTitle>
            <AlertDialogDescription>
              Retrieval sonuçları hemen değişir. Bu işlem geri alınamaz.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Vazgeç</AlertDialogCancel>
            <AlertDialogAction
              variant="destructive"
              onClick={() =>
                void run(
                  () => workspace.removeChunks(deleteChunkIds),
                  "Chunk'lar silindi.",
                ).then((succeeded) => {
                  if (succeeded) setDeleteChunkIds([]);
                })
              }
            >
              Kalıcı olarak sil
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
      <AlertDialog
        open={deleteTemplate !== null}
        onOpenChange={(open) => !open && setDeleteTemplate(null)}
      >
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Yapı grafiği silinsin mi?</AlertDialogTitle>
            <AlertDialogDescription>
              {deleteTemplate?.name ?? "Seçili grafik"} ve ilişkili graph
              satırları kalıcı olarak kaldırılır.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Vazgeç</AlertDialogCancel>
            <AlertDialogAction
              variant="destructive"
              onClick={() => {
                if (!deleteTemplate) return;
                void run(
                  () => workspace.removeStructureTemplate(deleteTemplate.id),
                  "Yapı grafiği silindi.",
                ).then((succeeded) => {
                  if (succeeded) setDeleteTemplate(null);
                });
              }}
            >
              Grafiği sil
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </div>
  );
}

function RetrievalWorkspace({
  datasetId,
  documents,
  onPreview,
}: {
  datasetId: string;
  documents: PlatformDocument[];
  onPreview: (document: PlatformDocument, pageNumber: number | null) => void;
}) {
  const workspace = useDatasetQualityWorkspace(datasetId, "");
  const [question, setQuestion] = useState("");
  const [documentId, setDocumentId] = useState("all");
  const [topK, setTopK] = useState(10);
  const [threshold, setThreshold] = useState(0.2);
  const [vectorWeight, setVectorWeight] = useState(0.3);
  const [highlight, setHighlight] = useState(true);
  const [rerank, setRerank] = useState(false);
  const [rerankId, setRerankId] = useState("");
  const valid = question.trim().length > 0 && (!rerank || rerankId.trim());
  const bestScore = workspace.retrieval?.items[0]?.normalizedScore ?? null;
  const scopeLabel =
    documentId === "all"
      ? `${documents.length.toLocaleString("tr-TR")} görünür belge`
      : (documents.find((document) => document.id === documentId)?.name ??
        "Seçili belge");

  const runSearch = () => {
    if (!valid) return;
    void workspace.runRetrieval({
      question,
      documentIds: documentId === "all" ? [] : [documentId],
      topK,
      pageSize: topK,
      similarityThreshold: threshold,
      vectorSimilarityWeight: vectorWeight,
      rerankId: rerank ? rerankId : undefined,
      highlight,
    });
  };

  return (
    <div className="grid gap-5 xl:grid-cols-[340px_minmax(0,1fr)]">
      <Card
        size="sm"
        className="h-fit gap-0 rounded-3xl border-0 bg-card py-0 ring-1 ring-foreground/10 xl:sticky xl:top-4"
      >
        <CardHeader className="border-b border-foreground/[0.06] px-4 py-4">
          <div className="flex items-center gap-3">
            <div className="flex size-10 items-center justify-center rounded-2xl bg-foreground text-background shadow-sm">
              <SlidersHorizontal className="size-4" />
            </div>
            <div>
              <CardTitle className="font-heading text-base font-semibold tracking-tight">
                Retrieval ayarları
              </CardTitle>
              <CardDescription className="mt-0.5 text-xs">
                Arama kalitesini gerçek sonuçlarla ölçün
              </CardDescription>
            </div>
          </div>
          <CardAction>
            <LiveBadge />
          </CardAction>
        </CardHeader>
        <CardContent className="space-y-4 px-4 py-4">
          <label className="block space-y-1.5 text-sm font-medium">
            <span className="flex items-center justify-between gap-2">
              <span>Arama sorusu</span>
              <span className="text-[10px] font-normal tabular-nums text-muted-foreground">
                {question.length.toLocaleString("tr-TR")} / 8.000
              </span>
            </span>
            <Textarea
              value={question}
              onChange={(event) => setQuestion(event.target.value)}
              rows={4}
              maxLength={8_000}
              placeholder="Örn. İade süresi kaç gündür?"
              aria-label="Retrieval sorgusu"
              className="resize-none bg-muted/20 leading-6 focus:bg-background"
            />
          </label>
          <label className="block space-y-1.5 text-sm font-medium">
            <span className="flex items-center gap-1.5">
              Belge kapsamı
              <InfoHint>
                Sorguyu tüm datasette veya yalnızca seçtiğiniz belgede
                çalıştırın.
              </InfoHint>
            </span>
            <Select value={documentId} onValueChange={setDocumentId}>
              <SelectTrigger aria-label="Retrieval belge kapsamı">
                <FileText className="size-4 text-muted-foreground" />
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">
                  Dataset içindeki tüm belgeler
                </SelectItem>
                {documents.map((document) => (
                  <SelectItem key={document.id} value={document.id}>
                    {document.name}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </label>
          <div className="space-y-4 rounded-2xl bg-muted/25 p-3.5">
            <SliderControl
              label="Sonuç sayısı"
              hint="Backend'in döndüreceği en yüksek sonuç adedi."
              value={topK}
              min={1}
              max={50}
              step={1}
              format={(value) => String(value)}
              onChange={setTopK}
            />
            <SliderControl
              label="Benzerlik eşiği"
              hint="Düşük değer daha fazla; yüksek değer daha seçici sonuç getirir."
              value={threshold}
              min={0}
              max={1}
              step={0.05}
              format={(value) => `%${(value * 100).toFixed(0)}`}
              onChange={setThreshold}
            />
            <SliderControl
              label="Vektör ağırlığı"
              hint="Anlamsal benzerliğin toplam skor içindeki payı."
              value={vectorWeight}
              min={0}
              max={1}
              step={0.05}
              format={(value) => `%${(value * 100).toFixed(0)}`}
              onChange={setVectorWeight}
            />
          </div>
          <div className="flex items-center justify-between rounded-2xl bg-muted/20 px-3 py-2.5 ring-1 ring-foreground/[0.05]">
            <div>
              <p className="text-sm font-medium">Eşleşmeleri vurgula</p>
              <p className="text-xs text-muted-foreground">
                Kaynak metindeki ilgili bölümleri gösterir
              </p>
            </div>
            <Switch
              checked={highlight}
              onCheckedChange={setHighlight}
              aria-label="Retrieval highlight"
            />
          </div>
          <div className="space-y-2 rounded-2xl bg-muted/20 px-3 py-2.5 ring-1 ring-foreground/[0.05]">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium">Sonuçları yeniden sırala</p>
                <p className="text-xs text-muted-foreground">
                  Seçili rerank modeliyle kaliteyi artırır
                </p>
              </div>
              <Switch
                checked={rerank}
                onCheckedChange={setRerank}
                aria-label="Retrieval rerank"
              />
            </div>
            {rerank ? (
              <Input
                value={rerankId}
                onChange={(event) => setRerankId(event.target.value)}
                placeholder="rerank model ID"
                aria-label="Rerank model ID"
              />
            ) : null}
          </div>
          <Button
            className="h-10 w-full shadow-sm transition-transform active:scale-[0.99]"
            disabled={!valid || workspace.retrieving}
            onClick={runSearch}
          >
            {workspace.retrieving ? <Spinner /> : <FileSearch />}
            Retrieval çalıştır
          </Button>
          <div className="flex items-center justify-between gap-3 border-t border-foreground/[0.06] pt-3 text-[11px] text-muted-foreground">
            <span className="flex min-w-0 items-center gap-1.5">
              <Target className="size-3.5 shrink-0" />
              <span className="truncate">{scopeLabel}</span>
            </span>
            <span className="shrink-0 tabular-nums">Top {topK}</span>
          </div>
        </CardContent>
      </Card>

      <Card
        size="sm"
        className="min-w-0 gap-0 rounded-3xl border-0 bg-card py-0 ring-1 ring-foreground/10"
      >
        <CardHeader className="border-b border-foreground/[0.06] px-4 py-4 sm:px-5">
          <div className="flex min-w-0 items-center gap-3">
            <div className="flex size-10 shrink-0 items-center justify-center rounded-2xl bg-muted text-muted-foreground">
              <Target className="size-4" />
            </div>
            <div className="min-w-0">
              <CardTitle className="font-heading text-base font-semibold tracking-tight">
                Arama sonuçları
              </CardTitle>
              <CardDescription className="mt-0.5 truncate text-xs">
                {workspace.retrieval
                  ? `${workspace.retrieval.total.toLocaleString("tr-TR")} eşleşme · ${scopeLabel}`
                  : "Sorunuzu yazın ve kaliteyi ölçün"}
              </CardDescription>
            </div>
          </div>
          <CardAction>
            {workspace.retrieval ? (
              <LiveBadge label="Tamamlandı" />
            ) : (
              <Badge variant="outline" className="gap-1.5 shadow-none">
                <SlidersHorizontal className="size-3" />%
                {(threshold * 100).toFixed(0)} eşik
              </Badge>
            )}
          </CardAction>
        </CardHeader>
        <CardContent className="px-4 py-4 sm:px-5">
          {workspace.retrieval && workspace.retrieval.items.length > 0 ? (
            <div className="mb-4 grid gap-2 sm:grid-cols-[1fr_1fr_1.35fr]">
              <MetricPill
                icon={FileSearch}
                label="Eşleşme"
                value={workspace.retrieval.total.toLocaleString("tr-TR")}
              />
              <MetricPill
                icon={Zap}
                label="En iyi skor"
                value={
                  bestScore === null ? "—" : `%${(bestScore * 100).toFixed(1)}`
                }
                tone={
                  bestScore !== null && bestScore >= 0.75
                    ? "success"
                    : "neutral"
                }
              />
              <MetricPill icon={Target} label="Kapsam" value={scopeLabel} />
            </div>
          ) : null}
          {workspace.retrievalError ? (
            <div>
              <ErrorNotice error={workspace.retrievalError} retry={runSearch} />
            </div>
          ) : workspace.retrieving ? (
            <QualityLoading label="Retrieval sonuçları yükleniyor" />
          ) : workspace.retrieval?.items.length === 0 ? (
            <Empty className="min-h-[460px] rounded-2xl bg-muted/15">
              <EmptyHeader>
                <EmptyMedia variant="icon">
                  <Binary />
                </EmptyMedia>
                <EmptyTitle>Sonuç bulunamadı</EmptyTitle>
                <EmptyDescription>
                  Bu bir hata değil. Eşiği düşürün, belge kapsamını genişletin
                  veya sorguyu değiştirin.
                </EmptyDescription>
              </EmptyHeader>
            </Empty>
          ) : !workspace.retrieval ? (
            <Empty className="min-h-[460px] rounded-2xl bg-muted/15">
              <EmptyHeader>
                <EmptyMedia variant="icon">
                  <Sparkles />
                </EmptyMedia>
                <EmptyTitle>Retrieval kalitesini test edin</EmptyTitle>
                <EmptyDescription>
                  Bir soru yazın. Sonuçları kalite skoru ve kaynak kırılımıyla
                  birlikte gösterelim.
                </EmptyDescription>
              </EmptyHeader>
            </Empty>
          ) : (
            <div className="max-h-[690px] space-y-3 overflow-auto pr-1 [scrollbar-gutter:stable]">
              {workspace.retrieval.items.map((chunk, index) => {
                const source =
                  documents.find(
                    (document) => document.id === chunk.documentId,
                  ) ?? chunkPreviewDocument(chunk, datasetId);
                return (
                  <article
                    key={`${chunk.id}-${index}`}
                    className="group/result animate-in rounded-2xl bg-background p-4 ring-1 ring-foreground/[0.07] fade-in slide-in-from-bottom-2 duration-300 transition-[transform,box-shadow] hover:-translate-y-0.5 hover:shadow-sm"
                  >
                    <div className="flex flex-wrap items-start gap-2">
                      <div className="flex size-8 shrink-0 items-center justify-center rounded-xl bg-foreground font-mono text-xs font-semibold text-background shadow-sm">
                        {String(index + 1).padStart(2, "0")}
                      </div>
                      <div className="min-w-0 flex-1">
                        <div className="flex flex-wrap items-center gap-2">
                          <ScoreBadge score={chunk.normalizedScore} />
                          <span className="text-xs font-medium text-muted-foreground">
                            {scoreLabel(chunk.normalizedScore)}
                          </span>
                        </div>
                        <div className="mt-1.5 flex min-w-0 items-center gap-2 text-xs text-muted-foreground">
                          <FileText className="size-3.5 shrink-0" />
                          <span className="truncate">
                            {chunk.documentName || source.name}
                          </span>
                          {chunk.pageNumber ? (
                            <span className="shrink-0 rounded-md bg-muted px-1.5 py-0.5 font-medium tabular-nums">
                              s. {chunk.pageNumber}
                            </span>
                          ) : null}
                        </div>
                      </div>
                      {chunk.pageNumber ? (
                        <span className="sr-only">
                          Sayfa {chunk.pageNumber}
                        </span>
                      ) : null}
                      <Button
                        size="xs"
                        variant="outline"
                        className="shrink-0 transition-transform active:scale-[0.98]"
                        disabled={!chunk.documentId}
                        onClick={() => onPreview(source, chunk.pageNumber)}
                      >
                        <Eye /> Kaynağı aç
                      </Button>
                    </div>
                    <p className="mt-4 whitespace-pre-wrap text-sm leading-6 text-foreground/90">
                      {chunk.content || "İçerik yok"}
                    </p>
                    <div className="mt-4">
                      <ScoreBreakdown chunk={chunk} />
                    </div>
                  </article>
                );
              })}
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}

function SliderControl({
  label,
  hint,
  value,
  min,
  max,
  step,
  onChange,
  format = (entry) => entry.toFixed(2),
}: {
  label: string;
  hint?: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (value: number) => void;
  format?: (value: number) => string;
}) {
  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between text-sm font-medium">
        <span className="flex items-center gap-1.5">
          {label}
          {hint ? <InfoHint>{hint}</InfoHint> : null}
        </span>
        <span className="rounded-md bg-background px-1.5 py-0.5 font-mono text-[11px] tabular-nums text-muted-foreground ring-1 ring-foreground/[0.06]">
          {format(value)}
        </span>
      </div>
      <Slider
        value={[value]}
        min={min}
        max={max}
        step={step}
        onValueChange={([next]) => onChange(next)}
        aria-label={label}
      />
    </div>
  );
}

export function DatasetQualityWorkspace({
  mode,
  datasetId,
  datasetName,
  documents,
  preferredDocumentId,
  onPreview,
}: DatasetQualityWorkspaceProps) {
  const [documentId, setDocumentId] = useState(preferredDocumentId ?? "");
  useEffect(() => {
    setDocumentId((current) => {
      if (current && documents.some((document) => document.id === current))
        return current;
      if (
        preferredDocumentId &&
        documents.some((document) => document.id === preferredDocumentId)
      )
        return preferredDocumentId;
      return documents[0]?.id ?? "";
    });
  }, [documents, preferredDocumentId]);

  if (!datasetId) {
    return (
      <Empty className="min-h-full">
        <EmptyHeader>
          <EmptyMedia variant="icon">
            <Binary />
          </EmptyMedia>
          <EmptyTitle>Önce bir dataset seçin</EmptyTitle>
          <EmptyDescription>
            Chunk ve retrieval araçları aktif bir dataset gerektirir.
          </EmptyDescription>
        </EmptyHeader>
      </Empty>
    );
  }

  return (
    <div className="h-full overflow-y-auto [scrollbar-gutter:stable_both-edges]">
      <div className="mx-auto w-full max-w-[var(--hub-measure)] px-5 pb-10 pt-6 sm:px-8">
        <header className="relative mb-5 overflow-hidden rounded-3xl bg-card px-5 py-5 ring-1 ring-foreground/10 sm:px-6">
          <div
            className="pointer-events-none absolute -right-16 -top-20 size-52 rounded-full bg-foreground/[0.025] blur-2xl"
            aria-hidden="true"
          />
          <div className="relative flex flex-col justify-between gap-4 sm:flex-row sm:items-center">
            <div className="flex min-w-0 items-center gap-4">
              <div className="flex size-11 shrink-0 items-center justify-center rounded-2xl bg-muted text-foreground ring-1 ring-foreground/[0.06]">
                {mode === "chunks" ? (
                  <Layers3 className="size-5" />
                ) : (
                  <Sparkles className="size-5" />
                )}
              </div>
              <div className="min-w-0">
                <p className="flex items-center gap-2 text-xs font-medium text-muted-foreground">
                  <Activity className="size-3.5" />
                  {mode === "chunks"
                    ? "Belge parçaları ve yapı"
                    : "Retrieval kalite laboratuvarı"}
                </p>
                <h1 className="mt-1 truncate font-heading text-xl font-semibold tracking-tight text-balance sm:text-2xl">
                  {datasetName || "Dataset"}
                </h1>
              </div>
            </div>
            <div className="flex flex-wrap items-center gap-2">
              <Badge
                variant="outline"
                className="gap-1.5 bg-background/60 shadow-none"
              >
                <FileText className="size-3" />
                {documents.length.toLocaleString("tr-TR")} görünür belge
              </Badge>
              <LiveBadge label="Backend bağlı" />
            </div>
          </div>
        </header>
        {mode === "chunks" ? (
          <ChunkWorkspace
            datasetId={datasetId}
            documentId={documentId}
            documents={documents}
            onDocumentChange={setDocumentId}
            onPreview={onPreview}
          />
        ) : (
          <RetrievalWorkspace
            datasetId={datasetId}
            documents={documents}
            onPreview={onPreview}
          />
        )}
      </div>
    </div>
  );
}
