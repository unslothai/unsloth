// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  Delete02Icon,
  Edit03Icon,
  PlusSignIcon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import {
  ChevronLeftIcon,
  ChevronRightIcon,
  SearchIcon,
  UploadIcon,
} from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";

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
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Spinner } from "@/components/ui/spinner";
import { Textarea } from "@/components/ui/textarea";
import { useSettingsDialogStore } from "@/features/settings/stores/settings-dialog-store";
import {
  type PlatformDatasetChunkMethod,
  type PlatformModel,
  getPlatformModelReadiness,
  isPlatformApiError,
} from "@/integrations/platform-backend";
import { toast } from "@/lib/toast";

import {
  datasetEmbeddingModelReference,
  getKnowledgeBase,
  listKnowledgeBasePage,
} from "../api/platform-dataset-adapter";
import {
  createKnowledgeBase,
  deleteKnowledgeBase,
  listKnowledgeBaseDocuments,
  updateKnowledgeBase,
} from "../api/rag-api";
import {
  type KnowledgeBase,
  RAG_UPLOAD_ACCEPT,
  isLinkedFolderManaged,
} from "../types/rag";
import { DocumentStatusChip } from "./document-status-chip";
import { LinkedFoldersManager } from "./linked-folders-manager";
import { PlatformPipelineSelect } from "./platform-pipeline-select";
import { useRagDocuments } from "./use-rag-documents";

const PAGE_SIZE = 8;
const CHUNK_METHODS: readonly {
  value: PlatformDatasetChunkMethod;
  label: string;
}[] = [
  { value: "naive", label: "General" },
  { value: "book", label: "Book" },
  { value: "email", label: "Email" },
  { value: "laws", label: "Legal" },
  { value: "manual", label: "Manual" },
  { value: "one", label: "One document" },
  { value: "paper", label: "Academic paper" },
  { value: "picture", label: "Image" },
  { value: "presentation", label: "Presentation" },
  { value: "qa", label: "Q&A" },
  { value: "table", label: "Table" },
  { value: "tag", label: "Tag" },
  { value: "resume", label: "Resume" },
];

type View =
  | { kind: "list" }
  | { kind: "create" }
  | { kind: "edit"; kb: KnowledgeBase }
  | { kind: "documents"; kb: KnowledgeBase };

type SortValue = "update-desc" | "update-asc" | "create-desc" | "create-asc";

interface FieldErrors {
  embeddingModel?: string;
  name?: string;
  parserConfig?: string;
}

export interface KnowledgeBaseDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

function errorMessage(error: unknown): string {
  if (isPlatformApiError(error) && error.isPermissionError) {
    return "Bu dataset işlemi için yetkiniz yok.";
  }
  return error instanceof Error ? error.message : String(error);
}

function errorsForWrite(error: unknown): {
  fields: FieldErrors;
  form?: string;
} {
  const message = errorMessage(error);
  const normalized = message.toLocaleLowerCase("tr");
  if (
    normalized.includes("duplicate") ||
    normalized.includes("duplicated") ||
    normalized.includes("already exists") ||
    normalized.includes("aynı ad")
  ) {
    return { fields: { name: message } };
  }
  if (normalized.includes("embedding")) {
    return { fields: { embeddingModel: message } };
  }
  if (
    normalized.includes("parser") ||
    normalized.includes("chunk") ||
    normalized.includes("pipeline")
  ) {
    return { fields: { parserConfig: message } };
  }
  return { fields: {}, form: message };
}

function modelLabel(model: PlatformModel): string {
  const owner = model.instanceName || model.providerName;
  return owner ? model.name + " · " + owner : model.name;
}

export function KnowledgeBaseDialog({
  open,
  onOpenChange,
}: KnowledgeBaseDialogProps) {
  const [view, setView] = useState<View>({ kind: "list" });
  const [kbs, setKbs] = useState<KnowledgeBase[]>([]);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(1);
  const [search, setSearch] = useState("");
  const [sort, setSort] = useState<SortValue>("update-desc");
  const [loading, setLoading] = useState(false);
  const [listError, setListError] = useState<string | null>(null);
  const [refreshKey, setRefreshKey] = useState(0);
  const [editingId, setEditingId] = useState<string | null>(null);

  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [embeddingModel, setEmbeddingModel] = useState("");
  const [permission, setPermission] = useState<"me" | "team">("me");
  const [chunkMethod, setChunkMethod] =
    useState<PlatformDatasetChunkMethod>("naive");
  const [pipelineId, setPipelineId] = useState("");
  const [parserConfig, setParserConfig] = useState("");
  const [embeddingModels, setEmbeddingModels] = useState<PlatformModel[]>([]);
  const [formLoading, setFormLoading] = useState(false);
  const [formError, setFormError] = useState<string | null>(null);
  const [fieldErrors, setFieldErrors] = useState<FieldErrors>({});
  const [saving, setSaving] = useState(false);

  const [confirmingDelete, setConfirmingDelete] =
    useState<KnowledgeBase | null>(null);
  const [deleting, setDeleting] = useState(false);
  const detailAbortRef = useRef<AbortController | undefined>(undefined);
  const mutationAbortRef = useRef<AbortController | undefined>(undefined);

  const resetForm = useCallback((kb?: KnowledgeBase) => {
    setName(kb?.name ?? "");
    setDescription(kb?.description ?? "");
    setEmbeddingModel(kb?.embeddingModel ?? "");
    setPermission(kb?.permission ?? "me");
    setChunkMethod(
      CHUNK_METHODS.some((method) => method.value === kb?.chunkMethod)
        ? (kb?.chunkMethod as PlatformDatasetChunkMethod)
        : "naive",
    );
    setPipelineId(kb?.pipelineId ?? "");
    setParserConfig(
      kb?.parserConfig ? JSON.stringify(kb.parserConfig, null, 2) : "",
    );
    setFormError(null);
    setFieldErrors({});
  }, []);

  useEffect(() => {
    if (!open || view.kind !== "list") return;
    const controller = new AbortController();
    const timer = window.setTimeout(() => {
      const [orderBy, direction] = sort.split("-") as [
        "update" | "create",
        "asc" | "desc",
      ];
      setLoading(true);
      setListError(null);
      void listKnowledgeBasePage(
        {
          page,
          pageSize: PAGE_SIZE,
          name: search,
          orderBy: orderBy === "create" ? "create_time" : "update_time",
          desc: direction === "desc",
        },
        controller.signal,
      )
        .then((result) => {
          setKbs(result.items);
          setTotal(result.total);
        })
        .catch((error: unknown) => {
          if (!isPlatformApiError(error) || !error.isAbort) {
            setKbs([]);
            setTotal(0);
            setListError(errorMessage(error));
          }
        })
        .finally(() => {
          if (!controller.signal.aborted) setLoading(false);
        });
    }, 250);
    return () => {
      window.clearTimeout(timer);
      controller.abort();
    };
  }, [open, page, refreshKey, search, sort, view.kind]);

  useEffect(() => {
    if (!open || (view.kind !== "create" && view.kind !== "edit")) return;
    const controller = new AbortController();
    setFormLoading(true);
    setFormError(null);
    void getPlatformModelReadiness("embedding", controller.signal)
      .then((readiness) => {
        const models = readiness.models.filter((model) =>
          model.capabilities.includes("embedding"),
        );
        setEmbeddingModels(models);
        setEmbeddingModel((current) => {
          if (current) return current;
          const selected = readiness.defaults.find(
            (item) => item.capability === "embedding" && item.enabled,
          );
          const model = models.find(
            (item) =>
              item.id === selected?.modelId ||
              item.name === selected?.modelName,
          );
          return model ? datasetEmbeddingModelReference(model) : "";
        });
        if (!readiness.ready) {
          setFormError(
            "Dataset oluşturmak için etkin bir embedding modeli ve doğrulanmış varsayılan gerekir.",
          );
        }
      })
      .catch((error: unknown) => {
        if (!isPlatformApiError(error) || !error.isAbort) {
          setFormError(errorMessage(error));
        }
      })
      .finally(() => {
        if (!controller.signal.aborted) setFormLoading(false);
      });
    return () => controller.abort();
  }, [open, view.kind]);

  useEffect(() => {
    if (open) return;
    detailAbortRef.current?.abort();
    mutationAbortRef.current?.abort();
    detailAbortRef.current = undefined;
    mutationAbortRef.current = undefined;
    setView({ kind: "list" });
    setConfirmingDelete(null);
    setEditingId(null);
    setSaving(false);
    setDeleting(false);
  }, [open]);

  function startCreate() {
    resetForm();
    setView({ kind: "create" });
  }

  async function startEdit(kb: KnowledgeBase) {
    detailAbortRef.current?.abort();
    const controller = new AbortController();
    detailAbortRef.current = controller;
    setEditingId(kb.id);
    try {
      const detail = await getKnowledgeBase(kb.id, controller.signal);
      resetForm(detail);
      setView({ kind: "edit", kb: detail });
    } catch (error) {
      if (!isPlatformApiError(error) || !error.isAbort) {
        toast.error("Dataset ayrıntıları yüklenemedi", {
          description: errorMessage(error),
        });
      }
    } finally {
      if (!controller.signal.aborted) setEditingId(null);
    }
  }

  function backToList() {
    mutationAbortRef.current?.abort();
    setView({ kind: "list" });
    setFormError(null);
    setFieldErrors({});
  }

  async function submitForm() {
    const nextErrors: FieldErrors = {};
    const trimmedName = name.trim();
    if (!trimmedName) nextErrors.name = "Ad zorunludur.";
    if (!embeddingModel.trim()) {
      nextErrors.embeddingModel = "Embedding modeli zorunludur.";
    }
    let parsedConfig: Record<string, unknown> | undefined;
    if (parserConfig.trim()) {
      try {
        const value = JSON.parse(parserConfig) as unknown;
        if (
          typeof value !== "object" ||
          value === null ||
          Array.isArray(value)
        ) {
          throw new TypeError("Parser ayarı bir JSON nesnesi olmalıdır.");
        }
        parsedConfig = value as Record<string, unknown>;
      } catch (error) {
        nextErrors.parserConfig =
          error instanceof Error ? error.message : "Geçersiz JSON.";
      }
    }
    setFieldErrors(nextErrors);
    setFormError(null);
    if (Object.keys(nextErrors).length > 0) return;

    mutationAbortRef.current?.abort();
    const controller = new AbortController();
    mutationAbortRef.current = controller;
    setSaving(true);
    try {
      const payload = {
        name: trimmedName,
        description: description.trim() || undefined,
        embeddingModel: embeddingModel.trim(),
        permission,
        chunkMethod,
        parserConfig: parsedConfig,
        pipelineId: pipelineId || undefined,
      };
      if (view.kind === "edit") {
        await updateKnowledgeBase(view.kb.id, payload, controller.signal);
        toast.success("Knowledge base güncellendi");
      } else {
        await createKnowledgeBase(payload, controller.signal);
        toast.success("Knowledge base oluşturuldu");
      }
      setPage(1);
      setView({ kind: "list" });
      setRefreshKey((value) => value + 1);
    } catch (error) {
      if (!isPlatformApiError(error) || !error.isAbort) {
        const mapped = errorsForWrite(error);
        setFieldErrors(mapped.fields);
        setFormError(mapped.form ?? null);
      }
    } finally {
      if (!controller.signal.aborted) setSaving(false);
    }
  }

  async function removeKb(kb: KnowledgeBase) {
    mutationAbortRef.current?.abort();
    const controller = new AbortController();
    mutationAbortRef.current = controller;
    setDeleting(true);
    try {
      await deleteKnowledgeBase(kb.id, controller.signal);
      toast.success("Knowledge base silindi");
      setConfirmingDelete(null);
      if (page > 1 && kbs.length === 1) setPage((value) => value - 1);
      else setRefreshKey((value) => value + 1);
    } catch (error) {
      if (!isPlatformApiError(error) || !error.isAbort) {
        toast.error("Knowledge base silinemedi", {
          description: errorMessage(error),
        });
      }
    } finally {
      if (!controller.signal.aborted) setDeleting(false);
    }
  }

  const showForm = view.kind === "create" || view.kind === "edit";
  const pageCount = Math.max(1, Math.ceil(total / PAGE_SIZE));

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-2xl">
        <DialogHeader>
          <DialogTitle>
            {view.kind === "create"
              ? "Knowledge base oluştur"
              : view.kind === "edit"
                ? "Knowledge base ayarları"
                : view.kind === "documents"
                  ? view.kb.name
                  : "Knowledge bases"}
          </DialogTitle>
          <DialogDescription>
            {view.kind === "documents"
              ? "Upload documents to index for retrieval in chat."
              : "Rag Platform dataset’lerini oluşturun, arayın ve yönetin."}
          </DialogDescription>
        </DialogHeader>

        {view.kind === "documents" ? (
          <KnowledgeBaseDocuments kb={view.kb} onBack={backToList} />
        ) : showForm ? (
          <div className="flex max-h-[70dvh] flex-col gap-4 overflow-y-auto pr-1">
            <div className="grid gap-2">
              <Label htmlFor="kb-name">Ad</Label>
              <Input
                id="kb-name"
                value={name}
                disabled={saving}
                aria-invalid={Boolean(fieldErrors.name)}
                onChange={(event) => setName(event.target.value)}
                placeholder="Örn. Ürün belgeleri"
              />
              {fieldErrors.name && (
                <p className="text-xs text-destructive">{fieldErrors.name}</p>
              )}
            </div>
            <div className="grid gap-2">
              <Label htmlFor="kb-description">Açıklama</Label>
              <Textarea
                id="kb-description"
                value={description}
                disabled={saving}
                onChange={(event) => setDescription(event.target.value)}
                placeholder="Bu bilgi tabanının içeriği"
                rows={3}
              />
            </div>
            <div className="grid gap-2">
              <Label htmlFor="kb-embedding-model">Embedding modeli</Label>
              {formLoading ? (
                <div className="flex items-center gap-2 text-sm text-muted-foreground">
                  <Spinner /> Modeller yükleniyor…
                </div>
              ) : (
                <select
                  id="kb-embedding-model"
                  className="h-9 rounded-full border bg-background px-3 text-sm"
                  value={embeddingModel}
                  disabled={saving || embeddingModels.length === 0}
                  aria-invalid={Boolean(fieldErrors.embeddingModel)}
                  onChange={(event) => setEmbeddingModel(event.target.value)}
                >
                  <option value="">Model seçin</option>
                  {embeddingModel &&
                  !embeddingModels.some(
                    (model) =>
                      datasetEmbeddingModelReference(model) === embeddingModel,
                  ) ? (
                    <option value={embeddingModel}>{embeddingModel}</option>
                  ) : null}
                  {embeddingModels.map((model) => (
                    <option
                      key={model.id}
                      value={datasetEmbeddingModelReference(model)}
                    >
                      {modelLabel(model)}
                    </option>
                  ))}
                </select>
              )}
              {fieldErrors.embeddingModel && (
                <p className="text-xs text-destructive">
                  {fieldErrors.embeddingModel}
                </p>
              )}
              {!formLoading && embeddingModels.length === 0 && (
                <Button
                  variant="outline"
                  size="sm"
                  className="w-fit"
                  onClick={() =>
                    useSettingsDialogStore.getState().openDialog("connections")
                  }
                >
                  Bağlantıları aç
                </Button>
              )}
            </div>
            <div className="grid gap-2">
              <Label htmlFor="kb-permission">Erişim</Label>
              <select
                id="kb-permission"
                className="h-9 rounded-full border bg-background px-3 text-sm"
                value={permission}
                disabled={saving}
                onChange={(event) =>
                  setPermission(event.target.value === "team" ? "team" : "me")
                }
              >
                <option value="me">Yalnızca ben</option>
                <option value="team">Ekibim</option>
              </select>
            </div>

            <details className="rounded-md border px-3 py-2">
              <summary className="cursor-pointer text-sm font-medium">
                Gelişmiş ayrıştırma ayarları
              </summary>
              <div className="mt-4 grid gap-4">
                <div className="grid gap-2">
                  <Label htmlFor="kb-chunk-method">Chunk yöntemi</Label>
                  <select
                    id="kb-chunk-method"
                    className="h-9 rounded-full border bg-background px-3 text-sm"
                    value={chunkMethod}
                    disabled={saving}
                    onChange={(event) =>
                      setChunkMethod(
                        event.target.value as PlatformDatasetChunkMethod,
                      )
                    }
                  >
                    {CHUNK_METHODS.map((method) => (
                      <option key={method.value} value={method.value}>
                        {method.label}
                      </option>
                    ))}
                  </select>
                </div>
                <PlatformPipelineSelect
                  value={pipelineId}
                  disabled={saving}
                  onChange={setPipelineId}
                />
                <div className="grid gap-2">
                  <Label htmlFor="kb-parser-config">Parser config (JSON)</Label>
                  <Textarea
                    id="kb-parser-config"
                    className="font-mono text-xs"
                    value={parserConfig}
                    disabled={saving}
                    aria-invalid={Boolean(fieldErrors.parserConfig)}
                    onChange={(event) => setParserConfig(event.target.value)}
                    placeholder={'{"chunk_token_num": 512}'}
                    rows={6}
                  />
                  {fieldErrors.parserConfig && (
                    <p className="text-xs text-destructive">
                      {fieldErrors.parserConfig}
                    </p>
                  )}
                </div>
              </div>
            </details>

            {formError && (
              <div
                role="alert"
                className="rounded-md border border-destructive/40 bg-destructive/5 px-3 py-2 text-sm text-destructive"
              >
                {formError}
              </div>
            )}
            <div className="flex justify-end gap-2 pt-2">
              <Button variant="ghost" onClick={backToList} disabled={saving}>
                İptal
              </Button>
              <Button
                onClick={() => void submitForm()}
                disabled={saving || formLoading || embeddingModels.length === 0}
              >
                {saving && <Spinner />}
                {view.kind === "edit" ? "Değişiklikleri kaydet" : "Oluştur"}
              </Button>
            </div>
          </div>
        ) : (
          <div className="flex min-w-0 flex-col gap-3">
            <div className="flex flex-col gap-2 sm:flex-row sm:items-center">
              <div className="relative min-w-0 flex-1">
                <SearchIcon className="absolute left-3 top-1/2 size-4 -translate-y-1/2 text-muted-foreground" />
                <Input
                  value={search}
                  className="pl-9"
                  placeholder="Knowledge base ara"
                  aria-label="Knowledge base ara"
                  onChange={(event) => {
                    setSearch(event.target.value);
                    setPage(1);
                  }}
                />
              </div>
              <select
                className="h-9 rounded-full border bg-background px-3 text-sm"
                value={sort}
                aria-label="Knowledge base sıralaması"
                onChange={(event) => {
                  setSort(event.target.value as SortValue);
                  setPage(1);
                }}
              >
                <option value="update-desc">Son güncellenen</option>
                <option value="update-asc">İlk güncellenen</option>
                <option value="create-desc">En yeni</option>
                <option value="create-asc">En eski</option>
              </select>
              <Button size="sm" onClick={startCreate}>
                <HugeiconsIcon icon={PlusSignIcon} size={14} />
                Yeni
              </Button>
            </div>

            {loading ? (
              <div className="flex justify-center py-8" aria-label="Yükleniyor">
                <Spinner />
              </div>
            ) : listError ? (
              <div
                role="alert"
                className="flex flex-col items-center gap-3 rounded-md border border-destructive/40 px-4 py-6 text-center text-sm text-destructive"
              >
                <p>{listError}</p>
                <Button
                  size="sm"
                  variant="outline"
                  onClick={() => setRefreshKey((value) => value + 1)}
                >
                  Yeniden dene
                </Button>
              </div>
            ) : kbs.length === 0 ? (
              <div className="rounded-md border border-dashed py-8 text-center text-sm text-muted-foreground">
                {search
                  ? "Aramayla eşleşen knowledge base bulunamadı."
                  : "Henüz knowledge base yok."}
              </div>
            ) : (
              <ul className="flex max-h-[52dvh] flex-col divide-y overflow-y-auto rounded-md border">
                {kbs.map((kb) => (
                  <li
                    key={kb.id}
                    className="flex items-center justify-between gap-3 px-3 py-2"
                  >
                    <button
                      type="button"
                      onClick={() => setView({ kind: "documents", kb })}
                      className="min-w-0 flex-1 text-left"
                    >
                      <div className="truncate font-medium">{kb.name}</div>
                      <div className="truncate text-xs text-muted-foreground">
                        {kb.documentCount ?? 0} belge
                        {kb.description ? " · " + kb.description : ""}
                      </div>
                    </button>
                    <div className="flex items-center gap-1">
                      <Button
                        type="button"
                        variant="ghost"
                        size="icon"
                        disabled={editingId === kb.id}
                        onClick={() => void startEdit(kb)}
                        aria-label={kb.name + " ayarlarını düzenle"}
                      >
                        {editingId === kb.id ? (
                          <Spinner />
                        ) : (
                          <HugeiconsIcon icon={Edit03Icon} size={14} />
                        )}
                      </Button>
                      <Button
                        type="button"
                        variant="ghost"
                        size="icon"
                        onClick={() => setConfirmingDelete(kb)}
                        aria-label={kb.name + " knowledge base’ini sil"}
                      >
                        <HugeiconsIcon icon={Delete02Icon} size={14} />
                      </Button>
                    </div>
                  </li>
                ))}
              </ul>
            )}

            {!loading && !listError && total > 0 && (
              <div className="flex items-center justify-between text-xs text-muted-foreground">
                <span>{total} knowledge base</span>
                <div className="flex items-center gap-2">
                  <Button
                    size="icon"
                    variant="ghost"
                    disabled={page <= 1}
                    aria-label="Önceki sayfa"
                    onClick={() => setPage((value) => Math.max(1, value - 1))}
                  >
                    <ChevronLeftIcon className="size-4" />
                  </Button>
                  <span>
                    {page} / {pageCount}
                  </span>
                  <Button
                    size="icon"
                    variant="ghost"
                    disabled={page >= pageCount}
                    aria-label="Sonraki sayfa"
                    onClick={() =>
                      setPage((value) => Math.min(pageCount, value + 1))
                    }
                  >
                    <ChevronRightIcon className="size-4" />
                  </Button>
                </div>
              </div>
            )}
          </div>
        )}
      </DialogContent>

      <AlertDialog
        open={confirmingDelete !== null}
        onOpenChange={(next) => {
          if (!next && !deleting) setConfirmingDelete(null);
        }}
      >
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Knowledge base silinsin mi?</AlertDialogTitle>
            <AlertDialogDescription>
              &quot;{confirmingDelete?.name}&quot; ve içindeki{" "}
              <span className="font-medium text-foreground">
                {confirmingDelete?.documentCount ?? 0} belge
              </span>{" "}
              kalıcı olarak silinecek. Bu işlem geri alınamaz.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel disabled={deleting}>İptal</AlertDialogCancel>
            <AlertDialogAction
              variant="destructive"
              disabled={deleting}
              onClick={(event) => {
                event.preventDefault();
                const kb = confirmingDelete;
                if (kb) void removeKb(kb);
              }}
            >
              {deleting && <Spinner />}
              Sil
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </Dialog>
  );
}

function KnowledgeBaseDocuments({
  kb,
  onBack,
}: {
  kb: KnowledgeBase;
  onBack: () => void;
}) {
  const lister = useCallback(() => listKnowledgeBaseDocuments(kb.id), [kb.id]);
  const { documents, loading, uploading, refresh, upload, remove } =
    useRagDocuments({ type: "kb", kbId: kb.id }, lister);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const handleLinkedSourcesChanged = useCallback(() => {
    void refresh({ quiet: true });
  }, [refresh]);

  return (
    <div className="flex min-w-0 flex-col gap-3">
      <div className="flex items-center justify-between">
        <Button variant="ghost" size="sm" onClick={onBack}>
          <ChevronLeftIcon className="size-4" />
          All knowledge bases
        </Button>
        <Button
          size="sm"
          onClick={() => fileInputRef.current?.click()}
          disabled={uploading}
        >
          {uploading ? <Spinner /> : <UploadIcon className="size-3.5" />}
          Upload
        </Button>
        <input
          ref={fileInputRef}
          type="file"
          multiple={true}
          accept={RAG_UPLOAD_ACCEPT}
          className="hidden"
          onChange={(event) => {
            if (event.target.files?.length) void upload(event.target.files);
            event.target.value = "";
          }}
        />
      </div>
      {loading && documents.length === 0 ? (
        <div className="flex justify-center py-6">
          <Spinner />
        </div>
      ) : documents.length === 0 ? (
        <div className="rounded-md border border-dashed py-6 text-center text-sm text-muted-foreground">
          No documents yet. Upload a PDF, Markdown, DOCX, HTML, or text file.
        </div>
      ) : (
        <div className="flex max-h-[55dvh] flex-wrap gap-1.5 overflow-y-auto pr-0.5">
          {documents.map((document) => (
            <DocumentStatusChip
              key={document.id}
              filename={document.filename}
              status={document.status}
              progress={document.progress}
              error={document.error}
              onRemove={
                document.id.startsWith("pending_") ||
                isLinkedFolderManaged(document)
                  ? undefined
                  : () => void remove(document.id)
              }
            />
          ))}
        </div>
      )}
      <div className="border-t pt-3">
        <LinkedFoldersManager
          scope={{ type: "knowledge_base", id: kb.id }}
          compact={true}
          onSourcesChanged={handleLinkedSourcesChanged}
        />
      </div>
    </div>
  );
}
