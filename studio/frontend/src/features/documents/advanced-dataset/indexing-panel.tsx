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
import { Progress } from "@/components/ui/progress";
import { Spinner } from "@/components/ui/spinner";
import {
  type PlatformEmbeddingCheckResponse,
  type PlatformIndexTask,
  type PlatformIndexType,
  type PlatformIngestionLog,
  checkDatasetEmbedding,
  deleteDatasetIndex,
  deleteDatasetIndexByQuery,
  getDatasetIndexStatus,
  getDatasetIngestionLog,
  getDatasetIngestionSummary,
  listDatasetIngestionLogs,
  runDatasetEmbedding,
  startDatasetIndex,
} from "@/integrations/platform-backend";
import { toast } from "@/lib/toast";
import { useCallback, useEffect, useRef, useState } from "react";
import {
  Field,
  PanelState,
  SectionCard,
  inputClass,
  useAbortableLoad,
} from "./shared";

const indexTypes: PlatformIndexType[] = ["graph", "raptor", "mindmap"];
interface Bundle {
  summary: Awaited<ReturnType<typeof getDatasetIngestionSummary>>;
  logs: Awaited<ReturnType<typeof listDatasetIngestionLogs>>;
  tasks: Record<PlatformIndexType, PlatformIndexTask | null>;
}

export default function IndexingPanel({
  datasetId,
}: { datasetId: string; datasetName: string }) {
  const loader = useCallback(
    async (signal: AbortSignal): Promise<Bundle> => {
      const [summary, logs, ...statuses] = await Promise.all([
        getDatasetIngestionSummary(datasetId, signal),
        listDatasetIngestionLogs(
          datasetId,
          { page: 1, pageSize: 30, logType: "dataset" },
          signal,
        ),
        ...indexTypes.map((type) =>
          getDatasetIndexStatus(datasetId, type, signal),
        ),
      ]);
      return {
        summary,
        logs,
        tasks: {
          graph: statuses[0],
          raptor: statuses[1],
          mindmap: statuses[2],
        },
      };
    },
    [datasetId],
  );
  const loaded = useAbortableLoad(loader);
  const generation = useRef(0);
  const [busy, setBusy] = useState<string | null>(null);
  const [wipe, setWipe] = useState<PlatformIndexType | null>(null);
  const [embeddingId, setEmbeddingId] = useState("");
  const [embeddingResult, setEmbeddingResult] =
    useState<PlatformEmbeddingCheckResponse | null>(null);
  const [selectedLog, setSelectedLog] = useState<PlatformIngestionLog | null>(
    null,
  );
  const active = indexTypes.some((type) => {
    const task = loaded.data?.tasks[type];
    return task && task.progress < 1;
  });
  const reload = loaded.load;

  useEffect(() => {
    if (!active) return;
    const local = ++generation.current;
    const timer = window.setInterval(() => {
      if (
        generation.current === local &&
        document.visibilityState === "visible"
      )
        reload();
    }, 3000);
    return () => {
      generation.current += 1;
      window.clearInterval(timer);
    };
  }, [active, reload]);
  const run = async (
    name: string,
    action: () => Promise<unknown>,
    success: string,
  ) => {
    setBusy(name);
    try {
      await action();
      toast.success(success);
      loaded.load();
    } catch (error) {
      toast.error("İşlem tamamlanamadı", {
        description: error instanceof Error ? error.message : String(error),
      });
    } finally {
      setBusy(null);
    }
  };
  if (loaded.state !== "ready" || !loaded.data)
    return (
      <PanelState
        state={loaded.state}
        error={loaded.error}
        onRetry={loaded.load}
      />
    );
  const data = loaded.data;
  return (
    <div className="grid min-w-0 gap-4 pb-5">
      <div className="grid min-w-0 gap-3 sm:grid-cols-3">
        {[
          ["Belge", loaded.data.summary.doc_num],
          ["Chunk", loaded.data.summary.chunk_num],
          ["Token", loaded.data.summary.token_num],
        ].map(([label, value]) => (
          <div key={label} className="rounded-2xl border bg-card p-4">
            <p className="text-xs text-muted-foreground">{label}</p>
            <p className="mt-1 text-2xl font-semibold tabular-nums">
              {String(value)}
            </p>
          </div>
        ))}
      </div>
      <SectionCard
        title="Dataset indeksleri"
        description="Graph, RAPTOR ve mindmap işleri ayrı izlenir. İptal mevcut çıktıyı korur; temizle kalıcı veriyi siler."
        actions={
          active ? (
            <span className="flex items-center gap-2 text-xs text-muted-foreground">
              <Spinner /> 3 sn polling
            </span>
          ) : undefined
        }
      >
        <div className="grid min-w-0 gap-3 lg:grid-cols-3">
          {indexTypes.map((type) => {
            const task = data.tasks[type];
            const progress = Math.max(
              0,
              Math.min(100, (task?.progress ?? 0) * 100),
            );
            return (
              <article key={type} className="rounded-xl border p-3">
                <div className="flex items-center justify-between">
                  <h4 className="text-sm font-semibold uppercase">{type}</h4>
                  <span className="text-xs tabular-nums text-muted-foreground">
                    {task ? `${progress.toFixed(0)}%` : "Başlatılmadı"}
                  </span>
                </div>
                <Progress className="my-3" value={progress} />
                <p className="min-h-8 text-xs text-muted-foreground">
                  {task?.progress_msg || "Aktif görev yok."}
                </p>
                <div className="mt-3 flex flex-wrap gap-2">
                  <Button
                    size="sm"
                    disabled={
                      busy !== null || Boolean(task && task.progress < 1)
                    }
                    onClick={() =>
                      void run(
                        `start-${type}`,
                        () => startDatasetIndex(datasetId, type),
                        `${type} indeksleme başlatıldı.`,
                      )
                    }
                  >
                    Başlat
                  </Button>
                  <Button
                    size="sm"
                    variant="outline"
                    disabled={busy !== null || !task || task.progress >= 1}
                    onClick={() =>
                      void run(
                        `cancel-${type}`,
                        () => deleteDatasetIndexByQuery(datasetId, type, false),
                        `${type} görevi iptal edildi; mevcut çıktı korundu.`,
                      )
                    }
                  >
                    İptal et
                  </Button>
                  <Button
                    size="sm"
                    variant="destructive"
                    disabled={busy !== null || !task}
                    onClick={() => setWipe(type)}
                  >
                    Temizle
                  </Button>
                </div>
              </article>
            );
          })}
        </div>
      </SectionCard>
      <SectionCard
        title="Embedding uyumluluğu"
        description="Yeni embedding modeli örnek chunk vektörleriyle doğrulanabilir; toplu embedding ayrı ve açık bir aksiyondur."
      >
        <div className="flex max-w-2xl flex-wrap items-end gap-2">
          <Field label="Embedding model kimliği">
            <input
              className={inputClass}
              value={embeddingId}
              onChange={(event) => setEmbeddingId(event.target.value)}
              placeholder="provider/model"
            />
          </Field>
          <Button
            size="sm"
            variant="outline"
            disabled={busy !== null || !embeddingId.trim()}
            onClick={() =>
              void (async () => {
                setBusy("check-embedding");
                try {
                  const result = await checkDatasetEmbedding(
                    datasetId,
                    embeddingId.trim(),
                    5,
                  );
                  setEmbeddingResult(result);
                  toast.success("Embedding uyumluluk kontrolü tamamlandı.");
                } catch (error) {
                  toast.error("Embedding kontrolü başarısız", {
                    description:
                      error instanceof Error ? error.message : String(error),
                  });
                } finally {
                  setBusy(null);
                }
              })()
            }
          >
            Uyumluluğu kontrol et
          </Button>
          <Button
            size="sm"
            disabled={busy !== null}
            onClick={() =>
              void run(
                "embedding",
                () => runDatasetEmbedding(datasetId),
                "Dataset embedding işlemi kuyruğa alındı.",
              )
            }
          >
            Tüm belgeleri embed et
          </Button>
        </div>
        {embeddingResult ? (
          <pre className="mt-3 max-h-72 overflow-auto rounded-xl bg-muted/50 p-3 text-[11px]">
            {JSON.stringify(embeddingResult, null, 2)}
          </pre>
        ) : null}
      </SectionCard>
      <SectionCard
        title="Ingestion günlükleri"
        description={`${loaded.data.logs.total} kayıt. Ayrıntı seçildiğinde DSL ve parser alanları backend'den ayrıca okunur.`}
        actions={
          <Button size="sm" variant="outline" onClick={loaded.load}>
            Yenile
          </Button>
        }
      >
        {loaded.data.logs.logs.length === 0 ? (
          <PanelState state="empty" empty="Ingestion günlüğü yok." />
        ) : (
          <div className="grid gap-2">
            {loaded.data.logs.logs.map((log) => (
              <button
                key={log.id}
                className="grid min-w-0 grid-cols-[minmax(0,1fr)_auto] gap-3 rounded-xl border px-3 py-2 text-left hover:bg-muted/40"
                onClick={() =>
                  void (async () => {
                    try {
                      setSelectedLog(
                        await getDatasetIngestionLog(datasetId, log.id),
                      );
                    } catch (error) {
                      toast.error("Günlük ayrıntısı alınamadı", {
                        description:
                          error instanceof Error
                            ? error.message
                            : String(error),
                      });
                    }
                  })()
                }
              >
                <span>
                  <span className="block text-sm font-medium">
                    {log.task_type || log.id}
                  </span>
                  <span className="text-xs text-muted-foreground">
                    {log.progress_msg || log.operation_status || "—"}
                  </span>
                </span>
                <span className="text-xs tabular-nums text-muted-foreground">
                  {typeof log.progress === "number"
                    ? `${Math.round(log.progress * 100)}%`
                    : ""}
                </span>
              </button>
            ))}
          </div>
        )}
        {selectedLog ? (
          <details className="mt-3" open={true}>
            <summary className="cursor-pointer text-xs font-semibold">
              Seçili günlük ayrıntısı
            </summary>
            <pre className="mt-2 max-h-96 overflow-auto rounded-xl bg-muted/50 p-3 text-[11px]">
              {JSON.stringify(selectedLog, null, 2)}
            </pre>
          </details>
        ) : null}
      </SectionCard>
      <AlertDialog
        open={wipe !== null}
        onOpenChange={(open) => {
          if (!open) setWipe(null);
        }}
      >
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>
              {wipe?.toUpperCase()} indeksi kalıcı olarak temizlensin mi?
            </AlertDialogTitle>
            <AlertDialogDescription>
              Görev iptal edilir ve persisted indeks çıktıları silinir. Bu işlem
              geri alınamaz.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Vazgeç</AlertDialogCancel>
            <AlertDialogAction
              variant="destructive"
              onClick={() => {
                const type = wipe;
                setWipe(null);
                if (type)
                  void run(
                    `wipe-${type}`,
                    () => deleteDatasetIndex(datasetId, type, true),
                    `${type} indeksi temizlendi.`,
                  );
              }}
            >
              Kalıcı olarak temizle
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </div>
  );
}
