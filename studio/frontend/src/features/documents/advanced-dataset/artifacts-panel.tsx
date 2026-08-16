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
  type PlatformArtifactPage,
  clearDatasetArtifacts,
  getDatasetArtifactPage,
  hasDatasetArtifacts,
  listDatasetArtifacts,
  updateDatasetArtifactPage,
} from "@/integrations/platform-backend";
import { toast } from "@/lib/toast";
import { useCallback, useState } from "react";
import {
  Field,
  PanelState,
  SectionCard,
  inputClass,
  textareaClass,
  useAbortableLoad,
} from "./shared";

export default function ArtifactsPanel({
  datasetId,
}: { datasetId: string; datasetName: string }) {
  const loader = useCallback(
    async (signal: AbortSignal) => {
      const [probe, pages] = await Promise.all([
        hasDatasetArtifacts(datasetId, signal),
        listDatasetArtifacts(datasetId, { page: 1, pageSize: 200 }, signal),
      ]);
      return { probe, pages };
    },
    [datasetId],
  );
  const loaded = useAbortableLoad(
    loader,
    (value) => !value.probe.has && value.pages.items.length === 0,
  );
  const [selected, setSelected] = useState<{
    pageType: string;
    slug: string;
  } | null>(null);
  const [page, setPage] = useState<PlatformArtifactPage | null>(null);
  const [content, setContent] = useState("");
  const [title, setTitle] = useState("");
  const [comments, setComments] = useState("");
  const [busy, setBusy] = useState(false);
  const [confirmClear, setConfirmClear] = useState(false);
  const open = async (pageType: string, slug: string) => {
    setBusy(true);
    try {
      const result = await getDatasetArtifactPage(datasetId, pageType, slug);
      setSelected({ pageType, slug });
      setPage(result);
      setContent(
        typeof result?.content_md === "string" ? result.content_md : "",
      );
      setTitle(typeof result?.title === "string" ? result.title : "");
    } catch (error) {
      toast.error("Artifact sayfası açılamadı", {
        description: error instanceof Error ? error.message : String(error),
      });
    } finally {
      setBusy(false);
    }
  };
  if (loaded.state !== "ready" && loaded.state !== "empty")
    return (
      <PanelState
        state={loaded.state}
        error={loaded.error}
        onRetry={loaded.load}
      />
    );
  return (
    <div className="grid min-w-0 gap-4 pb-5 lg:grid-cols-[minmax(16rem,0.75fr)_minmax(0,1.5fr)]">
      <SectionCard
        title="Artifact sayfaları"
        description="Derlenmiş entity/topic sayfaları. Bu runtime yalnızca sayfa CRUD ve graph yüzeyini sunuyor."
        actions={
          <Button
            size="sm"
            variant="destructive"
            disabled={!loaded.data?.probe.has}
            onClick={() => setConfirmClear(true)}
          >
            Tümünü temizle
          </Button>
        }
      >
        {(loaded.data?.pages.items.length ?? 0) === 0 ? (
          <PanelState state="empty" empty="Derlenmiş artifact yok." />
        ) : (
          <div className="grid gap-1">
            {loaded.data?.pages.items.map((item) => (
              <button
                key={`${item.page_type}:${item.slug}`}
                className="min-w-0 rounded-xl border px-3 py-2 text-left hover:bg-muted/40"
                onClick={() => void open(item.page_type, item.slug)}
              >
                <span className="block break-words text-sm font-medium">
                  {item.title || item.slug}
                </span>
                <span className="break-all text-[11px] text-muted-foreground">
                  {item.page_type} · {item.slug}
                </span>
              </button>
            ))}
          </div>
        )}
      </SectionCard>
      <SectionCard
        title={selected ? `Düzenle: ${selected.slug}` : "Artifact editörü"}
        description="Kaydetme yalnızca markdown sayfasını günceller; graph verisi bir sonraki compile işlemine kadar stale kalabilir."
        actions={
          selected ? (
            <Button
              size="sm"
              disabled={busy}
              onClick={() =>
                void (async () => {
                  setBusy(true);
                  try {
                    const result = await updateDatasetArtifactPage(
                      datasetId,
                      selected.pageType,
                      selected.slug,
                      {
                        content_md: content,
                        title: title.trim() || undefined,
                        comments: comments.trim() || undefined,
                      },
                    );
                    setPage(result);
                    toast.success("Artifact sayfası kaydedildi.");
                  } catch (error) {
                    toast.error("Artifact kaydedilemedi", {
                      description:
                        error instanceof Error ? error.message : String(error),
                    });
                  } finally {
                    setBusy(false);
                  }
                })()
              }
            >
              Kaydet
            </Button>
          ) : undefined
        }
      >
        {selected ? (
          <div className="grid gap-3">
            <Field label="Başlık">
              <input
                className={inputClass}
                value={title}
                onChange={(event) => setTitle(event.target.value)}
              />
            </Field>
            <Field label="Değişiklik notu">
              <input
                className={inputClass}
                value={comments}
                onChange={(event) => setComments(event.target.value)}
              />
            </Field>
            <Field label="Markdown">
              <textarea
                className={`${textareaClass} min-h-80`}
                value={content}
                onChange={(event) => setContent(event.target.value)}
              />
            </Field>
            {page?.content_md_rendered ? (
              <details>
                <summary className="cursor-pointer text-xs font-medium">
                  Backend rendered içerik
                </summary>
                <pre className="mt-2 max-h-64 overflow-auto rounded-xl bg-muted/50 p-3 text-[11px]">
                  {page.content_md_rendered}
                </pre>
              </details>
            ) : null}
          </div>
        ) : (
          <PanelState
            state="empty"
            empty="Düzenlemek için bir artifact sayfası seçin."
          />
        )}
      </SectionCard>
      <AlertDialog open={confirmClear} onOpenChange={setConfirmClear}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>
              Tüm artifact verisi temizlensin mi?
            </AlertDialogTitle>
            <AlertDialogDescription>
              Derlenmiş sayfalar, entity ve relation kayıtları kalıcı olarak
              silinir. Bu işlem geri alınamaz.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Vazgeç</AlertDialogCancel>
            <AlertDialogAction
              variant="destructive"
              onClick={() =>
                void (async () => {
                  try {
                    await clearDatasetArtifacts(datasetId);
                    setSelected(null);
                    setPage(null);
                    loaded.load();
                    toast.success("Artifact verisi temizlendi.");
                  } catch (error) {
                    toast.error("Artifact verisi temizlenemedi", {
                      description:
                        error instanceof Error ? error.message : String(error),
                    });
                  } finally {
                    setConfirmClear(false);
                  }
                })()
              }
            >
              Kalıcı olarak temizle
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </div>
  );
}
