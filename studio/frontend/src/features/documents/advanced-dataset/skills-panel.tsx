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
  type PlatformDatasetSkillPage,
  type PlatformSkillFieldConfig,
  type PlatformSkillSearchConfigRequest,
  type PlatformSkillSpace,
  type PlatformSkillTreeNode,
  createGlobalSkillSpace,
  deleteGlobalSkillIndex,
  deleteGlobalSkillSpace,
  getDatasetSkillPage,
  getDatasetSkillTree,
  getGlobalSkillSearchConfig,
  getGlobalSkillSpace,
  getGlobalSkillSpaceByFolder,
  hasDatasetSkills,
  indexGlobalSkills,
  listGlobalSkillSpaces,
  reindexGlobalSkills,
  searchGlobalSkills,
  updateGlobalSkillSearchConfig,
  updateGlobalSkillSpace,
} from "@/integrations/platform-backend";
import { toast } from "@/lib/toast";
import { useCallback, useMemo, useState } from "react";
import {
  Field,
  PanelState,
  SectionCard,
  inputClass,
  textareaClass,
  useAbortableLoad,
} from "./shared";

const defaultFields: PlatformSkillFieldConfig = {
  name: { enabled: true, weight: 3 },
  tags: { enabled: true, weight: 2 },
  description: { enabled: true, weight: 1 },
  content: { enabled: false, weight: 0.5 },
};
function nodes(value: PlatformSkillTreeNode | null): PlatformSkillTreeNode[] {
  if (!value) return [];
  const result: PlatformSkillTreeNode[] = [];
  const visit = (item: PlatformSkillTreeNode) => {
    result.push(item);
    if (Array.isArray(item.children)) item.children.forEach(visit);
  };
  visit(value);
  return result;
}

export default function SkillsPanel({
  datasetId,
}: { datasetId: string; datasetName: string }) {
  const loader = useCallback(
    async (signal: AbortSignal) => {
      const [probe, tree] = await Promise.all([
        hasDatasetSkills(datasetId, signal),
        getDatasetSkillTree(datasetId, signal),
      ]);
      return { probe, tree };
    },
    [datasetId],
  );
  const loaded = useAbortableLoad(loader);
  const treeNodes = useMemo(
    () => nodes(loaded.data?.tree ?? null),
    [loaded.data?.tree],
  );
  const [scope, setScope] = useState<"dataset" | "global">("dataset");
  const [spaces, setSpaces] = useState<Awaited<
    ReturnType<typeof listGlobalSkillSpaces>
  > | null>(null);
  const [globalState, setGlobalState] = useState<
    "idle" | "loading" | "ready" | "runtime-disabled" | "error"
  >("idle");
  const [globalError, setGlobalError] = useState<string | null>(null);
  const [page, setPage] = useState<PlatformDatasetSkillPage | null>(null);
  const [selectedSpace, setSelectedSpace] = useState<PlatformSkillSpace | null>(
    null,
  );
  const [spaceName, setSpaceName] = useState("");
  const [description, setDescription] = useState("");
  const [embeddingId, setEmbeddingId] = useState("");
  const [rerankId, setRerankId] = useState("");
  const [topK, setTopK] = useState(10);
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<Awaited<
    ReturnType<typeof searchGlobalSkills>
  > | null>(null);
  const [skillId, setSkillId] = useState("");
  const [skillName, setSkillName] = useState("");
  const [skillContent, setSkillContent] = useState("");
  const [busy, setBusy] = useState<string | null>(null);
  const [confirmDelete, setConfirmDelete] = useState(false);
  const probeGlobal = useCallback(async () => {
    setGlobalState("loading");
    setGlobalError(null);
    try {
      const [spaceResult] = await Promise.all([
        listGlobalSkillSpaces(),
        searchGlobalSkills({
          space_id: "default",
          query: "",
          page: 1,
          page_size: 1,
          sort_by: "update_time",
          sort_order: "desc",
        }),
      ]);
      setSpaces(spaceResult);
      setGlobalState("ready");
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      setGlobalError(message);
      setGlobalState(
        message.includes("skill_spaces") ||
          message.includes("connect: connection refused")
          ? "runtime-disabled"
          : "error",
      );
    }
  }, []);
  const run = async (
    name: string,
    action: () => Promise<unknown>,
    success: string,
    refresh = true,
  ) => {
    setBusy(name);
    try {
      await action();
      toast.success(success);
      if (refresh) await probeGlobal();
    } catch (error) {
      toast.error("Beceri işlemi tamamlanamadı", {
        description: error instanceof Error ? error.message : String(error),
      });
    } finally {
      setBusy(null);
    }
  };
  const selectSpace = async (space: PlatformSkillSpace) => {
    setBusy("space");
    try {
      const full = await getGlobalSkillSpace(space.id);
      setSelectedSpace(full);
      setSpaceName(full.name);
      setDescription(full.description ?? "");
      setEmbeddingId(full.embd_id ?? "");
      setRerankId(full.rerank_id ?? "");
      setTopK(full.top_k ?? 10);
    } catch (error) {
      toast.error("Beceri alanı okunamadı", {
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
  return (
    <div className="grid min-w-0 gap-4 pb-5">
      <div
        role="radiogroup"
        aria-label="Beceri sahiplik kapsamı"
        className="flex w-fit rounded-xl border bg-muted/35 p-1"
      >
        <button
          role="radio"
          aria-checked={scope === "dataset"}
          className={`rounded-lg px-3 py-1.5 text-xs font-medium ${scope === "dataset" ? "bg-background shadow-sm" : "text-muted-foreground"}`}
          onClick={() => setScope("dataset")}
        >
          Dataset-owned
        </button>
        <button
          role="radio"
          aria-checked={scope === "global"}
          className={`rounded-lg px-3 py-1.5 text-xs font-medium ${scope === "global" ? "bg-background shadow-sm" : "text-muted-foreground"}`}
          onClick={() => {
            setScope("global");
            if (globalState === "idle") void probeGlobal();
          }}
        >
          Global skill space
        </button>
      </div>
      {scope === "dataset" ? (
        <div className="grid min-w-0 gap-4 lg:grid-cols-[minmax(16rem,0.7fr)_minmax(0,1.5fr)]">
          <SectionCard
            title="Dataset beceri ağacı"
            description="Corpus2Skill tarafından bu datasete ait derlenen salt-okunur beceriler."
          >
            {!loaded.data.probe.has || treeNodes.length === 0 ? (
              <PanelState
                state="empty"
                empty="Bu dataset için derlenmiş beceri yok."
              />
            ) : (
              <div className="grid gap-1">
                {treeNodes.map((node, index) => {
                  const keyword =
                    typeof node.skill_kwd === "string"
                      ? node.skill_kwd
                      : typeof node.name === "string"
                        ? node.name
                        : "";
                  return (
                    <button
                      key={`${keyword}:${index}`}
                      disabled={!keyword}
                      className="rounded-xl border px-3 py-2 text-left text-sm hover:bg-muted/40 disabled:opacity-50"
                      onClick={() =>
                        void (async () => {
                          try {
                            setPage(
                              await getDatasetSkillPage(datasetId, keyword),
                            );
                          } catch (error) {
                            toast.error("Beceri sayfası açılamadı", {
                              description:
                                error instanceof Error
                                  ? error.message
                                  : String(error),
                            });
                          }
                        })()
                      }
                    >
                      {node.title ||
                        node.name ||
                        node.skill_kwd ||
                        `Node ${index + 1}`}
                    </button>
                  );
                })}
              </div>
            )}
          </SectionCard>
          <SectionCard
            title={page?.title || page?.name || "Beceri ayrıntısı"}
            description="Dataset-owned beceriler global skill indexinden ayrı tutulur."
          >
            {page ? (
              <pre className="max-h-[42rem] overflow-auto whitespace-pre-wrap rounded-xl bg-muted/50 p-4 text-xs">
                {page.content_md || JSON.stringify(page, null, 2)}
              </pre>
            ) : (
              <PanelState state="empty" empty="Bir beceri düğümü seçin." />
            )}
          </SectionCard>
        </div>
      ) : globalState === "ready" && spaces ? (
        <>
          <div className="grid min-w-0 gap-4 lg:grid-cols-[minmax(16rem,0.7fr)_minmax(0,1.5fr)]">
            <SectionCard
              title="Global skill space'ler"
              description={`${spaces.total} tenant-owned alan.`}
              actions={
                <Button
                  size="sm"
                  variant="outline"
                  onClick={() => {
                    setSelectedSpace(null);
                    setSpaceName("");
                    setDescription("");
                    setEmbeddingId("");
                    setRerankId("");
                    setTopK(10);
                  }}
                >
                  Yeni
                </Button>
              }
            >
              <div className="grid gap-1">
                {spaces.spaces.map((space) => (
                  <button
                    key={space.id}
                    className="rounded-xl border px-3 py-2 text-left hover:bg-muted/40"
                    onClick={() => void selectSpace(space)}
                  >
                    <span className="block text-sm font-medium">
                      {space.name}
                    </span>
                    <span className="text-[11px] text-muted-foreground">
                      {space.status} · {space.folder_id}
                    </span>
                  </button>
                ))}
              </div>
            </SectionCard>
            <SectionCard
              title={
                selectedSpace ? "Skill space düzenle" : "Skill space oluştur"
              }
              description="Global alanların tenant sahipliği dataset becerilerinden bağımsızdır."
              actions={
                <div className="flex gap-2">
                  {selectedSpace ? (
                    <Button
                      size="sm"
                      variant="destructive"
                      onClick={() => setConfirmDelete(true)}
                    >
                      Sil
                    </Button>
                  ) : null}
                  <Button
                    size="sm"
                    disabled={busy !== null || !spaceName.trim()}
                    onClick={() =>
                      void run(
                        "save-space",
                        async () => {
                          if (selectedSpace)
                            await updateGlobalSkillSpace(selectedSpace.id, {
                              name: spaceName.trim(),
                              description,
                              embd_id: embeddingId,
                              rerank_id: rerankId,
                              top_k: topK,
                            });
                          else {
                            const created = await createGlobalSkillSpace({
                              name: spaceName.trim(),
                              description,
                              embd_id: embeddingId,
                              rerank_id: rerankId,
                            });
                            setSelectedSpace(created);
                          }
                        },
                        selectedSpace
                          ? "Skill space güncellendi."
                          : "Skill space oluşturuldu.",
                      )
                    }
                  >
                    Kaydet
                  </Button>
                </div>
              }
            >
              <div className="grid gap-3 sm:grid-cols-2">
                <Field label="Ad">
                  <input
                    className={inputClass}
                    value={spaceName}
                    onChange={(event) => setSpaceName(event.target.value)}
                  />
                </Field>
                <Field label="Embedding ID">
                  <input
                    className={inputClass}
                    value={embeddingId}
                    onChange={(event) => setEmbeddingId(event.target.value)}
                  />
                </Field>
                <Field label="Rerank ID">
                  <input
                    className={inputClass}
                    value={rerankId}
                    onChange={(event) => setRerankId(event.target.value)}
                  />
                </Field>
                <Field label="Top K">
                  <input
                    type="number"
                    min={1}
                    className={inputClass}
                    value={topK}
                    onChange={(event) => setTopK(Number(event.target.value))}
                  />
                </Field>
                <Field label="Açıklama">
                  <input
                    className={inputClass}
                    value={description}
                    onChange={(event) => setDescription(event.target.value)}
                  />
                </Field>
              </div>
              {selectedSpace ? (
                <div className="mt-3 flex flex-wrap gap-2">
                  <Button
                    size="sm"
                    variant="outline"
                    onClick={() =>
                      void run(
                        "folder",
                        async () => {
                          const found = await getGlobalSkillSpaceByFolder(
                            selectedSpace.folder_id,
                          );
                          setSelectedSpace(found);
                        },
                        "Folder sahipliği doğrulandı.",
                        false,
                      )
                    }
                  >
                    Folder sahipliğini doğrula
                  </Button>
                  <Button
                    size="sm"
                    variant="outline"
                    disabled={!embeddingId.trim()}
                    onClick={() =>
                      void run(
                        "config",
                        async () => {
                          const current = await getGlobalSkillSearchConfig(
                            selectedSpace.id,
                            embeddingId,
                          );
                          const payload: PlatformSkillSearchConfigRequest = {
                            space_id: selectedSpace.id,
                            embd_id: embeddingId,
                            vector_similarity_weight:
                              current.vector_similarity_weight ?? 0.3,
                            similarity_threshold:
                              current.similarity_threshold ?? 0.2,
                            field_config: current.field_config ?? defaultFields,
                            rerank_id: rerankId,
                            top_k: topK,
                          };
                          await updateGlobalSkillSearchConfig(payload);
                        },
                        "Arama yapılandırması kaydedildi.",
                        false,
                      )
                    }
                  >
                    Arama configini kaydet
                  </Button>
                  <Button
                    size="sm"
                    variant="outline"
                    disabled={!embeddingId.trim()}
                    onClick={() =>
                      void run(
                        "reindex",
                        () =>
                          reindexGlobalSkills(selectedSpace.id, embeddingId),
                        "Global beceriler yeniden indekslendi.",
                        false,
                      )
                    }
                  >
                    Tümünü reindex et
                  </Button>
                </div>
              ) : null}
            </SectionCard>
          </div>
          {selectedSpace ? (
            <div className="grid min-w-0 gap-4 xl:grid-cols-2">
              <SectionCard
                title="Global beceri arama"
                description="Keyword/vector/hybrid sonuçları seçili space ile sınırlandırılır."
              >
                <div className="flex min-w-0 flex-col gap-2 sm:flex-row">
                  <input
                    className={`${inputClass} flex-1`}
                    value={query}
                    onChange={(event) => setQuery(event.target.value)}
                    placeholder="Beceri ara"
                  />
                  <Button
                    size="sm"
                    onClick={() =>
                      void (async () => {
                        setBusy("search");
                        try {
                          setResults(
                            await searchGlobalSkills({
                              space_id: selectedSpace.id,
                              query,
                              page: 1,
                              page_size: 25,
                              sort_by: query ? "relevance" : "update_time",
                              sort_order: "desc",
                            }),
                          );
                        } catch (error) {
                          toast.error("Beceri araması başarısız", {
                            description:
                              error instanceof Error
                                ? error.message
                                : String(error),
                          });
                        } finally {
                          setBusy(null);
                        }
                      })()
                    }
                  >
                    Ara
                  </Button>
                </div>
                {results ? (
                  <div className="mt-3 grid gap-2">
                    {results.skills.map((skill) => (
                      <div
                        key={skill.skill_id}
                        className="rounded-xl border p-3"
                      >
                        <div className="flex justify-between gap-3">
                          <span className="text-sm font-medium">
                            {skill.name}
                          </span>
                          <Button
                            size="xs"
                            variant="destructive"
                            onClick={() =>
                              void run(
                                "delete-index",
                                () =>
                                  deleteGlobalSkillIndex(
                                    selectedSpace.id,
                                    skill.skill_id,
                                  ),
                                "Beceri indeksi silindi.",
                                false,
                              )
                            }
                          >
                            İndeksten sil
                          </Button>
                        </div>
                        <p className="mt-1 text-xs text-muted-foreground">
                          {skill.description}
                        </p>
                      </div>
                    ))}
                  </div>
                ) : null}
              </SectionCard>
              <SectionCard
                title="Tek beceriyi indeksle"
                description="Beceri içeriği sadece istekte taşınır; token veya provider secret saklanmaz."
              >
                <div className="grid gap-3">
                  <Field label="Beceri ID">
                    <input
                      className={inputClass}
                      value={skillId}
                      onChange={(event) => setSkillId(event.target.value)}
                    />
                  </Field>
                  <Field label="Ad">
                    <input
                      className={inputClass}
                      value={skillName}
                      onChange={(event) => setSkillName(event.target.value)}
                    />
                  </Field>
                  <Field label="İçerik">
                    <textarea
                      className={textareaClass}
                      value={skillContent}
                      onChange={(event) => setSkillContent(event.target.value)}
                    />
                  </Field>
                  <Button
                    size="sm"
                    disabled={
                      !skillId.trim() ||
                      !skillName.trim() ||
                      !embeddingId.trim()
                    }
                    onClick={() =>
                      void run(
                        "index",
                        () =>
                          indexGlobalSkills(
                            selectedSpace.id,
                            [
                              {
                                id: skillId.trim(),
                                folder_id: selectedSpace.folder_id,
                                name: skillName.trim(),
                                description: "",
                                tags: [],
                                content: skillContent,
                                version: "1.0.0",
                              },
                            ],
                            embeddingId,
                          ),
                        "Beceri indekslendi.",
                        false,
                      )
                    }
                  >
                    İndeksle
                  </Button>
                </div>
              </SectionCard>
            </div>
          ) : null}
        </>
      ) : (
        <SectionCard
          title="Global skill runtime durumu"
          description="Dataset-owned beceriler kullanılabilir; global skill space yaşam döngüsü ayrı bir runtime yeteneğidir."
          actions={
            globalState !== "loading" ? (
              <Button
                size="sm"
                variant="outline"
                onClick={() => void probeGlobal()}
              >
                Yeniden dene
              </Button>
            ) : undefined
          }
        >
          {globalState === "loading" || globalState === "idle" ? (
            <PanelState state="loading" />
          ) : globalState === "runtime-disabled" ? (
            <div className="rounded-xl border border-amber-500/30 bg-amber-500/5 p-4">
              <p
                role="status"
                className="text-sm font-semibold text-amber-800 dark:text-amber-200"
              >
                Global skill space bu runtime'da kullanılamıyor.
              </p>
              <p className="mt-2 text-xs text-muted-foreground">
                Aktif v0.26.4 veritabanında <code>skill_spaces</code> tablosu
                yok; skill arama indeksi de Elasticsearch bağlantısı olmadan
                çalışmıyor. Oluşturma, düzenleme, silme, config, search ve index
                aksiyonları güvenli biçimde gizlendi.
              </p>
              {globalError ? (
                <p className="mt-2 text-xs text-muted-foreground">
                  {globalError}
                </p>
              ) : null}
            </div>
          ) : (
            <PanelState
              state="error"
              error={globalError}
              onRetry={probeGlobal}
            />
          )}
        </SectionCard>
      )}
      <AlertDialog open={confirmDelete} onOpenChange={setConfirmDelete}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Global skill space silinsin mi?</AlertDialogTitle>
            <AlertDialogDescription>
              Silme asenkron başlar; space önce deleting durumuna geçer ve
              ilişkili indeks/folder temizlenir.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Vazgeç</AlertDialogCancel>
            <AlertDialogAction
              variant="destructive"
              onClick={() => {
                const id = selectedSpace?.id;
                setConfirmDelete(false);
                if (id)
                  void run(
                    "delete-space",
                    async () => {
                      await deleteGlobalSkillSpace(id);
                      setSelectedSpace(null);
                    },
                    "Skill space silme kuyruğuna alındı.",
                  );
              }}
            >
              Silme işlemini başlat
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </div>
  );
}
