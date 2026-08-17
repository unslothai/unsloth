import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Checkbox } from "@/components/ui/checkbox";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Spinner } from "@/components/ui/spinner";
import { Switch } from "@/components/ui/switch";
import { Textarea } from "@/components/ui/textarea";
import {
  type PlatformChatReference,
  type PlatformDatasetDto,
  type PlatformModel,
  type PlatformSearchApp,
  createPlatformSearch,
  deletePlatformSearch,
  getPlatformSearch,
  getPlatformUiError,
  listPlatformDatasets,
  listPlatformSearches,
  listTenantModels,
  platformModelReference,
  resolvePlatformModelReference,
  streamPlatformSearchCompletion,
  updatePlatformSearch,
} from "@/integrations/platform-backend";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

const PAGE_SIZE = 20;
type SearchHistory = {
  id: string;
  question: string;
  answer: string;
  reference: PlatformChatReference | null;
};
const dtoText = (value: unknown) => (typeof value === "string" ? value : "");

function errorMessage(error: unknown) {
  const ui = getPlatformUiError(error);
  return ui.kind === "aborted" ? null : ui.message;
}

export function SearchPage() {
  const [items, setItems] = useState<PlatformSearchApp[]>([]);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(1);
  const [keywords, setKeywords] = useState("");
  const [selectedId, setSelectedId] = useState("");
  const [selected, setSelected] = useState<PlatformSearchApp | null>(null);
  const [datasets, setDatasets] = useState<PlatformDatasetDto[]>([]);
  const [models, setModels] = useState<PlatformModel[]>([]);
  const [loading, setLoading] = useState(true);
  const [detailLoading, setDetailLoading] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [createOpen, setCreateOpen] = useState(false);
  const [createName, setCreateName] = useState("");
  const [createDescription, setCreateDescription] = useState("");
  const [question, setQuestion] = useState("");
  const [streaming, setStreaming] = useState(false);
  const [answer, setAnswer] = useState("");
  const [reference, setReference] = useState<PlatformChatReference | null>(
    null,
  );
  const [history, setHistory] = useState<SearchHistory[]>([]);
  const streamController = useRef<AbortController | null>(null);

  const chatModels = useMemo(
    () => models.filter((model) => model.capabilities.includes("chat")),
    [models],
  );
  const rerankModels = useMemo(
    () => models.filter((model) => model.capabilities.includes("rerank")),
    [models],
  );
  const selectedChatModelReference = selected
    ? resolvePlatformModelReference(selected.config.chatModelId, chatModels)
    : "";
  const selectedRerankModelReference = selected
    ? resolvePlatformModelReference(selected.config.rerankId, rerankModels)
    : "";

  const loadList = useCallback(
    async (signal?: AbortSignal) => {
      setLoading(true);
      setError(null);
      try {
        const result = await listPlatformSearches(
          { page, pageSize: PAGE_SIZE, keywords },
          signal,
        );
        setItems(result.items);
        setTotal(result.total);
      } catch (reason) {
        setError(errorMessage(reason));
      } finally {
        setLoading(false);
      }
    },
    [keywords, page],
  );

  const loadDetail = useCallback(async (id: string, signal?: AbortSignal) => {
    setDetailLoading(true);
    setError(null);
    try {
      const detail = await getPlatformSearch(id, signal);
      setSelected(detail);
      setItems((current) =>
        current.map((item) => (item.id === detail.id ? detail : item)),
      );
    } catch (reason) {
      setError(errorMessage(reason));
    } finally {
      setDetailLoading(false);
    }
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    void Promise.all([
      loadList(controller.signal),
      listPlatformDatasets({ page: 1, pageSize: 100 }, controller.signal).then(
        (result) => setDatasets(result.items),
      ),
      listTenantModels(controller.signal).then(setModels),
    ]).catch((reason) => {
      const message = errorMessage(reason);
      if (message) setError(message);
    });
    return () => controller.abort();
  }, [loadList]);

  useEffect(() => {
    if (!selectedId) return;
    const controller = new AbortController();
    void loadDetail(selectedId, controller.signal);
    return () => controller.abort();
  }, [loadDetail, selectedId]);

  useEffect(() => () => streamController.current?.abort(), []);

  const mutate = async (action: () => Promise<unknown>, after?: () => void) => {
    setBusy(true);
    setError(null);
    try {
      await action();
      after?.();
    } catch (reason) {
      setError(errorMessage(reason));
    } finally {
      setBusy(false);
    }
  };

  const ask = async () => {
    if (!selected || !question.trim() || streaming) return;
    const asked = question.trim();
    const controller = new AbortController();
    streamController.current = controller;
    setStreaming(true);
    setError(null);
    setAnswer("");
    setReference(null);
    let completedAnswer = "";
    let completedReference: PlatformChatReference | null = null;
    try {
      for await (const event of streamPlatformSearchCompletion(
        selected.id,
        asked,
        selected.config.datasetIds,
        controller.signal,
      )) {
        if (event.type === "answer" && event.answer) {
          completedAnswer += event.answer;
          setAnswer(completedAnswer);
        }
        if (event.type === "reference" && event.reference) {
          completedReference = event.reference;
          setReference(event.reference);
        }
      }
      setHistory((current) => [
        {
          id: crypto.randomUUID(),
          question: asked,
          answer: completedAnswer,
          reference: completedReference,
        },
        ...current,
      ]);
      setQuestion("");
    } catch (reason) {
      setError(errorMessage(reason));
    } finally {
      if (streamController.current === controller)
        streamController.current = null;
      setStreaming(false);
    }
  };

  return (
    <main className="mx-auto flex w-full max-w-7xl flex-1 flex-col gap-6 overflow-auto p-6">
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <h1 className="font-heading font-semibold text-3xl">Arama</h1>
          <p className="mt-1 text-muted-foreground">
            Veri kümesi ve model kapsamı görünür, bağımsız Rag Platform arama
            deneyimi.
          </p>
        </div>
        <Button onClick={() => setCreateOpen((value) => !value)}>
          Yeni arama
        </Button>
      </div>
      {error ? (
        <Alert variant="destructive">
          <AlertTitle>İşlem tamamlanamadı</AlertTitle>
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      ) : null}

      {createOpen ? (
        <Card>
          <CardHeader>
            <CardTitle>Yeni arama</CardTitle>
            <CardDescription>
              Önce kaydı oluşturun, sonra veri kümesi ve provider kapsamını
              yapılandırın.
            </CardDescription>
          </CardHeader>
          <CardContent className="grid gap-4 md:grid-cols-2">
            <div className="space-y-2">
              <Label htmlFor="search-name">Ad</Label>
              <Input
                id="search-name"
                value={createName}
                onChange={(event) => setCreateName(event.target.value)}
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="search-description">Açıklama</Label>
              <Input
                id="search-description"
                value={createDescription}
                onChange={(event) => setCreateDescription(event.target.value)}
              />
            </div>
            <div className="flex gap-2 md:col-span-2">
              <Button
                disabled={busy || !createName.trim()}
                onClick={() =>
                  void mutate(async () => {
                    const id = await createPlatformSearch({
                      name: createName,
                      description: createDescription,
                    });
                    setCreateName("");
                    setCreateDescription("");
                    setCreateOpen(false);
                    await loadList();
                    setSelectedId(id);
                  })
                }
              >
                Oluştur
              </Button>
              <Button variant="outline" onClick={() => setCreateOpen(false)}>
                İptal
              </Button>
            </div>
          </CardContent>
        </Card>
      ) : null}

      <div className="grid min-h-0 gap-6 lg:grid-cols-[320px_minmax(0,1fr)]">
        <Card className="h-fit">
          <CardHeader>
            <CardTitle>Aramalar</CardTitle>
            <CardDescription>{total} kayıt</CardDescription>
          </CardHeader>
          <CardContent className="space-y-3">
            <form
              className="flex gap-2"
              onSubmit={(event) => {
                event.preventDefault();
                setPage(1);
                void loadList();
              }}
            >
              <Input
                aria-label="Arama uygulamalarında ara"
                placeholder="Ada göre ara"
                value={keywords}
                onChange={(event) => setKeywords(event.target.value)}
              />
              <Button type="submit" variant="outline">
                Ara
              </Button>
            </form>
            {loading ? (
              <div className="flex items-center gap-2 text-sm">
                <Spinner /> Yükleniyor
              </div>
            ) : items.length === 0 ? (
              <p className="text-muted-foreground text-sm">
                Henüz arama uygulaması yok.
              </p>
            ) : (
              items.map((app) => (
                <button
                  key={app.id}
                  type="button"
                  className={`w-full rounded-lg border p-3 text-left ${selectedId === app.id ? "border-primary bg-primary/5" : "hover:bg-muted/50"}`}
                  onClick={() => setSelectedId(app.id)}
                >
                  <p className="font-medium text-sm">{app.name}</p>
                  <p className="text-muted-foreground text-xs">
                    {app.ownerName || "Sahip"} ·{" "}
                    {selected?.id === app.id || app.hasConfig
                      ? `${selected?.id === app.id ? selected.config.datasetIds.length : app.config.datasetIds.length} veri kümesi`
                      : "Yapılandırmayı görmek için açın"}
                  </p>
                </button>
              ))
            )}
            <div className="flex justify-between">
              <Button
                size="sm"
                variant="outline"
                disabled={page <= 1 || loading}
                onClick={() => setPage((value) => value - 1)}
              >
                Önceki
              </Button>
              <span className="self-center text-xs">{page}</span>
              <Button
                size="sm"
                variant="outline"
                disabled={page * PAGE_SIZE >= total || loading}
                onClick={() => setPage((value) => value + 1)}
              >
                Sonraki
              </Button>
            </div>
          </CardContent>
        </Card>

        {!selectedId ? (
          <Card>
            <CardContent className="py-12 text-center text-muted-foreground">
              Yapılandırmak ve arama yapmak için bir kayıt seçin.
            </CardContent>
          </Card>
        ) : detailLoading || !selected ? (
          <Card>
            <CardContent className="flex items-center gap-2 py-12">
              <Spinner /> Arama yükleniyor
            </CardContent>
          </Card>
        ) : (
          <div className="space-y-6">
            <Card>
              <CardHeader>
                <div className="flex flex-wrap items-start justify-between gap-3">
                  <div>
                    <CardTitle>{selected.name}</CardTitle>
                    <CardDescription>
                      {selected.ownerName || selected.createdBy} · sahiplik
                      kontrollü
                    </CardDescription>
                  </div>
                  <Button
                    variant="destructive"
                    disabled={busy}
                    onClick={() => {
                      if (
                        window.confirm(
                          "Bu arama uygulamasını silmek istediğinize emin misiniz?",
                        )
                      )
                        void mutate(
                          () => deletePlatformSearch(selected.id),
                          () => {
                            setSelectedId("");
                            setSelected(null);
                            void loadList();
                          },
                        );
                    }}
                  >
                    Sil
                  </Button>
                </div>
              </CardHeader>
              <CardContent className="grid gap-4 md:grid-cols-2">
                <div className="space-y-2">
                  <Label htmlFor="search-edit-name">Ad</Label>
                  <Input
                    id="search-edit-name"
                    value={selected.name}
                    onChange={(event) =>
                      setSelected({ ...selected, name: event.target.value })
                    }
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="search-edit-description">Açıklama</Label>
                  <Input
                    id="search-edit-description"
                    value={selected.description}
                    onChange={(event) =>
                      setSelected({
                        ...selected,
                        description: event.target.value,
                      })
                    }
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="search-model">
                    Sohbet modeli / provider kapsamı
                  </Label>
                  <select
                    id="search-model"
                    className="h-9 w-full rounded-md border bg-background px-3 text-sm"
                    value={selectedChatModelReference}
                    onChange={(event) =>
                      setSelected({
                        ...selected,
                        config: {
                          ...selected.config,
                          chatModelId: event.target.value,
                        },
                      })
                    }
                  >
                    <option value="">Tenant varsayılan modeli</option>
                    {chatModels.map((model) => (
                      <option
                        key={model.id}
                        value={platformModelReference(model)}
                      >
                        {model.name} ·{" "}
                        {model.providerName || model.instanceName || "Provider"}
                      </option>
                    ))}
                  </select>
                </div>
                <div className="space-y-2">
                  <Label htmlFor="search-rerank">Rerank modeli</Label>
                  <select
                    id="search-rerank"
                    className="h-9 w-full rounded-md border bg-background px-3 text-sm"
                    value={selectedRerankModelReference}
                    onChange={(event) =>
                      setSelected({
                        ...selected,
                        config: {
                          ...selected.config,
                          rerankId: event.target.value,
                        },
                      })
                    }
                  >
                    <option value="">Rerank kullanma</option>
                    {rerankModels.map((model) => (
                      <option
                        key={model.id}
                        value={platformModelReference(model)}
                      >
                        {model.name} ·{" "}
                        {model.providerName || model.instanceName || "Provider"}
                      </option>
                    ))}
                  </select>
                </div>
                <fieldset className="space-y-2 md:col-span-2">
                  <legend className="text-sm font-medium">
                    Veri kümesi kapsamı
                  </legend>
                  <div className="grid gap-2 sm:grid-cols-2">
                    {datasets.length === 0 ? (
                      <p className="text-muted-foreground text-sm">
                        Erişilebilir veri kümesi yok.
                      </p>
                    ) : (
                      datasets.map((dataset) => {
                        const id = dtoText(dataset.id);
                        return (
                          <label
                            key={id}
                            className="flex items-center gap-2 rounded-lg border p-3 text-sm"
                          >
                            <Checkbox
                              checked={selected.config.datasetIds.includes(id)}
                              onCheckedChange={(checked) =>
                                setSelected({
                                  ...selected,
                                  config: {
                                    ...selected.config,
                                    datasetIds: checked
                                      ? [
                                          ...new Set([
                                            ...selected.config.datasetIds,
                                            id,
                                          ]),
                                        ]
                                      : selected.config.datasetIds.filter(
                                          (value) => value !== id,
                                        ),
                                  },
                                })
                              }
                            />
                            {dtoText(dataset.name) || id}
                          </label>
                        );
                      })
                    )}
                  </div>
                </fieldset>
                <div className="space-y-2">
                  <Label htmlFor="similarity">Benzerlik eşiği</Label>
                  <Input
                    id="similarity"
                    type="number"
                    min={0}
                    max={1}
                    step={0.05}
                    value={selected.config.similarityThreshold}
                    onChange={(event) =>
                      setSelected({
                        ...selected,
                        config: {
                          ...selected.config,
                          similarityThreshold: Number(event.target.value),
                        },
                      })
                    }
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="vector-weight">Vektör ağırlığı</Label>
                  <Input
                    id="vector-weight"
                    type="number"
                    min={0}
                    max={1}
                    step={0.05}
                    value={selected.config.vectorSimilarityWeight}
                    onChange={(event) =>
                      setSelected({
                        ...selected,
                        config: {
                          ...selected.config,
                          vectorSimilarityWeight: Number(event.target.value),
                        },
                      })
                    }
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="top-k">Aday chunk sayısı (top_k)</Label>
                  <Input
                    id="top-k"
                    type="number"
                    min={1}
                    max={4096}
                    value={selected.config.topK}
                    onChange={(event) =>
                      setSelected({
                        ...selected,
                        config: {
                          ...selected.config,
                          topK: Number(event.target.value),
                        },
                      })
                    }
                  />
                </div>
                <div className="grid gap-2 md:col-span-2 sm:grid-cols-2">
                  {(
                    [
                      ["highlight", "Kaynak vurgusu"],
                      ["keyword", "Anahtar kelime araması"],
                      ["useKnowledgeGraph", "Bilgi grafiği"],
                      ["summary", "Özet"],
                      ["webSearch", "Web araması"],
                      ["relatedSearch", "İlgili aramalar"],
                      ["queryMindMap", "Sorgu zihin haritası"],
                    ] as const
                  ).map(([key, label]) => (
                    <label
                      key={key}
                      className="flex items-center justify-between gap-3 rounded-lg border p-3 text-sm"
                    >
                      <span>{label}</span>
                      <Switch
                        checked={selected.config[key]}
                        onCheckedChange={(checked) =>
                          setSelected({
                            ...selected,
                            config: { ...selected.config, [key]: checked },
                          })
                        }
                      />
                    </label>
                  ))}
                </div>
                <Button
                  className="md:col-span-2"
                  disabled={busy || !selected.name.trim()}
                  onClick={() =>
                    void mutate(async () => {
                      const updated = await updatePlatformSearch(selected.id, {
                        name: selected.name,
                        description: selected.description,
                        config: {
                          ...selected.config,
                          chatModelId: selectedChatModelReference,
                          rerankId: selectedRerankModelReference,
                        },
                      });
                      setSelected(updated);
                      await loadList();
                    })
                  }
                >
                  Yapılandırmayı kaydet
                </Button>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Arama tamamlaması</CardTitle>
                <CardDescription>
                  SSE akışı iptal edilebilir. Yanıt ve kaynaklar görünür; geçmiş
                  yalnızca bu tarayıcı oturumunda tutulur ve kalıcı depoya
                  yazılmaz.
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="flex flex-wrap gap-2">
                  <Textarea
                    className="min-h-24 flex-1"
                    aria-label="Arama sorusu"
                    placeholder="Seçili veri kümelerinde sorunuzu yazın"
                    value={question}
                    onChange={(event) => setQuestion(event.target.value)}
                  />
                  <div className="flex flex-col gap-2">
                    <Button
                      disabled={
                        streaming ||
                        !question.trim() ||
                        selected.config.datasetIds.length === 0
                      }
                      onClick={() => void ask()}
                    >
                      {streaming ? "Aranıyor" : "Ara"}
                    </Button>
                    {streaming ? (
                      <Button
                        variant="outline"
                        onClick={() => streamController.current?.abort()}
                      >
                        İptal et
                      </Button>
                    ) : null}
                  </div>
                </div>
                {streaming ? (
                  <div className="flex items-center gap-2 text-sm">
                    <Spinner /> Yanıt akıyor
                  </div>
                ) : null}
                {answer ? (
                  <div className="rounded-xl border p-4">
                    <p className="whitespace-pre-wrap text-sm">{answer}</p>
                  </div>
                ) : null}
                <div>
                  <h3 className="mb-2 font-medium text-sm">Kaynaklar</h3>
                  {!reference?.chunks.length ? (
                    <p className="text-muted-foreground text-sm">
                      Henüz kaynak yok.
                    </p>
                  ) : (
                    <div className="grid gap-2">
                      {reference.chunks.map((source) => (
                        <div key={source.id} className="rounded-lg border p-3">
                          <div className="flex flex-wrap justify-between gap-2">
                            <p className="font-medium text-sm">
                              {source.filename}
                            </p>
                            <p className="text-muted-foreground text-xs">
                              Veri kümesi: {source.datasetId || "—"} · skor:{" "}
                              {source.score?.toFixed(3) ?? "—"}
                            </p>
                          </div>
                          {source.text ? (
                            <p className="mt-2 line-clamp-4 text-muted-foreground text-sm">
                              {source.text}
                            </p>
                          ) : null}
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Bu oturumun geçmişi</CardTitle>
                <CardDescription>
                  Backend kalıcı Search history endpoint’i sunmadığı için bu
                  liste sayfa yenilenince temizlenir.
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-3">
                {history.length === 0 ? (
                  <p className="text-muted-foreground text-sm">
                    Bu oturumda henüz arama yapılmadı.
                  </p>
                ) : (
                  history.map((entry) => (
                    <button
                      type="button"
                      key={entry.id}
                      className="w-full rounded-lg border p-3 text-left hover:bg-muted/50"
                      onClick={() => {
                        setAnswer(entry.answer);
                        setReference(entry.reference);
                      }}
                    >
                      <p className="font-medium text-sm">{entry.question}</p>
                      <p className="mt-1 line-clamp-2 text-muted-foreground text-sm">
                        {entry.answer || "Yanıt yok"}
                      </p>
                    </button>
                  ))
                )}
              </CardContent>
            </Card>
          </div>
        )}
      </div>
    </main>
  );
}
