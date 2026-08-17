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
  PLATFORM_MEMORY_TYPES,
  type PlatformMemory,
  type PlatformMemoryMessage,
  type PlatformMemoryType,
  type UpdatePlatformMemoryInput,
  addPlatformMemoryMessage,
  createPlatformMemory,
  deletePlatformMemory,
  forgetPlatformMemoryMessage,
  getPlatformMemoryConfig,
  getPlatformMemoryMessageContent,
  getPlatformUiError,
  hasPlatformMemoryConsent,
  listPlatformMemories,
  listPlatformMemoryMessages,
  listRecentPlatformMemoryMessages,
  listTenantModels,
  platformModelReference,
  resolvePlatformModelReference,
  searchPlatformMemoryMessages,
  setPlatformMemoryConsent,
  updatePlatformMemory,
  updatePlatformMemoryMessageStatus,
  type PlatformModel,
} from "@/integrations/platform-backend";
import { useCallback, useEffect, useMemo, useState } from "react";

const PAGE_SIZE = 20;

export function buildPlatformMemoryUpdateInput(
  current: PlatformMemory,
  baseline: PlatformMemory,
  modelReferences: { embedding: string; chat: string },
): UpdatePlatformMemoryInput {
  const memoryTypesChanged =
    [...current.memoryTypes].sort().join(",") !==
    [...baseline.memoryTypes].sort().join(",");
  return {
    ...(current.name !== baseline.name ? { name: current.name } : {}),
    ...(current.permissions !== baseline.permissions
      ? { permissions: current.permissions }
      : {}),
    ...(current.memorySize !== baseline.memorySize
      ? { memorySize: current.memorySize }
      : {}),
    ...(current.forgettingPolicy !== baseline.forgettingPolicy
      ? { forgettingPolicy: current.forgettingPolicy }
      : {}),
    ...(current.description !== baseline.description
      ? { description: current.description }
      : {}),
    ...(current.embeddingModelId !== baseline.embeddingModelId
      ? { embeddingModelId: modelReferences.embedding }
      : {}),
    ...(current.llmId !== baseline.llmId
      ? { llmId: modelReferences.chat }
      : {}),
    ...(memoryTypesChanged ? { memoryTypes: current.memoryTypes } : {}),
    ...(current.temperature !== baseline.temperature
      ? { temperature: current.temperature }
      : {}),
    ...(current.systemPrompt !== baseline.systemPrompt
      ? { systemPrompt: current.systemPrompt }
      : {}),
    ...(current.userPrompt !== baseline.userPrompt
      ? { userPrompt: current.userPrompt }
      : {}),
  };
}

export function phase13ErrorMessage(error: unknown): string | null {
  const ui = getPlatformUiError(error);
  return ui.kind === "aborted" ? null : ui.message;
}

function MemoryMessageCard({
  message,
  onChanged,
}: {
  message: PlatformMemoryMessage;
  onChanged: () => void;
}) {
  const [content, setContent] = useState(message.content);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const run = async (action: () => Promise<unknown>) => {
    setBusy(true);
    setError(null);
    try {
      await action();
      onChanged();
    } catch (reason) {
      setError(phase13ErrorMessage(reason));
    } finally {
      setBusy(false);
    }
  };
  return (
    <div className="space-y-3 rounded-xl border p-4">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div>
          <p className="font-medium text-sm">Mesaj #{message.messageId}</p>
          <p className="text-muted-foreground text-xs">
            Oturum: {message.sessionId || "—"} · Agent:{" "}
            {message.agentName || message.agentId || "—"}
          </p>
        </div>
        <span className="rounded-full bg-muted px-2 py-1 text-xs">
          {message.status ? "Etkin" : "Devre dışı"}
        </span>
      </div>
      {content ? (
        <p className="whitespace-pre-wrap text-sm">{content}</p>
      ) : null}
      {message.forgetAt ? (
        <p className="text-muted-foreground text-xs">
          Unutma zamanı: {message.forgetAt}
        </p>
      ) : null}
      {error ? <p className="text-destructive text-sm">{error}</p> : null}
      <div className="flex flex-wrap gap-2">
        <Button
          size="sm"
          variant="outline"
          disabled={busy}
          onClick={async () => {
            setBusy(true);
            setError(null);
            try {
              const full = await getPlatformMemoryMessageContent(
                message.memoryId,
                message.messageId,
              );
              setContent(full.content);
            } catch (reason) {
              setError(phase13ErrorMessage(reason));
            } finally {
              setBusy(false);
            }
          }}
        >
          İçeriği aç
        </Button>
        <Button
          size="sm"
          variant="outline"
          disabled={busy}
          onClick={() =>
            void run(() =>
              updatePlatformMemoryMessageStatus(
                message.memoryId,
                message.messageId,
                !message.status,
              ),
            )
          }
        >
          {message.status ? "Devre dışı bırak" : "Etkinleştir"}
        </Button>
        <Button
          size="sm"
          variant="destructive"
          disabled={busy}
          onClick={() => {
            if (
              window.confirm(
                "Bu mesajı unutmak istediğinize emin misiniz? İşlem mesajı soft-delete olarak işaretler.",
              )
            ) {
              void run(() =>
                forgetPlatformMemoryMessage(
                  message.memoryId,
                  message.messageId,
                ),
              );
            }
          }}
        >
          Unut
        </Button>
      </div>
    </div>
  );
}

export function MemoryPage() {
  const [items, setItems] = useState<PlatformMemory[]>([]);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(1);
  const [keywords, setKeywords] = useState("");
  const [selectedId, setSelectedId] = useState("");
  const [selected, setSelected] = useState<PlatformMemory | null>(null);
  const [selectedBaseline, setSelectedBaseline] =
    useState<PlatformMemory | null>(null);
  const [messages, setMessages] = useState<PlatformMemoryMessage[]>([]);
  const [messageTotal, setMessageTotal] = useState(0);
  const [messagePage, setMessagePage] = useState(1);
  const [messageQuery, setMessageQuery] = useState("");
  const [loading, setLoading] = useState(true);
  const [detailLoading, setDetailLoading] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [models, setModels] = useState<PlatformModel[]>([]);
  const [consent, setConsent] = useState(false);
  const [createOpen, setCreateOpen] = useState(false);
  const [createName, setCreateName] = useState("");
  const [createTypes, setCreateTypes] = useState<PlatformMemoryType[]>(["raw"]);
  const [embeddingModelId, setEmbeddingModelId] = useState("");
  const [llmId, setLlmId] = useState("");
  const [agentId, setAgentId] = useState("manual-entry");
  const [sessionId, setSessionId] = useState("");
  const [userInput, setUserInput] = useState("");
  const [agentResponse, setAgentResponse] = useState("");

  const embeddingModels = useMemo(
    () => models.filter((model) => model.capabilities.includes("embedding")),
    [models],
  );
  const chatModels = useMemo(
    () => models.filter((model) => model.capabilities.includes("chat")),
    [models],
  );
  const selectedEmbeddingModelReference = selected
    ? resolvePlatformModelReference(selected.embeddingModelId, embeddingModels)
    : "";
  const selectedChatModelReference = selected
    ? resolvePlatformModelReference(selected.llmId, chatModels)
    : "";

  const loadList = useCallback(
    async (signal?: AbortSignal) => {
      setLoading(true);
      setError(null);
      try {
        const result = await listPlatformMemories(
          { page, pageSize: PAGE_SIZE, keywords },
          signal,
        );
        setItems(result.items);
        setTotal(result.total);
        if (
          selectedId &&
          !result.items.some((item) => item.id === selectedId)
        ) {
          setSelectedId("");
          setSelected(null);
          setSelectedBaseline(null);
        }
      } catch (reason) {
        setError(phase13ErrorMessage(reason));
      } finally {
        setLoading(false);
      }
    },
    [keywords, page, selectedId],
  );

  const loadDetail = useCallback(
    async (memoryId: string, signal?: AbortSignal) => {
      setDetailLoading(true);
      setError(null);
      try {
        const [config, result] = await Promise.all([
          getPlatformMemoryConfig(memoryId, signal),
          listPlatformMemoryMessages(
            memoryId,
            { page: messagePage, pageSize: PAGE_SIZE, keywords: messageQuery },
            signal,
          ),
        ]);
        setSelected(config);
        setSelectedBaseline(config);
        setMessages(result.items);
        setMessageTotal(result.total);
        setConsent(hasPlatformMemoryConsent(memoryId));
      } catch (reason) {
        setError(phase13ErrorMessage(reason));
      } finally {
        setDetailLoading(false);
      }
    },
    [messagePage, messageQuery],
  );

  useEffect(() => {
    const controller = new AbortController();
    void Promise.all([
      loadList(controller.signal),
      listTenantModels(controller.signal)
        .then(setModels)
        .catch((reason) => {
          const message = phase13ErrorMessage(reason);
          if (message) setError(message);
        }),
    ]);
    return () => controller.abort();
  }, [loadList]);

  useEffect(() => {
    if (!selectedId) return;
    const controller = new AbortController();
    void loadDetail(selectedId, controller.signal);
    return () => controller.abort();
  }, [loadDetail, selectedId]);

  const refreshSelected = () => {
    if (selectedId) void loadDetail(selectedId);
  };

  const mutate = async (action: () => Promise<unknown>, after?: () => void) => {
    setBusy(true);
    setError(null);
    try {
      await action();
      after?.();
    } catch (reason) {
      setError(phase13ErrorMessage(reason));
    } finally {
      setBusy(false);
    }
  };

  return (
    <main className="mx-auto flex w-full max-w-7xl flex-1 flex-col gap-6 overflow-auto p-6">
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <h1 className="font-heading font-semibold text-3xl">Hafıza</h1>
          <p className="mt-1 text-muted-foreground">
            Sohbet kayıtlarını açık rıza, saklama ve unutma kontrolleriyle
            yönetin.
          </p>
        </div>
        <Button onClick={() => setCreateOpen((value) => !value)}>
          Yeni hafıza
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
            <CardTitle>Yeni hafıza</CardTitle>
            <CardDescription>
              Embedding ve sohbet modeli backend model kataloğundan seçilir.
            </CardDescription>
          </CardHeader>
          <CardContent className="grid gap-4 md:grid-cols-2">
            <div className="space-y-2">
              <Label htmlFor="memory-name">Ad</Label>
              <Input
                id="memory-name"
                value={createName}
                onChange={(event) => setCreateName(event.target.value)}
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="memory-embedding">Embedding modeli</Label>
              <select
                id="memory-embedding"
                className="h-9 w-full rounded-md border bg-background px-3 text-sm"
                value={embeddingModelId}
                onChange={(event) => setEmbeddingModelId(event.target.value)}
              >
                <option value="">Seçin</option>
                {embeddingModels.map((model) => (
                  <option key={model.id} value={platformModelReference(model)}>
                    {model.name} · {model.providerName || model.instanceName}
                  </option>
                ))}
              </select>
            </div>
            <div className="space-y-2">
              <Label htmlFor="memory-llm">Sohbet modeli</Label>
              <select
                id="memory-llm"
                className="h-9 w-full rounded-md border bg-background px-3 text-sm"
                value={llmId}
                onChange={(event) => setLlmId(event.target.value)}
              >
                <option value="">Seçin</option>
                {chatModels.map((model) => (
                  <option key={model.id} value={platformModelReference(model)}>
                    {model.name} · {model.providerName || model.instanceName}
                  </option>
                ))}
              </select>
            </div>
            <fieldset className="space-y-2">
              <legend className="text-sm font-medium">Hafıza türleri</legend>
              <div className="flex flex-wrap gap-3">
                {PLATFORM_MEMORY_TYPES.map((type) => (
                  <label key={type} className="flex items-center gap-2 text-sm">
                    <Checkbox
                      checked={createTypes.includes(type)}
                      onCheckedChange={(checked) =>
                        setCreateTypes((current) =>
                          checked
                            ? [...new Set([...current, type])]
                            : current.filter((value) => value !== type),
                        )
                      }
                    />
                    {type}
                  </label>
                ))}
              </div>
            </fieldset>
            <div className="flex gap-2 md:col-span-2">
              <Button
                disabled={
                  busy ||
                  !createName.trim() ||
                  !embeddingModelId ||
                  !llmId ||
                  createTypes.length === 0
                }
                onClick={() =>
                  void mutate(async () => {
                    const memory = await createPlatformMemory({
                      name: createName,
                      memoryTypes: createTypes,
                      embeddingModelId,
                      llmId,
                    });
                    setCreateName("");
                    setCreateOpen(false);
                    await loadList();
                    setSelectedId(memory.id);
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
            <CardTitle>Hafızalar</CardTitle>
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
                aria-label="Hafızalarda ara"
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
              <p className="text-muted-foreground text-sm">Henüz hafıza yok.</p>
            ) : (
              items.map((memory) => (
                <button
                  type="button"
                  key={memory.id}
                  onClick={() => {
                    setMessagePage(1);
                    setSelectedId(memory.id);
                  }}
                  className={`w-full rounded-lg border p-3 text-left ${selectedId === memory.id ? "border-primary bg-primary/5" : "hover:bg-muted/50"}`}
                >
                  <p className="font-medium text-sm">{memory.name}</p>
                  <p className="text-muted-foreground text-xs">
                    {memory.memoryTypes.join(", ") || "raw"} ·{" "}
                    {memory.permissions === "team" ? "Ekip" : "Yalnızca ben"}
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
              Ayarları ve mesajları görmek için bir hafıza seçin.
            </CardContent>
          </Card>
        ) : detailLoading || !selected ? (
          <Card>
            <CardContent className="flex items-center gap-2 py-12">
              <Spinner /> Hafıza yükleniyor
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
                      {selected.ownerName || "Sahip"} · {selected.storageType}
                    </CardDescription>
                  </div>
                  <Button
                    variant="destructive"
                    disabled={busy}
                    onClick={() => {
                      if (
                        window.confirm(
                          "Bu hafızayı ve ilişkili mesajları kalıcı olarak silmek istediğinize emin misiniz?",
                        )
                      )
                        void mutate(
                          () => deletePlatformMemory(selected.id),
                          () => {
                            setPlatformMemoryConsent(selected.id, false);
                            setSelectedId("");
                            setSelected(null);
                            void loadList();
                          },
                        );
                    }}
                  >
                    Hafızayı sil
                  </Button>
                </div>
              </CardHeader>
              <CardContent className="grid gap-4 md:grid-cols-2">
                <div className="space-y-2">
                  <Label htmlFor="memory-edit-name">Ad</Label>
                  <Input
                    id="memory-edit-name"
                    value={selected.name}
                    onChange={(event) =>
                      setSelected({ ...selected, name: event.target.value })
                    }
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="memory-permission">Erişim</Label>
                  <select
                    id="memory-permission"
                    className="h-9 w-full rounded-md border bg-background px-3 text-sm"
                    value={selected.permissions}
                    onChange={(event) =>
                      setSelected({
                        ...selected,
                        permissions:
                          event.target.value === "team" ? "team" : "me",
                      })
                    }
                  >
                    <option value="me">Yalnızca ben</option>
                    <option value="team">Ekip</option>
                  </select>
                </div>
                <div className="space-y-2">
                  <Label htmlFor="memory-policy">Unutma politikası</Label>
                  <select
                    id="memory-policy"
                    className="h-9 w-full rounded-md border bg-background px-3 text-sm"
                    value={selected.forgettingPolicy}
                    onChange={() =>
                      setSelected({
                        ...selected,
                        forgettingPolicy: "FIFO",
                      })
                    }
                  >
                    <option value="FIFO">FIFO · en eski önce</option>
                  </select>
                </div>
                <div className="space-y-2">
                  <Label htmlFor="memory-size">
                    Saklama sınırı (bayt, en fazla 5 MB)
                  </Label>
                  <Input
                    id="memory-size"
                    type="number"
                    min={1}
                    max={5 * 1024 * 1024}
                    value={selected.memorySize}
                    onChange={(event) =>
                      setSelected({
                        ...selected,
                        memorySize: Number(event.target.value),
                      })
                    }
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="memory-edit-embedding">
                    Embedding modeli
                  </Label>
                  <select
                    id="memory-edit-embedding"
                    className="h-9 w-full rounded-md border bg-background px-3 text-sm"
                    value={selectedEmbeddingModelReference}
                    onChange={(event) =>
                      setSelected({
                        ...selected,
                        embeddingModelId: event.target.value,
                      })
                    }
                  >
                    {embeddingModels.map((model) => (
                      <option
                        key={model.id}
                        value={platformModelReference(model)}
                      >
                        {model.name} ·{" "}
                        {model.providerName || model.instanceName}
                      </option>
                    ))}
                  </select>
                </div>
                <div className="space-y-2">
                  <Label htmlFor="memory-edit-llm">Sohbet modeli</Label>
                  <select
                    id="memory-edit-llm"
                    className="h-9 w-full rounded-md border bg-background px-3 text-sm"
                    value={selectedChatModelReference}
                    onChange={(event) =>
                      setSelected({ ...selected, llmId: event.target.value })
                    }
                  >
                    {chatModels.map((model) => (
                      <option
                        key={model.id}
                        value={platformModelReference(model)}
                      >
                        {model.name} ·{" "}
                        {model.providerName || model.instanceName}
                      </option>
                    ))}
                  </select>
                </div>
                <fieldset className="space-y-2 md:col-span-2">
                  <legend className="text-sm font-medium">
                    Hafıza türleri
                  </legend>
                  <div className="flex flex-wrap gap-3">
                    {PLATFORM_MEMORY_TYPES.map((type) => (
                      <label
                        key={type}
                        className="flex items-center gap-2 text-sm"
                      >
                        <Checkbox
                          checked={selected.memoryTypes.includes(type)}
                          onCheckedChange={(checked) =>
                            setSelected({
                              ...selected,
                              memoryTypes: checked
                                ? [...new Set([...selected.memoryTypes, type])]
                                : selected.memoryTypes.filter(
                                    (value) => value !== type,
                                  ),
                            })
                          }
                        />
                        {type}
                      </label>
                    ))}
                  </div>
                </fieldset>
                <div className="space-y-2 md:col-span-2">
                  <Label htmlFor="memory-temperature">
                    Çıkarım sıcaklığı ({selected.temperature.toFixed(2)})
                  </Label>
                  <Input
                    id="memory-temperature"
                    type="range"
                    min={0}
                    max={1}
                    step={0.05}
                    value={selected.temperature}
                    onChange={(event) =>
                      setSelected({
                        ...selected,
                        temperature: Number(event.target.value),
                      })
                    }
                  />
                </div>
                <div className="space-y-2 md:col-span-2">
                  <Label htmlFor="memory-description">Açıklama</Label>
                  <Textarea
                    id="memory-description"
                    value={selected.description}
                    onChange={(event) =>
                      setSelected({
                        ...selected,
                        description: event.target.value,
                      })
                    }
                  />
                </div>
                <div className="space-y-2 md:col-span-2">
                  <Label htmlFor="memory-system-prompt">
                    Sistem çıkarım prompt’u
                  </Label>
                  <Textarea
                    id="memory-system-prompt"
                    value={selected.systemPrompt}
                    onChange={(event) =>
                      setSelected({
                        ...selected,
                        systemPrompt: event.target.value,
                      })
                    }
                  />
                </div>
                <div className="space-y-2 md:col-span-2">
                  <Label htmlFor="memory-user-prompt">
                    Kullanıcı çıkarım prompt’u
                  </Label>
                  <Textarea
                    id="memory-user-prompt"
                    value={selected.userPrompt}
                    onChange={(event) =>
                      setSelected({
                        ...selected,
                        userPrompt: event.target.value,
                      })
                    }
                  />
                </div>
                <Alert className="md:col-span-2">
                  <AlertTitle>Model ve tür değişikliği</AlertTitle>
                  <AlertDescription>
                    Backend, mevcut hafıza verisiyle uyumsuz model veya tür
                    değişikliklerini reddedebilir. Reddedilen değişiklikler
                    uygulanmaz ve hata açıkça gösterilir.
                  </AlertDescription>
                </Alert>
                <div className="flex gap-2 md:col-span-2">
                  <Button
                    disabled={busy || selected.memoryTypes.length === 0}
                    onClick={() => {
                      if (!selectedBaseline) return;
                      void mutate(
                        () =>
                          updatePlatformMemory(
                            selected.id,
                            buildPlatformMemoryUpdateInput(
                              selected,
                              selectedBaseline,
                              {
                                embedding: selectedEmbeddingModelReference,
                                chat: selectedChatModelReference,
                              },
                            ),
                          ),
                        refreshSelected,
                      );
                    }}
                  >
                    Ayarları kaydet
                  </Button>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>İzinli sohbet kaydı</CardTitle>
                <CardDescription>
                  Rag Platform normal sohbet endpoint’i hafıza parametresi
                  almaz. Bu izin yalnızca aşağıdaki açık kayıt aksiyonunu
                  etkinleştirir; sohbetler otomatik aktarılmaz.
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <label className="flex items-center justify-between gap-4 rounded-lg border p-3 text-sm">
                  <span>Bu hafızaya sohbet kaydı eklememe izin ver</span>
                  <Switch
                    checked={consent}
                    onCheckedChange={(checked) => {
                      setConsent(checked);
                      setPlatformMemoryConsent(selected.id, checked);
                    }}
                  />
                </label>
                {!consent ? (
                  <Alert>
                    <AlertTitle>Kayıt kapalı</AlertTitle>
                    <AlertDescription>
                      Açık izin verilmeden kullanıcı veya asistan metni
                      backend’e gönderilmez.
                    </AlertDescription>
                  </Alert>
                ) : (
                  <div className="grid gap-3 md:grid-cols-2">
                    <div className="space-y-2">
                      <Label htmlFor="agent-id">Agent kimliği</Label>
                      <Input
                        id="agent-id"
                        value={agentId}
                        onChange={(event) => setAgentId(event.target.value)}
                      />
                    </div>
                    <div className="space-y-2">
                      <Label htmlFor="session-id">Oturum kimliği</Label>
                      <Input
                        id="session-id"
                        value={sessionId}
                        onChange={(event) => setSessionId(event.target.value)}
                      />
                    </div>
                    <div className="space-y-2">
                      <Label htmlFor="user-input">Kullanıcı mesajı</Label>
                      <Textarea
                        id="user-input"
                        value={userInput}
                        onChange={(event) => setUserInput(event.target.value)}
                      />
                    </div>
                    <div className="space-y-2">
                      <Label htmlFor="agent-response">Asistan yanıtı</Label>
                      <Textarea
                        id="agent-response"
                        value={agentResponse}
                        onChange={(event) =>
                          setAgentResponse(event.target.value)
                        }
                      />
                    </div>
                    <Button
                      className="md:col-span-2"
                      disabled={
                        busy ||
                        !agentId.trim() ||
                        !sessionId.trim() ||
                        !userInput.trim() ||
                        !agentResponse.trim()
                      }
                      onClick={() =>
                        void mutate(
                          () =>
                            addPlatformMemoryMessage({
                              memoryIds: [selected.id],
                              agentId,
                              sessionId,
                              userInput,
                              agentResponse,
                            }),
                          () => {
                            setUserInput("");
                            setAgentResponse("");
                            refreshSelected();
                          },
                        )
                      }
                    >
                      İzinli kaydı ekle
                    </Button>
                  </div>
                )}
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Mesajlar</CardTitle>
                <CardDescription>
                  {messageTotal} kayıt · içerik, durum ve unutma yaşam döngüsü
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <form
                  className="flex flex-wrap gap-2"
                  onSubmit={(event) => {
                    event.preventDefault();
                    setMessagePage(1);
                    void loadDetail(selected.id);
                  }}
                >
                  <Input
                    className="min-w-56 flex-1"
                    aria-label="Hafıza mesajlarında ara"
                    placeholder="Oturum anahtarı veya gelişmiş arama sorgusu"
                    value={messageQuery}
                    onChange={(event) => setMessageQuery(event.target.value)}
                  />
                  <Button type="submit">Oturuma göre ara</Button>
                  <Button
                    type="button"
                    variant="outline"
                    onClick={() =>
                      void mutate(async () => {
                        const result = await searchPlatformMemoryMessages(
                          [selected.id],
                          messageQuery,
                        );
                        setMessages(result);
                        setMessageTotal(result.length);
                      })
                    }
                  >
                    Anlamsal ara
                  </Button>
                  <Button
                    type="button"
                    variant="outline"
                    onClick={() =>
                      void mutate(async () => {
                        const result = await listRecentPlatformMemoryMessages(
                          [selected.id],
                          { limit: 20 },
                        );
                        setMessages(result);
                        setMessageTotal(result.length);
                      })
                    }
                  >
                    Son mesajlar
                  </Button>
                </form>
                {messages.length === 0 ? (
                  <p className="text-muted-foreground text-sm">
                    Bu filtrede mesaj yok.
                  </p>
                ) : (
                  messages.map((message) => (
                    <MemoryMessageCard
                      key={`${message.memoryId}:${message.messageId}`}
                      message={{
                        ...message,
                        memoryId: message.memoryId || selected.id,
                      }}
                      onChanged={refreshSelected}
                    />
                  ))
                )}
                <div className="flex justify-between">
                  <Button
                    size="sm"
                    variant="outline"
                    disabled={messagePage <= 1}
                    onClick={() => setMessagePage((value) => value - 1)}
                  >
                    Önceki
                  </Button>
                  <span className="self-center text-xs">{messagePage}</span>
                  <Button
                    size="sm"
                    variant="outline"
                    disabled={messagePage * PAGE_SIZE >= messageTotal}
                    onClick={() => setMessagePage((value) => value + 1)}
                  >
                    Sonraki
                  </Button>
                </div>
              </CardContent>
            </Card>
          </div>
        )}
      </div>
    </main>
  );
}
