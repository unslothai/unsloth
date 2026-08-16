import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Spinner } from "@/components/ui/spinner";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Textarea } from "@/components/ui/textarea";
import {
  EMPTY_PLATFORM_AGENT_DSL,
  PLATFORM_AGENT_FILE_MAX_BYTES,
  PlatformApiError,
  cancelAgentRun,
  cancelAgentSession,
  createAgent,
  createAgentSession,
  createMcpServer,
  debugAgentComponent,
  deleteAgent,
  deleteAgentSession,
  deleteAgentSessions,
  deleteAgentVersion,
  deleteMcpServer,
  downloadAgentAttachment,
  downloadAgentFile,
  getAgent,
  getAgentComponentInputForm,
  getAgentLogs,
  getAgentPrompts,
  getAgentSession,
  getAgentVersion,
  getAgentWebhookLogs,
  getMcpServer,
  importMcpServers,
  listAgentComponents,
  listAgentSessions,
  listAgentVersions,
  listAgents,
  listAgentTemplates,
  listAvailableAgentTags,
  listMcpServers,
  listPluginTools,
  previewAgentAttachment,
  publishAgent,
  redactAgentSecrets,
  rerunAgentDocument,
  resetAgent,
  streamAgentCompletion,
  streamAgentRun,
  testAgentDatabaseConnection,
  testAgentWebhook,
  testMcpServer,
  updateAgent,
  updateAgentTags,
  updateMcpServer,
  uploadAgentFiles,
  type PlatformAgent,
  type PlatformAgentComponent,
  type PlatformAgentDsl,
  type PlatformAgentSession,
  type PlatformAgentVersion,
  type PlatformMcpServer,
  type PlatformPluginTool,
} from "@/integrations/platform-backend";
import {
  type ReactNode,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";

type Action = string | null;

function messageOf(error: unknown): string {
  if (error instanceof PlatformApiError) {
    if (error.httpStatus === 401)
      return "Oturum doğrulanamadı. Lütfen yeniden giriş yapın.";
    if (error.httpStatus === 403) return "Bu işlem için yetkiniz yok.";
    return error.message;
  }
  return error instanceof Error
    ? error.message
    : "Beklenmeyen bir hata oluştu.";
}

function pretty(value: unknown): string {
  return JSON.stringify(value, null, 2);
}

function parseObject(value: string, label: string): Record<string, unknown> {
  const parsed = JSON.parse(value) as unknown;
  if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
    throw new Error(`${label} bir JSON nesnesi olmalıdır.`);
  }
  return parsed as Record<string, unknown>;
}

function downloadBlob(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  anchor.click();
  URL.revokeObjectURL(url);
}

function Output({ value }: { value: unknown }) {
  if (value === null || value === undefined || value === "") return null;
  return (
    <pre className="max-h-72 overflow-auto rounded-xl bg-muted p-3 text-xs whitespace-pre-wrap">
      {typeof value === "string" ? value : pretty(redactAgentSecrets(value))}
    </pre>
  );
}

function Field({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div className="grid gap-2">
      <Label>{label}</Label>
      {children}
    </div>
  );
}

export function AgentsPage() {
  const [agents, setAgents] = useState<PlatformAgent[]>([]);
  const [selectedId, setSelectedId] = useState("");
  const [title, setTitle] = useState("");
  const [description, setDescription] = useState("");
  const [tags, setTags] = useState("");
  const [dslText, setDslText] = useState(pretty(EMPTY_PLATFORM_AGENT_DSL));
  const [components, setComponents] = useState<PlatformAgentComponent[]>([]);
  const [templates, setTemplates] = useState<Record<string, unknown>[]>([]);
  const [prompts, setPrompts] = useState<Record<string, string>>({});
  const [availableTags, setAvailableTags] = useState<
    { tag: string; count: number }[]
  >([]);
  const [sessions, setSessions] = useState<PlatformAgentSession[]>([]);
  const [versions, setVersions] = useState<PlatformAgentVersion[]>([]);
  const [mcpServers, setMcpServers] = useState<PlatformMcpServer[]>([]);
  const [pluginTools, setPluginTools] = useState<PlatformPluginTool[]>([]);
  const [action, setAction] = useState<Action>(null);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [output, setOutput] = useState<unknown>(null);
  const [loading, setLoading] = useState(true);
  const activeRequest = useRef<AbortController | null>(null);

  const selected = useMemo(
    () => agents.find((item) => item.id === selectedId) ?? null,
    [agents, selectedId],
  );

  const run = useCallback(
    async (name: string, task: (signal: AbortSignal) => Promise<unknown>) => {
      if (action) return;
      const controller = new AbortController();
      activeRequest.current = controller;
      setAction(name);
      setError("");
      setNotice("");
      try {
        const result = await task(controller.signal);
        if (result !== undefined) setOutput(result);
        setNotice("İşlem tamamlandı.");
        return result;
      } catch (caught) {
        setError(messageOf(caught));
        return undefined;
      } finally {
        if (activeRequest.current === controller) activeRequest.current = null;
        setAction(null);
      }
    },
    [action],
  );

  const refreshAgents = useCallback(async (signal?: AbortSignal) => {
    const result = await listAgents({}, signal);
    setAgents(result.items);
    setSelectedId((current) => current || result.items[0]?.id || "");
    return result;
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    void Promise.all([
      refreshAgents(controller.signal),
      listAgentComponents(controller.signal).then(setComponents),
      listAgentTemplates(controller.signal).then(setTemplates),
      getAgentPrompts(controller.signal).then(setPrompts),
      listAvailableAgentTags(controller.signal).then(setAvailableTags),
      listMcpServers(controller.signal).then((value) =>
        setMcpServers(value.items),
      ),
      listPluginTools(controller.signal).then(setPluginTools),
    ])
      .catch((caught) => setError(messageOf(caught)))
      .finally(() => setLoading(false));
    return () => controller.abort();
  }, [refreshAgents]);

  useEffect(() => {
    if (!selectedId) {
      setSessions([]);
      setVersions([]);
      return;
    }
    const controller = new AbortController();
    void Promise.all([
      getAgent(selectedId, controller.signal),
      listAgentSessions(selectedId, controller.signal),
      listAgentVersions(selectedId, controller.signal),
    ])
      .then(([detail, nextSessions, nextVersions]) => {
        setTitle(detail.title ?? "");
        setDescription(detail.description ?? "");
        setTags((detail.tags ?? []).join(", "));
        setDslText(pretty(detail.dsl ?? EMPTY_PLATFORM_AGENT_DSL));
        setSessions(nextSessions);
        setVersions(nextVersions);
      })
      .catch((caught) => {
        if (!controller.signal.aborted) setError(messageOf(caught));
      });
    return () => controller.abort();
  }, [selectedId]);

  useEffect(
    () => () => {
      activeRequest.current?.abort();
    },
    [],
  );

  async function refreshSelected(signal?: AbortSignal) {
    if (!selectedId) return;
    const [detail, nextSessions, nextVersions] = await Promise.all([
      getAgent(selectedId, signal),
      listAgentSessions(selectedId, signal),
      listAgentVersions(selectedId, signal),
    ]);
    setTitle(detail.title ?? "");
    setDescription(detail.description ?? "");
    setTags((detail.tags ?? []).join(", "));
    setDslText(pretty(detail.dsl ?? EMPTY_PLATFORM_AGENT_DSL));
    setSessions(nextSessions);
    setVersions(nextVersions);
  }

  async function stream(
    name: "run" | "completion",
    input: string,
    sessionId?: string,
  ) {
    if (!selectedId || action || !input.trim()) return;
    const controller = new AbortController();
    activeRequest.current = controller;
    setAction(name);
    setError("");
    setNotice("");
    const events: unknown[] = [];
    try {
      const iterator =
        name === "run"
          ? streamAgentRun(
              { agentId: selectedId, userInput: input, sessionId },
              controller.signal,
            )
          : streamAgentCompletion(
              { agentId: selectedId, query: input, sessionId },
              controller.signal,
            );
      for await (const event of iterator) {
        events.push(event);
        setOutput([...events]);
      }
      setNotice("Stream tamamlandı.");
      await refreshSelected(controller.signal);
    } catch (caught) {
      setError(messageOf(caught));
    } finally {
      if (activeRequest.current === controller) activeRequest.current = null;
      setAction(null);
    }
  }

  if (loading) {
    return (
      <div className="flex flex-1 items-center justify-center gap-3">
        <Spinner /> Agents yükleniyor…
      </div>
    );
  }

  return (
    <main className="mx-auto flex w-full max-w-[1500px] flex-1 flex-col gap-5 overflow-auto p-6">
      <header className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <h1 className="font-heading text-3xl font-semibold">Agents</h1>
          <p className="mt-1 text-sm text-muted-foreground">
            Agent yaşam döngüsü, oturumlar, araçlar ve entegrasyonlar.
          </p>
        </div>
        <div className="flex gap-2">
          {action && (
            <Button
              variant="outline"
              onClick={() => activeRequest.current?.abort()}
            >
              İptal et
            </Button>
          )}
          <CreateAgentButton
            disabled={Boolean(action)}
            onCreate={(name) =>
              run("create", async (signal) => {
                const created = await createAgent(
                  { title: name, dsl: EMPTY_PLATFORM_AGENT_DSL },
                  signal,
                );
                await refreshAgents(signal);
                setSelectedId(created.id);
                return created;
              })
            }
          />
        </div>
      </header>

      {error && (
        <Alert variant="destructive">
          <AlertTitle>İşlem tamamlanamadı</AlertTitle>
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}
      {notice && (
        <Alert>
          <AlertTitle>Başarılı</AlertTitle>
          <AlertDescription>{notice}</AlertDescription>
        </Alert>
      )}

      <div className="grid min-h-0 flex-1 gap-5 lg:grid-cols-[280px_minmax(0,1fr)]">
        <Card className="h-fit">
          <CardHeader>
            <CardTitle>
              Agent listesi <Badge variant="secondary">{agents.length}</Badge>
            </CardTitle>
          </CardHeader>
          <CardContent className="grid gap-2">
            {agents.length === 0 ? (
              <p className="text-sm text-muted-foreground">
                Henüz agent yok. İlk agent’ı oluşturun.
              </p>
            ) : (
              agents.map((item) => (
                <Button
                  key={item.id}
                  variant={selectedId === item.id ? "secondary" : "ghost"}
                  className="h-auto justify-start py-3 text-left"
                  onClick={() => setSelectedId(item.id)}
                >
                  <span className="min-w-0">
                    <span className="block truncate">
                      {item.title || "Adsız agent"}
                    </span>
                    <span className="block truncate text-xs font-normal text-muted-foreground">
                      {item.id}
                    </span>
                  </span>
                </Button>
              ))
            )}
          </CardContent>
        </Card>

        {!selected ? (
          <Card>
            <CardContent className="py-12 text-center text-muted-foreground">
              Çalışmak için bir agent oluşturun veya seçin.
            </CardContent>
          </Card>
        ) : (
          <Tabs defaultValue="general" className="min-w-0">
            <TabsList className="max-w-full flex-wrap">
              <TabsTrigger value="general">Genel</TabsTrigger>
              <TabsTrigger value="editor">Canvas</TabsTrigger>
              <TabsTrigger value="runtime">Çalıştırma</TabsTrigger>
              <TabsTrigger value="sessions">Oturumlar</TabsTrigger>
              <TabsTrigger value="versions">Sürümler</TabsTrigger>
              <TabsTrigger value="integrations">Araçlar</TabsTrigger>
            </TabsList>
            <TabsContent value="general">
              <GeneralPanel
                title={title}
                setTitle={setTitle}
                description={description}
                setDescription={setDescription}
                tags={tags}
                setTags={setTags}
                disabled={Boolean(action)}
                onSave={() =>
                  run("save", async (signal) => {
                    await updateAgent(
                      selectedId,
                      { title, description },
                      signal,
                    );
                    await updateAgentTags(
                      selectedId,
                      tags
                        .split(",")
                        .map((tag) => tag.trim())
                        .filter(Boolean),
                      signal,
                    );
                    await refreshAgents(signal);
                    await refreshSelected(signal);
                  })
                }
                onPublish={() =>
                  run("publish", async (signal) =>
                    publishAgent(
                      selectedId,
                      { title, description, dsl: parseObject(dslText, "DSL") },
                      signal,
                    ),
                  )
                }
                onReset={() =>
                  run("reset", async (signal) => {
                    if (
                      !window.confirm(
                        "Agent çalışma durumunu sıfırlamak istiyor musunuz?",
                      )
                    )
                      return;
                    const dsl = await resetAgent(selectedId, signal);
                    setDslText(pretty(dsl));
                    return dsl;
                  })
                }
                onDelete={() =>
                  run("delete", async (signal) => {
                    if (
                      !window.confirm(
                        "Bu agent ve ilişkili sürümleri kalıcı olarak silmek istiyor musunuz?",
                      )
                    )
                      return;
                    await deleteAgent(selectedId, signal);
                    setSelectedId("");
                    await refreshAgents(signal);
                  })
                }
              />
            </TabsContent>
            <TabsContent value="editor">
              <EditorPanel
                dslText={dslText}
                setDslText={setDslText}
                components={components}
                templates={templates}
                prompts={prompts}
                availableTags={availableTags}
                disabled={Boolean(action)}
                output={output}
                onSave={() =>
                  run("save-dsl", async (signal) => {
                    const dsl = parseObject(dslText, "DSL") as PlatformAgentDsl;
                    await updateAgent(selectedId, { dsl }, signal);
                    await refreshSelected(signal);
                  })
                }
                onInputForm={(componentId) =>
                  run("input-form", (signal) =>
                    getAgentComponentInputForm(selectedId, componentId, signal),
                  )
                }
                onDebug={(componentId, params) =>
                  run("debug", (signal) =>
                    debugAgentComponent(
                      selectedId,
                      componentId,
                      params,
                      signal,
                    ),
                  )
                }
              />
            </TabsContent>
            <TabsContent value="runtime">
              <RuntimePanel
                sessions={sessions}
                disabled={Boolean(action)}
                output={output}
                onRun={(input, sessionId) => stream("run", input, sessionId)}
                onCompletion={(input, sessionId) =>
                  stream("completion", input, sessionId)
                }
                onCancel={() =>
                  run("cancel", async (signal) => {
                    if (
                      !window.confirm(
                        "Çalışan agent görevini iptal etmek istiyor musunuz?",
                      )
                    )
                      return;
                    return cancelAgentRun(selectedId, signal);
                  })
                }
                onSessionCancel={(sessionId) =>
                  run("cancel-session", (signal) =>
                    cancelAgentSession(sessionId, signal),
                  )
                }
                onRerun={(documentId, componentId) =>
                  run("rerun", (signal) =>
                    rerunAgentDocument(
                      {
                        id: documentId,
                        component_id: componentId,
                        dsl: parseObject(dslText, "DSL"),
                      },
                      signal,
                    ),
                  )
                }
                onLogs={(messageId) =>
                  run("logs", (signal) =>
                    getAgentLogs(selectedId, messageId, signal),
                  )
                }
              />
            </TabsContent>
            <TabsContent value="sessions">
              <SessionsPanel
                sessions={sessions}
                disabled={Boolean(action)}
                output={output}
                onCreate={(name) =>
                  run("create-session", async (signal) => {
                    const value = await createAgentSession(
                      selectedId,
                      name,
                      signal,
                    );
                    setSessions(await listAgentSessions(selectedId, signal));
                    return value;
                  })
                }
                onInspect={(id) =>
                  run("session-detail", (signal) =>
                    getAgentSession(selectedId, id, signal),
                  )
                }
                onDelete={(id) =>
                  run("delete-session", async (signal) => {
                    if (!window.confirm("Bu oturum silinsin mi?")) return;
                    await deleteAgentSession(selectedId, id, signal);
                    setSessions(await listAgentSessions(selectedId, signal));
                  })
                }
                onBulkDelete={(ids, deleteAll) =>
                  run("delete-sessions", async (signal) => {
                    if (
                      !window.confirm(
                        deleteAll
                          ? "Tüm oturumlar silinsin mi?"
                          : "Seçili oturumlar silinsin mi?",
                      )
                    )
                      return;
                    await deleteAgentSessions(
                      selectedId,
                      { ids, deleteAll },
                      signal,
                    );
                    setSessions(await listAgentSessions(selectedId, signal));
                  })
                }
              />
            </TabsContent>
            <TabsContent value="versions">
              <VersionsPanel
                versions={versions}
                disabled={Boolean(action)}
                output={output}
                onInspect={(id) =>
                  run("version-detail", (signal) =>
                    getAgentVersion(selectedId, id, signal),
                  )
                }
                onDelete={(id) =>
                  run("delete-version", async (signal) => {
                    if (!window.confirm("Bu sürüm kalıcı olarak silinsin mi?"))
                      return;
                    await deleteAgentVersion(selectedId, id, signal);
                    setVersions(await listAgentVersions(selectedId, signal));
                  })
                }
              />
            </TabsContent>
            <TabsContent value="integrations">
              <IntegrationsPanel
                agentId={selectedId}
                mcpServers={mcpServers}
                pluginTools={pluginTools}
                disabled={Boolean(action)}
                output={output}
                run={run}
                refreshMcp={async (signal) =>
                  setMcpServers((await listMcpServers(signal)).items)
                }
              />
            </TabsContent>
          </Tabs>
        )}
      </div>
    </main>
  );
}

function CreateAgentButton({
  disabled,
  onCreate,
}: {
  disabled: boolean;
  onCreate: (name: string) => void;
}) {
  const [name, setName] = useState("");
  return (
    <div className="flex gap-2">
      <Input
        aria-label="Yeni agent adı"
        placeholder="Yeni agent adı"
        value={name}
        onChange={(event) => setName(event.target.value)}
      />
      <Button
        disabled={disabled || !name.trim()}
        onClick={() => {
          onCreate(name.trim());
          setName("");
        }}
      >
        Oluştur
      </Button>
    </div>
  );
}

function GeneralPanel(props: {
  title: string;
  setTitle: (value: string) => void;
  description: string;
  setDescription: (value: string) => void;
  tags: string;
  setTags: (value: string) => void;
  disabled: boolean;
  onSave: () => void;
  onPublish: () => void;
  onReset: () => void;
  onDelete: () => void;
}) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Agent ayarları</CardTitle>
      </CardHeader>
      <CardContent className="grid gap-4">
        <Field label="Başlık">
          <Input
            value={props.title}
            onChange={(event) => props.setTitle(event.target.value)}
          />
        </Field>
        <Field label="Açıklama">
          <Textarea
            value={props.description}
            onChange={(event) => props.setDescription(event.target.value)}
          />
        </Field>
        <Field label="Etiketler (virgülle)">
          <Input
            value={props.tags}
            onChange={(event) => props.setTags(event.target.value)}
          />
        </Field>
        <div className="flex flex-wrap gap-2">
          <Button disabled={props.disabled} onClick={props.onSave}>
            Taslağı kaydet
          </Button>
          <Button
            variant="secondary"
            disabled={props.disabled}
            onClick={props.onPublish}
          >
            Yayınla
          </Button>
          <Button
            variant="outline"
            disabled={props.disabled}
            onClick={props.onReset}
          >
            Durumu sıfırla
          </Button>
          <Button
            variant="destructive"
            disabled={props.disabled}
            onClick={props.onDelete}
          >
            Agent’ı sil
          </Button>
        </div>
      </CardContent>
    </Card>
  );
}

function EditorPanel(props: {
  dslText: string;
  setDslText: (value: string) => void;
  components: PlatformAgentComponent[];
  templates: Record<string, unknown>[];
  prompts: Record<string, string>;
  availableTags: { tag: string; count: number }[];
  disabled: boolean;
  output: unknown;
  onSave: () => void;
  onInputForm: (id: string) => void;
  onDebug: (id: string, params: Record<string, { value: unknown }>) => void;
}) {
  const [componentId, setComponentId] = useState("begin");
  const [debugText, setDebugText] = useState("{}");
  const [validationError, setValidationError] = useState("");
  return (
    <div className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_340px]">
      <Card>
        <CardHeader>
          <CardTitle>Canonical Agent DSL editörü</CardTitle>
        </CardHeader>
        <CardContent className="grid gap-3">
          <Textarea
            aria-label="Agent DSL"
            className="min-h-[520px] font-mono text-xs"
            value={props.dslText}
            onChange={(event) => props.setDslText(event.target.value)}
          />
          <Button
            className="w-fit"
            disabled={props.disabled}
            onClick={props.onSave}
          >
            DSL’i doğrula ve kaydet
          </Button>
        </CardContent>
      </Card>
      <div className="grid content-start gap-4">
        <Card>
          <CardHeader>
            <CardTitle>Component kataloğu</CardTitle>
          </CardHeader>
          <CardContent className="max-h-64 overflow-auto">
            <ul className="grid gap-2">
              {props.components.length ? (
                props.components.map((item) => (
                  <li key={item.name} className="rounded-lg border p-2">
                    <b>{item.name}</b>
                    <span className="ml-2 text-xs text-muted-foreground">
                      {item.category}
                    </span>
                  </li>
                ))
              ) : (
                <li className="text-muted-foreground">Component bulunamadı.</li>
              )}
            </ul>
          </CardContent>
        </Card>
        <Card>
          <CardHeader>
            <CardTitle>Şablon, prompt ve tag kataloğu</CardTitle>
          </CardHeader>
          <CardContent className="grid gap-3 text-xs">
            <div>
              <b>Şablonlar</b>
              <p className="mt-1 text-muted-foreground">
                {props.templates.length
                  ? props.templates
                      .map((item) =>
                        String(item.title ?? item.name ?? "Şablon"),
                      )
                      .join(", ")
                  : "Şablon bulunamadı."}
              </p>
            </div>
            <div>
              <b>Prompt anahtarları</b>
              <p className="mt-1 text-muted-foreground">
                {Object.keys(props.prompts).join(", ") || "Prompt bulunamadı."}
              </p>
            </div>
            <div>
              <b>Kullanılabilir tag’ler</b>
              <p className="mt-1 text-muted-foreground">
                {props.availableTags
                  .map((item) => `${item.tag} (${item.count})`)
                  .join(", ") || "Tag bulunamadı."}
              </p>
            </div>
          </CardContent>
        </Card>
        <Card>
          <CardHeader>
            <CardTitle>Input form ve debug</CardTitle>
          </CardHeader>
          <CardContent className="grid gap-3">
            <Field label="DSL component id">
              <Input
                value={componentId}
                onChange={(event) => setComponentId(event.target.value)}
              />
            </Field>
            <Field label='Debug parametreleri (ör. {"sys.query":"Merhaba"})'>
              <Textarea
                value={debugText}
                onChange={(event) => setDebugText(event.target.value)}
              />
            </Field>
            <div className="flex gap-2">
              <Button
                variant="outline"
                disabled={props.disabled || !componentId}
                onClick={() => props.onInputForm(componentId)}
              >
                Input form
              </Button>
              <Button
                disabled={props.disabled || !componentId}
                onClick={() => {
                  try {
                    const raw = parseObject(debugText, "Debug parametreleri");
                    setValidationError("");
                    props.onDebug(
                      componentId,
                      Object.fromEntries(
                        Object.entries(raw).map(([key, value]) => [
                          key,
                          { value },
                        ]),
                      ),
                    );
                  } catch (caught) {
                    setValidationError(messageOf(caught));
                  }
                }}
              >
                Debug
              </Button>
            </div>
            {validationError && (
              <p className="text-xs text-destructive">{validationError}</p>
            )}
            <Output value={props.output} />
          </CardContent>
        </Card>
      </div>
    </div>
  );
}

function RuntimePanel(props: {
  sessions: PlatformAgentSession[];
  disabled: boolean;
  output: unknown;
  onRun: (input: string, sessionId?: string) => void;
  onCompletion: (input: string, sessionId?: string) => void;
  onCancel: () => void;
  onSessionCancel: (id: string) => void;
  onRerun: (documentId: string, componentId: string) => void;
  onLogs: (id: string) => void;
}) {
  const [input, setInput] = useState("");
  const [sessionId, setSessionId] = useState("");
  const [documentId, setDocumentId] = useState("");
  const [componentId, setComponentId] = useState("");
  const [messageId, setMessageId] = useState("");
  return (
    <div className="grid gap-4 xl:grid-cols-2">
      <Card>
        <CardHeader>
          <CardTitle>Run ve completion stream</CardTitle>
        </CardHeader>
        <CardContent className="grid gap-3">
          <Field label="Oturum">
            <select
              className="h-9 rounded-xl border bg-background px-3"
              value={sessionId}
              onChange={(event) => setSessionId(event.target.value)}
            >
              <option value="">Yeni / otomatik</option>
              {props.sessions.map((item) => (
                <option key={item.id} value={item.id}>
                  {item.name || item.id}
                </option>
              ))}
            </select>
          </Field>
          <Field label="Kullanıcı girdisi">
            <Textarea
              value={input}
              onChange={(event) => setInput(event.target.value)}
            />
          </Field>
          <div className="flex flex-wrap gap-2">
            <Button
              disabled={props.disabled || !input.trim()}
              onClick={() => props.onRun(input, sessionId || undefined)}
            >
              Run
            </Button>
            <Button
              variant="secondary"
              disabled={props.disabled || !input.trim()}
              onClick={() => props.onCompletion(input, sessionId || undefined)}
            >
              Completion
            </Button>
            <Button
              variant="destructive"
              disabled={props.disabled}
              onClick={props.onCancel}
            >
              Aktif run’ı iptal et
            </Button>
            {sessionId && (
              <Button
                variant="outline"
                disabled={props.disabled}
                onClick={() => props.onSessionCancel(sessionId)}
              >
                Oturum görevini iptal et
              </Button>
            )}
          </div>
          <Output value={props.output} />
        </CardContent>
      </Card>
      <Card>
        <CardHeader>
          <CardTitle>İleri düzey çalışma aksiyonları</CardTitle>
        </CardHeader>
        <CardContent className="grid gap-3">
          <p className="text-xs text-muted-foreground">
            Rerun, normal agent çalıştırmasını değil belge/dataflow component
            yeniden çalıştırma sözleşmesini kullanır.
          </p>
          <Field label="Belge id">
            <Input
              value={documentId}
              onChange={(event) => setDocumentId(event.target.value)}
            />
          </Field>
          <Field label="Component id">
            <Input
              value={componentId}
              onChange={(event) => setComponentId(event.target.value)}
            />
          </Field>
          <Button
            disabled={props.disabled || !documentId || !componentId}
            onClick={() => props.onRerun(documentId, componentId)}
          >
            Component’i rerun et
          </Button>
          <Field label="Message id">
            <Input
              value={messageId}
              onChange={(event) => setMessageId(event.target.value)}
            />
          </Field>
          <Button
            variant="outline"
            disabled={props.disabled || !messageId}
            onClick={() => props.onLogs(messageId)}
          >
            Çalışma logunu getir
          </Button>
        </CardContent>
      </Card>
    </div>
  );
}

function SessionsPanel(props: {
  sessions: PlatformAgentSession[];
  disabled: boolean;
  output: unknown;
  onCreate: (name: string) => void;
  onInspect: (id: string) => void;
  onDelete: (id: string) => void;
  onBulkDelete: (ids: string[], all: boolean) => void;
}) {
  const [name, setName] = useState("");
  const [selected, setSelected] = useState<string[]>([]);
  return (
    <Card>
      <CardHeader>
        <CardTitle>Oturum yönetimi</CardTitle>
      </CardHeader>
      <CardContent className="grid gap-4">
        <div className="flex gap-2">
          <Input
            placeholder="Oturum adı"
            value={name}
            onChange={(event) => setName(event.target.value)}
          />
          <Button
            disabled={props.disabled || !name.trim()}
            onClick={() => {
              props.onCreate(name);
              setName("");
            }}
          >
            Oluştur
          </Button>
        </div>
        {props.sessions.length === 0 ? (
          <p className="text-muted-foreground">Henüz oturum yok.</p>
        ) : (
          <div className="grid gap-2">
            {props.sessions.map((item) => (
              <div
                key={item.id}
                className="flex flex-wrap items-center gap-2 rounded-xl border p-3"
              >
                <input
                  aria-label={`${item.name || item.id} seç`}
                  type="checkbox"
                  checked={selected.includes(item.id)}
                  onChange={(event) =>
                    setSelected((current) =>
                      event.target.checked
                        ? [...current, item.id]
                        : current.filter((id) => id !== item.id),
                    )
                  }
                />
                <span className="mr-auto">{item.name || item.id}</span>
                <Button
                  size="sm"
                  variant="outline"
                  onClick={() => props.onInspect(item.id)}
                >
                  Aç
                </Button>
                <Button
                  size="sm"
                  variant="destructive"
                  onClick={() => props.onDelete(item.id)}
                >
                  Sil
                </Button>
              </div>
            ))}
          </div>
        )}
        <div className="flex gap-2">
          <Button
            variant="outline"
            disabled={props.disabled || selected.length === 0}
            onClick={() => props.onBulkDelete(selected, false)}
          >
            Seçilenleri sil
          </Button>
          <Button
            variant="destructive"
            disabled={props.disabled || props.sessions.length === 0}
            onClick={() => props.onBulkDelete([], true)}
          >
            Tümünü sil
          </Button>
        </div>
        <Output value={props.output} />
      </CardContent>
    </Card>
  );
}

function VersionsPanel(props: {
  versions: PlatformAgentVersion[];
  disabled: boolean;
  output: unknown;
  onInspect: (id: string) => void;
  onDelete: (id: string) => void;
}) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Sürümler</CardTitle>
      </CardHeader>
      <CardContent className="grid gap-3">
        {props.versions.length === 0 ? (
          <p className="text-muted-foreground">
            Henüz sürüm yok. Agent’ı yayınlayarak sürüm oluşturun.
          </p>
        ) : (
          props.versions.map((item) => (
            <div
              key={item.id}
              className="flex items-center gap-2 rounded-xl border p-3"
            >
              <span className="mr-auto truncate">{item.title || item.id}</span>
              {item.release && <Badge>Yayında</Badge>}
              <Button
                size="sm"
                variant="outline"
                disabled={props.disabled}
                onClick={() => props.onInspect(item.id)}
              >
                İncele
              </Button>
              <Button
                size="sm"
                variant="destructive"
                disabled={props.disabled}
                onClick={() => props.onDelete(item.id)}
              >
                Sil
              </Button>
            </div>
          ))
        )}
        <Output value={props.output} />
      </CardContent>
    </Card>
  );
}

function IntegrationsPanel(props: {
  agentId: string;
  mcpServers: PlatformMcpServer[];
  pluginTools: PlatformPluginTool[];
  disabled: boolean;
  output: unknown;
  run: (
    name: string,
    task: (signal: AbortSignal) => Promise<unknown>,
  ) => Promise<unknown>;
  refreshMcp: (signal: AbortSignal) => Promise<void>;
}) {
  const [db, setDb] = useState({
    db_type: "mysql",
    database: "",
    username: "",
    host: "",
    port: "3306",
    password: "",
  });
  const [webhookText, setWebhookText] = useState("{}");
  const [webhookMethod, setWebhookMethod] = useState<
    "GET" | "POST" | "PUT" | "PATCH" | "DELETE" | "HEAD"
  >("POST");
  const [attachment, setAttachment] = useState({
    attachmentId: "",
    filename: "",
    ext: "",
    mimeType: "application/octet-stream",
    fileId: "",
  });
  const [mcp, setMcp] = useState({
    id: "",
    name: "",
    url: "",
    serverType: "sse",
    description: "",
    variables: "{}",
    headers: "{}",
  });
  const fileInput = useRef<HTMLInputElement | null>(null);
  const previewUrl = useRef<string | null>(null);
  useEffect(
    () => () => {
      if (previewUrl.current) URL.revokeObjectURL(previewUrl.current);
    },
    [],
  );
  const mcpPayload = () => ({
    name: mcp.name,
    url: mcp.url,
    server_type: mcp.serverType,
    description: mcp.description,
    variables: parseObject(mcp.variables, "MCP variables"),
    headers: parseObject(mcp.headers, "MCP headers"),
    timeout: 10,
  });
  return (
    <div className="grid gap-4 xl:grid-cols-2">
      <Card>
        <CardHeader>
          <CardTitle>Veritabanı ve dosyalar</CardTitle>
        </CardHeader>
        <CardContent className="grid gap-3">
          <div className="grid grid-cols-2 gap-3">
            <Field label="DB türü">
              <Input
                value={db.db_type}
                onChange={(e) => setDb({ ...db, db_type: e.target.value })}
              />
            </Field>
            <Field label="Host">
              <Input
                value={db.host}
                onChange={(e) => setDb({ ...db, host: e.target.value })}
              />
            </Field>
            <Field label="Port">
              <Input
                inputMode="numeric"
                value={db.port}
                onChange={(e) => setDb({ ...db, port: e.target.value })}
              />
            </Field>
            <Field label="Database">
              <Input
                value={db.database}
                onChange={(e) => setDb({ ...db, database: e.target.value })}
              />
            </Field>
            <Field label="Kullanıcı">
              <Input
                value={db.username}
                onChange={(e) => setDb({ ...db, username: e.target.value })}
              />
            </Field>
            <Field label="Parola (saklanmaz)">
              <Input
                type="password"
                autoComplete="new-password"
                value={db.password}
                onChange={(e) => setDb({ ...db, password: e.target.value })}
              />
            </Field>
          </div>
          <Button
            disabled={props.disabled || !db.host || !db.database}
            onClick={() =>
              props.run("db-test", (signal) =>
                testAgentDatabaseConnection(
                  { ...db, port: Number(db.port) },
                  signal,
                ),
              )
            }
          >
            Bağlantıyı test et
          </Button>
          <hr />
          <Field
            label={`Dosya yükle (en çok ${PLATFORM_AGENT_FILE_MAX_BYTES / 1024 / 1024} MB)`}
          >
            <Input ref={fileInput} type="file" multiple />
          </Field>
          <Button
            variant="outline"
            disabled={props.disabled}
            onClick={() =>
              props.run("upload", async (signal) => {
                const files = Array.from(fileInput.current?.files ?? []);
                if (!files.length) throw new Error("En az bir dosya seçin.");
                if (
                  files.some(
                    (file) => file.size > PLATFORM_AGENT_FILE_MAX_BYTES,
                  )
                )
                  throw new Error("Dosya 64 MB sınırını aşıyor.");
                return uploadAgentFiles(props.agentId, files, signal);
              })
            }
          >
            Yükle
          </Button>
          <Field label="Yüklenen dosya id">
            <Input
              value={attachment.fileId}
              onChange={(e) =>
                setAttachment({ ...attachment, fileId: e.target.value })
              }
            />
          </Field>
          <Button
            variant="outline"
            disabled={props.disabled || !attachment.fileId}
            onClick={() =>
              props.run("file-download", async (signal) => {
                const blob = await downloadAgentFile(attachment.fileId, signal);
                downloadBlob(blob, attachment.filename || "agent-file");
              })
            }
          >
            Dosyayı indir
          </Button>
          <div className="grid grid-cols-3 gap-2">
            <Input
              placeholder="attachment id"
              value={attachment.attachmentId}
              onChange={(e) =>
                setAttachment({ ...attachment, attachmentId: e.target.value })
              }
            />
            <Input
              placeholder="filename"
              value={attachment.filename}
              onChange={(e) =>
                setAttachment({ ...attachment, filename: e.target.value })
              }
            />
            <Input
              placeholder="ext"
              value={attachment.ext}
              onChange={(e) =>
                setAttachment({ ...attachment, ext: e.target.value })
              }
            />
            <Input
              placeholder="mime type"
              value={attachment.mimeType}
              onChange={(e) =>
                setAttachment({ ...attachment, mimeType: e.target.value })
              }
            />
          </div>
          <div className="flex gap-2">
            <Button
              variant="outline"
              disabled={props.disabled || !attachment.attachmentId}
              onClick={() =>
                props.run("attachment-preview", async (signal) => {
                  const blob = await previewAgentAttachment(
                    attachment.attachmentId,
                    {
                      ext: attachment.ext,
                      mimeType: attachment.mimeType,
                      filename: attachment.filename,
                    },
                    signal,
                  );
                  if (previewUrl.current)
                    URL.revokeObjectURL(previewUrl.current);
                  previewUrl.current = URL.createObjectURL(blob);
                  window.open(
                    previewUrl.current,
                    "_blank",
                    "noopener,noreferrer",
                  );
                })
              }
            >
              Önizle
            </Button>
            <Button
              variant="outline"
              disabled={props.disabled || !attachment.attachmentId}
              onClick={() =>
                props.run("attachment-download", async (signal) => {
                  const blob = await downloadAgentAttachment(
                    attachment.attachmentId,
                    {
                      ext: attachment.ext,
                      mimeType: attachment.mimeType,
                      filename: attachment.filename,
                    },
                    signal,
                  );
                  downloadBlob(blob, attachment.filename);
                })
              }
            >
              Eki indir
            </Button>
          </div>
        </CardContent>
      </Card>
      <Card>
        <CardHeader>
          <CardTitle>Webhook</CardTitle>
        </CardHeader>
        <CardContent className="grid gap-3">
          <p className="text-xs text-muted-foreground">
            Üretim webhook adresi dış istemciler için callback sözleşmesidir.
            Burada yalnızca kimlik doğrulamalı test ve log aksiyonları
            çalıştırılır.
          </p>
          <Textarea
            value={webhookText}
            onChange={(e) => setWebhookText(e.target.value)}
          />
          <Field label="Test HTTP method">
            <select
              className="h-9 rounded-xl border bg-background px-3"
              value={webhookMethod}
              onChange={(event) =>
                setWebhookMethod(event.target.value as typeof webhookMethod)
              }
            >
              {(["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD"] as const).map(
                (method) => (
                  <option key={method}>{method}</option>
                ),
              )}
            </select>
          </Field>
          <div className="flex gap-2">
            <Button
              disabled={props.disabled}
              onClick={() =>
                props.run("webhook-test", (signal) =>
                  testAgentWebhook(
                    props.agentId,
                    JSON.parse(webhookText),
                    webhookMethod,
                    signal,
                  ),
                )
              }
            >
              Webhook test
            </Button>
            <Button
              variant="outline"
              disabled={props.disabled}
              onClick={() =>
                props.run("webhook-logs", (signal) =>
                  getAgentWebhookLogs(props.agentId, signal),
                )
              }
            >
              Logları getir
            </Button>
          </div>
        </CardContent>
      </Card>
      <Card>
        <CardHeader>
          <CardTitle>MCP sunucuları</CardTitle>
        </CardHeader>
        <CardContent className="grid gap-3">
          <Field label="Kayıt">
            <select
              className="h-9 rounded-xl border bg-background px-3"
              value={mcp.id}
              onChange={(event) => {
                const id = event.target.value;
                if (!id) {
                  setMcp({ ...mcp, id: "" });
                  return;
                }
                void props.run("mcp-detail", async (signal) => {
                  const found = await getMcpServer(id, signal);
                  setMcp({
                    id: found.id,
                    name: found.name,
                    url: found.url,
                    serverType: found.server_type,
                    description: found.description ?? "",
                    variables: pretty(found.variables ?? {}),
                    headers: "{}",
                  });
                  return redactAgentSecrets(found);
                });
              }}
            >
              <option value="">Yeni MCP sunucusu</option>
              {props.mcpServers.map((item) => (
                <option key={item.id} value={item.id}>
                  {item.name}
                </option>
              ))}
            </select>
          </Field>
          <Input
            placeholder="Ad"
            value={mcp.name}
            onChange={(e) => setMcp({ ...mcp, name: e.target.value })}
          />
          <Input
            placeholder="https://…"
            value={mcp.url}
            onChange={(e) => setMcp({ ...mcp, url: e.target.value })}
          />
          <Input
            placeholder="sse veya streamable-http"
            value={mcp.serverType}
            onChange={(e) => setMcp({ ...mcp, serverType: e.target.value })}
          />
          <Textarea
            aria-label="MCP variables"
            value={mcp.variables}
            onChange={(e) => setMcp({ ...mcp, variables: e.target.value })}
          />
          <Textarea
            aria-label="MCP headers (gizli değerler saklanmaz)"
            value={mcp.headers}
            onChange={(e) => setMcp({ ...mcp, headers: e.target.value })}
          />
          <div className="flex flex-wrap gap-2">
            <Button
              disabled={props.disabled || !mcp.name || !mcp.url}
              onClick={() =>
                props.run("mcp-save", async (signal) => {
                  const result = mcp.id
                    ? await updateMcpServer(mcp.id, mcpPayload(), signal)
                    : await createMcpServer(mcpPayload(), signal);
                  await props.refreshMcp(signal);
                  return result;
                })
              }
            >
              {mcp.id ? "Güncelle" : "Oluştur"}
            </Button>
            <Button
              variant="outline"
              disabled={props.disabled || !mcp.id}
              onClick={() =>
                props.run("mcp-test", (signal) =>
                  testMcpServer(mcp.id, mcpPayload(), signal),
                )
              }
            >
              Test
            </Button>
            <Button
              variant="outline"
              disabled={props.disabled || !mcp.name || !mcp.url}
              onClick={() =>
                props.run("mcp-import", async (signal) => {
                  const result = await importMcpServers(
                    {
                      [mcp.name]: {
                        url: mcp.url,
                        type: mcp.serverType,
                        variables: parseObject(mcp.variables, "MCP variables"),
                        headers: parseObject(mcp.headers, "MCP headers"),
                      },
                    },
                    10,
                    signal,
                  );
                  await props.refreshMcp(signal);
                  return result;
                })
              }
            >
              Import
            </Button>
            <Button
              variant="destructive"
              disabled={props.disabled || !mcp.id}
              onClick={() =>
                props.run("mcp-delete", async (signal) => {
                  if (!window.confirm("MCP sunucusu silinsin mi?")) return;
                  await deleteMcpServer(mcp.id, signal);
                  setMcp({ ...mcp, id: "" });
                  await props.refreshMcp(signal);
                })
              }
            >
              Sil
            </Button>
          </div>
        </CardContent>
      </Card>
      <Card>
        <CardHeader>
          <CardTitle>Plugin tools</CardTitle>
        </CardHeader>
        <CardContent className="grid gap-2">
          {props.pluginTools.length === 0 ? (
            <p className="text-muted-foreground">Kayıtlı plugin tool yok.</p>
          ) : (
            props.pluginTools.map((tool) => (
              <div key={tool.name} className="rounded-xl border p-3">
                <b>{tool.displayName || tool.name}</b>
                <p className="text-xs text-muted-foreground">
                  {tool.displayDescription || tool.description}
                </p>
              </div>
            ))
          )}
          <Output value={props.output} />
        </CardContent>
      </Card>
    </div>
  );
}
