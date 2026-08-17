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
  CONNECTOR_OAUTH_MESSAGE,
  PLATFORM_CONNECTOR_SOURCES,
  PlatformApiError,
  clearPendingConnectorOAuth,
  connectorOAuthRedirectUri,
  createPlatformCommit,
  createPlatformConnector,
  createPlatformFolder,
  deletePlatformConnector,
  deletePlatformFiles,
  diffPlatformCommits,
  downloadPlatformFile,
  getPlatformCommit,
  getPlatformCommitFileContent,
  getPlatformCommitTree,
  getPlatformConnector,
  getPlatformFileAncestors,
  getPlatformFileParent,
  getPlatformUncommittedChanges,
  isPlatformConnectorsEnabled,
  linkPlatformFilesToDatasets,
  linkAndRebuildPlatformConnector,
  listPlatformCommitFiles,
  listPlatformCommits,
  listPlatformConnectorLogs,
  listPlatformConnectors,
  listPlatformDatasets,
  listPlatformFiles,
  listPlatformFileVersions,
  movePlatformFiles,
  openConnectorOAuthWindow,
  readPendingConnectorOAuth,
  redactConnectorSecrets,
  savePendingConnectorOAuth,
  startBoxConnectorOAuth,
  startGoogleConnectorOAuth,
  testPlatformConnector,
  updatePlatformConnector,
  uploadPlatformFiles,
  waitForConnectorOAuthResult,
  type ConnectorOAuthMessage,
  type PlatformCommitScope,
  type PlatformConnector,
  type PlatformConnectorLog,
  type PlatformConnectorOAuthSource,
  type PlatformDatasetDto,
  type PlatformFileChange,
  type PlatformFileCommit,
  type PlatformFileDiff,
  type PlatformWorkspaceFile,
} from "@/integrations/platform-backend";
import { useVirtualizer } from "@tanstack/react-virtual";
import {
  type ReactNode,
  useCallback,
  useEffect,
  useRef,
  useState,
} from "react";

const MAX_COMMIT_TEXT_BYTES = 1_048_576;

function errorMessage(error: unknown): string {
  if (error instanceof PlatformApiError) {
    if (error.httpStatus === 401) return "Oturum süresi doldu. Yeniden giriş yapın.";
    if (error.httpStatus === 403) return "Bu işlem için yetkiniz yok.";
    if (error.isTimeout) return "İstek zaman aşımına uğradı. Yeniden deneyin.";
    if (error.isAbort) return "İşlem iptal edildi.";
    return error.message;
  }
  return error instanceof Error ? error.message : "Beklenmeyen bir hata oluştu.";
}

function objectJson(value: string, label: string): Record<string, unknown> {
  const parsed = JSON.parse(value) as unknown;
  if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
    throw new Error(`${label} bir JSON nesnesi olmalıdır.`);
  }
  return parsed as Record<string, unknown>;
}

function datasetId(value: PlatformDatasetDto): string {
  return typeof value.id === "string" ? value.id : "";
}

function datasetName(value: PlatformDatasetDto): string {
  return typeof value.name === "string" ? value.name : datasetId(value);
}

function pretty(value: unknown): string {
  return JSON.stringify(value, null, 2);
}

function formatBytes(value: number): string {
  if (value < 1024) return `${value} B`;
  if (value < 1_048_576) return `${(value / 1024).toFixed(1)} KB`;
  return `${(value / 1_048_576).toFixed(1)} MB`;
}

export function coalesceCommitMetadataChanges(
  changes: PlatformFileDiff[],
): PlatformFileDiff[] {
  const result: PlatformFileDiff[] = [];
  const byFile = new Map<string, number>();

  for (const change of changes) {
    const existingIndex = byFile.get(change.fileId);
    const existing = existingIndex === undefined ? undefined : result[existingIndex];
    const isRenameMovePair =
      existing &&
      new Set([existing.operation, change.operation]).size === 2 &&
      [existing.operation, change.operation].every((operation) =>
        operation === "rename" || operation === "move",
      );

    if (!existing || !isRenameMovePair || existingIndex === undefined) {
      byFile.set(change.fileId, result.length);
      result.push(change);
      continue;
    }

    const rename = existing.operation === "rename" ? existing : change;
    const move = existing.operation === "move" ? existing : change;
    result[existingIndex] = {
      ...move,
      operation: "move",
      fileName: rename.fileName || move.fileName,
      oldName: rename.oldName,
      newName: rename.newName,
      oldParentId: move.oldParentId ?? rename.oldParentId,
      newParentId: move.newParentId ?? rename.newParentId,
    };
  }

  return result;
}

function VirtualRows<T>({
  items,
  render,
  empty,
}: {
  items: T[];
  render: (item: T, index: number) => ReactNode;
  empty: string;
}) {
  const parentRef = useRef<HTMLDivElement>(null);
  const virtualizer = useVirtualizer({
    count: items.length,
    getScrollElement: () => parentRef.current,
    estimateSize: () => 52,
    overscan: 8,
  });
  if (!items.length)
    return <p className="py-8 text-center text-muted-foreground text-sm">{empty}</p>;
  return (
    <div ref={parentRef} className="max-h-[420px] overflow-auto rounded-xl border">
      <div className="relative" style={{ height: virtualizer.getTotalSize() }}>
        {virtualizer.getVirtualItems().map((row) => (
          <div
            key={row.key}
            className="absolute top-0 left-0 w-full"
            style={{ transform: `translateY(${row.start}px)` }}
          >
            {render(items[row.index], row.index)}
          </div>
        ))}
      </div>
    </div>
  );
}

function Status({ error, notice }: { error: string; notice: string }) {
  if (!error && !notice) return null;
  return (
    <Alert variant={error ? "destructive" : "default"}>
      <AlertTitle>{error ? "İşlem tamamlanamadı" : "İşlem tamamlandı"}</AlertTitle>
      <AlertDescription>{error || notice}</AlertDescription>
    </Alert>
  );
}

function FileManagerTab({ datasets }: { datasets: PlatformDatasetDto[] }) {
  const [files, setFiles] = useState<PlatformWorkspaceFile[]>([]);
  const [parentId, setParentId] = useState("");
  const [breadcrumbs, setBreadcrumbs] = useState<PlatformWorkspaceFile[]>([]);
  const [keywords, setKeywords] = useState("");
  const [selected, setSelected] = useState<string[]>([]);
  const [folderName, setFolderName] = useState("");
  const [moveTarget, setMoveTarget] = useState("");
  const [rename, setRename] = useState("");
  const [dataset, setDataset] = useState("");
  const [linkMode, setLinkMode] = useState<"add" | "replace">("add");
  const [output, setOutput] = useState<unknown>(null);
  const [loading, setLoading] = useState(true);
  const [action, setAction] = useState("");
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const controllerRef = useRef<AbortController | null>(null);

  const refresh = useCallback(async (signal?: AbortSignal) => {
    const page = await listPlatformFiles({ parentId, keywords }, signal);
    setFiles(page.files);
    if (parentId) setBreadcrumbs(await getPlatformFileAncestors(parentId, signal));
    else setBreadcrumbs([]);
  }, [keywords, parentId]);

  useEffect(() => {
    const controller = new AbortController();
    setLoading(true);
    setError("");
    void refresh(controller.signal)
      .catch((value) => {
        if (!(value instanceof PlatformApiError) || !value.isAbort) setError(errorMessage(value));
      })
      .finally(() => {
        if (!controller.signal.aborted) setLoading(false);
      });
    return () => controller.abort();
  }, [refresh]);

  const run = async (name: string, task: (signal: AbortSignal) => Promise<unknown>) => {
    if (action) return;
    const controller = new AbortController();
    controllerRef.current = controller;
    setAction(name);
    setError("");
    setNotice("");
    try {
      const result = await task(controller.signal);
      setOutput(result);
      setNotice("Değişiklik kaydedildi.");
      await refresh(controller.signal);
      setSelected([]);
    } catch (value) {
      setError(errorMessage(value));
    } finally {
      controllerRef.current = null;
      setAction("");
    }
  };

  const download = async (file: PlatformWorkspaceFile) => {
    await run("download", async (signal) => {
      const blob = await downloadPlatformFile(file.id, signal);
      const url = URL.createObjectURL(blob);
      const anchor = document.createElement("a");
      anchor.href = url;
      anchor.download = file.name;
      anchor.click();
      URL.revokeObjectURL(url);
    });
  };

  return (
    <div className="grid gap-4">
      <Status error={error} notice={notice} />
      <Card>
        <CardHeader><CardTitle>Dosya alanı</CardTitle></CardHeader>
        <CardContent className="grid gap-4">
          <div className="flex flex-wrap gap-2">
            <Button variant="outline" onClick={() => setParentId("")}>Kök</Button>
            {breadcrumbs.map((item) => (
              <Button key={item.id} variant="ghost" onClick={() => setParentId(item.id)}>{item.name}</Button>
            ))}
          </div>
          <div className="grid gap-2 md:grid-cols-[1fr_auto]">
            <Input value={keywords} onChange={(event) => setKeywords(event.target.value)} placeholder="Dosya veya klasör ara" />
            <Button variant="outline" onClick={() => void refresh()}>Yenile</Button>
          </div>
          {loading ? <div className="flex items-center gap-2"><Spinner className="size-4" />Yükleniyor</div> : (
            <VirtualRows
              items={files}
              empty="Bu klasör boş."
              render={(file) => (
                <div className="flex h-13 items-center gap-3 border-b px-3 text-sm last:border-0">
                  <input
                    type="checkbox"
                    aria-label={`${file.name} seç`}
                    checked={selected.includes(file.id)}
                    onChange={(event) => setSelected((value) => event.target.checked ? [...value, file.id] : value.filter((id) => id !== file.id))}
                  />
                  <button type="button" className="min-w-0 flex-1 truncate text-left font-medium" onClick={() => file.isFolder ? setParentId(file.id) : void download(file)}>
                    {file.isFolder ? "Klasör · " : ""}{file.name}
                  </button>
                  <span className="text-muted-foreground">{file.isFolder ? "—" : formatBytes(file.size)}</span>
                  {!file.isFolder ? <Button size="sm" variant="ghost" onClick={() => void run("versions", (signal) => listPlatformFileVersions(file.id, signal))}>Sürümler</Button> : null}
                  <Button size="sm" variant="ghost" onClick={() => void run("parent", async (signal) => ({ parent: await getPlatformFileParent(file.id, signal), ancestors: await getPlatformFileAncestors(file.id, signal) }))}>Konum</Button>
                </div>
              )}
            />
          )}
        </CardContent>
      </Card>

      <div className="grid gap-4 xl:grid-cols-2">
        <Card><CardHeader><CardTitle>Oluştur ve yükle</CardTitle></CardHeader><CardContent className="grid gap-3">
          <Label htmlFor="folder-name">Klasör adı</Label>
          <div className="flex gap-2"><Input id="folder-name" value={folderName} onChange={(event) => setFolderName(event.target.value)} /><Button disabled={!folderName.trim() || Boolean(action)} onClick={() => void run("folder", async (signal) => { const result = await createPlatformFolder(folderName.trim(), parentId || undefined, signal); setFolderName(""); return result; })}>Klasör oluştur</Button></div>
          <Label htmlFor="file-upload">Dosyalar</Label>
          <Input id="file-upload" type="file" multiple disabled={Boolean(action)} onChange={(event) => { const next = Array.from(event.target.files ?? []); if (next.length) void run("upload", (signal) => uploadPlatformFiles(next, parentId || undefined, signal)); event.currentTarget.value = ""; }} />
        </CardContent></Card>
        <Card><CardHeader><CardTitle>Seçili dosyalar</CardTitle></CardHeader><CardContent className="grid gap-3">
          <p className="text-muted-foreground text-sm">{selected.length} öğe seçildi.</p>
          <div className="grid gap-2 sm:grid-cols-2"><Input value={moveTarget} onChange={(event) => setMoveTarget(event.target.value)} placeholder="Hedef klasör kimliği (boş = kök)" /><Input value={rename} onChange={(event) => setRename(event.target.value)} placeholder="Tek öğe için yeni ad" /></div>
          <div className="flex flex-wrap gap-2"><Button variant="outline" disabled={!selected.length || Boolean(action)} onClick={() => void run("move", (signal) => movePlatformFiles(selected, { destinationFolderId: moveTarget || undefined, newName: selected.length === 1 && rename ? rename : undefined }, signal))}>Taşı / yeniden adlandır</Button><Button variant="destructive" disabled={!selected.length || Boolean(action)} onClick={() => window.confirm("Seçili öğeler silinsin mi?") && void run("delete", (signal) => deletePlatformFiles(selected, signal))}>Sil</Button></div>
          <div className="grid gap-2 sm:grid-cols-[1fr_150px_auto]"><select className="h-9 rounded-md border bg-background px-3" value={dataset} onChange={(event) => setDataset(event.target.value)}><option value="">Veri kümesi seçin</option>{datasets.map((item) => <option key={datasetId(item)} value={datasetId(item)}>{datasetName(item)}</option>)}</select><select className="h-9 rounded-md border bg-background px-3" value={linkMode} onChange={(event) => setLinkMode(event.target.value as "add" | "replace")}><option value="add">Ekle</option><option value="replace">Değiştir</option></select><Button disabled={!selected.length || !dataset || Boolean(action)} onClick={() => void run("link", (signal) => linkPlatformFilesToDatasets(selected, [dataset], linkMode, signal))}>Veri kümesine bağla</Button></div>
        </CardContent></Card>
      </div>
      {action ? <div className="flex items-center gap-2"><Spinner className="size-4" /><span>{action} çalışıyor</span><Button size="sm" variant="outline" onClick={() => controllerRef.current?.abort()}>İptal</Button></div> : null}
      {output !== null ? <pre className="max-h-64 overflow-auto rounded-xl bg-muted p-3 text-xs">{pretty(redactConnectorSecrets(output))}</pre> : null}
    </div>
  );
}

function ConnectorsTab({ datasets }: { datasets: PlatformDatasetDto[] }) {
  const [connectors, setConnectors] = useState<PlatformConnector[]>([]);
  const [selected, setSelected] = useState("");
  const [name, setName] = useState("");
  const [source, setSource] = useState("rest_api");
  const [config, setConfig] = useState("{}");
  const [configDirty, setConfigDirty] = useState(false);
  const [refreshFrequency, setRefreshFrequency] = useState("0");
  const [pruneFrequency, setPruneFrequency] = useState("0");
  const [timeoutSeconds, setTimeoutSeconds] = useState("30");
  const [dataset, setDataset] = useState("");
  const [logs, setLogs] = useState<PlatformConnectorLog[]>([]);
  const [logPage, setLogPage] = useState(1);
  const [action, setAction] = useState("");
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [oauthSource, setOauthSource] = useState<PlatformConnectorOAuthSource>("google-drive");
  const [boxClientId, setBoxClientId] = useState("");
  const [boxSecret, setBoxSecret] = useState("");
  const oauthCredentialRef = useRef<Record<string, unknown> | null>(null);
  const oauthResultRef = useRef<unknown>(null);
  const controllerRef = useRef<AbortController | null>(null);

  const refresh = useCallback(async (signal?: AbortSignal) => {
    const items = await listPlatformConnectors(signal);
    setConnectors(items);
    setSelected((current) => current || items[0]?.id || "");
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    void refresh(controller.signal).catch((value) => {
      if (!(value instanceof PlatformApiError) || !value.isAbort) setError(errorMessage(value));
    });
    return () => controller.abort();
  }, [refresh]);

  useEffect(() => {
    if (!selected) return;
    const controller = new AbortController();
    void getPlatformConnector(selected, controller.signal)
      .then((detail) => {
        setName(detail.name);
        setSource(detail.source);
        setRefreshFrequency(String(detail.refreshFrequency));
        setPruneFrequency(String(detail.pruneFrequency));
        setTimeoutSeconds(String(detail.timeoutSeconds));
        setConfig("{}");
        setConfigDirty(false);
      })
      .catch((value) => {
        if (!(value instanceof PlatformApiError) || !value.isAbort) setError(errorMessage(value));
      });
    return () => controller.abort();
  }, [selected]);

  const run = useCallback(async (name: string, task: (signal: AbortSignal) => Promise<unknown>) => {
    if (action) return;
    const controller = new AbortController();
    controllerRef.current = controller;
    setAction(name); setError(""); setNotice("");
    try {
      await task(controller.signal);
      setNotice("İşlem tamamlandı.");
      await refresh(controller.signal);
    } catch (value) { setError(errorMessage(value)); }
    finally { controllerRef.current = null; setAction(""); }
  }, [action, refresh]);

  const collectOAuthResult = useCallback((nextSource: PlatformConnectorOAuthSource, flowId: string) => {
    void run("oauth-result", async (signal) => {
      oauthResultRef.current = await waitForConnectorOAuthResult(nextSource, flowId, { signal });
      clearPendingConnectorOAuth(flowId);
      setNotice("OAuth kimlik bilgisi alındı; kaydedilene kadar yalnızca bellekte tutuluyor.");
    });
  }, [run]);

  useEffect(() => {
    const receive = (event: MessageEvent<ConnectorOAuthMessage>) => {
      if (event.origin !== window.location.origin || event.data?.type !== CONNECTOR_OAUTH_MESSAGE) return;
      const pending = readPendingConnectorOAuth();
      if (!pending || pending.flowId !== event.data.flowId || pending.source !== event.data.source) return;
      if (event.data.status === "success") collectOAuthResult(event.data.source, event.data.flowId);
      else { clearPendingConnectorOAuth(event.data.flowId); setError("OAuth yetkilendirmesi tamamlanamadı."); }
    };
    window.addEventListener("message", receive);
    const query = new URLSearchParams(window.location.search);
    const flowId = query.get("oauth_flow");
    const returnedSource = query.get("oauth_source") as PlatformConnectorOAuthSource | null;
    if (flowId && returnedSource && query.get("oauth_status") === "success") collectOAuthResult(returnedSource, flowId);
    return () => window.removeEventListener("message", receive);
  }, [collectOAuthResult]);

  const startOAuth = () => void run("oauth-start", async (signal) => {
    const redirectUri = connectorOAuthRedirectUri(oauthSource);
    const started = oauthSource === "box"
      ? await startBoxConnectorOAuth(boxClientId, boxSecret, redirectUri, signal)
      : await startGoogleConnectorOAuth(oauthSource, oauthCredentialRef.current ?? {}, redirectUri, signal);
    savePendingConnectorOAuth({ source: oauthSource, flowId: started.flowId, returnTo: "/files", startedAt: Date.now() });
    setSource(oauthSource === "google-drive" ? "google_drive" : oauthSource);
    const popup = openConnectorOAuthWindow(oauthSource, started.flowId, started.authorizationUrl);
    if (!popup) window.location.assign(started.authorizationUrl);
    boxSecretRefCleanup();
  });

  const boxSecretRefCleanup = () => setBoxSecret("");
  const create = () => void run("create", async (signal) => {
    const base = objectJson(config, "Yapılandırma");
    const credentials = oauthResultRef.current;
    const created = await createPlatformConnector({
      name: name.trim(),
      source: source as (typeof PLATFORM_CONNECTOR_SOURCES)[number],
      config: credentials === null ? base : { ...base, credentials },
      refreshFrequency: Number(refreshFrequency),
      pruneFrequency: Number(pruneFrequency),
      timeoutSeconds: Number(timeoutSeconds),
    }, signal);
    oauthResultRef.current = null;
    oauthCredentialRef.current = null;
    setConfigDirty(false);
    setSelected(created.id);
  });

  const loadLogs = (page = logPage) => selected ? void run("logs", async (signal) => {
    const result = await listPlatformConnectorLogs(selected, { page, pageSize: 20 }, signal);
    setLogs(result.logs); setLogPage(page);
  }) : undefined;

  return (
    <div className="grid gap-4">
      <Status error={error} notice={notice} />
      <div className="grid gap-4 xl:grid-cols-[minmax(260px,0.7fr)_1.3fr]">
        <Card><CardHeader><CardTitle>Connector’lar</CardTitle></CardHeader><CardContent>
          <VirtualRows items={connectors} empty="Henüz connector yok." render={(item) => <button type="button" onClick={() => setSelected(item.id)} className={`flex h-13 w-full items-center justify-between border-b px-3 text-left ${selected === item.id ? "bg-muted" : ""}`}><span><strong>{item.name}</strong><small className="block text-muted-foreground">{item.source}</small></span><Badge variant="outline">{item.status || "—"}</Badge></button>} />
        </CardContent></Card>
        <Card><CardHeader><CardTitle>Connector yapılandırması</CardTitle></CardHeader><CardContent className="grid gap-3">
          <div className="grid gap-3 sm:grid-cols-2"><div><Label htmlFor="connector-name">Ad</Label><Input id="connector-name" value={name} onChange={(event) => setName(event.target.value)} /></div><div><Label htmlFor="connector-source">Kaynak</Label><select id="connector-source" className="h-9 w-full rounded-md border bg-background px-3" value={source} onChange={(event) => setSource(event.target.value)}>{PLATFORM_CONNECTOR_SOURCES.map((item) => <option key={item}>{item}</option>)}</select></div></div>
          <Label htmlFor="connector-config">Yeni yapılandırma JSON’u (mevcut secret alanları ekranda geri gösterilmez)</Label><Textarea id="connector-config" className="min-h-32 font-mono" value={config} onChange={(event) => { setConfig(event.target.value); setConfigDirty(true); }} />
          <div className="grid gap-3 sm:grid-cols-3"><Input aria-label="Yenileme sıklığı" type="number" min="0" value={refreshFrequency} onChange={(event) => setRefreshFrequency(event.target.value)} /><Input aria-label="Budama sıklığı" type="number" min="0" value={pruneFrequency} onChange={(event) => setPruneFrequency(event.target.value)} /><Input aria-label="Zaman aşımı" type="number" min="1" value={timeoutSeconds} onChange={(event) => setTimeoutSeconds(event.target.value)} /></div>
          <div className="flex flex-wrap gap-2"><Button disabled={!name.trim() || Boolean(action)} onClick={create}>Oluştur</Button><Button variant="outline" disabled={!selected || Boolean(action)} onClick={() => void run("update", async (signal) => { const result = await updatePlatformConnector(selected, { ...(configDirty ? { config: objectJson(config, "Yapılandırma") } : {}), refreshFrequency: Number(refreshFrequency), pruneFrequency: Number(pruneFrequency), timeoutSeconds: Number(timeoutSeconds), reschedule: true }, signal); setConfigDirty(false); return result; })}>Güncelle</Button><Button variant="outline" disabled={!selected || Boolean(action)} onClick={() => void run("test", (signal) => testPlatformConnector(selected, signal))}>Bağlantıyı test et</Button><Button variant="destructive" disabled={!selected || Boolean(action)} onClick={() => window.confirm("Connector silinsin mi?") && void run("delete", async (signal) => { const result = await deletePlatformConnector(selected, signal); setSelected(""); return result; })}>Sil</Button></div>
          <div className="grid gap-2 sm:grid-cols-[1fr_auto]"><select className="h-9 rounded-md border bg-background px-3" value={dataset} onChange={(event) => setDataset(event.target.value)}><option value="">Veri kümesi seçin</option>{datasets.map((item) => <option key={datasetId(item)} value={datasetId(item)}>{datasetName(item)}</option>)}</select><Button disabled={!selected || !dataset || Boolean(action)} onClick={() => void run("rebuild", async (signal) => { const result = await linkAndRebuildPlatformConnector(selected, dataset, signal); const page = await listPlatformConnectorLogs(selected, { page: 1, pageSize: 20 }, signal); setLogs(page.logs); setLogPage(1); return result; })}>Bağla ve yeniden indeksle</Button></div>
        </CardContent></Card>
      </div>

      <Card><CardHeader><CardTitle>Google, Gmail ve Box OAuth</CardTitle></CardHeader><CardContent className="grid gap-3">
        <select className="h-9 rounded-md border bg-background px-3" value={oauthSource} onChange={(event) => setOauthSource(event.target.value as PlatformConnectorOAuthSource)}><option value="google-drive">Google Drive</option><option value="gmail">Gmail</option><option value="box">Box</option></select>
        {oauthSource === "box" ? <div className="grid gap-3 sm:grid-cols-2"><Input value={boxClientId} onChange={(event) => setBoxClientId(event.target.value)} placeholder="Box client ID" autoComplete="off" /><Input type="password" value={boxSecret} onChange={(event) => setBoxSecret(event.target.value)} placeholder="Box client secret" autoComplete="new-password" /></div> : <div><Label htmlFor="google-credentials">Google client credential JSON dosyası</Label><Input id="google-credentials" type="file" accept="application/json,.json" onChange={(event) => { const file = event.target.files?.[0]; if (!file) return; void file.text().then((text) => { oauthCredentialRef.current = objectJson(text, "Credential dosyası"); setNotice("Credential dosyası belleğe alındı; tarayıcı deposuna yazılmadı."); }).catch((value) => setError(errorMessage(value))); event.currentTarget.value = ""; }} /></div>}
        <Button className="w-fit" disabled={Boolean(action) || (oauthSource === "box" ? !boxClientId || !boxSecret : !oauthCredentialRef.current)} onClick={startOAuth}>Yetkilendirmeyi başlat</Button>
      </CardContent></Card>

      <Card><CardHeader><CardTitle>Çalışma günlükleri</CardTitle></CardHeader><CardContent className="grid gap-3"><div className="flex gap-2"><Button variant="outline" disabled={!selected || Boolean(action)} onClick={() => loadLogs(1)}>Günlükleri getir</Button><Button variant="ghost" disabled={logPage <= 1 || Boolean(action)} onClick={() => loadLogs(logPage - 1)}>Önceki</Button><Button variant="ghost" disabled={!logs.length || Boolean(action)} onClick={() => loadLogs(logPage + 1)}>Sonraki</Button></div><VirtualRows items={logs} empty="Günlük kaydı yok." render={(log) => <div className="flex h-13 items-center justify-between border-b px-3 text-sm"><span><strong>{log.taskType || "sync"}</strong><small className="block text-muted-foreground">{log.datasetName || log.datasetId}</small></span><span className="text-right"><Badge variant="outline">{log.status}</Badge><small className="block text-muted-foreground">{log.errorCount ? `${log.errorCount} hata` : `${log.totalDocuments} belge`}</small></span></div>} /></CardContent></Card>
      {action ? <div className="flex items-center gap-2"><Spinner className="size-4" />{action} çalışıyor<Button size="sm" variant="outline" onClick={() => controllerRef.current?.abort()}>İptal</Button></div> : null}
    </div>
  );
}

function CommitsTab({ datasets }: { datasets: PlatformDatasetDto[] }) {
  const [scope, setScope] = useState<PlatformCommitScope>("workspace");
  const [scopeId, setScopeId] = useState("");
  const [commits, setCommits] = useState<PlatformFileCommit[]>([]);
  const [changes, setChanges] = useState<PlatformFileDiff[]>([]);
  const [selectedCommit, setSelectedCommit] = useState("");
  const [compareFrom, setCompareFrom] = useState("");
  const [compareTo, setCompareTo] = useState("");
  const [message, setMessage] = useState("");
  const [output, setOutput] = useState<unknown>(null);
  const [action, setAction] = useState("");
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const controllerRef = useRef<AbortController | null>(null);

  const run = async (name: string, task: (signal: AbortSignal) => Promise<unknown>) => {
    if (action || !scopeId.trim()) return;
    const controller = new AbortController(); controllerRef.current = controller;
    setAction(name); setError(""); setNotice("");
    try { const result = await task(controller.signal); setOutput(result); setNotice("İşlem tamamlandı."); }
    catch (value) { setError(errorMessage(value)); }
    finally { controllerRef.current = null; setAction(""); }
  };

  const refresh = () => void run("refresh", async (signal) => {
    const [page, pending] = await Promise.all([listPlatformCommits(scope, scopeId.trim(), {}, signal), getPlatformUncommittedChanges(scope, scopeId.trim(), signal)]);
    setCommits(page.commits); setChanges(pending); return { total: page.total, pending };
  });

  const create = () => void run("commit", async (signal) => {
    const payload: PlatformFileChange[] = [];
    for (const change of coalesceCommitMetadataChanges(changes)) {
      const operation = change.operation as PlatformFileChange["operation"];
      const next: PlatformFileChange = { fileId: change.fileId, fileName: change.fileName, operation };
      if (operation === "add" || operation === "modify") {
        const blob = await downloadPlatformFile(change.fileId, signal);
        if (blob.size > MAX_COMMIT_TEXT_BYTES) throw new Error(`${change.fileName} commit sınırı olan 1 MB’tan büyük.`);
        if (blob.type && !blob.type.startsWith("text/") && !/json|xml|javascript|yaml/.test(blob.type)) throw new Error(`${change.fileName} metin dosyası değil; içerik commit’e eklenmedi.`);
        next.content = await blob.text();
      }
      if (operation === "rename" || (operation === "move" && change.oldName && change.newName)) {
        if (!change.oldName || !change.newName) {
          throw new Error(`${change.fileName} için yeniden adlandırma bilgisi eksik.`);
        }
        next.oldName = change.oldName;
        next.newName = change.newName;
      }
      if (operation === "rename" || operation === "move") {
        next.oldParentId = change.oldParentId ?? undefined;
        next.newParentId = change.newParentId ?? undefined;
      }
      payload.push(next);
    }
    const created = await createPlatformCommit(scope, scopeId.trim(), message.trim(), payload, signal);
    setSelectedCommit(created.id); setMessage(""); setChanges([]);
    const page = await listPlatformCommits(scope, scopeId.trim(), {}, signal); setCommits(page.commits);
    return created;
  });

  const inspect = (kind: "detail" | "files" | "tree") => selectedCommit ? void run(kind, (signal) => kind === "detail" ? getPlatformCommit(scope, scopeId.trim(), selectedCommit, signal) : kind === "files" ? listPlatformCommitFiles(scope, scopeId.trim(), selectedCommit, signal) : getPlatformCommitTree(scope, scopeId.trim(), selectedCommit, signal)) : undefined;

  return (
    <div className="grid gap-4">
      <Status error={error} notice={notice} />
      <Card><CardHeader><CardTitle>Commit kapsamı</CardTitle></CardHeader><CardContent className="grid gap-3">
        <div className="grid gap-3 md:grid-cols-[180px_1fr_auto]"><select className="h-9 rounded-md border bg-background px-3" value={scope} onChange={(event) => { setScope(event.target.value as PlatformCommitScope); setScopeId(""); }}><option value="workspace">Workspace</option><option value="folders">Klasör</option><option value="datasets">Veri kümesi</option></select>{scope === "datasets" ? <select className="h-9 rounded-md border bg-background px-3" value={scopeId} onChange={(event) => setScopeId(event.target.value)}><option value="">Veri kümesi seçin</option>{datasets.map((item) => <option key={datasetId(item)} value={datasetId(item)}>{datasetName(item)}</option>)}</select> : <Input value={scopeId} onChange={(event) => setScopeId(event.target.value)} placeholder={scope === "workspace" ? "Workspace kimliği" : "Klasör kimliği"} />}<Button disabled={!scopeId.trim() || Boolean(action)} onClick={refresh}>Geçmişi getir</Button></div>
      </CardContent></Card>
      <div className="grid gap-4 xl:grid-cols-2"><Card><CardHeader><CardTitle>Commit geçmişi</CardTitle></CardHeader><CardContent><VirtualRows items={commits} empty="Commit kaydı yok." render={(commit) => <button type="button" onClick={() => setSelectedCommit(commit.id)} className={`flex h-13 w-full items-center justify-between border-b px-3 text-left ${selectedCommit === commit.id ? "bg-muted" : ""}`}><span className="truncate"><strong>{commit.message || "Mesajsız commit"}</strong><small className="block text-muted-foreground">{commit.id}</small></span><Badge variant="outline">{commit.fileCount} dosya</Badge></button>} /></CardContent></Card><Card><CardHeader><CardTitle>Bekleyen değişiklikler</CardTitle></CardHeader><CardContent><VirtualRows items={changes} empty="Bekleyen değişiklik yok." render={(change) => <div className="flex h-13 items-center justify-between border-b px-3 text-sm"><span className="truncate">{change.fileName || change.fileId}</span><Badge variant="outline">{change.operation}</Badge></div>} /></CardContent></Card></div>
      <Card><CardHeader><CardTitle>Commit oluştur</CardTitle></CardHeader><CardContent className="grid gap-3"><Input value={message} onChange={(event) => setMessage(event.target.value)} placeholder="Commit mesajı" /><p className="text-muted-foreground text-sm">Add/modify içerikleri dosya servisinden alınır; ikili veya 1 MB üzeri içerik güvenli biçimde reddedilir.</p><Button className="w-fit" disabled={!message.trim() || !changes.length || Boolean(action)} onClick={create}>Commit oluştur</Button></CardContent></Card>
      <Card><CardHeader><CardTitle>İnceleme ve karşılaştırma</CardTitle></CardHeader><CardContent className="grid gap-3"><div className="flex flex-wrap gap-2"><Button variant="outline" disabled={!selectedCommit || Boolean(action)} onClick={() => inspect("detail")}>Detay</Button><Button variant="outline" disabled={!selectedCommit || Boolean(action)} onClick={() => inspect("files")}>Dosyalar</Button><Button variant="outline" disabled={!selectedCommit || Boolean(action)} onClick={() => inspect("tree")}>Ağaç</Button></div><div className="grid gap-2 md:grid-cols-[1fr_1fr_auto]"><Input value={compareFrom} onChange={(event) => setCompareFrom(event.target.value)} placeholder="Başlangıç commit kimliği" /><Input value={compareTo} onChange={(event) => setCompareTo(event.target.value)} placeholder="Bitiş commit kimliği" /><Button variant="outline" disabled={!compareFrom || !compareTo || Boolean(action)} onClick={() => void run("diff", (signal) => diffPlatformCommits(scope, scopeId.trim(), compareFrom, compareTo, signal))}>Karşılaştır</Button></div><div className="grid gap-2 md:grid-cols-[1fr_1fr_auto]"><Input value={selectedCommit} onChange={(event) => setSelectedCommit(event.target.value)} placeholder="Commit kimliği" /><Input id="commit-file-id" placeholder="Commit dosya kimliği" /><Button variant="outline" disabled={!selectedCommit || Boolean(action)} onClick={() => { const id = (document.getElementById("commit-file-id") as HTMLInputElement | null)?.value ?? ""; if (id) void run("content", (signal) => getPlatformCommitFileContent(scope, scopeId.trim(), selectedCommit, id, signal)); }}>İçeriği getir</Button></div></CardContent></Card>
      {action ? <div className="flex items-center gap-2"><Spinner className="size-4" />{action} çalışıyor<Button size="sm" variant="outline" onClick={() => controllerRef.current?.abort()}>İptal</Button></div> : null}
      {output !== null ? <pre className="max-h-80 overflow-auto rounded-xl bg-muted p-3 text-xs whitespace-pre-wrap">{pretty(output)}</pre> : null}
    </div>
  );
}

export function FilesPage() {
  const [datasets, setDatasets] = useState<PlatformDatasetDto[]>([]);
  const [error, setError] = useState("");
  const connectorsEnabled = isPlatformConnectorsEnabled();
  useEffect(() => {
    const controller = new AbortController();
    void listPlatformDatasets({ page: 1, pageSize: 100 }, controller.signal)
      .then((result) => setDatasets(result.items))
      .catch((value) => {
        if (!(value instanceof PlatformApiError) || !value.isAbort) setError(errorMessage(value));
      });
    return () => controller.abort();
  }, []);
  return (
    <main className="mx-auto flex w-full max-w-[1500px] flex-1 flex-col gap-5 p-4 md:p-7">
      <header><h1 className="font-heading text-2xl font-semibold">Dosyalar ve connector’lar</h1><p className="text-muted-foreground text-sm">Dosya alanını, veri kaynaklarını ve sürümlenmiş commit geçmişini yönetin.</p></header>
      {error ? <Status error={error} notice="" /> : null}
      <Tabs defaultValue="files"><TabsList><TabsTrigger value="files">Dosyalar</TabsTrigger><TabsTrigger value="connectors" disabled={!connectorsEnabled}>Connector’lar</TabsTrigger><TabsTrigger value="commits">Commit geçmişi</TabsTrigger></TabsList><TabsContent value="files"><FileManagerTab datasets={datasets} /></TabsContent><TabsContent value="connectors">{connectorsEnabled ? <ConnectorsTab datasets={datasets} /> : <Alert><AlertTitle>Connector’lar kapalı</AlertTitle><AlertDescription>Bu ortamda connector yeteneği etkin değil.</AlertDescription></Alert>}</TabsContent><TabsContent value="commits"><CommitsTab datasets={datasets} /></TabsContent></Tabs>
    </main>
  );
}
