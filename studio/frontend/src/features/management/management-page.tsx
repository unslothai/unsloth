import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Spinner } from "@/components/ui/spinner";
import {
  PHASE14_OPERATIONS,
  acceptPlatformTenantInvite,
  createPlatformUserPublicToken,
  executeManagementOperation,
  getPlatformAdminDashboard,
  getPlatformDifyHealth,
  getPlatformTenant,
  getPlatformUiError,
  invitePlatformTenantMember,
  listPlatformChatChannels,
  listPlatformCompilationBuiltins,
  listPlatformCompilationTemplateGroups,
  listPlatformCompilationWikiPresets,
  listPlatformTenantMembers,
  listPlatformTenants,
  listPlatformUserPublicTokens,
  loginPlatformAdmin,
  logoutPlatformAdmin,
  parseManagementJson,
  pollPlatformAimlapiAuthorization,
  redactManagementData,
  revokePlatformUserPublicToken,
  rotatePlatformUserPublicToken,
  startPlatformAimlapiAuthorization,
  toManagementRecords,
  usePlatformSessionStore,
  type AdminSession,
  type ManagementArea,
  type ManagementJson,
  type ManagementRecord,
  type ManagementSnapshot,
  type PublicTokenPair,
} from "@/integrations/platform-backend";
import { cn } from "@/lib/utils";
import { getPlatformManagementConfig } from "@/integrations/platform-backend/config";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

type ManagementTab = "system" | "tenants" | "bots" | "integrations";

const TAB_LABELS: Record<ManagementTab, string> = {
  system: "Sistem ve admin",
  tenants: "Tenant ve ekip",
  bots: "Bot, kanal ve template",
  integrations: "Compatibility",
};

const MANAGEMENT_CONFIG = getPlatformManagementConfig();

function errorMessage(error: unknown): string {
  return getPlatformUiError(error).message;
}

function JsonValue({ value }: { value: ManagementJson }) {
  return (
    <pre className="max-h-72 overflow-auto whitespace-pre-wrap break-words rounded-md bg-muted/45 p-3 font-mono text-xs">
      {JSON.stringify(value, null, 2)}
    </pre>
  );
}

function StatePanel({
  empty,
  error,
  loading,
}: {
  empty?: string;
  error?: string | null;
  loading?: boolean;
}) {
  if (loading) {
    return (
      <div className="flex items-center gap-2 rounded-lg border p-4 text-sm">
        <Spinner className="size-4" /> Yükleniyor…
      </div>
    );
  }
  if (error) {
    return (
      <div role="alert" className="rounded-lg border border-destructive/40 bg-destructive/5 p-4 text-sm text-destructive">
        {error}
      </div>
    );
  }
  if (empty) return <div className="rounded-lg border border-dashed p-5 text-muted-foreground text-sm">{empty}</div>;
  return null;
}

function Records({ records }: { records: ManagementRecord[] }) {
  if (records.length === 0) return <StatePanel empty="Kayıt bulunamadı." />;
  return (
    <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-3">
      {records.map((record) => (
        <article key={record.id} className="min-w-0 rounded-lg border bg-card p-4">
          <h3 className="truncate font-medium text-sm">{record.label}</h3>
          <p className="mb-3 truncate text-muted-foreground text-xs">{record.id}</p>
          <JsonValue value={record.values} />
        </article>
      ))}
    </div>
  );
}

function OperationRunner({
  adminSession,
  areas,
  onCompleted,
}: {
  adminSession: AdminSession | null;
  areas: ManagementArea[];
  onCompleted?: () => void;
}) {
  const operations = useMemo(
    () => PHASE14_OPERATIONS.filter((operation) => areas.includes(operation.area)),
    [areas],
  );
  const [selectedId, setSelectedId] = useState(operations[0]?.id ?? "");
  const selected = operations.find((operation) => operation.id === selectedId) ?? operations[0];
  const [pathValues, setPathValues] = useState<Record<string, string>>({});
  const [queryValues, setQueryValues] = useState<Record<string, string>>({});
  const [body, setBody] = useState("{}");
  const [auditReason, setAuditReason] = useState("");
  const [confirmation, setConfirmation] = useState("");
  const [result, setResult] = useState<ManagementJson | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [running, setRunning] = useState(false);
  const abortRef = useRef<AbortController | null>(null);

  useEffect(() => () => abortRef.current?.abort(), []);
  useEffect(() => {
    if (!selected) return;
    setPathValues({});
    setQueryValues({});
    setBody(JSON.stringify(selected.bodyTemplate ?? {}, null, 2));
    setAuditReason("");
    setConfirmation("");
    setResult(null);
    setError(null);
  }, [selected]);

  if (!selected) return null;

  const run = async () => {
    if (selected.needsAdminToken && !adminSession) {
      setError("Bu işlem için yönetici oturumu açılmalıdır.");
      return;
    }
    if (selected.danger && confirmation !== "ONAYLA") {
      setError("Tehlikeli işlemi çalıştırmak için ONAYLA yazın.");
      return;
    }
    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;
    setRunning(true);
    setError(null);
    setResult(null);
    try {
      const value = await executeManagementOperation(selected, {
        adminToken: adminSession?.token,
        auditReason,
        body: selected.bodyTemplate ? parseManagementJson(body) : undefined,
        pathParameters: pathValues,
        query: queryValues,
        signal: controller.signal,
      });
      setResult(value);
      onCompleted?.();
    } catch (cause) {
      setError(errorMessage(cause));
    } finally {
      if (abortRef.current === controller) abortRef.current = null;
      setRunning(false);
    }
  };

  return (
    <section className="rounded-xl border bg-card p-5">
      <div className="mb-4">
        <h2 className="font-heading font-semibold text-lg">Güvenli işlem merkezi</h2>
        <p className="text-muted-foreground text-sm">Mutasyonlar otomatik retry edilmez. Tehlikeli işlemler gerekçe ve açık onay ister.</p>
      </div>
      <label className="grid gap-1 text-sm">
        İşlem
        <select className="h-10 rounded-md border bg-background px-3" value={selected.id} onChange={(event) => setSelectedId(event.target.value)}>
          {operations.map((operation) => <option key={operation.id} value={operation.id}>{operation.label}</option>)}
        </select>
      </label>
      <p className="mt-2 text-muted-foreground text-sm">{selected.method} {selected.endpoint} — {selected.description}</p>
      <div className="mt-4 grid gap-3 md:grid-cols-2">
        {selected.pathParameters?.map((parameter) => (
          <label key={parameter} className="grid gap-1 text-sm">{parameter}<Input value={pathValues[parameter] ?? ""} onChange={(event) => setPathValues((current) => ({ ...current, [parameter]: event.target.value }))} /></label>
        ))}
        {selected.queryParameters?.map((parameter) => (
          <label key={parameter} className="grid gap-1 text-sm">Sorgu: {parameter}<Input value={queryValues[parameter] ?? ""} onChange={(event) => setQueryValues((current) => ({ ...current, [parameter]: event.target.value }))} /></label>
        ))}
      </div>
      {selected.bodyTemplate ? (
        <label className="mt-4 grid gap-1 text-sm">Doğrulanmış JSON sözleşmesi<textarea className="min-h-36 rounded-md border bg-background p-3 font-mono text-xs" value={body} onChange={(event) => setBody(event.target.value)} spellCheck={false} /></label>
      ) : null}
      {selected.requiresAuditReason ? (
        <label className="mt-4 grid gap-1 text-sm">Denetim gerekçesi<Input value={auditReason} onChange={(event) => setAuditReason(event.target.value)} placeholder="İş gerekçesi ve talep kaydı" /></label>
      ) : null}
      {selected.danger ? (
        <label className="mt-4 grid gap-1 text-sm">Onay<Input value={confirmation} onChange={(event) => setConfirmation(event.target.value)} placeholder="ONAYLA" /></label>
      ) : null}
      <div className="mt-4 flex gap-2">
        <Button onClick={() => void run()} disabled={running}>{running ? <Spinner className="mr-2 size-4" /> : null}{selected.label}</Button>
        {running ? <Button variant="outline" onClick={() => abortRef.current?.abort()}>İptal et</Button> : null}
      </div>
      {error ? <div role="alert" className="mt-4 text-destructive text-sm">{error}</div> : null}
      {result !== null ? <div className="mt-4"><JsonValue value={result} /></div> : null}
    </section>
  );
}

function AdminLogin({ onLogin }: { onLogin: (session: AdminSession) => void }) {
  const platformEmail = usePlatformSessionStore((state) => state.user?.email ?? "");
  const [email, setEmail] = useState(platformEmail);
  const [password, setPassword] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  useEffect(() => () => abortRef.current?.abort(), []);
  const submit = async () => {
    const controller = new AbortController();
    abortRef.current = controller;
    setLoading(true);
    setError(null);
    try {
      const session = await loginPlatformAdmin(email, password, { signal: controller.signal });
      setPassword("");
      onLogin(session);
    } catch (cause) {
      setPassword("");
      setError(errorMessage(cause));
    } finally {
      setLoading(false);
    }
  };
  return (
    <section className="max-w-xl rounded-xl border bg-card p-6">
      <h2 className="font-heading font-semibold text-lg">Yönetici yeniden doğrulaması</h2>
      <p className="mb-5 text-muted-foreground text-sm">Admin yüzeyi ayrı, yalnızca bellekte tutulan bir oturum kullanır. Parola veya token kalıcı store'a yazılmaz.</p>
      <div className="grid gap-3">
        <label className="grid gap-1 text-sm">E-posta<Input type="email" value={email} onChange={(event) => setEmail(event.target.value)} autoComplete="username" /></label>
        <label className="grid gap-1 text-sm">Parola<Input type="password" value={password} onChange={(event) => setPassword(event.target.value)} autoComplete="current-password" /></label>
        <Button disabled={loading || !email.trim() || !password} onClick={() => void submit()}>{loading ? <Spinner className="mr-2 size-4" /> : null}Yönetici oturumu aç</Button>
        {error ? <div role="alert" className="text-destructive text-sm">{error}</div> : null}
      </div>
    </section>
  );
}

function PublicTokenLifecycle({ session }: { session: AdminSession }) {
  const [username, setUsername] = useState(session.email);
  const [currentToken, setCurrentToken] = useState("");
  const [auditReason, setAuditReason] = useState("");
  const [confirmation, setConfirmation] = useState("");
  const [tokens, setTokens] = useState<ManagementJson | null>(null);
  const [issued, setIssued] = useState<PublicTokenPair | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [running, setRunning] = useState(false);
  const abortRef = useRef<AbortController | null>(null);
  const clearTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  useEffect(
    () => () => {
      abortRef.current?.abort();
      if (clearTimerRef.current) clearTimeout(clearTimerRef.current);
    },
    [],
  );
  const exposeOnce = (pair: PublicTokenPair) => {
    setIssued(pair);
    if (clearTimerRef.current) clearTimeout(clearTimerRef.current);
    clearTimerRef.current = setTimeout(() => setIssued(null), 60_000);
  };
  const run = async (action: "create" | "list" | "revoke" | "rotate") => {
    if (!username.trim()) return setError("Kullanıcı e-postası zorunludur.");
    if (action !== "list" && !auditReason.trim()) return setError("Denetim gerekçesi zorunludur.");
    if ((action === "revoke" || action === "rotate") && (!currentToken || confirmation !== "ONAYLA")) {
      return setError("Rotate/revoke için mevcut normal token ve ONAYLA doğrulaması zorunludur.");
    }
    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;
    setRunning(true);
    setError(null);
    setIssued(null);
    try {
      if (action === "list") {
        setTokens(await listPlatformUserPublicTokens(session.token, username.trim(), controller.signal));
      } else if (action === "create") {
        exposeOnce(await createPlatformUserPublicToken(session.token, username.trim(), controller.signal));
      } else if (action === "rotate") {
        exposeOnce(
          await rotatePlatformUserPublicToken(
            session.token,
            username.trim(),
            currentToken,
            controller.signal,
          ),
        );
        setCurrentToken("");
        setConfirmation("");
      } else {
        await revokePlatformUserPublicToken(
          session.token,
          username.trim(),
          currentToken,
          controller.signal,
        );
        setCurrentToken("");
        setConfirmation("");
      }
      if (action !== "list") {
        setTokens(await listPlatformUserPublicTokens(session.token, username.trim(), controller.signal));
      }
    } catch (cause) {
      setError(errorMessage(cause));
    } finally {
      if (abortRef.current === controller) abortRef.current = null;
      setRunning(false);
    }
  };
  return (
    <section className="rounded-xl border bg-card p-5">
      <h2 className="font-heading font-semibold text-lg">Public/embed token yaşam döngüsü</h2>
      <p className="mb-4 text-muted-foreground text-sm">
        Normal API tokenı ve beta/public token aynı backend kaydında oluşturulur. Rotate önce yeni çifti üretir, sonra eski kaydı revoke eder; kısmi hata durumunda yeni kayıt geri alınır.
      </p>
      <div className="grid gap-3 md:grid-cols-2">
        <label className="grid gap-1 text-sm">Kullanıcı e-postası<Input type="email" value={username} onChange={(event) => setUsername(event.target.value)} /></label>
        <label className="grid gap-1 text-sm">Mevcut normal token<Input type="password" value={currentToken} onChange={(event) => setCurrentToken(event.target.value)} autoComplete="off" /></label>
        <label className="grid gap-1 text-sm">Denetim gerekçesi<Input value={auditReason} onChange={(event) => setAuditReason(event.target.value)} /></label>
        <label className="grid gap-1 text-sm">Tehlikeli işlem onayı<Input value={confirmation} onChange={(event) => setConfirmation(event.target.value)} placeholder="ONAYLA" /></label>
      </div>
      <div className="mt-4 flex flex-wrap gap-2">
        <Button variant="outline" disabled={running} onClick={() => void run("list")}>Listele</Button>
        <Button disabled={running} onClick={() => void run("create")}>Oluştur</Button>
        <Button variant="outline" disabled={running} onClick={() => void run("rotate")}>Rotate</Button>
        <Button variant="destructive" disabled={running} onClick={() => void run("revoke")}>Revoke</Button>
        {running ? <Button variant="outline" onClick={() => abortRef.current?.abort()}>İptal et</Button> : null}
      </div>
      {error ? <div role="alert" className="mt-4 text-destructive text-sm">{error}</div> : null}
      {issued ? (
        <div className="mt-4 rounded-lg border border-amber-500/40 bg-amber-500/5 p-4">
          <p className="font-medium text-sm">Yalnızca bir kez gösterilir; 60 saniye sonra bellekten temizlenir.</p>
          <div className="mt-2 grid gap-2 font-mono text-xs"><div>API: {issued.token}</div><div>Public/beta: {issued.beta}</div><div>Tenant: {issued.tenantId}</div></div>
          <Button className="mt-3" size="sm" variant="outline" onClick={() => setIssued(null)}>Şimdi temizle</Button>
        </div>
      ) : null}
      {tokens !== null ? <div className="mt-4"><JsonValue value={tokens} /></div> : null}
    </section>
  );
}

function SystemTab({ operationsEnabled }: { operationsEnabled: boolean }) {
  const user = usePlatformSessionStore((state) => state.user);
  const [adminSession, setAdminSession] = useState<AdminSession | null>(null);
  const [snapshots, setSnapshots] = useState<ManagementSnapshot[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  const load = useCallback(async (session = adminSession) => {
    if (!session) return;
    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;
    setLoading(true);
    setError(null);
    try {
      setSnapshots(await getPlatformAdminDashboard({ token: session.token, signal: controller.signal }));
    } catch (cause) {
      setError(errorMessage(cause));
    } finally {
      if (abortRef.current === controller) abortRef.current = null;
      setLoading(false);
    }
  }, [adminSession]);
  useEffect(() => () => abortRef.current?.abort(), []);
  const login = (session: AdminSession) => {
    setAdminSession(session);
    void load(session);
  };
  const logout = async () => {
    const session = adminSession;
    setAdminSession(null);
    setSnapshots([]);
    if (session) await logoutPlatformAdmin(session.token).catch(() => undefined);
  };
  if (!user?.superuser) {
    return <StatePanel error="Bu alan yalnızca backend tarafından superuser olarak doğrulanan kullanıcılara açıktır. Menü görünürlüğü tek güvenlik katmanı değildir; bütün çağrılar backend 403 kontrolünden geçer." />;
  }
  if (!adminSession) return <AdminLogin onLogin={login} />;
  return (
    <div className="space-y-5">
      <div className="flex flex-wrap items-center justify-between gap-3 rounded-xl border bg-card p-4">
        <div><strong>Yönetici doğrulandı</strong><p className="text-muted-foreground text-sm">{adminSession.email}</p></div>
        <div className="flex gap-2"><Button variant="outline" onClick={() => void load()}>Yenile</Button><Button variant="outline" onClick={() => void logout()}>Admin oturumunu kapat</Button></div>
      </div>
      <StatePanel loading={loading} error={error} empty={!loading && !error && snapshots.length === 0 ? "Yönetim verisi bulunamadı." : undefined} />
      <div className="grid gap-4 lg:grid-cols-2">
        {snapshots.map((snapshot) => <section key={snapshot.key} className="min-w-0 rounded-xl border bg-card p-4"><h2 className="mb-3 font-heading font-semibold">{snapshot.label}</h2><JsonValue value={snapshot.data} /></section>)}
      </div>
      <PublicTokenLifecycle session={adminSession} />
      {operationsEnabled ? <OperationRunner adminSession={adminSession} areas={["admin"]} onCompleted={() => void load()} /> : <StatePanel empty="Admin operasyonları rollout flag'i ile kapalı; bu durumda mutation network çağrısı yapılmaz." />}
    </div>
  );
}

function TenantsTab() {
  const [tenants, setTenants] = useState<ManagementRecord[]>([]);
  const [members, setMembers] = useState<ManagementRecord[]>([]);
  const [detail, setDetail] = useState<ManagementJson | null>(null);
  const [selectedId, setSelectedId] = useState("");
  const [email, setEmail] = useState("");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const controllerRef = useRef<AbortController | null>(null);
  const load = useCallback(async () => {
    controllerRef.current?.abort();
    const controller = new AbortController();
    controllerRef.current = controller;
    setLoading(true);
    setError(null);
    try {
      const list = toManagementRecords(await listPlatformTenants(controller.signal));
      setTenants(list);
      const tenantId = selectedId || list[0]?.id || "";
      setSelectedId(tenantId);
      if (tenantId) {
        const [memberData, detailData] = await Promise.all([
          listPlatformTenantMembers(tenantId, controller.signal),
          getPlatformTenant(tenantId, controller.signal),
        ]);
        setMembers(toManagementRecords(memberData));
        setDetail(redactManagementData(detailData));
      } else {
        setMembers([]);
        setDetail(null);
      }
    } catch (cause) {
      setError(errorMessage(cause));
    } finally {
      setLoading(false);
    }
  }, [selectedId]);
  useEffect(() => { void load(); return () => controllerRef.current?.abort(); }, [load]);
  const select = async (tenantId: string) => {
    setSelectedId(tenantId);
    setMembers([]);
  };
  const invite = async () => {
    if (!selectedId || !email.trim()) return;
    try { await invitePlatformTenantMember(selectedId, email); setEmail(""); await load(); } catch (cause) { setError(errorMessage(cause)); }
  };
  return (
    <div className="space-y-5">
      <StatePanel loading={loading} error={error} empty={!loading && tenants.length === 0 ? "Bu kullanıcıya bağlı tenant bulunamadı." : undefined} />
      <section className="rounded-xl border bg-card p-5">
        <h2 className="font-heading font-semibold text-lg">Tenant kapsamı</h2>
        <p className="mb-4 text-muted-foreground text-sm">Backend membership ve ownership denetimi her detail/mutation çağrısında tekrar yapılır.</p>
        <div className="flex flex-wrap gap-2">{tenants.map((tenant) => <Button key={tenant.id} variant={selectedId === tenant.id ? "default" : "outline"} onClick={() => void select(tenant.id)}>{tenant.label}</Button>)}</div>
      </section>
      {selectedId ? <section className="rounded-xl border bg-card p-5">{detail !== null ? <div className="mb-4"><JsonValue value={detail} /></div> : null}<div className="mb-4 flex flex-wrap items-end gap-3"><label className="grid min-w-64 flex-1 gap-1 text-sm">Üye e-postası<Input type="email" value={email} onChange={(event) => setEmail(event.target.value)} /></label><Button onClick={() => void invite()}>Davet gönder</Button><Button variant="outline" onClick={() => void acceptPlatformTenantInvite(selectedId).then(load).catch((cause) => setError(errorMessage(cause)))}>Bekleyen daveti kabul et</Button></div><Records records={members} /></section> : null}
      <OperationRunner adminSession={null} areas={["tenant"]} onCompleted={() => void load()} />
    </div>
  );
}

function BotsTab({ botsEnabled, channelsEnabled }: { botsEnabled: boolean; channelsEnabled: boolean }) {
  const [channels, setChannels] = useState<ManagementRecord[]>([]);
  const [groups, setGroups] = useState<ManagementRecord[]>([]);
  const [builtins, setBuiltins] = useState<ManagementRecord[]>([]);
  const [wiki, setWiki] = useState<ManagementRecord[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const controllerRef = useRef<AbortController | null>(null);
  const load = useCallback(async () => {
    controllerRef.current?.abort();
    const controller = new AbortController();
    controllerRef.current = controller;
    setLoading(true);
    setError(null);
    try {
      const [channelData, groupData, builtinData, wikiData] = await Promise.all([
        channelsEnabled ? listPlatformChatChannels(controller.signal) : Promise.resolve([]),
        botsEnabled ? listPlatformCompilationTemplateGroups(controller.signal) : Promise.resolve([]),
        botsEnabled ? listPlatformCompilationBuiltins(controller.signal) : Promise.resolve([]),
        botsEnabled ? listPlatformCompilationWikiPresets(controller.signal) : Promise.resolve([]),
      ]);
      setChannels(toManagementRecords(channelData));
      setGroups(toManagementRecords(groupData));
      setBuiltins(toManagementRecords(builtinData));
      setWiki(toManagementRecords(wikiData));
    } catch (cause) { setError(errorMessage(cause)); } finally { setLoading(false); }
  }, [botsEnabled, channelsEnabled]);
  useEffect(() => { void load(); return () => controllerRef.current?.abort(); }, [load]);
  return (
    <div className="space-y-5">
      <StatePanel loading={loading} error={error} />
      {channelsEnabled ? <section><h2 className="mb-3 font-heading font-semibold text-lg">Yayın kanalları</h2><Records records={channels} /></section> : null}
      {botsEnabled ? <><section><h2 className="mb-3 font-heading font-semibold text-lg">Compilation template grupları</h2><Records records={groups} /></section><section className="grid gap-5 lg:grid-cols-2"><div><h2 className="mb-3 font-heading font-semibold text-lg">Builtin template'ler</h2><Records records={builtins} /></div><div><h2 className="mb-3 font-heading font-semibold text-lg">Wiki preset'leri</h2><Records records={wiki} /></div></section></> : null}
      <OperationRunner adminSession={null} areas={[...(botsEnabled ? ["bots", "templates"] as ManagementArea[] : []), ...(channelsEnabled ? ["channels"] as ManagementArea[] : [])]} onCompleted={() => void load()} />
    </div>
  );
}

function IntegrationsTab() {
  const [health, setHealth] = useState<ManagementJson | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [aimlapiRequest, setAimlapiRequest] = useState<{ requestId: string; verificationUri: string } | null>(null);
  const [aimlapiStatus, setAimlapiStatus] = useState<string | null>(null);
  const [aimlapiKey, setAimlapiKey] = useState<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  const clearTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  useEffect(() => () => {
    abortRef.current?.abort();
    if (clearTimerRef.current) clearTimeout(clearTimerRef.current);
  }, []);
  const check = async () => {
    setLoading(true); setError(null);
    try { setHealth(redactManagementData(await getPlatformDifyHealth())); } catch (cause) { setError(errorMessage(cause)); } finally { setLoading(false); }
  };
  const startAimlapi = async () => {
    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;
    setLoading(true); setError(null); setAimlapiKey(null); setAimlapiStatus(null);
    try {
      const authorization = await startPlatformAimlapiAuthorization(controller.signal);
      setAimlapiRequest({ requestId: authorization.requestId, verificationUri: authorization.verificationUri });
      setAimlapiStatus("pending");
    } catch (cause) { setError(errorMessage(cause)); } finally { setLoading(false); }
  };
  const pollAimlapi = async () => {
    if (!aimlapiRequest) return;
    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;
    setLoading(true); setError(null);
    try {
      const result = await pollPlatformAimlapiAuthorization(aimlapiRequest.requestId, controller.signal);
      setAimlapiStatus(result.status);
      if (result.apiKey) {
        setAimlapiKey(result.apiKey);
        setAimlapiRequest(null);
        if (clearTimerRef.current) clearTimeout(clearTimerRef.current);
        clearTimerRef.current = setTimeout(() => setAimlapiKey(null), 60_000);
      }
    } catch (cause) { setError(errorMessage(cause)); } finally { setLoading(false); }
  };
  return (
    <div className="space-y-5">
      <section className="rounded-xl border bg-card p-5"><h2 className="font-heading font-semibold text-lg">Uyumluluk protokolleri</h2><p className="mt-1 text-muted-foreground text-sm">Dify retrieval, OpenAI chat ve agent compatibility endpoint'leri çekirdek Rag Platform chat adapter'ını değiştirmez. Credential değerleri bu sayfada saklanmaz.</p><div className="mt-4 grid gap-2 text-sm"><code>GET|POST /api/v1/dify/retrieval</code><code>POST /api/v1/openai/:chat_id/chat/completions</code><code>POST /api/v1/chats_openai/:chat_id/chat/completions</code><code>POST /api/v1/agents_openai/:agent_id/chat/completions</code></div><Button className="mt-4" variant="outline" onClick={() => void check()} disabled={loading}>{loading ? <Spinner className="mr-2 size-4" /> : null}Dify health sözleşmesini doğrula</Button>{error ? <div role="alert" className="mt-3 text-destructive text-sm">{error}</div> : null}{health !== null ? <div className="mt-3"><JsonValue value={health} /></div> : null}</section>
      <section className="rounded-xl border bg-card p-5"><h2 className="font-heading font-semibold text-lg">Public/embed durumu</h2><p className="mt-1 text-muted-foreground text-sm">Beta-token MCP protokolü API-only sunulur. Preview/thumbnail sözleşmeleri JWT, API ve beta-token auth kabul eder; anonim erişim reddedilir. Token create/rotate/revoke Yönetim → Sistem ve admin alanındaki tek-seferlik secret görünümüyle yönetilir. AIMLAPI device authorization kullanıcı kapsamlı Redis kaydıyla çalışır; device code tarayıcıya verilmez.</p></section>
      <section className="rounded-xl border bg-card p-5"><h2 className="font-heading font-semibold text-lg">AIMLAPI güvenli yetkilendirme</h2><p className="mt-1 text-muted-foreground text-sm">Device code yalnızca backend Redis kaydında tutulur. Tarayıcı yalnızca izin URL'sini ve request kimliğini görür; provider anahtarı kalıcı store'a yazılmaz.</p><div className="mt-4 flex flex-wrap gap-2"><Button variant="outline" disabled={loading} onClick={() => void startAimlapi()}>Yetkilendirmeyi başlat</Button>{aimlapiRequest ? <a className="inline-flex h-9 items-center rounded-md border px-4 text-sm" href={aimlapiRequest.verificationUri} target="_blank" rel="noreferrer">AIMLAPI iznini aç</a> : null}{aimlapiRequest ? <Button disabled={loading} onClick={() => void pollAimlapi()}>Durumu kontrol et</Button> : null}{loading ? <Button variant="outline" onClick={() => abortRef.current?.abort()}>İptal et</Button> : null}</div>{aimlapiStatus ? <p className="mt-3 text-sm">Durum: {aimlapiStatus}</p> : null}{aimlapiKey ? <div className="mt-3 rounded-lg border border-amber-500/40 bg-amber-500/5 p-4"><p className="text-sm">API anahtarı yalnızca bir kez gösterilir ve 60 saniye sonra bellekten temizlenir.</p><code className="mt-2 block break-all text-xs">{aimlapiKey}</code><Button className="mt-3" size="sm" variant="outline" onClick={() => setAimlapiKey(null)}>Şimdi temizle</Button></div> : null}</section>
      <OperationRunner adminSession={null} areas={["compatibility"]} />
    </div>
  );
}

export function ManagementPage() {
  const user = usePlatformSessionStore((state) => state.user);
  const tabs = (Object.keys(TAB_LABELS) as ManagementTab[]).filter((item) =>
    item === "system"
      ? MANAGEMENT_CONFIG.adminEnabled && Boolean(user?.superuser)
      : item === "tenants"
        ? MANAGEMENT_CONFIG.tenantsEnabled
        : item === "bots"
          ? MANAGEMENT_CONFIG.botsEnabled || MANAGEMENT_CONFIG.channelsEnabled
          : true,
  );
  const [tab, setTab] = useState<ManagementTab>(tabs[0] ?? "integrations");
  return (
    <main className="h-full overflow-y-auto">
      <div className="mx-auto w-full max-w-[1500px] p-5 md:p-8">
        <header className="mb-6"><p className="font-medium text-primary text-sm">Rag Platform</p><h1 className="font-heading font-semibold text-3xl tracking-tight">Yönetim merkezi</h1><p className="mt-2 max-w-3xl text-muted-foreground">Sistem operasyonları, tenant üyeliği, bot/kanal yayınlama ve dış protokoller için rol kontrollü çalışma alanı.</p></header>
        <nav aria-label="Yönetim bölümleri" className="mb-6 flex flex-wrap gap-2">{tabs.map((item) => <Button key={item} variant={tab === item ? "default" : "outline"} onClick={() => setTab(item)}>{TAB_LABELS[item]}</Button>)}</nav>
        <div className={cn(tab !== "system" && "hidden")}>{tab === "system" ? <SystemTab operationsEnabled={MANAGEMENT_CONFIG.adminOperationsEnabled} /> : null}</div>
        {tab === "tenants" ? <TenantsTab /> : null}
        {tab === "bots" ? <BotsTab botsEnabled={MANAGEMENT_CONFIG.botsEnabled} channelsEnabled={MANAGEMENT_CONFIG.channelsEnabled} /> : null}
        {tab === "integrations" ? <IntegrationsTab /> : null}
      </div>
    </main>
  );
}
