import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogMedia,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Spinner } from "@/components/ui/spinner";
import {
  type PlatformDefaultModel,
  type PlatformModel,
  type PlatformPipeline,
  type PlatformProvider,
  type PlatformProviderInstance,
  addInstanceModel,
  addProvider,
  createProviderInstance,
  deleteInstanceModels,
  deleteProviderInstances,
  getCurrentPlatformTenantModels,
  getDefaultModels,
  getProviderInstance,
  getProviderModel,
  isPlatformApiError,
  listAvailableProviders,
  listConfiguredProviders,
  listInstanceModels,
  listPipelines,
  listProviderInstances,
  listProviderModels,
  listSupportedInstanceModels,
  listTenantModels,
  mergePlatformDefaultModels,
  setDefaultModel,
  testProviderConnection,
  testProviderInstanceConnection,
  updateInstanceModel,
  updateProviderInstance,
  usePlatformSessionStore,
} from "@/integrations/platform-backend";
import {
  Alert02Icon,
  ArrowDown01Icon,
  ArrowRight01Icon,
  Delete02Icon,
  Edit03Icon,
  Wifi02Icon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useCallback, useEffect, useMemo, useState } from "react";
import { toast } from "sonner";
import { ApiProviderLogo } from "../../chat/api-provider-logo";
import { PlatformModelTools } from "./platform-model-tools";

type LoadState = "loading" | "ready" | "empty" | "error";

interface ProviderView extends PlatformProvider {
  instances: PlatformProviderInstance[];
}

const errorMessage = (error: unknown) =>
  error instanceof Error
    ? error.message
    : "Rag Platform isteği başarısız oldu.";

const OPENAI_COMPATIBLE_PROVIDER = "OpenAI-API-Compatible";
const CUSTOM_OPENAI_PROVIDER = "VLLM";

const providerOptionLabel = (providerName: string) => {
  if (providerName === CUSTOM_OPENAI_PROVIDER)
    return "OpenAI compatible / Custom (VLLM)";
  if (providerName === OPENAI_COMPATIBLE_PROVIDER)
    return `${providerName} (runtime uyumsuz)`;
  return providerName;
};

const defaultMatchesModel = (
  savedDefault: PlatformDefaultModel | undefined,
  model: PlatformModel,
) =>
  (Boolean(savedDefault?.modelId) && model.id === savedDefault?.modelId) ||
  (Boolean(savedDefault?.modelName) &&
    model.name === savedDefault?.modelName &&
    (!savedDefault?.providerName ||
      model.providerName === savedDefault.providerName) &&
    (!savedDefault?.instanceName ||
      model.instanceName === savedDefault.instanceName));

const sameConfiguredModel = (left: PlatformModel, right: PlatformModel) =>
  left.name === right.name &&
  left.providerName === right.providerName &&
  left.instanceName === right.instanceName;

function Section({
  children,
  description,
  title,
}: {
  children: React.ReactNode;
  description: string;
  title: string;
}) {
  return (
    <section className="space-y-4 overflow-hidden rounded-[8px] border border-border/70 bg-muted/[0.12] p-4">
      <div className="flex min-w-0 flex-col gap-0.5">
        <h3 className="text-sm font-medium text-foreground">{title}</h3>
        <p className="text-xs leading-snug text-muted-foreground">
          {description}
        </p>
      </div>
      {children}
    </section>
  );
}

export function PlatformModelsSettings({
  mode = "manage",
  onCreated,
  onSummaryChange,
  refreshKey = 0,
}: {
  mode?: "create" | "manage";
  onCreated?: () => void;
  onSummaryChange?: (summary: { connections: number; models: number }) => void;
  refreshKey?: number;
}) {
  const user = usePlatformSessionStore((state) => state.user);
  const [state, setState] = useState<LoadState>("loading");
  const [error, setError] = useState("");
  const [catalog, setCatalog] = useState<PlatformProvider[]>([]);
  const [providers, setProviders] = useState<ProviderView[]>([]);
  const [models, setModels] = useState<PlatformModel[]>([]);
  const [defaults, setDefaults] = useState<PlatformDefaultModel[]>([]);
  const [canManage, setCanManage] = useState(false);
  const [pipelineState, setPipelineState] = useState<
    "loading" | "ready" | "runtime-disabled" | "error"
  >("loading");
  const [pipelines, setPipelines] = useState<PlatformPipeline[]>([]);
  const [busy, setBusy] = useState("");

  const load = useCallback(
    async (signal?: AbortSignal, background = false) => {
      if (!background) {
        setState("loading");
        setError("");
      }
      try {
        const [available, configured, tenantModels, selectedDefaults, tenant] =
          await Promise.all([
            listAvailableProviders(signal),
            listConfiguredProviders(signal),
            listTenantModels(signal),
            getDefaultModels(signal),
            getCurrentPlatformTenantModels(signal),
          ]);
        const [withInstances, configuredCatalogResults] = await Promise.all([
          Promise.all(
            configured.map(async (provider) => ({
              ...provider,
              instances: await listProviderInstances(provider.name, signal),
            })),
          ),
          Promise.allSettled(
            configured.map((provider) =>
              listProviderModels(provider.name, signal),
            ),
          ),
        ]);
        if (signal?.aborted) return;
        // Provider catalogs are enrichment only. A pinned Go runtime can lack a
        // provider that Python already has configured; that must not make the
        // whole Add connection screen unusable.
        const catalogModels = configuredCatalogResults.flatMap((result) =>
          result.status === "fulfilled" ? result.value : [],
        );
        const enrichedModels = tenantModels.map((model) => {
          const catalogModel = catalogModels.find(
            (candidate) =>
              candidate.name === model.name &&
              (!candidate.providerName ||
                candidate.providerName === model.providerName),
          );
          return catalogModel
            ? {
                ...model,
                capabilities: Array.from(
                  new Set([
                    ...model.capabilities,
                    ...catalogModel.capabilities,
                  ]),
                ),
              }
            : model;
        });
        setCatalog(available);
        setProviders(withInstances);
        setModels(enrichedModels);
        setDefaults(mergePlatformDefaultModels(selectedDefaults, tenant));
        setCanManage(
          user?.superuser === true ||
            ["owner", "admin"].includes(tenant.role.toLowerCase()),
        );
        setState(
          available.length || configured.length || enrichedModels.length
            ? "ready"
            : "empty",
        );
      } catch (loadError) {
        if (isPlatformApiError(loadError) && loadError.isAbort) return;
        setError(errorMessage(loadError));
        setState("error");
      }
    },
    [user?.superuser],
  );

  const loadPipelines = useCallback(async (signal?: AbortSignal) => {
    setPipelineState("loading");
    try {
      setPipelines(await listPipelines(signal));
      setPipelineState("ready");
    } catch (loadError) {
      if (isPlatformApiError(loadError) && loadError.isAbort) return;
      setPipelineState(
        isPlatformApiError(loadError) && loadError.httpStatus === 404
          ? "runtime-disabled"
          : "error",
      );
    }
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    void load(controller.signal);
    if (mode === "manage") void loadPipelines(controller.signal);
    return () => controller.abort();
  }, [load, loadPipelines, mode, refreshKey]);

  useEffect(() => {
    if (mode !== "manage" || state === "loading") return;
    onSummaryChange?.({
      connections: providers.reduce(
        (total, provider) => total + provider.instances.length,
        0,
      ),
      models: models.length,
    });
  }, [mode, models.length, onSummaryChange, providers, state]);

  const runMutation = async (
    key: string,
    action: () => Promise<unknown>,
  ): Promise<boolean> => {
    if (!canManage) return false;
    setBusy(key);
    try {
      await action();
      toast.success("Rag Platform ayarı kaydedildi.");
      await load(undefined, true);
      return true;
    } catch (mutationError) {
      toast.error(errorMessage(mutationError));
      return false;
    } finally {
      setBusy("");
    }
  };

  if (state === "loading") {
    return (
      <div className="flex items-center gap-2 border-border/60 border-b px-4 py-5 text-sm text-muted-foreground last:border-b-0">
        <Spinner /> Rag Platform model yapılandırması yükleniyor…
      </div>
    );
  }
  if (state === "error") {
    return (
      <div className="space-y-3 border-border/60 border-b px-4 py-4 last:border-b-0">
        <p className="text-sm text-destructive">{error}</p>
        <Button variant="outline" onClick={() => void load()}>
          Yeniden dene
        </Button>
      </div>
    );
  }

  if (mode === "create") {
    return (
      <ProviderConfiguration
        busy={busy}
        canManage={canManage}
        catalog={catalog}
        providers={providers}
        onCreated={onCreated}
        runMutation={runMutation}
      />
    );
  }

  return (
    <PlatformConnectionRows
      busy={busy}
      canManage={canManage}
      defaults={defaults}
      models={models}
      pipelineState={pipelineState}
      pipelines={pipelines}
      providers={providers}
      loadPipelines={loadPipelines}
      runMutation={runMutation}
    />
  );
}

function ProviderConfiguration({
  busy,
  canManage,
  catalog,
  onCreated,
  providers,
  runMutation,
}: {
  busy: string;
  canManage: boolean;
  catalog: PlatformProvider[];
  onCreated?: () => void;
  providers: ProviderView[];
  runMutation: (
    key: string,
    action: () => Promise<unknown>,
  ) => Promise<boolean>;
}) {
  const [providerName, setProviderName] = useState("");
  const [instanceName, setInstanceName] = useState("");
  const [apiKey, setApiKey] = useState("");
  const [baseUrl, setBaseUrl] = useState("");
  const [draftTesting, setDraftTesting] = useState(false);

  const create = async () => {
    const name = providerName.trim();
    const instance = instanceName.trim();
    if (!name || !instance) return;
    const secret = apiKey;
    const created = await runMutation(`create:${name}`, async () => {
      if (!providers.some((provider) => provider.name === name)) {
        await addProvider(name);
      }
      await createProviderInstance(name, {
        apiKey: secret,
        baseUrl,
        instanceName: instance,
        region: "",
      });
    });
    if (!created) return;
    setApiKey("");
    setInstanceName("");
    onCreated?.();
  };

  const testDraftConnection = async () => {
    const secret = apiKey;
    setDraftTesting(true);
    try {
      await testProviderConnection(providerName, {
        apiKey: secret,
        baseUrl,
        region: "",
      });
      toast.success("Bağlantı doğrulandı.");
    } catch (connectionError) {
      toast.error(errorMessage(connectionError));
    } finally {
      setDraftTesting(false);
    }
  };

  return (
    <section
      aria-labelledby="platform-connection-heading"
      className="overflow-hidden rounded-[8px] border border-border/70 bg-muted/[0.12]"
    >
      {!canManage ? (
        <output className="block border-border/60 border-b bg-muted/20 px-4 py-3 text-xs text-muted-foreground">
          Yeni bağlantı eklemek için owner veya admin yetkisi gerekir.
        </output>
      ) : null}
      <div className="divide-y divide-border/60">
        <div className="grid grid-cols-[minmax(140px,0.8fr)_minmax(0,1.2fr)] items-center gap-4 px-4 py-3 @max-[520px]:grid-cols-1">
          <div className="flex min-w-0 flex-col gap-0.5">
            <Label
              id="platform-connection-heading"
              htmlFor="platform-provider"
              className="text-sm font-medium"
            >
              Connection
            </Label>
            <p className="text-xs leading-snug text-muted-foreground">
              Choose a provider from the Rag Platform catalog.
            </p>
          </div>
          <Select
            value={providerName}
            disabled={!canManage}
            onValueChange={setProviderName}
          >
            <SelectTrigger
              id="platform-provider"
              className="h-9 w-full text-sm"
            >
              <SelectValue placeholder="Choose a connection">
                {providerName ? (
                  <span className="flex min-w-0 items-center gap-2">
                    <ApiProviderLogo
                      providerType={providerName.toLowerCase()}
                      className="size-4"
                      title={providerName}
                    />
                    <span className="truncate">
                      {providerOptionLabel(providerName)}
                    </span>
                  </span>
                ) : null}
              </SelectValue>
            </SelectTrigger>
            <SelectContent>
              <SelectGroup>
                {catalog
                  .filter(
                    (provider) => provider.name !== OPENAI_COMPATIBLE_PROVIDER,
                  )
                  .map((provider) => (
                    <SelectItem key={provider.id} value={provider.name}>
                      <span className="flex min-w-0 items-center gap-2">
                        <ApiProviderLogo
                          providerType={provider.name.toLowerCase()}
                          className="size-4"
                          title={provider.name}
                        />
                        <span className="truncate">
                          {providerOptionLabel(provider.name)}
                        </span>
                      </span>
                    </SelectItem>
                  ))}
              </SelectGroup>
            </SelectContent>
          </Select>
        </div>

        {providerName ? (
          <>
            {providerName === CUSTOM_OPENAI_PROVIDER ? (
              <output className="block bg-background/35 px-4 py-3 text-xs leading-relaxed text-muted-foreground">
                OpenAI-compatible ve custom endpoint’ler bu runtime’da VLLM
                sözleşmesini kullanır. Base URL otomatik olarak /v1 köküne
                normalize edilir.
              </output>
            ) : null}
            <div className="grid grid-cols-[minmax(140px,0.8fr)_minmax(0,1.2fr)] items-center gap-4 px-4 py-3 @max-[520px]:grid-cols-1">
              <div className="flex min-w-0 flex-col gap-0.5">
                <Label
                  htmlFor="platform-instance-name"
                  className="text-sm font-medium"
                >
                  Instance name
                </Label>
                <p className="text-xs leading-snug text-muted-foreground">
                  Bu bağlantıyı listede ayırt edeceğiniz kısa ad.
                </p>
              </div>
              <Input
                id="platform-instance-name"
                className="h-9 text-sm"
                placeholder="Production"
                value={instanceName}
                disabled={!canManage}
                onChange={(event) => setInstanceName(event.target.value)}
              />
            </div>
            <div className="grid grid-cols-[minmax(140px,0.8fr)_minmax(0,1.2fr)] items-center gap-4 px-4 py-3 @max-[520px]:grid-cols-1">
              <div className="flex min-w-0 flex-col gap-0.5">
                <Label
                  htmlFor="platform-api-key"
                  className="text-sm font-medium"
                >
                  API key
                </Label>
                <p className="text-xs leading-snug text-muted-foreground">
                  Browser storage’a yazılmaz; doğrudan Rag Platform’a
                  gönderilir.
                </p>
              </div>
              <Input
                id="platform-api-key"
                className="h-9 text-sm"
                type="password"
                autoComplete="new-password"
                placeholder="Enter API key"
                value={apiKey}
                disabled={!canManage}
                onChange={(event) => setApiKey(event.target.value)}
              />
            </div>
            <div className="grid grid-cols-[minmax(140px,0.8fr)_minmax(0,1.2fr)] items-center gap-4 px-4 py-3 @max-[520px]:grid-cols-1">
              <div className="flex min-w-0 flex-col gap-0.5">
                <Label
                  htmlFor="platform-base-url"
                  className="text-sm font-medium"
                >
                  Base URL
                </Label>
                <p className="text-xs leading-snug text-muted-foreground">
                  {providerName === CUSTOM_OPENAI_PROVIDER
                    ? "OpenAI-compatible API root; /v1 suffix is added when missing."
                    : "Optional custom endpoint for this provider."}
                </p>
              </div>
              <Input
                id="platform-base-url"
                className="h-9 text-sm"
                inputMode="url"
                placeholder={
                  providerName === CUSTOM_OPENAI_PROVIDER
                    ? "https://llm.example.com/v1"
                    : "https://api.example.com"
                }
                value={baseUrl}
                disabled={!canManage}
                onChange={(event) => setBaseUrl(event.target.value)}
              />
            </div>
            <div className="flex flex-wrap items-center justify-end gap-2 bg-background/25 px-4 py-3">
              <Button
                type="button"
                variant="outline"
                size="sm"
                className="h-8"
                disabled={
                  !canManage ||
                  !apiKey ||
                  (providerName === CUSTOM_OPENAI_PROVIDER &&
                    !baseUrl.trim()) ||
                  Boolean(busy) ||
                  draftTesting
                }
                onClick={() => void testDraftConnection()}
              >
                {draftTesting ? <Spinner /> : "Test connection"}
              </Button>
              <Button
                type="button"
                size="sm"
                className="h-8"
                disabled={
                  !canManage ||
                  !providerName ||
                  !instanceName.trim() ||
                  (providerName === CUSTOM_OPENAI_PROVIDER &&
                    (!apiKey || !baseUrl.trim())) ||
                  Boolean(busy)
                }
                onClick={() => void create()}
              >
                {busy.startsWith("create:") ? <Spinner /> : "Add connection"}
              </Button>
            </div>
          </>
        ) : null}
      </div>
    </section>
  );
}

function PipelineCatalog({
  loadPipelines,
  pipelines,
  state,
}: {
  loadPipelines: (signal?: AbortSignal) => Promise<void>;
  pipelines: PlatformPipeline[];
  state: "loading" | "ready" | "runtime-disabled" | "error";
}) {
  return (
    <Section
      title="Pipeline kataloğu"
      description="Dataset oluştururken kullanılabilen backend pipeline tanımları."
    >
      {state === "loading" && (
        <div className="flex items-center gap-2 text-sm text-muted-foreground">
          <Spinner /> Pipeline kataloğu yükleniyor…
        </div>
      )}
      {state === "runtime-disabled" && (
        <p role="status" className="text-sm text-amber-600">
          Pipeline kataloğu bu runtime sürümünde kullanılamıyor. Mevcut parser
          akışı etkilenmez.
        </p>
      )}
      {state === "error" && (
        <div className="flex flex-wrap items-center gap-2">
          <p className="text-sm text-destructive">
            Pipeline kataloğu okunamadı.
          </p>
          <Button
            type="button"
            size="sm"
            variant="outline"
            onClick={() => void loadPipelines()}
          >
            Yeniden dene
          </Button>
        </div>
      )}
      {state === "ready" && pipelines.length === 0 && (
        <p className="text-sm text-muted-foreground">Pipeline bulunamadı.</p>
      )}
      {state === "ready" && pipelines.length > 0 && (
        <ul className="grid gap-2 sm:grid-cols-2">
          {pipelines.map((pipeline) => (
            <li
              key={pipeline.id}
              className="rounded-[8px] border border-border/70 bg-background/50 p-3 text-sm"
            >
              <div className="font-medium">{pipeline.title}</div>
              <div className="mt-0.5 text-xs text-muted-foreground">
                {pipeline.description || pipeline.id}
              </div>
            </li>
          ))}
        </ul>
      )}
    </Section>
  );
}

function PlatformConnectionRows({
  busy,
  canManage,
  defaults,
  loadPipelines,
  models,
  pipelines,
  pipelineState,
  providers,
  runMutation,
}: {
  busy: string;
  canManage: boolean;
  defaults: PlatformDefaultModel[];
  loadPipelines: (signal?: AbortSignal) => Promise<void>;
  models: PlatformModel[];
  pipelines: PlatformPipeline[];
  pipelineState: "loading" | "ready" | "runtime-disabled" | "error";
  providers: ProviderView[];
  runMutation: (
    key: string,
    action: () => Promise<unknown>,
  ) => Promise<boolean>;
}) {
  const [expandedId, setExpandedId] = useState("");
  const [editingId, setEditingId] = useState("");
  const [pendingDeleteId, setPendingDeleteId] = useState("");
  const [editName, setEditName] = useState("");
  const [editApiKey, setEditApiKey] = useState("");
  const [editBaseUrl, setEditBaseUrl] = useState("");
  const [editRegion, setEditRegion] = useState("");
  const [connectionChecks, setConnectionChecks] = useState<
    Record<string, { message: string; state: "loading" | "success" | "error" }>
  >({});

  const connections = providers.flatMap((provider) =>
    provider.instances.map((instance) => ({ provider, instance })),
  );

  const openEdit = async (
    provider: PlatformProvider,
    instance: PlatformProviderInstance,
  ) => {
    try {
      const detail = await getProviderInstance(provider.name, instance.name);
      setExpandedId(instance.id);
      setEditingId(instance.id);
      setPendingDeleteId("");
      setEditName(instance.name);
      setEditBaseUrl(detail.baseUrl);
      setEditRegion(detail.region);
      setEditApiKey("");
    } catch (detailError) {
      toast.error(errorMessage(detailError));
    }
  };

  const testConnection = async (
    provider: PlatformProvider,
    instance: PlatformProviderInstance,
  ) => {
    const key = instance.id;
    setConnectionChecks((current) => ({
      ...current,
      [key]: {
        message: "Bağlantı ve model kataloğu sınanıyor…",
        state: "loading",
      },
    }));
    try {
      await testProviderInstanceConnection(provider.name, instance.name);
      const discovered = await listSupportedInstanceModels(
        provider.name,
        instance.name,
      );
      const message = `Bağlantı doğrulandı; ${discovered.length} model bulundu.`;
      setConnectionChecks((current) => ({
        ...current,
        [key]: { message, state: "success" },
      }));
      toast.success(message);
    } catch (connectionError) {
      const message = errorMessage(connectionError);
      setConnectionChecks((current) => ({
        ...current,
        [key]: { message, state: "error" },
      }));
      toast.error(message);
    }
  };

  if (connections.length === 0) return null;

  return connections.map(({ provider, instance }) => {
    const expanded = expandedId === instance.id;
    const editing = editingId === instance.id;
    const deleting = pendingDeleteId === instance.id;
    const instanceModels = models.filter(
      (model) =>
        model.providerName === provider.name &&
        model.instanceName === instance.name,
    );
    const check = connectionChecks[instance.id];
    const fixedProvider: ProviderView = {
      ...provider,
      instances: [instance],
    };

    return (
      <article
        key={instance.id}
        className="border-border/60 border-b bg-background/15 last:border-b-0"
      >
        <div className="grid min-w-0 grid-cols-[minmax(0,1fr)_auto] items-center gap-3 px-3 py-3 transition-colors hover:bg-muted/30 max-sm:grid-cols-1">
          <button
            type="button"
            className="group flex min-w-0 items-start gap-3 rounded-[8px] text-left focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
            aria-expanded={expanded}
            onClick={() => {
              setExpandedId(expanded ? "" : instance.id);
              if (expanded) {
                setEditingId("");
                setPendingDeleteId("");
              }
            }}
          >
            <span className="mt-1 flex size-8 shrink-0 items-center justify-center rounded-[8px] border border-border/70 bg-background/80">
              <ApiProviderLogo
                providerType={provider.name.toLowerCase()}
                className="size-5"
                title={provider.name}
              />
            </span>
            <span className="min-w-0 pt-px">
              <span className="flex min-w-0 flex-wrap items-center gap-2">
                <span className="truncate text-sm font-medium text-foreground">
                  {instance.name}
                </span>
                <span className="rounded-[6px] border border-control-accent/15 bg-control-accent/8 px-1.5 py-0.5 text-ui-10 leading-none tabular-nums text-control-accent">
                  {instanceModels.length} model
                </span>
              </span>
              <span className="mt-0.5 block truncate text-xs text-muted-foreground">
                {providerOptionLabel(provider.name)}
                {instance.baseUrl ? ` · ${instance.baseUrl}` : ""}
              </span>
              <span className="mt-1 block text-ui-11 leading-4 text-muted-foreground/75">
                {instance.hasCredential
                  ? "Kimlik bilgisi güvenli biçimde kayıtlı"
                  : "Kimlik bilgisi kayıtlı değil"}
              </span>
            </span>
          </button>

          <div className="flex shrink-0 items-center justify-end gap-0.5 text-muted-foreground">
            <Button
              type="button"
              size="icon-sm"
              variant="ghost"
              className="size-7 rounded-[8px] hover:text-foreground"
              disabled={
                !canManage || Boolean(busy) || check?.state === "loading"
              }
              onClick={() => void testConnection(provider, instance)}
              title="Bağlantıyı test et"
              aria-label={`${instance.name} bağlantısını test et`}
            >
              {check?.state === "loading" ? (
                <Spinner className="size-3.5" />
              ) : (
                <HugeiconsIcon icon={Wifi02Icon} className="size-4" />
              )}
            </Button>
            <Button
              type="button"
              size="icon-sm"
              variant="ghost"
              className="size-7 rounded-[8px] hover:text-foreground"
              disabled={!canManage || Boolean(busy)}
              onClick={() => void openEdit(provider, instance)}
              title="Bağlantıyı düzenle"
              aria-label={`${instance.name} bağlantısını düzenle`}
            >
              <HugeiconsIcon icon={Edit03Icon} className="size-4" />
            </Button>
            <Button
              type="button"
              size="icon-sm"
              variant="ghost"
              className="size-7 rounded-[8px] hover:text-destructive"
              disabled={!canManage || Boolean(busy)}
              onClick={() => {
                setExpandedId(instance.id);
                setEditingId("");
                setPendingDeleteId(instance.id);
              }}
              title="Bağlantıyı kaldır"
              aria-label={`${instance.name} bağlantısını kaldır`}
            >
              <HugeiconsIcon icon={Delete02Icon} className="size-4" />
            </Button>
            <Button
              type="button"
              size="icon-sm"
              variant="ghost"
              className="size-7 rounded-[8px]"
              onClick={() => setExpandedId(expanded ? "" : instance.id)}
              aria-label={expanded ? "Ayrıntıları kapat" : "Ayrıntıları aç"}
            >
              <HugeiconsIcon
                icon={ArrowDown01Icon}
                className={`size-4 transition-transform duration-200 ${expanded ? "rotate-180" : ""}`}
              />
            </Button>
          </div>
        </div>

        {check ? (
          <p
            role="status"
            className={`border-border/60 border-t px-4 py-2 text-xs ${
              check.state === "error"
                ? "text-destructive"
                : "text-muted-foreground"
            }`}
          >
            {check.message}
          </p>
        ) : null}

        {expanded ? (
          <div
            data-testid="platform-connection-details"
            className="space-y-4 border-border/60 border-t bg-background/45 px-3 py-5 sm:px-4 sm:py-6"
          >
            {deleting ? (
              <div className="flex flex-wrap items-center justify-between gap-3 rounded-[8px] border border-destructive/25 bg-destructive/5 p-3">
                <div className="min-w-0">
                  <div className="text-sm font-medium text-foreground">
                    Bu bağlantı kaldırılsın mı?
                  </div>
                  <p className="mt-0.5 text-xs text-muted-foreground">
                    Instance ve ona bağlı model kayıtları artık kullanılamaz.
                  </p>
                </div>
                <div className="flex gap-2">
                  <Button
                    type="button"
                    size="sm"
                    variant="ghost"
                    onClick={() => setPendingDeleteId("")}
                  >
                    Vazgeç
                  </Button>
                  <Button
                    type="button"
                    size="sm"
                    variant="destructive"
                    disabled={Boolean(busy)}
                    onClick={() =>
                      void runMutation(`drop:${instance.id}`, () =>
                        deleteProviderInstances(provider.name, [instance.name]),
                      ).then((removed) => {
                        if (!removed) return;
                        setExpandedId("");
                        setPendingDeleteId("");
                      })
                    }
                  >
                    Bağlantıyı kaldır
                  </Button>
                </div>
              </div>
            ) : editing ? (
              <section className="overflow-hidden rounded-[8px] border border-border/70 bg-muted/[0.12]">
                <div className="border-border/60 border-b px-4 py-3">
                  <h3 className="text-sm font-medium text-foreground">
                    Bağlantıyı düzenle
                  </h3>
                  <p className="mt-0.5 text-xs text-muted-foreground">
                    Kayıtlı API key gösterilmez. Yalnız değiştirecekseniz yeni
                    bir değer girin.
                  </p>
                </div>
                <div className="divide-y divide-border/60">
                  <div className="grid grid-cols-[minmax(140px,0.8fr)_minmax(0,1.2fr)] items-center gap-4 px-4 py-3 @max-[520px]:grid-cols-1">
                    <Label htmlFor={`edit-instance-${instance.id}`}>
                      Instance name
                    </Label>
                    <Input
                      id={`edit-instance-${instance.id}`}
                      value={editName}
                      onChange={(event) => setEditName(event.target.value)}
                    />
                  </div>
                  <div className="grid grid-cols-[minmax(140px,0.8fr)_minmax(0,1.2fr)] items-center gap-4 px-4 py-3 @max-[520px]:grid-cols-1">
                    <div>
                      <Label htmlFor={`edit-key-${instance.id}`}>API key</Label>
                      <p className="mt-0.5 text-xs text-muted-foreground">
                        Boş bırakırsanız mevcut key korunur.
                      </p>
                    </div>
                    <Input
                      id={`edit-key-${instance.id}`}
                      type="password"
                      autoComplete="new-password"
                      placeholder="Değiştirmek için yeni key girin"
                      value={editApiKey}
                      onChange={(event) => setEditApiKey(event.target.value)}
                    />
                  </div>
                  <div className="grid grid-cols-[minmax(140px,0.8fr)_minmax(0,1.2fr)] items-center gap-4 px-4 py-3 @max-[520px]:grid-cols-1">
                    <Label htmlFor={`edit-url-${instance.id}`}>Base URL</Label>
                    <Input
                      id={`edit-url-${instance.id}`}
                      inputMode="url"
                      value={editBaseUrl}
                      onChange={(event) => setEditBaseUrl(event.target.value)}
                    />
                  </div>
                </div>
                <div className="flex flex-wrap justify-end gap-2 border-border/60 border-t px-4 py-3">
                  <Button
                    type="button"
                    size="sm"
                    variant="ghost"
                    onClick={() => {
                      setEditApiKey("");
                      setEditingId("");
                    }}
                  >
                    İptal
                  </Button>
                  <Button
                    type="button"
                    size="sm"
                    disabled={!editName.trim() || Boolean(busy)}
                    onClick={() => {
                      const secret = editApiKey;
                      void runMutation(`update:${instance.id}`, () =>
                        updateProviderInstance(provider.name, instance.name, {
                          instanceName: editName,
                          apiKey: secret,
                          baseUrl: editBaseUrl,
                          region: editRegion,
                          verify: true,
                        }),
                      ).then((updated) => {
                        if (!updated) return;
                        setEditApiKey("");
                        setEditingId("");
                      });
                    }}
                  >
                    {busy === `update:${instance.id}` ? (
                      <Spinner />
                    ) : (
                      "Değişiklikleri kaydet"
                    )}
                  </Button>
                </div>
              </section>
            ) : (
              <>
                {!canManage ? (
                  <p role="status" className="text-sm text-muted-foreground">
                    Model ve varsayılan değişiklikleri owner veya admin yetkisi
                    gerektirir.
                  </p>
                ) : null}
                <ModelConfiguration
                  busy={busy}
                  canManage={canManage}
                  defaults={defaults}
                  fixedTarget={`${provider.name}\u0000${instance.name}`}
                  models={instanceModels}
                  providers={[fixedProvider]}
                  runMutation={runMutation}
                />
                <PipelineCatalog
                  loadPipelines={loadPipelines}
                  pipelines={pipelines}
                  state={pipelineState}
                />
                {canManage ? (
                  <PlatformModelTools models={instanceModels} />
                ) : (
                  <div className="rounded-[8px] border border-border/70 bg-muted/[0.12] p-4 text-sm text-muted-foreground">
                    Model araçları owner veya admin yetkisi gerektirir.
                  </div>
                )}
              </>
            )}
          </div>
        ) : null}
      </article>
    );
  });
}

function ModelConfiguration({
  busy,
  canManage,
  defaults,
  fixedTarget,
  models,
  providers,
  runMutation,
}: {
  busy: string;
  canManage: boolean;
  defaults: PlatformDefaultModel[];
  fixedTarget?: string;
  models: PlatformModel[];
  providers: ProviderView[];
  runMutation: (
    key: string,
    action: () => Promise<unknown>,
  ) => Promise<boolean>;
}) {
  const firstInstance = providers.flatMap((provider) =>
    provider.instances.map((instance) => ({
      provider: provider.name,
      instance: instance.name,
    })),
  )[0];
  const [target, setTarget] = useState("");
  const [modelName, setModelName] = useState("");
  const [capability, setCapability] = useState("chat");
  const [catalogModels, setCatalogModels] = useState<PlatformModel[]>([]);
  const [catalogState, setCatalogState] = useState<
    "idle" | "loading" | "ready" | "empty" | "error"
  >("idle");
  const [catalogError, setCatalogError] = useState("");
  const [catalogReload, setCatalogReload] = useState(0);
  const [configuredModelSnapshot, setConfiguredModelSnapshot] = useState<{
    models: PlatformModel[];
    target: string;
  } | null>(null);
  const [pendingModelDeleteId, setPendingModelDeleteId] = useState("");
  const [confirmedDefaultIds, setConfirmedDefaultIds] = useState<
    Record<string, string>
  >({});
  const [pendingEmbeddingDefault, setPendingEmbeddingDefault] = useState<{
    currentModel?: PlatformModel;
    nextModel: PlatformModel;
  } | null>(null);
  const selectedTarget =
    fixedTarget ||
    target ||
    (firstInstance
      ? `${firstInstance.provider}\u0000${firstInstance.instance}`
      : "");
  const managedModels = useMemo(() => {
    if (configuredModelSnapshot?.target !== selectedTarget) return models;

    const grouped = new Map<string, PlatformModel>();
    for (const configured of configuredModelSnapshot.models) {
      const activeRecord = models.find(
        (candidate) =>
          candidate.name === configured.name &&
          candidate.providerName === configured.providerName &&
          candidate.instanceName === configured.instanceName,
      );
      const previous = grouped.get(configured.name);
      const capabilities = Array.from(
        new Set([
          ...(previous?.capabilities ?? []),
          ...(activeRecord?.capabilities ?? []),
          ...configured.capabilities,
        ]),
      );
      const statuses = [previous?.status, configured.status].filter(Boolean);
      grouped.set(configured.name, {
        ...configured,
        id: activeRecord?.id || previous?.id || configured.id,
        providerId:
          activeRecord?.providerId ||
          previous?.providerId ||
          configured.providerId,
        instanceId:
          activeRecord?.instanceId ||
          previous?.instanceId ||
          configured.instanceId,
        capabilities,
        status: statuses.includes("active") ? "active" : "inactive",
      });
    }
    return Array.from(grouped.values());
  }, [configuredModelSnapshot, models, selectedTarget]);
  const availableCapabilities = useMemo(
    () =>
      Array.from(
        new Set(managedModels.flatMap((model) => model.capabilities)),
      ).sort(),
    [managedModels],
  );
  const defaultCapabilities = Array.from(
    new Set(["chat", "embedding", "rerank", ...availableCapabilities]),
  );
  const activeModelCount = managedModels.filter(
    (model) => model.status !== "inactive",
  ).length;
  const assignedDefaultCount = defaultCapabilities.filter((item) => {
    if (confirmedDefaultIds[item]) return true;
    const saved = defaults.find(
      (modelDefault) => modelDefault.capability === item,
    );
    return managedModels.some(
      (model) =>
        model.status !== "inactive" &&
        model.capabilities.includes(item) &&
        defaultMatchesModel(saved, model),
    );
  }).length;

  useEffect(() => {
    const controller = new AbortController();
    const [provider, instance] = selectedTarget.split("\u0000");
    if (!provider || !instance) {
      setCatalogModels([]);
      setConfiguredModelSnapshot(null);
      setCatalogState("idle");
      setCatalogError("");
      return () => controller.abort();
    }
    setCatalogState("loading");
    setCatalogError("");
    void Promise.allSettled([
      listSupportedInstanceModels(provider, instance, controller.signal),
      listProviderModels(provider, controller.signal),
      listInstanceModels(provider, instance, controller.signal),
    ])
      .then(
        ([
          supportedModelsResult,
          providerModelsResult,
          instanceModelsResult,
        ]) => {
          if (controller.signal.aborted) return;
          const supportedModels =
            supportedModelsResult.status === "fulfilled"
              ? supportedModelsResult.value
              : [];
          const providerModels =
            providerModelsResult.status === "fulfilled"
              ? providerModelsResult.value
              : [];
          const instanceModels =
            instanceModelsResult.status === "fulfilled"
              ? instanceModelsResult.value
              : [];
          if (instanceModelsResult.status === "fulfilled") {
            setConfiguredModelSnapshot((current) => {
              if (current?.target !== selectedTarget) {
                return {
                  models: instanceModels,
                  target: selectedTarget,
                };
              }

              // Some deployed runtimes omit inactive rows from this inventory.
              // Keep models already known by this mounted connection so a
              // successful disable remains visible and can be reversed. A
              // confirmed delete removes the local row before this merge.
              const retainedModels = current.models.filter(
                (currentModel) =>
                  !instanceModels.some((instanceModel) =>
                    sameConfiguredModel(currentModel, instanceModel),
                  ),
              );
              return {
                models: [...instanceModels, ...retainedModels],
                target: selectedTarget,
              };
            });
          }
          const seen = new Set<string>();
          const nextModels = [
            ...instanceModels,
            ...supportedModels,
            ...providerModels,
          ].filter((model) => {
            const key = model.name;
            if (seen.has(key)) return false;
            seen.add(key);
            return true;
          });
          setCatalogModels(nextModels);
          if (
            supportedModelsResult.status === "rejected" &&
            instanceModelsResult.status === "rejected" &&
            providerModelsResult.status === "rejected"
          ) {
            setCatalogError(errorMessage(supportedModelsResult.reason));
            setCatalogState("error");
            return;
          }
          setCatalogState(nextModels.length ? "ready" : "empty");
        },
      )
      .catch((catalogLoadError) => {
        if (controller.signal.aborted) return;
        setCatalogModels([]);
        setCatalogError(errorMessage(catalogLoadError));
        setCatalogState("error");
      });
    return () => controller.abort();
  }, [catalogReload, selectedTarget]);

  const capabilityOptions = Array.from(
    new Set([
      "chat",
      "embedding",
      "rerank",
      ...catalogModels.flatMap((model) => model.capabilities),
    ]),
  );

  const selectModel = async (value: string) => {
    const model = catalogModels.find((candidate) => candidate.name === value);
    setModelName(value);
    if (model?.capabilities[0]) setCapability(model.capabilities[0]);
    const [provider] = selectedTarget.split("\u0000");
    if (!provider || !model) return;
    // Instance models already carry the fields needed by this form. Provider
    // detail is optional enrichment and may be unavailable in a mixed-version
    // hybrid runtime, so it must not surface as a user-facing connection error.
    if (!model.instanceName) {
      try {
        await getProviderModel(provider, model.name);
      } catch (detailError) {
        if (!isPlatformApiError(detailError) || detailError.httpStatus !== 404)
          setCapability(model.capabilities[0] || "chat");
      }
    }
  };

  const selectedModelAlreadyAdded = managedModels.some(
    (model) =>
      model.name === modelName &&
      `${model.providerName}\u0000${model.instanceName}` === selectedTarget,
  );

  const saveDefaultSelection = async (
    model: PlatformModel,
    capability: string,
  ) => {
    const saved = await runMutation(`default:${capability}`, () =>
      setDefaultModel(model, capability),
    );
    if (!saved) return false;
    setConfirmedDefaultIds((current) => ({
      ...current,
      [capability]: model.id,
    }));
    return true;
  };

  return (
    <Section
      title="Models ve varsayılanlar"
      description="Bu bağlantıda kullanılacak modelleri ekleyin, durumlarını yönetin ve işlem türlerine göre varsayılanları belirleyin."
    >
      <div className="overflow-hidden rounded-[10px] border border-border/70 bg-background/45">
        <div className="flex flex-wrap items-center gap-x-3 gap-y-1 border-border/60 border-b bg-muted/[0.12] px-3 py-2 text-ui-10 text-muted-foreground">
          <span>
            <strong className="font-medium tabular-nums text-foreground">
              {catalogModels.length}
            </strong>{" "}
            katalog modeli
          </span>
          <span aria-hidden="true" className="size-1 rounded-full bg-border" />
          <span>
            <strong className="font-medium tabular-nums text-foreground">
              {activeModelCount}
            </strong>{" "}
            etkin
          </span>
          <span aria-hidden="true" className="size-1 rounded-full bg-border" />
          <span>
            <strong className="font-medium tabular-nums text-foreground">
              {assignedDefaultCount}/{defaultCapabilities.length}
            </strong>{" "}
            varsayılan atandı
          </span>
        </div>

        <section
          aria-labelledby="add-model-heading"
          className="overflow-hidden border-border/60 border-b"
        >
          <div className="flex flex-wrap items-center justify-between gap-2 border-border/60 border-b bg-control-accent/[0.03] px-3 py-2.5">
            <div className="flex min-w-0 items-center gap-2">
              <span className="flex size-5 shrink-0 items-center justify-center rounded-[5px] bg-control-accent/[0.09] text-ui-10 font-semibold tabular-nums text-control-accent">
                01
              </span>
              <h4
                id="add-model-heading"
                className="text-xs font-medium text-foreground"
              >
                Model ekle
              </h4>
            </div>
            <span className="shrink-0 text-ui-10 tabular-nums text-muted-foreground">
              {catalogState === "loading"
                ? "Katalog yükleniyor…"
                : `${catalogModels.length} kullanılabilir`}
            </span>
          </div>

          <div
            className={`grid items-end gap-2.5 p-3 ${fixedTarget ? "@min-[600px]:grid-cols-[minmax(0,1.45fr)_minmax(10rem,0.75fr)_auto]" : "@min-[520px]:grid-cols-2 @min-[720px]:grid-cols-[minmax(11rem,0.9fr)_minmax(0,1.35fr)_minmax(9rem,0.65fr)_auto]"}`}
          >
            {!fixedTarget ? (
              <div className="space-y-1.5">
                <Label htmlFor="model-provider-instance">Bağlantı</Label>
                <Select
                  value={selectedTarget}
                  disabled={!canManage}
                  onValueChange={(value) => {
                    setTarget(value);
                    setModelName("");
                  }}
                >
                  <SelectTrigger
                    id="model-provider-instance"
                    className="h-8 w-full bg-background"
                  >
                    <SelectValue placeholder="Bağlantı seçin" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectGroup>
                      {providers.flatMap((provider) =>
                        provider.instances.map((instance) => (
                          <SelectItem
                            key={instance.id}
                            value={`${provider.name}\u0000${instance.name}`}
                          >
                            {provider.name} / {instance.name}
                          </SelectItem>
                        )),
                      )}
                    </SelectGroup>
                  </SelectContent>
                </Select>
              </div>
            ) : null}
            <div className="space-y-1.5">
              <Label htmlFor="platform-model-name">Model</Label>
              <Select
                value={modelName}
                disabled={
                  !canManage ||
                  !selectedTarget ||
                  catalogState === "loading" ||
                  catalogModels.length === 0
                }
                onValueChange={(value) => void selectModel(value)}
              >
                <SelectTrigger
                  id="platform-model-name"
                  aria-label="Model name"
                  className="h-8 w-full bg-background"
                >
                  <SelectValue
                    placeholder={
                      catalogState === "loading"
                        ? "Modeller yükleniyor…"
                        : selectedTarget
                          ? "Model seçin"
                          : "Önce bağlantı seçin"
                    }
                  />
                </SelectTrigger>
                <SelectContent>
                  <SelectGroup>
                    {catalogModels.map((model) => (
                      <SelectItem
                        key={`${model.providerName}:${model.id}`}
                        value={model.name}
                      >
                        {model.name}
                      </SelectItem>
                    ))}
                  </SelectGroup>
                </SelectContent>
              </Select>
            </div>
            <div className="space-y-1.5">
              <Label htmlFor="platform-model-capability">Kullanım alanı</Label>
              <Select
                value={capability}
                disabled={!canManage || !modelName}
                onValueChange={setCapability}
              >
                <SelectTrigger
                  id="platform-model-capability"
                  aria-label="Capability"
                  className="h-8 w-full bg-background"
                >
                  <SelectValue placeholder="Kullanım alanı seçin" />
                </SelectTrigger>
                <SelectContent>
                  <SelectGroup>
                    {capabilityOptions.map((item) => (
                      <SelectItem key={item} value={item}>
                        {item}
                      </SelectItem>
                    ))}
                  </SelectGroup>
                </SelectContent>
              </Select>
            </div>
            <Button
              className="h-8 min-w-24 px-3 text-xs transition-transform active:translate-y-px"
              disabled={
                !canManage ||
                !selectedTarget ||
                !modelName.trim() ||
                selectedModelAlreadyAdded ||
                Boolean(busy)
              }
              onClick={() => {
                const [provider, instance] = selectedTarget.split("\u0000");
                const selectedModelName = modelName;
                void runMutation(`model-add:${selectedModelName}`, () =>
                  addInstanceModel(provider, instance, {
                    modelName: selectedModelName,
                    capabilities: [capability],
                  }),
                ).then((added) => {
                  if (added) {
                    setModelName("");
                    setCatalogReload((value) => value + 1);
                  }
                });
              }}
            >
              {busy.startsWith("model-add:") ? (
                <>
                  <Spinner /> Ekleniyor…
                </>
              ) : selectedModelAlreadyAdded ? (
                "Model zaten ekli"
              ) : (
                "Model ekle"
              )}
            </Button>
          </div>

          {catalogState === "loading" ? (
            <output className="flex items-center gap-2 border-border/60 border-t px-3 py-2 text-xs text-muted-foreground">
              <Spinner /> Servisteki modeller alınıyor…
            </output>
          ) : null}
          {catalogState === "empty" ? (
            <output className="block border-border/60 border-t px-3 py-2 text-xs leading-relaxed text-muted-foreground">
              Bu bağlantı için servis model döndürmedi. Bağlantı testini
              çalıştırın veya API key ve Base URL değerlerini kontrol edin.
            </output>
          ) : null}
          {catalogState === "error" ? (
            <div
              role="alert"
              className="flex flex-wrap items-center justify-between gap-2 border-border/60 border-t bg-destructive/5 px-3 py-2 text-xs text-destructive"
            >
              <span>{catalogError || "Model kataloğu alınamadı."}</span>
              <Button
                type="button"
                size="sm"
                variant="outline"
                onClick={() => setCatalogReload((value) => value + 1)}
              >
                Yeniden dene
              </Button>
            </div>
          ) : null}
        </section>

        <section
          aria-labelledby="configured-models-heading"
          className="overflow-hidden border-border/60 border-b"
        >
          <div className="flex items-center justify-between gap-2 border-border/60 border-b bg-muted/[0.08] px-3 py-2.5">
            <div className="flex min-w-0 items-center gap-2">
              <span className="flex size-5 shrink-0 items-center justify-center rounded-[5px] bg-muted/60 text-ui-10 font-semibold tabular-nums text-muted-foreground">
                02
              </span>
              <h4
                id="configured-models-heading"
                className="text-xs font-medium text-foreground"
              >
                Ekli modeller
              </h4>
            </div>
            <span className="shrink-0 text-ui-10 tabular-nums text-muted-foreground">
              {managedModels.length} model
            </span>
          </div>

          {managedModels.length === 0 ? (
            <div className="m-3 rounded-[8px] border border-dashed border-border/80 bg-muted/[0.1] px-4 py-5 text-center">
              <p className="text-sm font-medium text-foreground">
                Henüz model eklenmedi
              </p>
              <p className="mt-1 text-xs text-muted-foreground">
                Yukarıdan bir servis modeli ve capability seçerek ekleyin.
              </p>
            </div>
          ) : (
            <div className="divide-y divide-border/55 bg-background/20">
              {managedModels.map((model) => {
                const inactive = model.status === "inactive";
                const deleting = pendingModelDeleteId === model.id;
                return (
                  <article key={`${model.instanceId}:${model.id}`}>
                    <div className="grid min-w-0 grid-cols-[minmax(0,1fr)_auto] items-center gap-2.5 px-3 py-2.5 transition-colors duration-200 hover:bg-muted/[0.16] @max-[520px]:grid-cols-1">
                      <div className="flex min-w-0 items-start gap-2.5">
                        <span className="flex size-7 shrink-0 items-center justify-center rounded-[7px] border border-border/70 bg-background/80">
                          <ApiProviderLogo
                            providerType={model.providerName.toLowerCase()}
                            className="size-4"
                            title={model.providerName}
                          />
                        </span>
                        <div className="min-w-0">
                          <div className="flex min-w-0 flex-wrap items-center gap-2">
                            <h5 className="truncate text-sm font-medium text-foreground">
                              {model.name}
                            </h5>
                            <span
                              className={`inline-flex items-center gap-1.5 text-ui-10 ${
                                inactive
                                  ? "text-muted-foreground"
                                  : "text-emerald-600 dark:text-emerald-400"
                              }`}
                            >
                              <span
                                aria-hidden="true"
                                className={`size-1.5 rounded-full ${
                                  inactive
                                    ? "bg-muted-foreground/50"
                                    : "bg-emerald-500"
                                }`}
                              />
                              {inactive ? "Devre dışı" : "Etkin"}
                            </span>
                          </div>
                          <p className="mt-0.5 truncate text-xs text-muted-foreground">
                            {providerOptionLabel(model.providerName)} ·{" "}
                            {model.instanceName}
                          </p>
                          <div className="mt-1.5 flex flex-wrap gap-1">
                            {model.capabilities.length ? (
                              model.capabilities.map((item) => (
                                <Badge
                                  key={`${model.id}:${item}`}
                                  variant="outline"
                                  className="h-[18px] rounded-[5px] bg-muted/30 px-1.5 text-ui-10 font-normal text-muted-foreground"
                                >
                                  {item}
                                </Badge>
                              ))
                            ) : (
                              <span className="text-ui-10 text-muted-foreground">
                                Capability tanımlı değil
                              </span>
                            )}
                          </div>
                        </div>
                      </div>

                      <div className="flex shrink-0 items-center justify-end gap-0.5">
                        <Button
                          type="button"
                          size="sm"
                          variant="ghost"
                          className="h-7 rounded-[8px] px-2 text-xs text-muted-foreground hover:text-foreground"
                          disabled={!canManage || Boolean(busy)}
                          onClick={() =>
                            void runMutation(`toggle:${model.id}`, () =>
                              updateInstanceModel(
                                model.providerName,
                                model.instanceName,
                                model.name,
                                { status: inactive ? "active" : "inactive" },
                              ),
                            ).then((updated) => {
                              if (!updated) return;
                              const nextStatus = inactive
                                ? "active"
                                : "inactive";
                              setConfiguredModelSnapshot((current) => {
                                if (current?.target !== selectedTarget) {
                                  return {
                                    models: [{ ...model, status: nextStatus }],
                                    target: selectedTarget,
                                  };
                                }
                                return {
                                  ...current,
                                  models: current.models.map((currentModel) =>
                                    sameConfiguredModel(currentModel, model)
                                      ? {
                                          ...currentModel,
                                          status: nextStatus,
                                        }
                                      : currentModel,
                                  ),
                                };
                              });
                              setCatalogReload((value) => value + 1);
                            })
                          }
                        >
                          {inactive ? "Etkinleştir" : "Devre dışı bırak"}
                        </Button>
                        <Button
                          type="button"
                          size="icon-sm"
                          variant="ghost"
                          className="size-7 rounded-[8px] text-muted-foreground hover:text-destructive"
                          disabled={!canManage || Boolean(busy)}
                          onClick={() => setPendingModelDeleteId(model.id)}
                          title="Modeli kaldır"
                          aria-label={`${model.name} modelini kaldır`}
                        >
                          <HugeiconsIcon
                            icon={Delete02Icon}
                            className="size-4"
                          />
                        </Button>
                      </div>
                    </div>

                    {deleting ? (
                      <div className="flex flex-wrap items-center justify-between gap-3 border-border/60 border-t bg-destructive/5 px-3 py-3 sm:px-4">
                        <div className="min-w-0">
                          <p className="text-sm font-medium text-foreground">
                            {model.name} kaldırılsın mı?
                          </p>
                          <p className="mt-0.5 text-xs text-muted-foreground">
                            Bu model artık varsayılan veya araç modeli olarak
                            kullanılamaz.
                          </p>
                        </div>
                        <div className="flex gap-2">
                          <Button
                            type="button"
                            size="sm"
                            variant="ghost"
                            onClick={() => setPendingModelDeleteId("")}
                          >
                            Vazgeç
                          </Button>
                          <Button
                            type="button"
                            size="sm"
                            variant="destructive"
                            disabled={Boolean(busy)}
                            onClick={() =>
                              void runMutation(`model-delete:${model.id}`, () =>
                                deleteInstanceModels(
                                  model.providerName,
                                  model.instanceName,
                                  [model.name],
                                ),
                              ).then((removed) => {
                                if (removed) {
                                  setConfiguredModelSnapshot((current) =>
                                    current?.target === selectedTarget
                                      ? {
                                          ...current,
                                          models: current.models.filter(
                                            (currentModel) =>
                                              !sameConfiguredModel(
                                                currentModel,
                                                model,
                                              ),
                                          ),
                                        }
                                      : current,
                                  );
                                  setPendingModelDeleteId("");
                                  setCatalogReload((value) => value + 1);
                                }
                              })
                            }
                          >
                            Modeli kaldır
                          </Button>
                        </div>
                      </div>
                    ) : null}
                  </article>
                );
              })}
            </div>
          )}
        </section>

        <section
          aria-labelledby="default-models-heading"
          className="overflow-hidden"
        >
          <div className="flex flex-wrap items-center justify-between gap-2 border-border/60 border-b bg-muted/[0.08] px-3 py-2.5">
            <div className="flex min-w-0 items-center gap-2">
              <span className="flex size-5 shrink-0 items-center justify-center rounded-[5px] bg-muted/60 text-ui-10 font-semibold tabular-nums text-muted-foreground">
                03
              </span>
              <h4
                id="default-models-heading"
                className="text-xs font-medium text-foreground"
              >
                Varsayılan modeller
              </h4>
            </div>
            <span className="shrink-0 text-ui-10 tabular-nums text-muted-foreground">
              {assignedDefaultCount}/{defaultCapabilities.length} atandı
            </span>
          </div>
          <div className="divide-y divide-border/55 bg-background/20">
            {defaultCapabilities.map((capability) => {
              const savedDefault = defaults.find(
                (item) => item.capability === capability,
              );
              const options = managedModels.filter(
                (model) =>
                  model.status !== "inactive" &&
                  model.capabilities.includes(capability),
              );
              const selectedFromServer = options.find((model) =>
                defaultMatchesModel(savedDefault, model),
              );
              const confirmedSelection = options.find(
                (model) => model.id === confirmedDefaultIds[capability],
              );
              const selectedId =
                confirmedSelection?.id || selectedFromServer?.id || "";
              return (
                <div
                  key={capability}
                  className="grid items-center gap-3 px-3 py-3 transition-colors duration-200 hover:bg-muted/[0.16] @min-[600px]:grid-cols-[minmax(0,1fr)_minmax(14rem,0.9fr)]"
                >
                  <div className="min-w-0">
                    <div className="flex flex-wrap items-center gap-2">
                      <Label
                        htmlFor={`default-${capability}`}
                        className="text-xs font-medium text-foreground"
                      >
                        {capability} varsayılanı
                      </Label>
                      {selectedId ? (
                        <span className="inline-flex items-center gap-1.5 text-ui-10 text-emerald-600 dark:text-emerald-400">
                          <span
                            aria-hidden="true"
                            className="size-1.5 rounded-full bg-emerald-500"
                          />
                          Atandı
                        </span>
                      ) : null}
                    </div>
                    <p className="mt-0.5 max-w-[54ch] text-ui-11 leading-snug text-muted-foreground">
                      {capability === "chat"
                        ? "Sohbet yanıtları ve metin üretimi için kullanılır."
                        : capability === "embedding"
                          ? "Dataset indeksleme ve benzerlik araması için kullanılır."
                          : capability === "rerank"
                            ? "Arama sonuçlarını yeniden sıralamak için kullanılır."
                            : `${capability} işlemleri için öncelikli model.`}
                    </p>
                  </div>
                  <div className="min-w-0">
                    <Select
                      disabled={
                        !canManage || options.length === 0 || Boolean(busy)
                      }
                      value={selectedId}
                      onValueChange={(value) => {
                        const model = options.find((item) => item.id === value);
                        if (!model) return;
                        if (model.id === selectedId) return;
                        if (capability === "embedding") {
                          setPendingEmbeddingDefault({
                            currentModel:
                              confirmedSelection || selectedFromServer,
                            nextModel: model,
                          });
                          return;
                        }
                        void saveDefaultSelection(model, capability);
                      }}
                    >
                      <SelectTrigger
                        id={`default-${capability}`}
                        className="h-8 w-full bg-background text-xs"
                      >
                        <SelectValue
                          placeholder={
                            options.length
                              ? "Varsayılan model seçin"
                              : "Uyumlu model yok"
                          }
                        />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectGroup>
                          {options.map((model) => (
                            <SelectItem
                              key={`${capability}:${model.id}`}
                              value={model.id}
                            >
                              {model.name} — {model.providerName}
                            </SelectItem>
                          ))}
                        </SelectGroup>
                      </SelectContent>
                    </Select>
                  </div>
                </div>
              );
            })}
          </div>
        </section>
      </div>

      <AlertDialog
        open={pendingEmbeddingDefault !== null}
        onOpenChange={(open) => {
          if (!open && busy !== "default:embedding") {
            setPendingEmbeddingDefault(null);
          }
        }}
      >
        <AlertDialogContent className="sm:!max-w-[28rem]">
          <AlertDialogHeader>
            <AlertDialogMedia className="size-12 rounded-[14px] bg-amber-500/10 text-amber-700 dark:bg-amber-400/10 dark:text-amber-300">
              <HugeiconsIcon icon={Alert02Icon} className="size-6" />
            </AlertDialogMedia>
            <AlertDialogTitle>
              Embedding varsayılanını değiştir?
            </AlertDialogTitle>
            <AlertDialogDescription className="leading-relaxed">
              Embedding varsayılanını değiştirmek mevcut dataset indeksleriyle
              uyumsuzluk yaratabilir. Değişiklikten sonra mevcut dataset’leri
              yeniden indekslemeniz gerekebilir.
            </AlertDialogDescription>
          </AlertDialogHeader>

          <div className="grid grid-cols-[minmax(0,1fr)_auto_minmax(0,1fr)] items-center gap-3 rounded-[12px] border border-border/70 bg-muted/30 p-3">
            <div className="min-w-0">
              <p className="text-ui-10 font-medium tracking-wide text-muted-foreground uppercase">
                Mevcut
              </p>
              <p className="mt-1 truncate text-sm font-medium text-foreground">
                {pendingEmbeddingDefault?.currentModel?.name || "Atanmamış"}
              </p>
            </div>
            <span className="flex size-7 items-center justify-center rounded-[8px] border border-border/70 bg-background text-muted-foreground">
              <HugeiconsIcon
                icon={ArrowRight01Icon}
                className="size-4"
                aria-hidden="true"
              />
            </span>
            <div className="min-w-0 text-right">
              <p className="text-ui-10 font-medium tracking-wide text-muted-foreground uppercase">
                Yeni
              </p>
              <p className="mt-1 truncate text-sm font-medium text-foreground">
                {pendingEmbeddingDefault?.nextModel.name}
              </p>
            </div>
          </div>

          <AlertDialogFooter>
            <AlertDialogCancel disabled={busy === "default:embedding"}>
              Vazgeç
            </AlertDialogCancel>
            <AlertDialogAction
              disabled={busy === "default:embedding"}
              onClick={(event) => {
                event.preventDefault();
                const pending = pendingEmbeddingDefault;
                if (!pending) return;
                void saveDefaultSelection(pending.nextModel, "embedding").then(
                  (saved) => {
                    if (saved) setPendingEmbeddingDefault(null);
                  },
                );
              }}
            >
              {busy === "default:embedding" ? (
                <>
                  <Spinner /> Kaydediliyor…
                </>
              ) : (
                "Varsayılanı değiştir"
              )}
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </Section>
  );
}
