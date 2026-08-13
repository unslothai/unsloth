import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import { Spinner } from "@/components/ui/spinner";
import { useSettingsDialogStore } from "@/features/settings/stores/settings-dialog-store";
import {
  type PlatformDefaultModel,
  type PlatformModel,
  getCurrentPlatformTenantModels,
  getDefaultModels,
  isPlatformApiError,
  isPlatformAuthEnabled,
  isPlatformModelToolsEnabled,
  listTenantModels,
  mergePlatformDefaultModels,
  setDefaultModel,
} from "@/integrations/platform-backend";
import { Tick02Icon } from "@/lib/tick-icon";
import { cn } from "@/lib/utils";
import {
  ArrowDown01Icon,
  CloudIcon,
  Search01Icon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { ApiProviderLogo } from "../api-provider-logo";

type LoadState = "loading" | "ready" | "error";

interface PlatformChatModelSelectorProps {
  className?: string;
  contentDataTour?: string;
  onOpenChange?: (open: boolean) => void;
  open?: boolean;
  triggerDataTour?: string;
}

const modelKey = (model: PlatformModel) =>
  [model.providerName, model.instanceName, model.id].join("\u0000");

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

function providerLogoType(providerName: string): string {
  const normalized = providerName.trim().toLowerCase();
  if (normalized.includes("openai")) return "openai";
  if (normalized.includes("anthropic")) return "anthropic";
  if (normalized.includes("gemini") || normalized.includes("google"))
    return "gemini";
  if (normalized.includes("vllm")) return "vllm";
  if (normalized.includes("openrouter")) return "openrouter";
  if (normalized.includes("ollama")) return "ollama";
  return normalized;
}

const errorMessage = (error: unknown) =>
  error instanceof Error
    ? error.message
    : "Rag Platform model listesi alınamadı.";

/**
 * Chat's compact model control for the Phase 3 Rag Platform contract.
 * It intentionally manages the backend chat default; completion streaming is
 * owned by the separate Phase 8 transport and is not mixed into this picker.
 */
export function PlatformChatModelSelector({
  className,
  contentDataTour,
  onOpenChange,
  open: controlledOpen,
  triggerDataTour,
}: PlatformChatModelSelectorProps) {
  const enabled = isPlatformAuthEnabled() && isPlatformModelToolsEnabled();
  const [uncontrolledOpen, setUncontrolledOpen] = useState(false);
  const open = controlledOpen ?? uncontrolledOpen;
  const setOpen = onOpenChange ?? setUncontrolledOpen;
  const [state, setState] = useState<LoadState>("loading");
  const [models, setModels] = useState<PlatformModel[]>([]);
  const [defaults, setDefaults] = useState<PlatformDefaultModel[]>([]);
  const [query, setQuery] = useState("");
  const [error, setError] = useState("");
  const [savingKey, setSavingKey] = useState("");
  const loadAbortRef = useRef<AbortController | undefined>(undefined);
  const saveAbortRef = useRef<AbortController | undefined>(undefined);

  const load = useCallback(async (background = false) => {
    loadAbortRef.current?.abort();
    const controller = new AbortController();
    loadAbortRef.current = controller;
    if (!background) setState("loading");
    setError("");
    try {
      const [tenantModels, selectedDefaults, tenant] = await Promise.all([
        listTenantModels(controller.signal),
        getDefaultModels(controller.signal),
        getCurrentPlatformTenantModels(controller.signal),
      ]);
      if (controller.signal.aborted) return;
      setModels(tenantModels);
      setDefaults(mergePlatformDefaultModels(selectedDefaults, tenant));
      setState("ready");
    } catch (loadError) {
      if (
        controller.signal.aborted ||
        (isPlatformApiError(loadError) && loadError.isAbort)
      )
        return;
      setError(errorMessage(loadError));
      setState("error");
    }
  }, []);

  useEffect(() => {
    if (enabled) void load();
    else setState("ready");
    return () => {
      loadAbortRef.current?.abort();
      saveAbortRef.current?.abort();
    };
  }, [enabled, load]);

  const chatModels = useMemo(
    () =>
      models
        .filter(
          (model) =>
            model.status !== "inactive" && model.capabilities.includes("chat"),
        )
        .sort(
          (left, right) =>
            left.providerName.localeCompare(right.providerName) ||
            left.instanceName.localeCompare(right.instanceName) ||
            left.name.localeCompare(right.name),
        ),
    [models],
  );

  const selectedModel = useMemo(() => {
    const chatDefault = defaults.find(
      (item) => item.capability === "chat" && item.enabled,
    );
    return chatModels.find((model) => defaultMatchesModel(chatDefault, model));
  }, [chatModels, defaults]);

  const groups = useMemo(() => {
    const needle = query.trim().toLocaleLowerCase();
    const grouped = new Map<string, PlatformModel[]>();
    for (const model of chatModels) {
      const searchable = `${model.name} ${model.providerName} ${model.instanceName}`.toLocaleLowerCase();
      if (needle && !searchable.includes(needle)) continue;
      const key = `${model.providerName}\u0000${model.instanceName}`;
      const current = grouped.get(key);
      if (current) current.push(model);
      else grouped.set(key, [model]);
    }
    return [...grouped.values()];
  }, [chatModels, query]);

  const handleOpenChange = (nextOpen: boolean) => {
    if (nextOpen && enabled && state !== "loading") void load(true);
    if (!nextOpen) setQuery("");
    setOpen(nextOpen);
  };

  const selectModel = async (model: PlatformModel) => {
    const key = modelKey(model);
    saveAbortRef.current?.abort();
    const controller = new AbortController();
    saveAbortRef.current = controller;
    setSavingKey(key);
    setError("");
    try {
      await setDefaultModel(model, "chat", controller.signal);
      if (controller.signal.aborted) return;
      setDefaults((current) => [
        ...current.filter((item) => item.capability !== "chat"),
        {
          capability: "chat",
          enabled: true,
          instanceName: model.instanceName,
          modelId: model.id,
          modelName: model.name,
          providerName: model.providerName,
        },
      ]);
      setOpen(false);
      setQuery("");
    } catch (saveError) {
      if (
        controller.signal.aborted ||
        (isPlatformApiError(saveError) && saveError.isAbort)
      )
        return;
      setError(errorMessage(saveError));
    } finally {
      if (!controller.signal.aborted) setSavingKey("");
    }
  };

  return (
    <Popover open={open} onOpenChange={handleOpenChange}>
      <PopoverTrigger asChild>
        <button
          type="button"
          aria-label={
            selectedModel
              ? `Chat model: ${selectedModel.name} — ${selectedModel.providerName}`
              : "Select model"
          }
          data-tour={triggerDataTour}
          className={cn(
            "group flex h-[var(--studio-chat-control-height,34px)] min-w-0 max-w-[62vw] items-center gap-2 rounded-full px-3 text-sm transition-colors hover:bg-accent sm:max-w-none",
            className,
          )}
        >
          {selectedModel ? (
            <ApiProviderLogo
              providerType={providerLogoType(selectedModel.providerName)}
              className="size-4"
              title={selectedModel.providerName}
            />
          ) : (
            <HugeiconsIcon
              icon={CloudIcon}
              strokeWidth={1.75}
              className="size-4 shrink-0 text-muted-foreground"
            />
          )}
          <span className="min-w-0 truncate font-heading text-ui-16 font-medium leading-tight text-foreground">
            {selectedModel?.name ?? "Select model"}
          </span>
          {selectedModel ? (
            <span className="hidden max-w-28 truncate text-xs text-muted-foreground sm:block">
              {selectedModel.providerName}
            </span>
          ) : null}
          <HugeiconsIcon
            icon={ArrowDown01Icon}
            strokeWidth={1.75}
            className="size-3.5 shrink-0 text-muted-foreground transition-transform group-data-[state=open]:rotate-180"
          />
        </button>
      </PopoverTrigger>

      <PopoverContent
        align="start"
        alignOffset={8}
        sideOffset={4}
        data-tour={contentDataTour}
        className="menu-soft-surface w-[min(380px,calc(100vw-1rem))] gap-0 overflow-hidden rounded-[12px] border border-border/70 p-0 shadow-xl ring-0"
      >
        <div className="border-b border-border/60 px-4 py-3.5">
          <div className="flex items-start justify-between gap-3">
            <div className="min-w-0">
              <h2 className="text-sm font-medium text-foreground">
                Connected models
              </h2>
              <p className="mt-0.5 text-xs leading-relaxed text-muted-foreground">
                Choose the default chat model from Connections.
              </p>
            </div>
            {state === "ready" && chatModels.length > 0 ? (
              <span className="shrink-0 rounded-full bg-muted px-2 py-1 text-ui-10 tabular-nums text-muted-foreground">
                {chatModels.length} model{chatModels.length === 1 ? "" : "s"}
              </span>
            ) : null}
          </div>
        </div>

        {state === "loading" && models.length === 0 ? (
          <div className="flex min-h-36 items-center justify-center gap-2 text-sm text-muted-foreground">
            <Spinner className="size-4" />
            Loading connected models…
          </div>
        ) : (
          <>
            {chatModels.length > 4 ? (
              <div className="border-b border-border/60 p-3">
                <div className="relative">
                  <HugeiconsIcon
                    icon={Search01Icon}
                    strokeWidth={1.75}
                    className="pointer-events-none absolute left-2.5 top-1/2 size-4 -translate-y-1/2 text-muted-foreground"
                  />
                  <Input
                    data-model-picker-search-input
                    value={query}
                    onChange={(event) => setQuery(event.target.value)}
                    placeholder="Search connected models"
                    className="h-9 rounded-[8px] border-border/70 bg-background/70 pl-8"
                  />
                </div>
              </div>
            ) : null}

            <div className="max-h-[min(420px,var(--radix-popover-content-available-height))] overflow-y-auto p-2">
              {groups.length > 0 ? (
                <div className="space-y-1.5">
                  {groups.map((group) => {
                    const first = group[0];
                    if (!first) return null;
                    const groupKey = `${first.providerName}:${first.instanceName}`;
                    return (
                      <section key={groupKey} className="rounded-[9px] border border-border/55 bg-background/35 p-1">
                        <div className="flex min-w-0 items-center gap-2 px-2.5 py-2">
                          <span className="flex size-7 shrink-0 items-center justify-center rounded-[7px] border border-border/60 bg-background">
                            <ApiProviderLogo
                              providerType={providerLogoType(first.providerName)}
                              className="size-4"
                              title={first.providerName}
                            />
                          </span>
                          <div className="min-w-0">
                            <p className="truncate text-xs font-medium text-foreground">
                              {first.providerName}
                            </p>
                            <p className="truncate text-ui-10 text-muted-foreground">
                              {first.instanceName || "Default instance"}
                            </p>
                          </div>
                        </div>
                        <div className="space-y-0.5">
                          {group.map((model) => {
                            const key = modelKey(model);
                            const selected = selectedModel
                              ? modelKey(selectedModel) === key
                              : false;
                            const saving = savingKey === key;
                            return (
                              <button
                                key={key}
                                type="button"
                                aria-label={`${model.name} — ${model.providerName} — ${model.instanceName || "Default instance"}`}
                                data-model-picker-option
                                data-model-picker-active-option={selected || undefined}
                                disabled={Boolean(savingKey)}
                                onClick={() => void selectModel(model)}
                                className={cn(
                                  "flex w-full min-w-0 items-center gap-3 rounded-[7px] px-2.5 py-2.5 text-left transition-colors hover:bg-accent disabled:cursor-wait disabled:opacity-70",
                                  selected && "bg-accent/65",
                                )}
                              >
                                <span className="min-w-0 flex-1">
                                  <span className="block truncate text-sm font-medium text-foreground">
                                    {model.name}
                                  </span>
                                  <span className="mt-0.5 block truncate text-ui-10 text-muted-foreground">
                                    {model.maxTokens
                                      ? `${model.maxTokens.toLocaleString()} max tokens`
                                      : "Chat model"}
                                  </span>
                                </span>
                                {saving ? (
                                  <Spinner className="size-4 text-muted-foreground" />
                                ) : selected ? (
                                  <span className="flex size-5 shrink-0 items-center justify-center rounded-full bg-emerald-500/12 text-emerald-600 dark:text-emerald-400">
                                    <HugeiconsIcon
                                      icon={Tick02Icon}
                                      strokeWidth={2}
                                      className="size-3.5"
                                    />
                                  </span>
                                ) : null}
                              </button>
                            );
                          })}
                        </div>
                      </section>
                    );
                  })}
                </div>
              ) : (
                <div className="px-4 py-8 text-center">
                  <span className="mx-auto flex size-9 items-center justify-center rounded-[9px] border border-border/60 bg-muted/30 text-muted-foreground">
                    <HugeiconsIcon icon={CloudIcon} className="size-4" />
                  </span>
                  <p className="mt-3 text-sm font-medium text-foreground">
                    {query ? "No matching models" : "No active chat models"}
                  </p>
                  <p className="mx-auto mt-1 max-w-64 text-xs leading-relaxed text-muted-foreground">
                    {query
                      ? "Try a different model, provider, or instance name."
                      : "Add a model with the chat capability in Connections first."}
                  </p>
                  {!query ? (
                    <Button
                      type="button"
                      variant="outline"
                      size="sm"
                      className="mt-4 rounded-[8px]"
                      onClick={() => {
                        setOpen(false);
                        useSettingsDialogStore
                          .getState()
                          .openDialog("connections");
                      }}
                    >
                      Open Connections
                    </Button>
                  ) : null}
                </div>
              )}
            </div>
          </>
        )}

        {error ? (
          <div role="alert" className="border-t border-destructive/20 bg-destructive/5 px-4 py-3">
            <p className="text-xs leading-relaxed text-destructive">{error}</p>
            {state === "error" ? (
              <Button
                type="button"
                variant="ghost"
                size="sm"
                className="mt-1 h-7 rounded-[7px] px-2 text-xs"
                onClick={() => void load()}
              >
                Try again
              </Button>
            ) : null}
          </div>
        ) : null}
      </PopoverContent>
    </Popover>
  );
}
