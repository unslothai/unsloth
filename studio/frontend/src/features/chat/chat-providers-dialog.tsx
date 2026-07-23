// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectSeparator,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Switch } from "@/components/ui/switch";
import { Spinner } from "@/components/ui/spinner";
import { Textarea } from "@/components/ui/textarea";
import {
  ArrowLeft02Icon,
  Delete02Icon,
  Edit03Icon,
  PlusSignIcon,
  Wifi02Icon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { Eye, EyeOff } from "lucide-react";
import { AnimatePresence, motion, useReducedMotion } from "motion/react";
import { useEffect, useMemo, useRef, useState } from "react";
import { toast } from "sonner";
import { ApiProviderLogo } from "./api-provider-logo";
import {
  type ProviderRegistryEntry,
  createProviderConfig,
  deleteProviderConfig,
  listProviderConfigs,
  listProviderModels,
  listProviderRegistry,
  testProviderConnection,
  updateProviderConfig,
} from "./api/providers-api";
import type { ExternalProviderConfig } from "./external-providers";
import {
  CUSTOM_BACKEND_PROVIDER_TYPE,
  CUSTOM_PROVIDER_PRESETS,
  allowsManualModelIdsWithCatalog,
  customProviderBaseUrlPlaceholder,
  customProviderDisplayName,
  customProviderModelIdsPlaceholder,
  customPresetSkipsApiKeyField,
  getExternalProviderApiKey,
  isCustomProviderType,
  LEGACY_CUSTOM_PROVIDER_TYPE,
  CUSTOM_PROVIDER_DISPLAY_NAME,
  removeExternalProviderApiKey,
  setExternalProviderApiKey,
  supportsProviderPromptCaching,
  supportsProviderReasoningToggle,
  supportsRemoteModelCatalog,
  toExternalBackendProviderType,
} from "./external-providers";
import { useExternalProvidersStore } from "./stores/external-providers-store";

/** Matches navbar / thread layout easing (see index.css --ease-out-quart) */
const PROVIDER_FORM_EASE: [number, number, number, number] = [
  0.165, 0.84, 0.44, 1,
];
const PROVIDER_FORM_DURATION = 0.2;
const CUSTOM_PROVIDER_MISSING_KEY_MESSAGE =
  "No API key found. Add a valid API key for this connection.";
const ANTHROPIC_DATED_SNAPSHOT_SUFFIX = /-\d{8}$/;
const OPENAI_DEPRECATED_MODELS = new Set(["gpt-5.3"]);
const HIDDEN_PROVIDER_TYPES = new Set(["qwen"]);
const OPENROUTER_EXCLUDED_MODELS = new Set([
  "google/chirp-3",
  "kwaivgi/kling-v3.0-pro",
  "openai/whisper-1",
  "openai/gpt-4o-mini-transcribe",
  "recraft/recraft-v4-pro",
]);

function normalizeUrl(input: string): string {
  return input.trim().replace(/\/+$/, "");
}

function resolveUiProviderTypeFromConfig(
  configProviderType: string,
  configDisplayName: string | null | undefined,
  configBaseUrl: string | null | undefined,
  registryRows: ProviderRegistryEntry[],
  existingProviderType: string | undefined,
): string {
  if (existingProviderType && isCustomProviderType(existingProviderType)) {
    return existingProviderType;
  }
  if (configProviderType !== CUSTOM_BACKEND_PROVIDER_TYPE) {
    return configProviderType;
  }
  const displayName = (configDisplayName ?? "").trim().toLowerCase();
  const matchingCustomPreset = CUSTOM_PROVIDER_PRESETS.find(
    (preset) => preset.displayName.toLowerCase() === displayName,
  );
  if (matchingCustomPreset) {
    return matchingCustomPreset.providerType;
  }
  const openAiRegistry = registryRows.find(
    (entry) => entry.provider_type === CUSTOM_BACKEND_PROVIDER_TYPE,
  );
  if (!openAiRegistry) {
    return configProviderType;
  }
  const openAiDisplayName = openAiRegistry.display_name.trim().toLowerCase();
  if (displayName.length > 0 && displayName !== openAiDisplayName) {
    return LEGACY_CUSTOM_PROVIDER_TYPE;
  }
  const configUrl = normalizeUrl(configBaseUrl ?? "");
  const defaultUrl = normalizeUrl(openAiRegistry.base_url ?? "");
  if (configUrl.length > 0 && configUrl !== defaultUrl) {
    return LEGACY_CUSTOM_PROVIDER_TYPE;
  }
  return configProviderType;
}

function parseManualModelIds(text: string): string[] {
  const seen = new Set<string>();
  const out: string[] = [];
  for (const raw of text.split(/[\n,]+/)) {
    const id = raw.trim();
    if (!id || seen.has(id)) continue;
    seen.add(id);
    out.push(id);
  }
  return out;
}

// Remote providers that support both catalog load and manual model IDs.
const EMPTY_CATALOG_HINTS: Record<string, { title: string; description: string }> =
  {
    ollama: {
      title: "No local Ollama models found.",
      description:
        "Run `ollama pull <model>` in a terminal, then reload — or enter a model ID manually below.",
    },
    llama_cpp: {
      title: "No llama.cpp models found.",
      description:
        "Ensure llama-server is running with models loaded, then reload — or enter model IDs manually below.",
    },
    vllm: {
      title: "No vLLM models found.",
      description:
        "Ensure the vLLM server is running and models are loaded, then reload — or enter model IDs manually below.",
    },
  };

function emptyCatalogHint(providerType: string): {
  title: string;
  description: string;
} {
  return (
    EMPTY_CATALOG_HINTS[providerType] ?? {
      title: "No models returned by this connection.",
      description: "Enter model IDs manually below, or check the server.",
    }
  );
}

function shouldAppendOpenAiVersionPath(providerType: string): boolean {
  return (
    providerType === "ollama" ||
    providerType === "llama_cpp" ||
    providerType === "vllm" ||
    providerType === LEGACY_CUSTOM_PROVIDER_TYPE
  );
}

function pruneProviderModelIds(providerType: string, modelIds: string[]): string[] {
  if (providerType === "anthropic") {
    return modelIds.filter((id) => !ANTHROPIC_DATED_SNAPSHOT_SUFFIX.test(id));
  }
  if (providerType === "openai") {
    return modelIds.filter((id) => !OPENAI_DEPRECATED_MODELS.has(id));
  }
  if (providerType === "openrouter") {
    return modelIds.filter((id) => !OPENROUTER_EXCLUDED_MODELS.has(id));
  }
  return modelIds;
}

function formatModelSummary(models: string[]): string {
  if (models.length === 0) {
    return "No models enabled";
  }
  const visible = models.slice(0, 4);
  const remaining = models.length - visible.length;
  return `${visible.join(", ")}${remaining > 0 ? ` +${remaining}` : ""}`;
}

interface ChatProvidersSettingsProps {
  providers: ExternalProviderConfig[];
  onProvidersChange: (providers: ExternalProviderConfig[]) => void;
}

export function ChatProvidersSettings({
  providers,
  onProvidersChange,
}: ChatProvidersSettingsProps) {
  const providersRef = useRef(providers);
  const seededProviderTypeRef = useRef<string | null>(null);
  const [page, setPage] = useState<"list" | "form">("list");
  const [providerType, setProviderType] = useState<string>("");
  const [apiKey, setApiKey] = useState("");
  const [showApiKey, setShowApiKey] = useState(false);
  const [baseUrlDraft, setBaseUrlDraft] = useState("");
  const [editingProviderId, setEditingProviderId] = useState<string | null>(
    null,
  );
  const [registry, setRegistry] = useState<ProviderRegistryEntry[]>([]);
  const [availableModels, setAvailableModels] = useState<string[]>([]);
  const [selectedModelIds, setSelectedModelIds] = useState<string[]>([]);
  const [syncingProviders, setSyncingProviders] = useState(false);
  const [registryLoading, setRegistryLoading] = useState(false);
  const [modelsLoading, setModelsLoading] = useState(false);
  const [mutatingProvider, setMutatingProvider] = useState(false);
  const [manualModelIds, setManualModelIds] = useState("");
  const [modelSearchQuery, setModelSearchQuery] = useState("");
  const [customProviderName, setCustomProviderName] = useState(
    CUSTOM_PROVIDER_DISPLAY_NAME,
  );
  const [isReasoningModel, setIsReasoningModel] = useState(false);
  const reduceMotion = useReducedMotion();
  const connectionsEnabled = useExternalProvidersStore(
    (s) => s.connectionsEnabled,
  );
  const setConnectionsEnabled = useExternalProvidersStore(
    (s) => s.setConnectionsEnabled,
  );
  const isCustomProvider = isCustomProviderType(providerType);
  // llama.cpp hides the key field. Ollama and vLLM show an optional key:
  // Ollama cloud and secured vLLM need one; local servers leave it empty.
  const showApiKeyField = !customPresetSkipsApiKeyField(providerType);
  const showReasoningToggle = supportsProviderReasoningToggle(providerType);

  const registryByType = useMemo(
    () => new Map(registry.map((entry) => [entry.provider_type, entry])),
    [registry],
  );
  const isCuratedModelList = useMemo(() => {
    return registryByType.get(providerType)?.model_list_mode === "curated";
  }, [registryByType, providerType]);
  const isManualModelList =
    (isCustomProvider && !supportsRemoteModelCatalog(providerType)) ||
    isCuratedModelList;

  const modelsPanelKey = isCustomProvider
    ? providerType || "custom"
    : isCuratedModelList
      ? "curated"
      : "remote";
  const remoteAllowsManual = allowsManualModelIdsWithCatalog(providerType);
  const formModelCount =
    isManualModelList || remoteAllowsManual
      ? new Set([...selectedModelIds, ...parseManualModelIds(manualModelIds)])
          .size
      : selectedModelIds.length;
  const modelStatusLabel =
    !isManualModelList &&
    !remoteAllowsManual &&
    availableModels.length === 0
      ? "No models loaded"
      : `${formModelCount} ${formModelCount === 1 ? "model" : "models"} selected`;
  const showModelsBody =
    isManualModelList ||
    remoteAllowsManual ||
    availableModels.length > 0;
  const missingModelCatalogBaseUrl =
    supportsRemoteModelCatalog(providerType) && baseUrlDraft.trim().length === 0;
  const missingModelCatalogApiKey =
    !isCustomProvider && !isCuratedModelList && apiKey.trim().length === 0;
  const loadModelsDisabled =
    modelsLoading ||
    mutatingProvider ||
    isManualModelList ||
    missingModelCatalogBaseUrl ||
    missingModelCatalogApiKey;
  const loadModelsTitle =
    isManualModelList && isCustomProvider
      ? "This connection uses manual model IDs"
      : isCuratedModelList
        ? "Full catalog is not fetched for this connection"
        : missingModelCatalogBaseUrl
          ? "Enter a Base URL before loading models"
          : missingModelCatalogApiKey
            ? "Enter an API key before loading models"
            : undefined;
  const filteredAvailableModels = useMemo(() => {
    const query = modelSearchQuery.trim().toLowerCase();
    if (!query) {
      return availableModels;
    }
    return availableModels.filter((model) =>
      model.toLowerCase().includes(query),
    );
  }, [availableModels, modelSearchQuery]);
  const availableModelsLabel = modelSearchQuery.trim()
    ? `${filteredAvailableModels.length} of ${availableModels.length} models`
    : `${availableModels.length} models`;
  const modelSearchInputClassName =
    "h-8 w-full bg-background/55 text-xs placeholder:text-muted-foreground/65 focus-visible:border-border focus-visible:ring-0";

  useEffect(() => {
    providersRef.current = providers;
  }, [providers]);

  useEffect(() => {
    if (!providerType || editingProviderId) return;
    if (seededProviderTypeRef.current === providerType) return;
    seededProviderTypeRef.current = providerType;
    const entry = registryByType.get(providerType);
    if (!entry) {
      if (isCustomProviderType(providerType)) {
        setCustomProviderName(customProviderDisplayName(providerType));
        setBaseUrlDraft("");
      }
      return;
    }
    // Seed default_models only for curated providers (catalog too large to
    // enumerate). Remote cloud providers and local OpenAI-compat presets stay
    // empty until the user clicks "Load available models".
    const seedDefaults = entry.model_list_mode === "curated";
    setAvailableModels(seedDefaults ? [...entry.default_models] : []);
    setSelectedModelIds([]);
    setManualModelIds("");
    setModelSearchQuery("");
    setBaseUrlDraft("");
  }, [providerType, editingProviderId, registryByType]);

  const totalModels = useMemo(
    () =>
      providers.reduce((count, provider) => count + provider.models.length, 0),
    [providers],
  );

  useEffect(() => {
    let isMounted = true;
    const syncFromBackend = async ({
      showSpinner = true,
    }: { showSpinner?: boolean } = {}) => {
      if (showSpinner) {
        setRegistryLoading(true);
        setSyncingProviders(true);
      }
      let syncSucceeded = false;
      try {
        const [registryRows, configRows] = await Promise.all([
          listProviderRegistry(),
          listProviderConfigs(),
        ]);
        if (!isMounted) return;
        syncSucceeded = true;
        setRegistry(registryRows);
        setProviderType((current) => {
          if (
            current &&
            (isCustomProviderType(current) ||
              registryRows.some((entry) => entry.provider_type === current))
          ) {
            return current;
          }
          return registryRows[0]?.provider_type ?? "";
        });
        const existingById = new Map<string, ExternalProviderConfig>();
        for (const provider of providersRef.current) {
          existingById.set(provider.id, provider);
        }
        const syncedProviders: ExternalProviderConfig[] = configRows
          .filter((config) => config.is_enabled)
          .map((config) => {
            const existing = existingById.get(config.id);
            const uiProviderType = resolveUiProviderTypeFromConfig(
              config.provider_type,
              config.display_name,
              config.base_url,
              registryRows,
              existing?.providerType,
            );
            const createdAt = Number.isFinite(Date.parse(config.created_at))
              ? Date.parse(config.created_at)
              : Date.now();
            const updatedAt = Number.isFinite(Date.parse(config.updated_at))
              ? Date.parse(config.updated_at)
              : Date.now();
            const registryEntry =
              registryRows.find((entry) => entry.provider_type === uiProviderType) ??
              registryRows.find((entry) => entry.provider_type === config.provider_type);
            const defaultModels = pruneProviderModelIds(
              uiProviderType,
              registryEntry?.default_models ?? [],
            );
            const savedModels = existing?.models ?? [];
            const savedAvailableModels = existing?.availableModels ?? [];
            const existingModels = pruneProviderModelIds(
              uiProviderType,
              savedModels.length > 0 ? savedModels : defaultModels,
            );
            const existingAvailableModels = pruneProviderModelIds(
              uiProviderType,
              savedAvailableModels.length > 0 ? savedAvailableModels : defaultModels,
            );
            return {
              id: config.id,
              providerType: uiProviderType,
              name: config.display_name,
              baseUrl: config.base_url ?? "",
              models: existingModels,
              availableModels: existingAvailableModels,
              enablePromptCaching: supportsProviderPromptCaching(uiProviderType)
                ? (existing?.enablePromptCaching ?? true)
                : undefined,
              isReasoningModel: supportsProviderReasoningToggle(uiProviderType)
                ? existing?.isReasoningModel === true
                : undefined,
              createdAt: existing?.createdAt ?? createdAt,
              updatedAt,
            };
          });
        // Trust the backend response. An empty array means every connection was
        // removed (often from another tab); mirror that locally, else stale
        // entries become un-removable here until localStorage is cleared.
        onProvidersChange(syncedProviders);
      } catch (error) {
        // Only surface a toast for real failures, not for the silent
        // background re-sync on tab focus.
        if (showSpinner) {
          const message =
            error instanceof Error ? error.message : "Unknown error";
          toast.error(`Failed to load connections: ${message}`);
        }
      } finally {
        if (isMounted && showSpinner) {
          setRegistryLoading(false);
          setSyncingProviders(false);
        }
      }
      return syncSucceeded;
    };
    void syncFromBackend();
    // Re-sync silently when the tab regains focus so deletes made in
    // another browser propagate without forcing the user to reopen the
    // dialog. Skip when the document is hidden to avoid background work.
    const handleVisibilityChange = () => {
      if (typeof document === "undefined" || document.hidden) return;
      void syncFromBackend({ showSpinner: false });
    };
    if (typeof window !== "undefined") {
      window.addEventListener("focus", handleVisibilityChange);
      document.addEventListener("visibilitychange", handleVisibilityChange);
    }
    return () => {
      isMounted = false;
      if (typeof window !== "undefined") {
        window.removeEventListener("focus", handleVisibilityChange);
        document.removeEventListener("visibilitychange", handleVisibilityChange);
      }
    };
  }, [onProvidersChange]);

  function resetForm() {
    setEditingProviderId(null);
    setApiKey("");
    setShowApiKey(false);
    setBaseUrlDraft("");
    setAvailableModels([]);
    setSelectedModelIds([]);
    setManualModelIds("");
    setModelSearchQuery("");
    setCustomProviderName(customProviderDisplayName(providerType));
    setIsReasoningModel(false);
  }

  function openAddProvider() {
    resetForm();
    const entry = providerType ? registryByType.get(providerType) : null;
    if (entry?.model_list_mode === "curated") {
      setAvailableModels([...entry.default_models]);
    }
    setPage("form");
  }

  function closeForm() {
    resetForm();
    setPage("list");
  }

  function toggleModel(modelId: string) {
    setSelectedModelIds((prev) =>
      prev.includes(modelId)
        ? prev.filter((id) => id !== modelId)
        : [...prev, modelId],
    );
  }

  function selectAllModels() {
    setSelectedModelIds([...availableModels]);
  }

  function clearModelSelection() {
    setSelectedModelIds([]);
  }

  function parseOptionalBaseUrl(
    input: string,
    options: { appendOpenAiVersionPath?: boolean } = {},
  ): string | null {
    const trimmed = input.trim();
    if (!trimmed) return null;
    let parsed: URL;
    try {
      parsed = new URL(trimmed);
    } catch {
      throw new Error("Base URL must be a valid URL.");
    }
    if (parsed.protocol !== "http:" && parsed.protocol !== "https:") {
      throw new Error("Base URL must use http or https.");
    }
    if (options.appendOpenAiVersionPath) {
      const pathname = parsed.pathname.replace(/\/+$/, "");
      if (!pathname) {
        parsed.pathname = "/v1";
      }
    }
    return parsed.toString().replace(/\/+$/, "");
  }

  function parseBaseUrlForProvider(
    input: string,
    required: boolean,
    providerTypeForUrl: string,
  ): string | null {
    const trimmed = input.trim();
    if (!trimmed) {
      if (required) {
        throw new Error("Base URL is required for this connection.");
      }
      return null;
    }
    return parseOptionalBaseUrl(trimmed, {
      appendOpenAiVersionPath: shouldAppendOpenAiVersionPath(providerTypeForUrl),
    });
  }

  async function loadModels() {
    if (!providerType) {
      toast.error("Choose a connection first.");
      return;
    }
    if (isCustomProvider && !supportsRemoteModelCatalog(providerType)) {
      toast.info("This connection uses manual model IDs.");
      return;
    }
    if (isCuratedModelList) {
      toast.info(
        "This connection has a very large model catalog. Use the suggestions and add model IDs manually — full list is not fetched.",
      );
      return;
    }
    if (!isCustomProvider && !apiKey.trim()) {
      toast.error("Add an API key first.");
      return;
    }
    setModelsLoading(true);
    try {
      const baseUrl = parseBaseUrlForProvider(
        baseUrlDraft,
        supportsRemoteModelCatalog(providerType),
        providerType,
      );
      const backendProviderType =
        toExternalBackendProviderType(providerType) ?? providerType;
      const models = await listProviderModels({
        providerType: backendProviderType,
        apiKey: apiKey.trim(),
        baseUrl,
      });
      const registryDefaults = supportsRemoteModelCatalog(providerType)
        ? []
        : (registryByType.get(providerType)?.default_models ?? []);
      // Union of registry defaults + fetched models, defaults first so any
      // curated picks (e.g. claude-haiku-4-5) always show even when the
      // provider's /models endpoint omits them.
      const modelIds = pruneProviderModelIds(providerType, [
        ...new Set(
          [
            ...registryDefaults,
            ...models.map((model) => model.id.trim()),
          ].filter((id) => id.length > 0),
        ),
      ]);
      setAvailableModels(modelIds);
      setSelectedModelIds((prev) =>
        prev.filter((id) => modelIds.includes(id)),
      );
      if (modelIds.length === 0) {
        const hint = emptyCatalogHint(providerType);
        toast.info(hint.title, { description: hint.description });
      } else {
        toast.success(
          `Found ${modelIds.length} ${modelIds.length === 1 ? "model" : "models"}.`,
        );
      }
      if (editingProviderId) {
        onProvidersChange(
          providersRef.current.map((provider) =>
            provider.id === editingProviderId
              ? { ...provider, availableModels: modelIds }
              : provider,
          ),
        );
      }
      setModelSearchQuery("");
    } catch (error) {
      const message = error instanceof Error ? error.message : "Unknown error";
      toast.error(`Could not load models: ${message}`);
    } finally {
      setModelsLoading(false);
    }
  }

  async function addProvider() {
    if (!providerType) {
      toast.error("Choose a connection first.");
      return;
    }
    const backendProviderType = toExternalBackendProviderType(providerType);
    const selectedRegistryEntry = registryByType.get(backendProviderType);
    const displayName = isCustomProvider
      ? customProviderName.trim() || customProviderDisplayName(providerType)
      : (selectedRegistryEntry?.display_name ?? providerType);
    if (!isCustomProvider && !apiKey.trim()) {
      toast.error("API key is required.");
      return;
    }
    const curated = selectedRegistryEntry?.model_list_mode === "curated";
    const manualOnly =
      (isCustomProvider && !supportsRemoteModelCatalog(providerType)) ||
      curated;
    const remoteAllowsManual = allowsManualModelIdsWithCatalog(providerType);
    const manualIds = parseManualModelIds(manualModelIds);
    const allowManual = manualOnly || remoteAllowsManual;
    const modelsToSave = pruneProviderModelIds(
      providerType,
      allowManual
      ? [
          ...new Set([
            ...selectedModelIds,
            ...manualIds,
          ]),
        ]
      : [...selectedModelIds],
    );
    if (manualOnly) {
      if (modelsToSave.length === 0) {
        toast.error("Add at least one model ID.");
        return;
      }
    } else if (remoteAllowsManual) {
      if (modelsToSave.length === 0) {
        toast.error("Add at least one model ID.");
        return;
      }
    } else {
      if (availableModels.length === 0) {
        toast.error(
          "Load available models first, then choose which to enable.",
        );
        return;
      }
      if (selectedModelIds.length === 0) {
        toast.error("Select at least one model.");
        return;
      }
    }
    setMutatingProvider(true);
    try {
      const baseUrl = parseBaseUrlForProvider(
        baseUrlDraft,
        isCustomProvider,
        providerType,
      );
      const created = await createProviderConfig({
        providerType: backendProviderType,
        displayName,
        baseUrl,
      });
      const createdAt = Number.isFinite(Date.parse(created.created_at))
        ? Date.parse(created.created_at)
        : Date.now();
      const updatedAt = Number.isFinite(Date.parse(created.updated_at))
        ? Date.parse(created.updated_at)
        : Date.now();
      const uiProviderType = isCustomProvider
        ? providerType
        : created.provider_type;
      const provider: ExternalProviderConfig = {
        id: created.id,
        providerType: uiProviderType,
        name: created.display_name,
        baseUrl: created.base_url ?? "",
        models: modelsToSave,
        availableModels: manualOnly
          ? []
          : pruneProviderModelIds(providerType, availableModels),
        isReasoningModel: supportsProviderReasoningToggle(uiProviderType)
          ? isReasoningModel
          : undefined,
        createdAt,
        updatedAt,
      };
      if (apiKey.trim()) {
        setExternalProviderApiKey(created.id, apiKey.trim());
      }
      onProvidersChange([
        ...providers.filter((p) => p.id !== created.id),
        provider,
      ]);
      resetForm();
      setPage("list");
      toast.success("Connection added.");
    } catch (error) {
      const message = error instanceof Error ? error.message : "Unknown error";
      toast.error(`Failed to add connection: ${message}`);
    } finally {
      setMutatingProvider(false);
    }
  }

  async function saveProviderEdits() {
    if (!editingProviderId) return;
    const existing = providers.find(
      (provider) => provider.id === editingProviderId,
    );
    if (!existing) {
      toast.error("Connection not found.");
      return;
    }
    const isEditingCustomProvider =
      isCustomProviderType(existing.providerType);
    if (!isEditingCustomProvider && !apiKey.trim()) {
      toast.error("API key is required.");
      return;
    }
    const entry = registryByType.get(existing.providerType);
    const curated = entry?.model_list_mode === "curated";
    const manualOnly =
      (isEditingCustomProvider &&
        !supportsRemoteModelCatalog(existing.providerType)) ||
      curated;
    const remoteAllowsManual = allowsManualModelIdsWithCatalog(
      existing.providerType,
    );
    const manualIds = parseManualModelIds(manualModelIds);
    const allowManual = manualOnly || remoteAllowsManual;
    const modelsToSave = pruneProviderModelIds(
      existing.providerType,
      allowManual
      ? [
          ...new Set([
            ...selectedModelIds,
            ...manualIds,
          ]),
        ]
      : [...selectedModelIds],
    );
    if (manualOnly) {
      if (modelsToSave.length === 0) {
        toast.error("Add at least one model ID.");
        return;
      }
    } else if (remoteAllowsManual) {
      if (modelsToSave.length === 0) {
        toast.error("Add at least one model ID.");
        return;
      }
    } else {
      if (availableModels.length === 0) {
        toast.error(
          "Load available models first, then choose which to enable.",
        );
        return;
      }
      if (selectedModelIds.length === 0) {
        toast.error("Select at least one model.");
        return;
      }
    }
    setMutatingProvider(true);
    try {
      const baseUrl = parseBaseUrlForProvider(
        baseUrlDraft,
        isEditingCustomProvider,
        existing.providerType,
      );
      const updated = await updateProviderConfig(editingProviderId, {
        displayName: isEditingCustomProvider
          ? customProviderName.trim() ||
            customProviderDisplayName(existing.providerType)
          : existing.name,
        baseUrl,
      });
      if (apiKey.trim()) {
        setExternalProviderApiKey(editingProviderId, apiKey.trim());
      } else if (isEditingCustomProvider) {
        removeExternalProviderApiKey(editingProviderId);
      }
      const updatedAt = Number.isFinite(Date.parse(updated.updated_at))
        ? Date.parse(updated.updated_at)
        : Date.now();
      onProvidersChange(
        providers.map((provider) =>
          provider.id === editingProviderId
            ? {
                ...provider,
                name: updated.display_name,
                baseUrl: updated.base_url ?? "",
                models: modelsToSave,
                availableModels: manualOnly
                  ? []
                  : pruneProviderModelIds(existing.providerType, availableModels),
                isReasoningModel: supportsProviderReasoningToggle(
                  existing.providerType,
                )
                  ? isReasoningModel
                  : undefined,
                updatedAt,
              }
            : provider,
        ),
      );
      toast.success("Connection updated.");
      resetForm();
      setPage("list");
    } catch (error) {
      const message = error instanceof Error ? error.message : "Unknown error";
      toast.error(`Failed to update connection: ${message}`);
    } finally {
      setMutatingProvider(false);
    }
  }

  async function editProvider(provider: ExternalProviderConfig) {
    setEditingProviderId(provider.id);
    setPage("form");
    setProviderType(provider.providerType);
    setCustomProviderName(
      provider.name || customProviderDisplayName(provider.providerType),
    );
    setApiKey(getExternalProviderApiKey(provider.id));
    setShowApiKey(false);
    setBaseUrlDraft(provider.baseUrl);
    setModelSearchQuery("");
    setIsReasoningModel(
      supportsProviderReasoningToggle(provider.providerType)
        ? provider.isReasoningModel === true
        : false,
    );
    if (
      isCustomProviderType(provider.providerType) &&
      !supportsRemoteModelCatalog(provider.providerType)
    ) {
      setAvailableModels([]);
      setSelectedModelIds([]);
      setManualModelIds(provider.models.join("\n"));
      return;
    }
    if (supportsRemoteModelCatalog(provider.providerType)) {
      const cachedCatalog = provider.availableModels ?? [];
      const catalogModels = pruneProviderModelIds(provider.providerType, [
        ...new Set(
          cachedCatalog
            .map((model) => model.trim())
            .filter((model) => model.length > 0),
        ),
      ]);
      setAvailableModels(catalogModels);
      const catalogSet = new Set(catalogModels);
      setSelectedModelIds(
        provider.models.filter((model) => catalogSet.has(model)),
      );
      setManualModelIds(
        provider.models.filter((model) => !catalogSet.has(model)).join("\n"),
      );
      return;
    }
    const entry = registryByType.get(provider.providerType);
    if (entry?.model_list_mode === "curated") {
      const defaults = new Set(entry.default_models);
      const inDefaults = provider.models.filter((m) => defaults.has(m));
      const custom = provider.models.filter((m) => !defaults.has(m));
      setAvailableModels([...entry.default_models]);
      setSelectedModelIds(inDefaults);
      setManualModelIds(custom.join("\n"));
    } else {
      const shortlist = entry?.default_models ?? [];
      const cachedCatalog = provider.availableModels ?? [];
      const mergedModels = pruneProviderModelIds(provider.providerType, [
        ...new Set(
          [...shortlist, ...cachedCatalog, ...provider.models]
            .map((model) => model.trim())
            .filter((model) => model.length > 0),
        ),
      ]);
      setAvailableModels(mergedModels);
      setSelectedModelIds(
        provider.models.filter((model) => mergedModels.includes(model)),
      );
      setManualModelIds("");
    }
  }

  async function deleteProvider(providerId: string) {
    setMutatingProvider(true);
    try {
      await deleteProviderConfig(providerId);
      removeExternalProviderApiKey(providerId);
      onProvidersChange(
        providers.filter((provider) => provider.id !== providerId),
      );
    } catch (error) {
      const message = error instanceof Error ? error.message : "Unknown error";
      toast.error(`Failed to delete connection: ${message}`);
    } finally {
      setMutatingProvider(false);
    }
  }

  async function testProvider(provider: ExternalProviderConfig) {
    const savedKey = getExternalProviderApiKey(provider.id).trim();
    // Hosted registry providers require keys. Local OpenAI-compatible presets
    // may be keyless.
    if (
      !savedKey &&
      !supportsRemoteModelCatalog(provider.providerType)
    ) {
      if (isCustomProviderType(provider.providerType)) {
        await editProvider(provider);
        toast.info(CUSTOM_PROVIDER_MISSING_KEY_MESSAGE);
        return;
      }
      await editProvider(provider);
      toast.info(`No API key for ${provider.name}. Add one in Connections and save.`);
      return;
    }
    try {
      const result = await testProviderConnection({
        providerType:
          toExternalBackendProviderType(provider.providerType) ??
          provider.providerType,
        apiKey: savedKey,
        baseUrl: provider.baseUrl || null,
        modelId:
          provider.providerType === LEGACY_CUSTOM_PROVIDER_TYPE
            ? (provider.models[0] ?? null)
            : null,
      });
      if (result.success) {
        toast.success(result.message);
      } else {
        if (
          isCustomProviderType(provider.providerType) &&
          result.message.includes("Illegal header value b'Bearer '")
        ) {
          toast.error(CUSTOM_PROVIDER_MISSING_KEY_MESSAGE);
          return;
        }
        toast.error(result.message);
      }
    } catch (error) {
      const message = error instanceof Error ? error.message : "Unknown error";
      if (
        isCustomProviderType(provider.providerType) &&
        message.includes("Illegal header value b'Bearer '")
      ) {
        toast.error(CUSTOM_PROVIDER_MISSING_KEY_MESSAGE);
        return;
      }
      toast.error(`Test failed: ${message}`);
    }
  }

  if (page === "form") {
    return (
      <div className="@container -mt-3 flex min-h-0 flex-col gap-2">
        <header className="flex items-center gap-2 pr-8">
          <Button
            type="button"
            variant="ghost"
            size="icon-sm"
            className="size-8 rounded-[8px]"
            onClick={closeForm}
            aria-label="Back to connections"
            title="Back to connections"
          >
            <HugeiconsIcon icon={ArrowLeft02Icon} className="size-4" />
          </Button>
          <div className="flex min-w-0 items-center gap-2 leading-none">
            <span className="text-xs font-medium text-muted-foreground">
              Connections
            </span>
            <span className="size-1 rounded-full bg-muted-foreground/35" />
            <span className="truncate text-xs font-medium text-muted-foreground">
              {editingProviderId ? "Edit" : "New"}
            </span>
          </div>
        </header>

        <div className="flex max-w-[760px] flex-col gap-3">
          <section className="overflow-hidden rounded-[8px] border border-border/70 bg-muted/[0.12]">
            <div className="divide-y divide-border/60">
              <div className="grid grid-cols-[minmax(140px,0.8fr)_minmax(0,1.2fr)] items-center gap-4 px-4 py-3 @max-[520px]:grid-cols-1">
                <div className="flex min-w-0 flex-col gap-0.5">
                  <Label
                    htmlFor="provider-preset"
                    className="text-sm font-medium"
                  >
                    Connection
                  </Label>
                  <p className="text-xs leading-snug text-muted-foreground">
                    OpenAI, Anthropic, or a compatible local endpoint.
                  </p>
                </div>
                <Select
                  value={providerType}
                  onValueChange={(value) => {
                    if (editingProviderId) return;
                    setProviderType(value);
                    setAvailableModels([]);
                    setSelectedModelIds([]);
                    setManualModelIds("");
                    setModelSearchQuery("");
                    if (isCustomProviderType(value)) {
                      setCustomProviderName(customProviderDisplayName(value));
                    }
                  }}
                >
                  <SelectTrigger
                    id="provider-preset"
                    className="h-9 w-full text-sm"
                    disabled={editingProviderId != null}
                  >
                    <SelectValue placeholder="Choose a connection" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectGroup>
                      {CUSTOM_PROVIDER_PRESETS.map((preset) => (
                        <SelectItem
                          key={preset.providerType}
                          value={preset.providerType}
                        >
                          <span className="flex items-center gap-2">
                            <ApiProviderLogo
                              providerType={preset.providerType}
                              className="size-4"
                              title={preset.displayName}
                            />
                            {preset.displayName}
                          </span>
                        </SelectItem>
                      ))}
                      <SelectItem value={LEGACY_CUSTOM_PROVIDER_TYPE}>
                        <span className="flex items-center gap-2">
                          <ApiProviderLogo
                            providerType={LEGACY_CUSTOM_PROVIDER_TYPE}
                            className="size-4"
                            title={CUSTOM_PROVIDER_DISPLAY_NAME}
                          />
                          {CUSTOM_PROVIDER_DISPLAY_NAME}
                        </span>
                      </SelectItem>
                    </SelectGroup>
                    <SelectSeparator />
                    <SelectGroup>
                      {registry
                        .filter(
                          (entry) =>
                            !HIDDEN_PROVIDER_TYPES.has(entry.provider_type),
                        )
                        .map((entry) => (
                          <SelectItem
                            key={entry.provider_type}
                            value={entry.provider_type}
                          >
                            <span className="flex items-center gap-2">
                              <ApiProviderLogo
                                providerType={entry.provider_type}
                                className="size-4"
                                title={entry.display_name}
                              />
                              {entry.display_name}
                            </span>
                          </SelectItem>
                        ))}
                    </SelectGroup>
                  </SelectContent>
                </Select>
              </div>

              {showApiKeyField ? (
                <div className="grid grid-cols-[minmax(140px,0.8fr)_minmax(0,1.2fr)] items-center gap-4 px-4 py-3 @max-[520px]:grid-cols-1">
                  <div className="flex min-w-0 flex-col gap-0.5">
                    <Label
                      htmlFor="provider-api-key"
                      className="text-sm font-medium"
                    >
                      API key {isCustomProvider ? "(optional)" : ""}
                    </Label>
                    <p className="text-xs leading-snug text-muted-foreground">
                      Stored locally.
                    </p>
                  </div>
                  <div className="relative min-w-0">
                    <Input
                      id="provider-api-key"
                      type={showApiKey ? "text" : "password"}
                      value={apiKey}
                      onChange={(event) => setApiKey(event.target.value)}
                      placeholder="Enter API key"
                      className="h-9 pr-9 text-sm"
                    />
                    <button
                      type="button"
                      onClick={() => setShowApiKey((visible) => !visible)}
                      className="absolute top-1/2 right-1.5 flex size-5 -translate-y-1/2 items-center justify-center rounded text-muted-foreground transition-colors hover:text-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
                      aria-label={showApiKey ? "Hide API key" : "Show API key"}
                      aria-pressed={showApiKey}
                    >
                      {showApiKey ? (
                        <Eye className="size-3.5" />
                      ) : (
                        <EyeOff className="size-3.5" />
                      )}
                    </button>
                  </div>
                </div>
              ) : null}

              {isCustomProvider ? (
                <div className="grid grid-cols-[minmax(140px,0.8fr)_minmax(0,1.2fr)] items-center gap-4 px-4 py-3 @max-[520px]:grid-cols-1">
                  <Label
                    htmlFor="provider-custom-name"
                    className="text-sm font-medium"
                  >
                    Connection name
                  </Label>
                  <Input
                    id="provider-custom-name"
                    type="text"
                    value={customProviderName}
                    onChange={(event) =>
                      setCustomProviderName(event.target.value)
                    }
                    placeholder={CUSTOM_PROVIDER_DISPLAY_NAME}
                    className="h-9 text-sm"
                  />
                </div>
              ) : null}

              {isCustomProvider ? (
                <div className="grid grid-cols-[minmax(140px,0.8fr)_minmax(0,1.2fr)] items-center gap-4 px-4 py-3 @max-[520px]:grid-cols-1">
                  <div className="flex min-w-0 flex-col gap-0.5">
                    <Label
                      htmlFor="provider-base-url"
                      className="text-sm font-medium"
                    >
                      Base URL
                    </Label>
                    <p className="text-xs leading-snug text-muted-foreground">
                      OpenAI-compatible endpoint.
                    </p>
                  </div>
                  <Input
                    id="provider-base-url"
                    type="text"
                    value={baseUrlDraft}
                    onChange={(event) => setBaseUrlDraft(event.target.value)}
                    placeholder={customProviderBaseUrlPlaceholder(providerType)}
                    className="h-9 text-sm"
                  />
                </div>
              ) : null}

              {showReasoningToggle ? (
                <div className="grid grid-cols-[minmax(140px,0.8fr)_minmax(0,1.2fr)] items-center gap-4 px-4 py-3 @max-[520px]:grid-cols-1">
                  <Label
                    htmlFor="provider-is-reasoning"
                    className="text-sm font-medium"
                  >
                    Reasoning model
                  </Label>
                  <label
                    htmlFor="provider-is-reasoning"
                    className="flex cursor-pointer items-center gap-2 text-sm"
                  >
                    <Checkbox
                      id="provider-is-reasoning"
                      checked={isReasoningModel}
                      onCheckedChange={(checked) =>
                        setIsReasoningModel(checked === true)
                      }
                    />
                    This server runs a reasoning model
                  </label>
                </div>
              ) : null}
            </div>
          </section>

          <section className="overflow-hidden rounded-[8px] border border-border/70 bg-muted/[0.12]">
            <AnimatePresence initial={false} mode="wait">
              <motion.div
                key={modelsPanelKey}
                className="origin-top overflow-hidden"
                initial={reduceMotion ? false : { opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: "auto" }}
                exit={reduceMotion ? undefined : { opacity: 0, height: 0 }}
                transition={{
                  height: {
                    duration: PROVIDER_FORM_DURATION,
                    ease: PROVIDER_FORM_EASE,
                  },
                  opacity: {
                    duration: reduceMotion ? 0 : 0.14,
                    ease: PROVIDER_FORM_EASE,
                  },
                }}
              >
                <div
                  className={`flex flex-wrap items-center justify-between gap-3 px-4 py-3 ${showModelsBody ? "border-border/60 border-b" : ""}`}
                >
                  <div className="flex min-w-0 flex-col gap-0.5">
                    <Label className="text-sm font-medium">Models</Label>
                    <p className="text-xs leading-snug text-muted-foreground">
                      {modelStatusLabel}
                    </p>
                  </div>
                  <Button
                    type="button"
                    variant="outline"
                    size="sm"
                    className={
                      availableModels.length > 0
                        ? "h-7 shrink-0 border-transparent bg-transparent px-2 text-xs text-muted-foreground shadow-none hover:bg-muted/45 hover:text-foreground"
                        : "h-8 shrink-0 px-3"
                    }
                    disabled={loadModelsDisabled}
                    title={loadModelsTitle}
                    onClick={() => void loadModels()}
                  >
                    {modelsLoading ? (
                      <>
                        <Spinner className="mr-2 size-3.5" />
                        Loading…
                      </>
                    ) : availableModels.length > 0 ? (
                      "Reload models"
                    ) : (
                      "Load available models"
                    )}
                  </Button>
                </div>
                {isCustomProvider && !supportsRemoteModelCatalog(providerType) ? (
                  <div className="space-y-3 px-4 py-4">
                    <div className="space-y-2">
                      <Label
                        htmlFor="provider-manual-models"
                        className="text-sm font-medium"
                      >
                        Model IDs (one per line or comma-separated)
                      </Label>
                      <Textarea
                        id="provider-manual-models"
                        value={manualModelIds}
                        onChange={(event) =>
                          setManualModelIds(event.target.value)
                        }
                        placeholder={customProviderModelIdsPlaceholder(providerType)}
                        rows={5}
                        className="min-h-[100px] resize-y font-mono text-sm"
                      />
                    </div>
                  </div>
                ) : isCuratedModelList ? (
                  <div className="space-y-3 px-4 py-4">
                    <p className="text-xs leading-relaxed text-muted-foreground">
                      Select from suggestions below or enter exact model IDs.
                    </p>
                    {availableModels.length > 0 ? (
                      <div className="space-y-3 rounded-[8px] border border-border/70 bg-background/50 p-3">
                        <div className="grid grid-cols-[minmax(90px,auto)_minmax(0,1fr)_auto] items-center gap-3 @max-[520px]:grid-cols-1">
                          <span className="whitespace-nowrap text-xs font-medium text-muted-foreground">
                            {availableModelsLabel}
                          </span>
                          <Input
                            id={`provider-model-search-${modelsPanelKey}`}
                            type="search"
                            value={modelSearchQuery}
                            onChange={(event) =>
                              setModelSearchQuery(event.target.value)
                            }
                            placeholder="Search"
                            aria-label="Search models"
                            className={modelSearchInputClassName}
                          />
                          <div className="flex items-center justify-end gap-2">
                            <Button
                              type="button"
                              variant="ghost"
                              size="sm"
                              className="h-8 px-2 text-xs font-medium text-foreground/80 hover:bg-muted/45"
                              onClick={selectAllModels}
                            >
                              Select all
                            </Button>
                            <Button
                              type="button"
                              variant="ghost"
                              size="sm"
                              className="h-8 px-2 text-xs font-medium text-foreground/80 hover:bg-muted/45"
                              onClick={() => {
                                clearModelSelection();
                                setManualModelIds("");
                              }}
                            >
                              Clear
                            </Button>
                          </div>
                        </div>
                        <ul className="max-h-56 overflow-y-auto rounded-[8px] border border-border/70 bg-background/50">
                          {filteredAvailableModels.length === 0 ? (
                            <li className="px-3 py-3 text-xs text-muted-foreground">
                              No matching models
                            </li>
                          ) : (
                            filteredAvailableModels.map((model, index) => (
                              <li
                                key={model}
                                className="flex cursor-pointer items-center gap-2.5 border-border/60 border-b px-3 py-2 last:border-b-0 hover:bg-muted/35"
                                onClick={() => toggleModel(model)}
                              >
                                <Checkbox
                                  id={`provider-model-curated-${modelsPanelKey}-${index}`}
                                  checked={selectedModelIds.includes(model)}
                                  onCheckedChange={() => toggleModel(model)}
                                  onClick={(event) => event.stopPropagation()}
                                />
                                <span
                                  className="min-w-0 break-all text-sm leading-tight"
                                >
                                  {model}
                                </span>
                              </li>
                            ))
                          )}
                        </ul>
                      </div>
                    ) : null}
                    <div className="space-y-2">
                      <Label
                        htmlFor="provider-manual-models"
                        className="text-sm font-medium"
                      >
                        Model IDs (one per line or comma-separated)
                      </Label>
                      <Textarea
                        id="provider-manual-models"
                        value={manualModelIds}
                        onChange={(event) =>
                          setManualModelIds(event.target.value)
                        }
                        placeholder={"model-id-1\nmodel-id-2"}
                        rows={5}
                        className="min-h-[100px] resize-y font-mono text-sm"
                      />
                    </div>
                  </div>
                ) : availableModels.length === 0 &&
                  !allowsManualModelIdsWithCatalog(providerType) ? null : (
                  <div className="space-y-3 px-4 py-4">
                    {availableModels.length === 0 ? null : (
                      <>
                        <div className="grid grid-cols-[minmax(90px,auto)_minmax(0,1fr)_auto] items-center gap-3 @max-[520px]:grid-cols-1">
                          <span className="whitespace-nowrap text-xs font-medium text-muted-foreground">
                            {availableModelsLabel}
                          </span>
                          <Input
                            id={`provider-model-search-${modelsPanelKey}`}
                            type="search"
                            value={modelSearchQuery}
                            onChange={(event) =>
                              setModelSearchQuery(event.target.value)
                            }
                            placeholder="Search"
                            aria-label="Search models"
                            className={modelSearchInputClassName}
                          />
                          <div className="flex items-center justify-end gap-2">
                            <Button
                              type="button"
                              variant="ghost"
                              size="sm"
                              className="h-8 px-2 text-xs font-medium text-foreground/80 hover:bg-muted/45"
                              onClick={selectAllModels}
                            >
                              Select all
                            </Button>
                            <Button
                              type="button"
                              variant="ghost"
                              size="sm"
                              className="h-8 px-2 text-xs font-medium text-foreground/80 hover:bg-muted/45"
                              onClick={clearModelSelection}
                            >
                              Clear
                            </Button>
                          </div>
                        </div>
                        <ul className="max-h-56 overflow-y-auto rounded-[8px] border border-border/70 bg-background/50">
                          {filteredAvailableModels.length === 0 ? (
                            <li className="px-3 py-3 text-xs text-muted-foreground">
                              No matching models
                            </li>
                          ) : (
                            filteredAvailableModels.map((model, index) => (
                              <li
                                key={model}
                                className="flex cursor-pointer items-center gap-2.5 border-border/60 border-b px-3 py-2 last:border-b-0 hover:bg-muted/35"
                                onClick={() => toggleModel(model)}
                              >
                                <Checkbox
                                  id={`provider-model-remote-${modelsPanelKey}-${index}`}
                                  checked={selectedModelIds.includes(model)}
                                  onCheckedChange={() => toggleModel(model)}
                                  onClick={(event) => event.stopPropagation()}
                                />
                                <span
                                  className="min-w-0 break-all text-sm leading-tight"
                                >
                                  {model}
                                </span>
                              </li>
                            ))
                          )}
                        </ul>
                      </>
                    )}
                    {/* Manual IDs allowed alongside catalog load. */}
                    {allowsManualModelIdsWithCatalog(providerType) ? (
                      <div className="space-y-2">
                        <Label
                          htmlFor="provider-manual-models"
                          className="text-sm font-medium"
                        >
                          {availableModels.length === 0
                            ? "Or enter model IDs manually (one per line or comma-separated)"
                            : "Additional model IDs (one per line or comma-separated)"}
                        </Label>
                        <Textarea
                          id="provider-manual-models"
                          value={manualModelIds}
                          onChange={(event) =>
                            setManualModelIds(event.target.value)
                          }
                          placeholder={customProviderModelIdsPlaceholder(
                            providerType,
                          )}
                          rows={4}
                          className="min-h-[80px] resize-y font-mono text-sm"
                        />
                      </div>
                    ) : null}
                  </div>
                )}
              </motion.div>
            </AnimatePresence>
          </section>

          <div className="mb-3 flex flex-wrap items-center justify-end gap-3">
            <div className="flex flex-wrap gap-2">
              <Button
                type="button"
                size="sm"
                className="h-8"
                disabled={
                  registryLoading ||
                  syncingProviders ||
                  modelsLoading ||
                  mutatingProvider
                }
                onClick={() =>
                  editingProviderId
                    ? void saveProviderEdits()
                    : void addProvider()
                }
              >
                {editingProviderId ? "Save connection" : "Add connection"}
              </Button>
              <Button
                type="button"
                variant="outline"
                size="sm"
                className="h-8"
                onClick={editingProviderId ? closeForm : resetForm}
              >
                {editingProviderId ? "Cancel" : "Clear"}
              </Button>
            </div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="flex min-h-0 flex-col gap-6">
      <header className="flex flex-col gap-1 pr-8">
        <div className="flex min-w-0 flex-col gap-1">
          <h1 className="font-heading text-lg font-semibold">Connections</h1>
          <p className="text-xs leading-relaxed text-muted-foreground">
            Manage model connections for chat.
          </p>
        </div>
      </header>

      <div className="flex w-full max-w-[760px] flex-col gap-2 sm:flex-row sm:items-center sm:justify-between sm:gap-x-6">
        <div className="flex items-center gap-2">
          <Label
            htmlFor="chat-connections-enabled"
            className="cursor-pointer text-xs text-muted-foreground"
          >
            Enable connections
          </Label>
          <Switch
            id="chat-connections-enabled"
            checked={connectionsEnabled}
            onCheckedChange={setConnectionsEnabled}
            aria-label="Enable connections"
            aria-describedby="chat-connections-description"
          />
        </div>
        <p
          id="chat-connections-description"
          className="max-w-md text-[0.6875rem] leading-snug text-muted-foreground/65 sm:text-right"
        >
          When off, all connections are disabled.
        </p>
      </div>

      <section className="flex max-w-[760px] flex-col gap-2">
        <div className="overflow-hidden rounded-[10px] border border-border/70 bg-muted/[0.12]">
          <button
            type="button"
            onClick={openAddProvider}
            className="group/add flex w-full items-center justify-between gap-3 border-border/60 border-b px-3 py-2.5 text-left text-sm font-medium text-muted-foreground transition-colors hover:bg-muted/35 focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring focus-visible:ring-inset"
          >
            <span className="flex min-w-0 items-center gap-2 rounded-full border border-border bg-background/50 px-3 py-1.5 transition-colors group-hover/add:border-control-accent/25 group-hover/add:text-control-accent">
              <HugeiconsIcon icon={PlusSignIcon} className="size-4 shrink-0" />
              <span>Add connection</span>
            </span>
            <span className="shrink-0 text-xs tabular-nums text-muted-foreground/90">
              {providers.length} connections · {totalModels} models
            </span>
          </button>
          {providers.length === 0 ? (
            <div className="px-3 py-4">
              <div className="flex min-w-0 flex-col gap-0.5">
                <span className="text-sm font-medium text-foreground">
                  No connections yet
                </span>
                <span className="text-xs leading-snug text-muted-foreground">
                  Add a connection to use hosted models from chat.
                </span>
              </div>
            </div>
          ) : (
            <>
              {providers.map((provider) => {
                const registryEntry = registryByType.get(provider.providerType);
                const detail =
                  provider.baseUrl || registryEntry?.base_url || "";
                const providerLabel =
                  registryEntry?.display_name ??
                  customProviderDisplayName(provider.providerType);
                const modelSummary = formatModelSummary(provider.models);
                return (
                  <div
                    key={provider.id}
                    className="group grid min-w-0 grid-cols-[minmax(0,1fr)_auto] items-center gap-3 border-border/60 border-b px-3 py-3 transition-colors last:border-b-0 hover:bg-muted/35 max-sm:grid-cols-1"
                  >
                    <div className="flex min-w-0 items-start gap-3">
                      <div className="mt-1 flex size-8 shrink-0 items-center justify-center rounded-[8px] border border-border/70 bg-background/80">
                        <ApiProviderLogo
                          providerType={provider.providerType}
                          className="size-5"
                          title={provider.name}
                        />
                      </div>
                      <div className="min-w-0 pt-px">
                        <div className="flex min-w-0 items-center gap-2">
                          <span className="truncate text-sm font-medium text-foreground">
                            {provider.name}
                          </span>
                          <span className="shrink-0 rounded-[6px] border border-control-accent/15 bg-control-accent/8 px-1.5 py-0.5 text-[0.625rem] leading-none text-control-accent">
                            {provider.models.length}{" "}
                            {provider.models.length === 1 ? "model" : "models"}
                          </span>
                        </div>
                        <div className="mt-0.5 truncate text-xs text-muted-foreground">
                          <span>{providerLabel}</span>
                          {detail ? (
                            <>
                              {" · "}
                              <span>{detail}</span>
                            </>
                          ) : null}
                        </div>
                        <div
                          className="mt-1 truncate text-[0.6875rem] leading-4 text-muted-foreground/80"
                          title={provider.models.join(", ")}
                        >
                          {modelSummary}
                        </div>
                      </div>
                    </div>
                    <div className="flex shrink-0 items-center justify-end gap-0.5 text-muted-foreground">
                      <Button
                        type="button"
                        size="icon-sm"
                        variant="ghost"
                        className="size-7 rounded-[8px] hover:text-foreground"
                        disabled={mutatingProvider}
                        onClick={() => editProvider(provider)}
                        title="Edit connection"
                        aria-label={`Edit ${provider.name}`}
                      >
                        <HugeiconsIcon icon={Edit03Icon} className="size-4" />
                      </Button>
                      <Button
                        type="button"
                        size="icon-sm"
                        variant="ghost"
                        className="size-7 rounded-[8px] hover:text-foreground"
                        disabled={mutatingProvider}
                        onClick={() => void testProvider(provider)}
                        title="Check connection"
                        aria-label={`Check ${provider.name}`}
                      >
                        <HugeiconsIcon icon={Wifi02Icon} className="size-4" />
                      </Button>
                      <Button
                        type="button"
                        size="icon-sm"
                        variant="ghost"
                        className="size-7 rounded-[8px] hover:text-destructive"
                        disabled={mutatingProvider}
                        onClick={() => void deleteProvider(provider.id)}
                        title="Delete connection"
                        aria-label={`Delete ${provider.name}`}
                      >
                        <HugeiconsIcon icon={Delete02Icon} className="size-4" />
                      </Button>
                    </div>
                  </div>
                );
              })}
            </>
          )}
        </div>
      </section>
    </div>
  );
}

interface ChatProvidersDialogProps extends ChatProvidersSettingsProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

export function ChatProvidersDialog({
  open,
  onOpenChange,
  providers,
  onProvidersChange,
}: ChatProvidersDialogProps) {
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent
        overlayClassName="bg-black/50 backdrop-blur-sm"
        className="flex max-h-[90dvh] w-[96vw] flex-col gap-0 overflow-y-auto p-8 sm:max-w-none md:max-w-[704px]"
      >
        <DialogHeader className="sr-only">
          <DialogTitle>Connections</DialogTitle>
          <DialogDescription>
            Manage model connections for chat.
          </DialogDescription>
        </DialogHeader>
        <ChatProvidersSettings
          providers={providers}
          onProvidersChange={onProvidersChange}
        />
      </DialogContent>
    </Dialog>
  );
}
