


import type {
  BaseModelSource,
  LocalSource,
  ModelInventoryCapabilities,
  ModelInventoryFormat,
} from "@/features/hub/inventory";
import type { HfModelResult } from "@/features/hub/hooks/use-hub-model-search";
import type { Capability, CapabilityKey } from "./lib/model-capabilities";

export type {
  CachedInventoryRow,
  InventoryHint,
  InventoryRow,
  LocalInventoryRow,
  LocalSource,
} from "@/features/hub/inventory";

export type ModelsTab = "discover" | "downloaded";

export type ResourceTypeFilter = "models" | "datasets";

export type HubModelType = "text" | "vision" | "audio" | "embeddings";

export type ModelFormatFilter = "all" | "gguf" | "checkpoint" | "mlx";

export type CapabilityFilter = "all" | CapabilityKey;

export interface DiscoverRow {
  id: string;
  owner: string;
  repo: string;
  result: HfModelResult;
  isAvailableOnDevice: boolean;
  isPartialOnDevice: boolean;
  summary: string;
  capabilities: Capability[];
}

export type SelectedResourceSource = "huggingface" | "hub_cache" | LocalSource;

export type SelectedResourceCacheState =
  | "remote"
  | "cached"
  | "local"
  | "partial";

export interface SelectedResourceRef {
  repoId: string | null;
  localPath: string | null;
  source: SelectedResourceSource;
  cacheState: SelectedResourceCacheState;
  runId: string;
  trainId: string;
}

export interface SelectedModelView {
  id: string;
  kind: "discover" | "cache" | "local";
  resource: SelectedResourceRef;
  displayId: string;
  hubRepoId: string | null;
  owner: string;
  title: string;
  summary: string;
  sourceLabel: string;
  path: string | null;
  localSource?: LocalSource;
  isLocal: boolean;
  isGguf: boolean;
  requiresVariant?: boolean;
  modelFormat: ModelInventoryFormat | null;
  baseModel?: string | null;
  baseModelSource?: BaseModelSource | null;
  baseModelHubId?: string | null;
  baseModelSummary?: string | null;
  adapterType?: string | null;
  trainingMethod?: string | null;
  isDownloaded: boolean;
  isPartial?: boolean;
  partialTransport?: string | null;
  partialResumable?: boolean;
  runtimeCapabilities?: ModelInventoryCapabilities;
  capabilities: Capability[];
  license: string | null;
  pipelineTag?: string;
  /** The backend's inferred pipeline task for an on-device row. A cached GGUF repo carries
   *  this and NOT `pipelineTag`, so deciding a GGUF row's modality needs both. */
  task?: string | null;
  libraryName?: string;
  gated?: false | "auto" | "manual";
  private?: boolean;
  downloads?: number;
  downloadsAllTime?: number;
  likes?: number;
  totalParams?: number;
  estimatedSizeBytes?: number;
  cachedBytes?: number;
  updatedAt?: string;
  createdAt?: string;
  localUpdatedAt?: number | null;
  tags?: string[];
  quantMethod?: string;
}
