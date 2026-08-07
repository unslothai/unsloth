


// eslint-disable-next-line no-restricted-imports -- Keep this state helper independent of the Hub barrel's browser modules.
import { normalizeModelIdentity } from "@/features/hub/lib/model-identity";

export type DatasetCacheUsabilityIdentity = Readonly<{
  dataset: string;
  cachePath: string | null;
  subset: string | null;
  split: string;
  streaming: boolean;
}>;

export type DatasetCacheInventoryIdentity = Readonly<{
  cachePath?: string | null;
  sizeBytes?: number | null;
  partial?: boolean;
  partialTransport?: string | null;
}>;

type ObservedDatasetCache = {
  identity: DatasetCacheUsabilityIdentity;
  fingerprint: string;
};

type RejectedDatasetCache = {
  identity: DatasetCacheUsabilityIdentity;
  fingerprint: string | null;
};

export type DatasetCacheValidationToken = Readonly<{
  identity: DatasetCacheUsabilityIdentity;
  generation: number;
}>;

function normalizeOptional(value: string | null | undefined): string | null {
  const normalized = value?.trim() ?? "";
  return normalized || null;
}

function normalizeCachePath(path: string | null | undefined): string | null {
  const normalized = normalizeOptional(path);
  return normalized ? normalizeModelIdentity(normalized) : null;
}

export function createDatasetCacheUsabilityIdentity({
  dataset,
  cachePath,
  subset,
  split,
  streaming,
}: {
  dataset: string;
  cachePath?: string | null;
  subset?: string | null;
  split?: string | null;
  streaming: boolean;
}): DatasetCacheUsabilityIdentity {
  return {
    dataset: normalizeModelIdentity(dataset),
    cachePath: normalizeCachePath(cachePath),
    subset: normalizeOptional(subset),
    split: normalizeOptional(split) ?? "train",
    streaming,
  };
}

export function datasetCacheUsabilityIdentitiesEqual(
  left: DatasetCacheUsabilityIdentity,
  right: DatasetCacheUsabilityIdentity,
): boolean {
  return (
    left.dataset === right.dataset &&
    left.cachePath === right.cachePath &&
    left.subset === right.subset &&
    left.split === right.split &&
    left.streaming === right.streaming
  );
}

function targetsSameCacheUsability(
  left: DatasetCacheUsabilityIdentity,
  right: DatasetCacheUsabilityIdentity,
): boolean {
  return (
    targetsSameDatasetInputs(left, right) &&
    (left.cachePath === null ||
      right.cachePath === null ||
      left.cachePath === right.cachePath)
  );
}

function targetsSameDatasetInputs(
  left: DatasetCacheUsabilityIdentity,
  right: DatasetCacheUsabilityIdentity,
): boolean {
  return (
    left.dataset === right.dataset &&
    left.subset === right.subset &&
    left.split === right.split &&
    left.streaming === right.streaming
  );
}

function inventoryFingerprint(identity: DatasetCacheInventoryIdentity): string {
  const sizeBytes =
    typeof identity.sizeBytes === "number" &&
    Number.isFinite(identity.sizeBytes) &&
    identity.sizeBytes >= 0
      ? identity.sizeBytes
      : null;
  return JSON.stringify([
    normalizeCachePath(identity.cachePath),
    sizeBytes,
    Boolean(identity.partial),
    normalizeOptional(identity.partialTransport),
  ]);
}

export class DatasetCacheRejectionTracker {
  private generation = 0;
  private observed: ObservedDatasetCache | null = null;
  private rejected: RejectedDatasetCache | null = null;

  observe(
    identity: DatasetCacheUsabilityIdentity,
    inventory: DatasetCacheInventoryIdentity,
  ): void {
    const fingerprint = inventoryFingerprint(inventory);
    if (
      this.observed &&
      targetsSameDatasetInputs(this.observed.identity, identity) &&
      this.observed.fingerprint !== fingerprint
    ) {
      this.generation += 1;
      if (
        this.rejected &&
        targetsSameDatasetInputs(this.rejected.identity, identity)
      ) {
        this.rejected = null;
      }
    }
    this.observed = { identity, fingerprint };
    if (
      this.rejected &&
      this.rejected.fingerprint === null &&
      targetsSameCacheUsability(this.rejected.identity, identity)
    ) {
      this.rejected.fingerprint = fingerprint;
    }
  }

  beginValidation(
    identity: DatasetCacheUsabilityIdentity,
  ): DatasetCacheValidationToken {
    return { identity, generation: this.generation };
  }

  isValidationCurrent(token: DatasetCacheValidationToken): boolean {
    return token.generation === this.generation;
  }

  rejectValidation(token: DatasetCacheValidationToken): boolean {
    if (token.generation !== this.generation) {
      return false;
    }
    const fingerprint =
      this.observed &&
      targetsSameCacheUsability(this.observed.identity, token.identity)
        ? this.observed.fingerprint
        : null;
    this.rejected = { identity: token.identity, fingerprint };
    return true;
  }

  shouldPromote(
    identity: DatasetCacheUsabilityIdentity,
    inventory: DatasetCacheInventoryIdentity,
  ): boolean {
    const fingerprint = inventoryFingerprint(inventory);
    this.observe(identity, inventory);
    if (!this.rejected) {
      return true;
    }
    if (!targetsSameDatasetInputs(this.rejected.identity, identity)) {
      return true;
    }
    if (!targetsSameCacheUsability(this.rejected.identity, identity)) {
      this.generation += 1;
      this.rejected = null;
      return true;
    }
    if (this.rejected.fingerprint === fingerprint) {
      return false;
    }
    this.generation += 1;
    this.rejected = null;
    return true;
  }

  reset(dataset?: string | null): void {
    this.generation += 1;
    if (!dataset) {
      this.observed = null;
      this.rejected = null;
      return;
    }
    const normalizedDataset = normalizeModelIdentity(dataset);
    if (this.observed?.identity.dataset === normalizedDataset) {
      this.observed = null;
    }
    if (this.rejected?.identity.dataset === normalizedDataset) {
      this.rejected = null;
    }
  }
}

export const trainingDatasetCacheRejections =
  new DatasetCacheRejectionTracker();
