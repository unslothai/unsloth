// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export interface HfCredentialStatus {
  has_token: boolean;
  token: string | null;
}

export interface LegacyHfTokenReconciliationDependencies<
  T extends HfCredentialStatus,
> {
  loadSavedToken: () => Promise<T>;
  getLegacyToken: () => string;
  saveLegacyToken: (token: string) => Promise<T>;
  removeLegacyToken: (expectedToken: string) => void;
  applyToken: (token: string) => void;
}

export async function reconcileLegacyHfToken<T extends HfCredentialStatus>(
  dependencies: LegacyHfTokenReconciliationDependencies<T>,
): Promise<T> {

  const legacyTokenBeforeLoad = dependencies.getLegacyToken().trim();
  let saved: T;
  try {
    saved = await dependencies.loadSavedToken();
  } catch (error) {
    const retainedToken = dependencies.getLegacyToken().trim();
    if (retainedToken) dependencies.applyToken(retainedToken);
    throw error;
  }
  if (saved.has_token && saved.token) {
    dependencies.applyToken(saved.token);
    if (legacyTokenBeforeLoad) {
      dependencies.removeLegacyToken(legacyTokenBeforeLoad);
    }
    return saved;
  }

  const legacyToken = dependencies.getLegacyToken().trim();
  if (!legacyToken) {
    dependencies.applyToken("");
    return saved;
  }

  let migrated: T;
  try {
    migrated = await dependencies.saveLegacyToken(legacyToken);
  } catch (error) {
    // Keep upgraded users working from their retained legacy token while the
    // backend is unavailable. The caller still sees the failure and can retry.
    dependencies.applyToken(legacyToken);
    throw error;
  }
  if (migrated.has_token) {
    dependencies.applyToken(migrated.token ?? legacyToken);
    dependencies.removeLegacyToken(legacyToken);
  }
  return migrated;
}


export interface ProviderCredentialStatus {
  id: string;
  has_api_key: boolean;
}

export interface LegacyProviderKeyReconciliationDependencies<
  T extends ProviderCredentialStatus,
> {
  getLegacyKey: (providerId: string) => string;
  saveLegacyKey: (providerId: string, apiKey: string) => Promise<T>;
  removeLegacyKey: (providerId: string, expectedApiKey: string) => void;

  isCurrent?: () => boolean;
}

export async function reconcileLegacyProviderKeys<
  T extends ProviderCredentialStatus,
>(
  configs: T[],
  dependencies: LegacyProviderKeyReconciliationDependencies<T>,
): Promise<T[]> {
  const reconciled = [...configs];
  for (let index = 0; index < reconciled.length; index += 1) {

    if (dependencies.isCurrent && !dependencies.isCurrent()) return reconciled;
    const config = reconciled[index];
    if (config.has_api_key) {
      const legacyKey = dependencies.getLegacyKey(config.id);
      if (legacyKey) dependencies.removeLegacyKey(config.id, legacyKey);
      continue;
    }
    const legacyKey = dependencies.getLegacyKey(config.id);
    const normalizedLegacyKey = legacyKey.trim();
    if (!normalizedLegacyKey) continue;
    try {
      const saved = await dependencies.saveLegacyKey(config.id, normalizedLegacyKey);
      if (dependencies.isCurrent && !dependencies.isCurrent()) return reconciled;
      reconciled[index] = saved;
      if (saved.has_api_key) {
        dependencies.removeLegacyKey(config.id, legacyKey);
      }
    } catch {
      // Preserve local input for retry and per-request fallback.
    }
  }
  return reconciled;
}


export async function settleTasksIfCurrent(
  tasks: Array<() => Promise<unknown>>,
  isCurrent?: () => boolean,
): Promise<void> {
  if (isCurrent && !isCurrent()) return;
  await Promise.allSettled(tasks.map((task) => task()));
}

export interface CredentialBootstrapDependencies<T> {
  hydrateHfToken: () => Promise<void>;
  getProviders: () => T[];
  syncProviders: (providers: T[]) => Promise<T[]>;
  setProviders: (providers: T[]) => void;

  isCurrent?: () => boolean;
}

/** Wait for both server-first reconciliations before releasing app content. */
export async function runCredentialBootstrap<T>(
  dependencies: CredentialBootstrapDependencies<T>,
): Promise<void> {

  if (dependencies.isCurrent && !dependencies.isCurrent()) return;
  const providerSync = dependencies.syncProviders(dependencies.getProviders());
  const [, providerResult] = await Promise.allSettled([
    dependencies.hydrateHfToken(),
    providerSync,
  ]);
  if (
    providerResult.status === "fulfilled" &&
    (!dependencies.isCurrent || dependencies.isCurrent())
  ) {
    dependencies.setProviders(providerResult.value);
  }
}
