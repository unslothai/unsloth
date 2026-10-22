// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useEffect, useSyncExternalStore } from "react";
import { subscribeInventory } from "../stores/inventory-events";
import {
  type ScanFolderInfo,
  addScanFolder,
  listScanFolders,
  removeScanFolder,
} from "./api";

/** One copy of the custom folder list for every surface (folders dialog, model picker, Download
 *  to menu), so a change in one shows in the others and opening any of them is one request. */
export interface ScanFoldersState {
  folders: ScanFolderInfo[];
  loaded: boolean;
  loading: boolean;
  error: string | null;
}

const EMPTY_STATE: ScanFoldersState = {
  folders: [],
  loaded: false,
  loading: false,
  error: null,
};

let state = EMPTY_STATE;
let inFlight: Promise<ScanFolderInfo[]> | null = null;
// Bumped by every local mutation, so a list fetched before it cannot overwrite its result.
let mutationSeq = 0;
// The mutation's own inventory bump must not refetch the list it is about to update in place.
let mutating = 0;
const listeners = new Set<() => void>();

function setState(next: Partial<ScanFoldersState>): void {
  state = { ...state, ...next };
  for (const listener of [...listeners]) listener();
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

export function getScanFoldersState(): ScanFoldersState {
  return state;
}

export function subscribeScanFolders(listener: () => void): () => void {
  listeners.add(listener);
  return () => listeners.delete(listener);
}

/** Concurrent callers share one request. */
export function refreshScanFolders(): Promise<ScanFolderInfo[]> {
  if (inFlight) return inFlight;
  const seq = mutationSeq;
  let superseded = false;
  setState({ loading: true });
  const request = listScanFolders()
    .then((folders) => {
      // May predate a mutation that landed meanwhile: reread rather than drop it.
      superseded = seq !== mutationSeq;
      if (!superseded) setState({ folders, loaded: true, error: null });
      return state.folders;
    })
    .catch((error: unknown) => {
      setState({ error: errorMessage(error) });
      throw error;
    })
    .finally(() => {
      inFlight = null;
      setState({ loading: false });
      if (superseded) void refreshScanFolders().catch(() => undefined);
    });
  inFlight = request;
  return request;
}

function revalidateScanFolders(): void {
  if (inFlight || mutating > 0) return;
  void refreshScanFolders().catch(() => undefined);
}

async function mutate<T>(request: () => Promise<T>): Promise<T> {
  mutating += 1;
  try {
    return await request();
  } finally {
    mutating -= 1;
  }
}

function upsert(folder: ScanFolderInfo): void {
  mutationSeq += 1;
  const rest = state.folders.filter((row) => row.id !== folder.id);
  setState({ folders: [...rest, folder] });
}

export interface AddScanFolderResult {
  folder: ScanFolderInfo;
  alreadyAdded: boolean;
}

/** `recursive` undefined keeps the stored flag of a folder that is already registered. */
export async function addScanFolderToStore(
  path: string,
  recursive?: boolean,
): Promise<AddScanFolderResult> {
  const known = new Map(state.folders.map((row) => [row.id, row]));
  const folder = await mutate(() => addScanFolder(path, recursive));
  const previous = known.get(folder.id);
  // The add route answers "ok" without scanning; keep a known row's last status.
  upsert({ ...folder, status: previous?.status ?? folder.status });
  return {
    folder,
    alreadyAdded:
      previous !== undefined &&
      Boolean(previous.recursive) === Boolean(folder.recursive),
  };
}

export async function setScanFolderRecursive(
  folder: ScanFolderInfo,
  recursive: boolean,
): Promise<void> {
  await addScanFolderToStore(folder.path, recursive);
}

export async function removeScanFolderFromStore(id: number): Promise<void> {
  await mutate(() => removeScanFolder(id));
  mutationSeq += 1;
  setState({ folders: state.folders.filter((row) => row.id !== id) });
}

export function useScanFolders(enabled = true): ScanFoldersState {
  const snapshot = useSyncExternalStore(
    subscribeScanFolders,
    getScanFoldersState,
    getScanFoldersState,
  );
  // Read on every open: health changes without an inventory bump (a scan finishing, a fix outside Studio).
  useEffect(() => {
    if (!enabled) return;
    revalidateScanFolders();
    return subscribeInventory(revalidateScanFolders);
  }, [enabled]);
  return snapshot;
}

export function resetScanFoldersStoreForTests(): void {
  state = EMPTY_STATE;
  inFlight = null;
  mutationSeq = 0;
  mutating = 0;
}
