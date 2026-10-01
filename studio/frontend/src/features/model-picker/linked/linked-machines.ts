// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useEffect } from "react";
import { create } from "zustand";
import {
  type LinkedInstance,
  type LinkedInstanceInfo,
  type LinkedInstanceStatus,
  fetchLinkedInstances,
  fetchLinkedInstancesInfo,
  fetchLinkedInstancesStatus,
} from "@/features/settings/api/linked-instances";

/** Which picker a machine choice belongs to. Chat and Images keep their own. */
export type LinkedPickerKind = "chat" | "image";

const MACHINE_KEY = "unsloth_picker_machine";
// A reopened picker reuses the last answer and refreshes behind it.
const STALE_MS = 15_000;

function loadMachines(): Record<LinkedPickerKind, string | null> {
  try {
    const raw = JSON.parse(localStorage.getItem(MACHINE_KEY) ?? "{}");
    return {
      chat: typeof raw.chat === "string" ? raw.chat : null,
      image: typeof raw.image === "string" ? raw.image : null,
    };
  } catch {
    return { chat: null, image: null };
  }
}

type LinkedMachinesState = {
  instances: LinkedInstance[];
  statuses: Record<string, LinkedInstanceStatus>;
  infos: Record<string, LinkedInstanceInfo>;
  loadedAt: number;
  loading: boolean;
  /** Instance id each picker is browsing, or null for this machine. */
  machine: Record<LinkedPickerKind, string | null>;
  setMachine: (kind: LinkedPickerKind, id: string | null) => void;
};

const byId = <T extends { id: string }>(rows: T[]) =>
  Object.fromEntries(rows.map((r) => [r.id, r])) as Record<string, T>;

export const useLinkedMachinesStore = create<LinkedMachinesState>((set) => ({
  instances: [],
  statuses: {},
  infos: {},
  loadedAt: 0,
  loading: false,
  machine: loadMachines(),
  setMachine: (kind, id) =>
    set((state) => {
      const machine = { ...state.machine, [kind]: id };
      try {
        localStorage.setItem(MACHINE_KEY, JSON.stringify(machine));
      } catch {
        // Ignore unavailable storage.
      }
      return { machine };
    }),
}));

let inFlight: Promise<void> | null = null;

export function refreshLinkedMachines(force = false): Promise<void> {
  const { loadedAt } = useLinkedMachinesStore.getState();
  if (!force && Date.now() - loadedAt < STALE_MS) return Promise.resolve();
  if (inFlight) return inFlight;
  useLinkedMachinesStore.setState({ loading: true });
  inFlight = (async () => {
    try {
      const instances = await fetchLinkedInstances();
      useLinkedMachinesStore.setState({ instances, loadedAt: Date.now() });
      if (instances.length === 0) return;
      const [statuses, infos] = await Promise.all([
        fetchLinkedInstancesStatus(),
        fetchLinkedInstancesInfo(),
      ]);
      useLinkedMachinesStore.setState({
        statuses: byId(statuses),
        infos: byId(infos),
        loadedAt: Date.now(),
      });
    } catch {
      // Not the owner, or offline: the picker just shows this machine.
    } finally {
      useLinkedMachinesStore.setState({ loading: false });
      inFlight = null;
    }
  })();
  return inFlight;
}

/** Linked machines for a picker, refreshed while `active`. A choice whose instance was removed
 *  reads as this machine. */
export function useLinkedMachines(kind: LinkedPickerKind, active: boolean) {
  const state = useLinkedMachinesStore();
  useEffect(() => {
    if (active) void refreshLinkedMachines();
  }, [active]);
  const chosen = state.machine[kind];
  const instance = chosen
    ? (state.instances.find((i) => i.id === chosen) ?? null)
    : null;
  return {
    instances: state.instances,
    statuses: state.statuses,
    infos: state.infos,
    loading: state.loading,
    instance,
    setMachine: (id: string | null) => state.setMachine(kind, id),
  };
}
