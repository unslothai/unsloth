// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useEffect } from "react";
import { toast } from "sonner";
import { FlowsApiError, getFlow, saveFlow } from "../api";
import {
  documentFromState,
  documentSignature,
  useDocStore,
} from "../stores/doc-store";
import { useUiStore } from "../stores/ui-store";

const SAVE_DEBOUNCE_MS = 800;

interface SaveSession {
  flowId: string;
  baseUpdatedAt: number | null;
  savedSignature: string;
  failedSignature: string | null;
  inflight: Promise<void> | null;
  // A PUT is on the wire: a second one sent beside it would carry a stale baseUpdatedAt.
  saving: boolean;
  timer: ReturnType<typeof setTimeout> | null;
}

let session: SaveSession | null = null;

export function beginSaveSession(
  flowId: string,
  updatedAt: number,
  signature: string,
): void {
  if (session?.timer) clearTimeout(session.timer);
  session = {
    flowId,
    baseUpdatedAt: updatedAt,
    savedSignature: signature,
    failedSignature: null,
    inflight: null,
    saving: false,
    timer: null,
  };
  useUiStore.getState().setSaveState("saved");
}

export function endSaveSession(): void {
  const current = session;
  if (!current) return;
  if (current.timer) {
    clearTimeout(current.timer);
    current.timer = null;
  }
  const state = useDocStore.getState();
  const doc = state.flowId === current.flowId ? documentFromState(state) : null;
  if (doc) {
    current.inflight = (current.inflight ?? Promise.resolve()).then(() =>
      saveClosing(current, doc),
    );
  }
  void (current.inflight ?? Promise.resolve()).finally(() => {
    if (session === current) session = null;
  });
}

async function saveClosing(
  current: SaveSession,
  doc: ReturnType<typeof documentFromState>,
): Promise<void> {
  if (documentSignature(doc) === current.savedSignature) return;
  current.saving = true;
  try {
    const record = await saveFlow(current.flowId, doc, current.baseUpdatedAt);
    current.baseUpdatedAt = record.updatedAt;
  } catch (error) {
    toast.error(
      error instanceof Error ? error.message : "Could not save the flow",
    );
  } finally {
    current.saving = false;
  }
}

async function performSave(force = false, keepalive = false): Promise<boolean> {
  const current = session;
  if (!current) return true;
  const state = useDocStore.getState();
  if (state.flowId !== current.flowId) return true;
  const doc = documentFromState(state);
  const signature = documentSignature(doc);
  if (signature === current.savedSignature) {
    useUiStore.getState().setSaveState("saved");
    return true;
  }
  if (!force && signature === current.failedSignature) return false;
  useUiStore.getState().setSaveState("saving");
  current.saving = true;
  try {
    const record = await saveFlow(
      current.flowId,
      doc,
      current.baseUpdatedAt,
      keepalive,
    );
    current.saving = false;
    current.baseUpdatedAt = record.updatedAt;
    current.savedSignature = signature;
    if (session !== current) return true;
    current.failedSignature = null;
    const now = useDocStore.getState();
    const latest =
      now.graph === state.graph && now.meta === state.meta
        ? signature
        : documentSignature(documentFromState(now));
    useUiStore
      .getState()
      .setSaveState(latest === signature ? "saved" : "dirty");
    if (latest !== signature) scheduleSave();
    return true;
  } catch (error) {
    current.saving = false;
    if (session !== current) return false;
    current.failedSignature = signature;
    if (error instanceof FlowsApiError && error.status === 409) {
      useUiStore.getState().setSaveState("conflict");
      toast.error("This flow changed somewhere else", {
        description: "Keep your version, or reload the saved one.",
        duration: 20_000,
        action: {
          label: "Keep mine",
          onClick: () => {
            void getFlow(current.flowId).then((record) => {
              current.baseUpdatedAt = record.updatedAt;
              void performSave(true);
            });
          },
        },
        cancel: {
          label: "Reload",
          onClick: () => {
            void getFlow(current.flowId).then((record) => {
              useDocStore.getState().load(record.doc);
              beginSaveSession(
                record.id,
                record.updatedAt,
                documentSignature(record.doc),
              );
            });
          },
        },
      });
      return false;
    }
    useUiStore.getState().setSaveState("error");
    toast.error(
      error instanceof Error ? error.message : "Could not save the flow",
    );
    return false;
  }
}

function scheduleSave(): void {
  const current = session;
  if (!current) return;
  if (current.timer) clearTimeout(current.timer);
  current.timer = setTimeout(() => {
    current.timer = null;
    current.inflight = (current.inflight ?? Promise.resolve()).then(
      async () => {
        await performSave();
      },
    );
  }, SAVE_DEBOUNCE_MS);
}

export async function flushSave(): Promise<boolean> {
  const current = session;
  if (!current) return true;
  if (current.timer) {
    clearTimeout(current.timer);
    current.timer = null;
  }
  const saved = (current.inflight ?? Promise.resolve()).then(() =>
    performSave(true),
  );
  current.inflight = saved.then(() => undefined);
  return saved;
}

/** Tab closing: send any unsaved edit now, as a keepalive PUT the browser finishes after the page is gone. */
export function saveOnUnload(): void {
  const current = session;
  if (!current) return;
  const state = useDocStore.getState();
  if (state.flowId !== current.flowId) return;
  if (documentSignature(documentFromState(state)) === current.savedSignature)
    return;
  if (current.saving) {
    void flushSave();
    return;
  }
  if (current.timer) {
    clearTimeout(current.timer);
    current.timer = null;
  }
  // Started synchronously, not behind `inflight`: a chained save never runs once the page unloads.
  const saved = performSave(true, true);
  current.inflight = (current.inflight ?? Promise.resolve()).then(() =>
    saved.then(() => undefined),
  );
}

export function useAutosave(enabled: boolean): void {
  useEffect(() => {
    if (!enabled) return;
    let lastVersion = useDocStore.getState().version;
    const unsubscribe = useDocStore.subscribe((state) => {
      if (state.version === lastVersion) return;
      lastVersion = state.version;
      if (!session || state.flowId !== session.flowId) return;
      useUiStore.getState().setSaveState("dirty");
      scheduleSave();
    });
    const onBeforeUnload = () => saveOnUnload();
    window.addEventListener("beforeunload", onBeforeUnload);
    return () => {
      unsubscribe();
      window.removeEventListener("beforeunload", onBeforeUnload);
      void flushSave();
    };
  }, [enabled]);
}
