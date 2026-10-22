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
    timer: null,
  };
  useUiStore.getState().setSaveState("saved");
}

/** Leave the editor: a pending edit is still saved, then the session closes. */
export function endSaveSession(): void {
  const current = session;
  if (!current) return;
  if (current.timer) {
    clearTimeout(current.timer);
    current.timer = null;
    current.inflight = (current.inflight ?? Promise.resolve()).then(() =>
      performSave(true),
    );
  }
  void (current.inflight ?? Promise.resolve()).finally(() => {
    if (session === current) session = null;
  });
}

async function performSave(force = false): Promise<void> {
  const current = session;
  if (!current) return;
  const state = useDocStore.getState();
  if (state.flowId !== current.flowId) return;
  const doc = documentFromState(state);
  const signature = documentSignature(doc);
  if (signature === current.savedSignature) {
    useUiStore.getState().setSaveState("saved");
    return;
  }
  // A signature that just failed is not retried until it changes or the user saves explicitly.
  if (!force && signature === current.failedSignature) return;
  useUiStore.getState().setSaveState("saving");
  try {
    const record = await saveFlow(current.flowId, doc, current.baseUpdatedAt);
    if (session !== current) return;
    current.baseUpdatedAt = record.updatedAt;
    current.savedSignature = signature;
    current.failedSignature = null;
    const latest = documentSignature(documentFromState(useDocStore.getState()));
    useUiStore
      .getState()
      .setSaveState(latest === signature ? "saved" : "dirty");
    if (latest !== signature) scheduleSave();
  } catch (error) {
    if (session !== current) return;
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
      return;
    }
    useUiStore.getState().setSaveState("error");
    toast.error(
      error instanceof Error ? error.message : "Could not save the flow",
    );
  }
}

function scheduleSave(): void {
  const current = session;
  if (!current) return;
  if (current.timer) clearTimeout(current.timer);
  current.timer = setTimeout(() => {
    current.timer = null;
    current.inflight = (current.inflight ?? Promise.resolve()).then(() =>
      performSave(),
    );
  }, SAVE_DEBOUNCE_MS);
}

/** Save now (Ctrl+S, before playing); resolves once the document on the server is current. */
export async function flushSave(): Promise<void> {
  const current = session;
  if (!current) return;
  if (current.timer) {
    clearTimeout(current.timer);
    current.timer = null;
  }
  current.inflight = (current.inflight ?? Promise.resolve()).then(() =>
    performSave(true),
  );
  await current.inflight;
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
    const onBeforeUnload = () => {
      if (session?.timer) void flushSave();
    };
    window.addEventListener("beforeunload", onBeforeUnload);
    return () => {
      unsubscribe();
      window.removeEventListener("beforeunload", onBeforeUnload);
      void flushSave();
    };
  }, [enabled]);
}
