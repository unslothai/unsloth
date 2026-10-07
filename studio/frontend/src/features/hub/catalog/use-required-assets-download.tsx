// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useCallback, useEffect, useRef, useState } from "react";
import {
  getDiffusionDownloadPlan,
  type DiffusionDownloadPlan,
} from "@/features/images/api";
import { getVideoDownloadPlan } from "@/features/video/api";
import { getAudioDownloadPlan } from "@/features/audio/api";
import { toast } from "@/lib/toast";
import { useInventoryVersion } from "../stores/inventory-events";
import { useHfTokenStore } from "../stores/hf-token-store";
import {
  additionalAssetDownloads,
  selectDownloadEntries,
} from "../download-manager/required-assets";
import { RequiredAssetsDownloadDialog } from "../download-manager/required-assets-dialog";
import { withCachedCheckpoint } from "../download-manager/download-breakdown";
import { enqueueHubDownload } from "../download-manager/use-hub-download-queue";
import type { StagedDownloadEntry } from "../download-manager/use-staged-download";
export type AssetRuntime = "images" | "video" | "audio";

export function useRequiredAssetsDownload({
  repoId,
  filename,
  runtime,
  modelLabel,
}: {
  repoId: string;
  filename?: string;
  runtime?: AssetRuntime;
  modelLabel?: string;
}) {
  const token = useHfTokenStore((s) => s.token);
  const [checking, setChecking] = useState(false);
  const [pending, setPending] = useState<StagedDownloadEntry[] | null>(null);
  const inventoryVersion = useInventoryVersion();
  const sequence = useRef(0);
  const starting = useRef(false);
  const fetchPlan = useCallback(async () => {
    const body = {
      model_path: repoId,
      gguf_filename: filename,
      model_kind: filename ? ("gguf" as const) : ("pipeline" as const),
      hf_token: token || undefined,
    };
    const result =
      runtime === "audio"
        ? await getAudioDownloadPlan(repoId, token || undefined)
        : runtime === "video"
          ? await getVideoDownloadPlan(body)
          : await getDiffusionDownloadPlan(body);
    const resultPlan: DiffusionDownloadPlan = result;
    if (resultPlan.plan_failed)
      throw new Error(
        "Required file information is incomplete. Please try again.",
      );
    if (resultPlan.incompatible_reason)
      throw new Error(resultPlan.incompatible_reason);
    return resultPlan;
  }, [repoId, filename, token, runtime]);
  const cached = useRef<{
    fetch: typeof fetchPlan;
    inventory: number;
    expires: number;
    promise: ReturnType<typeof fetchPlan>;
  } | null>(null);
  const resolvePlan = useCallback(() => {
    const previous = cached.current;
    if (
      previous &&
      previous.fetch === fetchPlan &&
      previous.inventory === inventoryVersion &&
      previous.expires > Date.now()
    )
      return previous.promise;
    const record = {
      fetch: fetchPlan,
      inventory: inventoryVersion,
      expires: Date.now() + 30_000,
      promise: fetchPlan(),
    };
    cached.current = record;
    void record.promise.catch(() => {
      if (cached.current === record) cached.current = null;
    });
    return record.promise;
  }, [fetchPlan, inventoryVersion]);
  useEffect(() => {
    if (!runtime) return;
    const timer = window.setTimeout(() => {
      void resolvePlan().catch(() => {});
    }, 200);
    return () => window.clearTimeout(timer);
  }, [resolvePlan, runtime]);
  useEffect(() => {
    ++sequence.current;
    setPending(null);
    setChecking(false);
    starting.current = false;
    return () => {
      ++sequence.current;
    };
  }, [fetchPlan, runtime]);
  const request = async (fallback: () => void) => {
    if (!runtime) {
      fallback();
      return;
    }
    if (starting.current) return;
    starting.current = true;
    setChecking(true);
    const id = ++sequence.current;
    try {
      const p = await resolvePlan();
      if (sequence.current !== id) return;
      const entries = withCachedCheckpoint(
        p.entries.map((e) => ({
          repoId: e.repo_id,
          files: e.files,
          bytes: e.bytes,
          fileBytes: e.file_bytes,
          ggufFilename: e.gguf_filename,
          checkpoint:
            e.checkpoint ??
            (filename ? e.files.includes(filename) : e.repo_id === repoId),
        })),
        p.checkpoint_bytes,
      );
      if (additionalAssetDownloads(entries).length) setPending(entries);
      else if (entries.length)
        enqueueHubDownload(entries, { repoId, filename });
      else fallback();
    } catch (e) {
      if (sequence.current !== id) return;
      toast.error("Could not check required files", {
        description: e instanceof Error ? e.message : "Please try again.",
      });
    } finally {
      if (sequence.current === id) {
        starting.current = false;
        setChecking(false);
      }
    }
  };
  return {
    checking,
    request,
    dialog: (
      <RequiredAssetsDownloadDialog
        entries={pending}
        checking={checking}
        modelLabel={modelLabel ?? repoId}
        onCancel={() => {
          ++sequence.current;
          starting.current = false;
          setChecking(false);
          setPending(null);
        }}
        onConfirm={(include) => {
          if (pending)
            enqueueHubDownload(selectDownloadEntries(pending, include), {
              repoId,
              filename,
            });
          setPending(null);
        }}
      />
    ),
  };
}
