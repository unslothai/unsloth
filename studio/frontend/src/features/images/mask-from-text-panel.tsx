// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Cancel01Icon, Search01Icon, Download01Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useCallback, useEffect, useRef, useState } from "react";

import { Button } from "@/components/ui/button";
import { InfoHint } from "@/components/ui/info-hint";
import { Input } from "@/components/ui/input";
import { Progress } from "@/components/ui/progress";
import { Spinner } from "@/components/ui/spinner";
import { Switch } from "@/components/ui/switch";
import { useIsAccountOwner } from "@/features/auth";
import { formatBytes, getHfToken, hfApiToken } from "@/features/hub";
import { toast } from "@/lib/toast";
import { cn } from "@/lib/utils";

import {
  MASK_FROM_TEXT_HINT,
  MASK_FROM_TEXT_LABEL,
  MASK_FROM_TEXT_PLACEHOLDER,
  parseMaskPrompt,
  segmenterDownloadFraction,
  segmenterDownloadNotice,
  segmenterNeedsDownload,
  segmenterReady,
  segmentSummary,
  type SegmenterStatus,
} from "./lib/mask-from-text";
import {
  cancelSegmenterDownload,
  getSegmenterStatus,
  releaseSegmenter,
  segmentImage,
  startSegmenterDownload,
} from "./mask-segment-api";

/** Segmenter status, read once on mount and every second while a download runs. */
function useSegmenterStatus() {
  const [status, setStatus] = useState<SegmenterStatus | null>(null);
  const mounted = useRef(true);
  const refresh = useCallback(() => {
    getSegmenterStatus()
      .then((next) => {
        if (mounted.current) setStatus(next);
      })
      .catch(() => {});
  }, []);
  const downloading = status?.download.downloading === true;
  useEffect(() => {
    mounted.current = true;
    refresh();
    return () => {
      mounted.current = false;
    };
  }, [refresh]);
  useEffect(() => {
    if (!downloading) return;
    const id = window.setInterval(refresh, 1000);
    return () => window.clearInterval(id);
  }, [downloading, refresh]);
  // Re-read once a download ends, so the notice clears without a reload.
  const wasDownloading = useRef(false);
  useEffect(() => {
    if (wasDownloading.current && !downloading) refresh();
    wasDownloading.current = downloading;
  }, [downloading, refresh]);
  return { status, refresh };
}

function SegmenterDownloadNotice({
  status,
  onChanged,
}: {
  status: SegmenterStatus;
  onChanged: () => void;
}) {
  const [busy, setBusy] = useState(false);
  // The download and cancel routes are owner-only; a managed account is told who can fetch it instead.
  const owner = useIsAccountOwner();
  const { download } = status;
  const fraction = segmenterDownloadFraction(download);
  const size = status.size_bytes ? formatBytes(status.size_bytes) : null;
  const label = status.label;
  const notice = segmenterDownloadNotice(status, size);
  const start = async () => {
    setBusy(true);
    try {
      await startSegmenterDownload(hfApiToken(getHfToken()));
    } catch (err) {
      toast.error(`Could not download ${label}`, {
        description: err instanceof Error ? err.message : undefined,
      });
    } finally {
      setBusy(false);
      onChanged();
    }
  };
  const cancel = async () => {
    try {
      await cancelSegmenterDownload();
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Could not cancel the download");
    } finally {
      onChanged();
    }
  };
  return (
    <div
      data-testid="mask-segmenter-download"
      className="flex flex-col gap-1.5 rounded-lg border border-border/60 bg-muted/30 px-2.5 py-2 text-ui-11"
    >
      {download.downloading ? (
        <>
          <div className="flex items-center justify-between gap-2">
            <span className="text-muted-foreground">
              {download.bytes_total
                ? `Downloading ${label}: ${formatBytes(download.bytes_done ?? 0)} of ${formatBytes(download.bytes_total)}`
                : `Downloading ${label}...`}
            </span>
            <Button
              type="button"
              size="sm"
              variant="ghost"
              className={cn("h-6 gap-1 px-1.5 text-xs", !owner && "hidden")}
              onClick={() => void cancel()}
            >
              <HugeiconsIcon icon={Cancel01Icon} className="size-3.5" />
              Cancel
            </Button>
          </div>
          <Progress
            className="h-1.5"
            value={fraction === null ? undefined : fraction * 100}
            indeterminate={fraction === null}
          />
        </>
      ) : (
        <>
          <span className="text-muted-foreground">
            {notice.text}
            {notice.needsLicense ? (
              <>
                {" Its license must be accepted on Hugging Face first"}
                {status.license_url ? (
                  <>
                    {": "}
                    <a
                      href={status.license_url}
                      target="_blank"
                      rel="noreferrer"
                      className="underline underline-offset-2"
                    >
                      {status.repo || "the model page"}
                    </a>
                  </>
                ) : null}
                {", and your Hugging Face token must be set."}
              </>
            ) : null}
          </span>
          {download.error && !download.cancelled ? (
            <span className="text-destructive">{download.error}</span>
          ) : null}
          {owner ? (
            <Button
              type="button"
              size="sm"
              variant="secondary"
              className="gap-1.5 self-start text-xs"
              disabled={busy}
              onClick={() => void start()}
            >
              {busy ? <Spinner className="size-3.5" /> : <HugeiconsIcon icon={Download01Icon} className="size-3.5" />}
              {size ? `Download (${size})` : "Download"}
            </Button>
          ) : (
            <span className="text-muted-foreground">Ask the owner of this Unsloth install to download it.</span>
          )}
        </>
      )}
    </div>
  );
}

/** "Mask from text": finds what the text names in ``image`` and hands the mask to the painter. */
export function MaskFromTextPanel({
  image,
  onMask,
}: {
  image: string;
  onMask: (mask: { dataUrl: string; additive: boolean; empty?: boolean }) => void;
}) {
  const { status, refresh } = useSegmenterStatus();
  const [prompt, setPrompt] = useState("");
  const [additive, setAdditive] = useState(false);
  const [running, setRunning] = useState(false);
  // The last outcome, kept with the source it was for: a new source hides it instead of an effect clearing it.
  const [outcome, setOutcome] = useState<{ image: string; error: string | null; summary: string | null } | null>(
    null,
  );
  const error = outcome?.image === image ? outcome.error : null;
  const summary = outcome?.image === image ? outcome.summary : null;
  const controllerRef = useRef<AbortController | null>(null);
  const latestImage = useRef(image);

  // A new source drops a run still in flight for the old one, so its mask never lands on the new image.
  useEffect(() => {
    latestImage.current = image;
    controllerRef.current?.abort();
  }, [image]);

  // Closing the painter frees the segmenter instead of holding it for its keep-alive.
  useEffect(
    () => () => {
      controllerRef.current?.abort();
      releaseSegmenter().catch(() => {});
    },
    [],
  );

  const ready = segmenterReady(status);
  const concepts = parseMaskPrompt(prompt);
  const canRun = ready && !running && concepts.length > 0;

  const run = async () => {
    if (!canRun) return;
    controllerRef.current?.abort();
    const controller = new AbortController();
    controllerRef.current = controller;
    const source = image;
    setRunning(true);
    setOutcome(null);
    try {
      const result = await segmentImage({ image: source, prompt }, controller.signal);
      if (controller.signal.aborted || latestImage.current !== source) return;
      setOutcome({ image: source, error: null, summary: segmentSummary(result) });
      if (result.instances.length > 0 || additive === false) {
        // Nothing found with Replace still clears the mask: the painter shows exactly what the text selected.
        onMask({ dataUrl: result.mask, additive, empty: result.instances.length === 0 });
      }
    } catch (err) {
      if (controller.signal.aborted) return;
      setOutcome({
        image: source,
        error: err instanceof Error ? err.message : "Could not make a mask from the text.",
        summary: null,
      });
      // A 409 can mean the weights went missing; re-read so the download notice comes back.
      refresh();
    } finally {
      if (controllerRef.current === controller) {
        controllerRef.current = null;
        setRunning(false);
      }
    }
  };

  return (
    <div data-testid="mask-from-text" className="flex flex-col gap-1.5">
      <div className="flex items-center gap-1">
        <label htmlFor="mask-from-text-input" className="text-xs font-medium text-muted-foreground">
          {MASK_FROM_TEXT_LABEL}
        </label>
        <InfoHint>{MASK_FROM_TEXT_HINT}</InfoHint>
      </div>
      {status && !status.supported ? (
        <p className="text-ui-11 leading-snug text-muted-foreground">
          {status.reason || "Mask from text is not available on this install."}
        </p>
      ) : null}
      {segmenterNeedsDownload(status) && status ? (
        <SegmenterDownloadNotice status={status} onChanged={refresh} />
      ) : null}
      <div className="flex gap-2">
        <Input
          id="mask-from-text-input"
          value={prompt}
          placeholder={MASK_FROM_TEXT_PLACEHOLDER}
          disabled={!ready}
          onChange={(e) => setPrompt(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter") {
              e.preventDefault();
              void run();
            }
          }}
          className="h-8 flex-1 text-xs"
        />
        <Button
          type="button"
          size="sm"
          variant="secondary"
          className="gap-1.5 text-xs"
          disabled={!canRun}
          onClick={() => void run()}
        >
          {running ? <Spinner className="size-3.5" /> : <HugeiconsIcon icon={Search01Icon} className="size-3.5" />}
          {running ? "Finding" : "Find"}
        </Button>
      </div>
      <div className="flex items-center gap-2">
        <Switch
          size="sm"
          checked={additive}
          onCheckedChange={setAdditive}
          disabled={!ready}
          aria-label="Add to current mask"
        />
        <span className="text-ui-11 text-muted-foreground">Add to current mask</span>
      </div>
      {error ? <p className="text-ui-11 leading-snug text-destructive">{error}</p> : null}
      {summary && !error ? (
        <p data-testid="mask-from-text-summary" className="text-ui-11 leading-snug text-muted-foreground">
          {summary}
        </p>
      ) : null}
    </div>
  );
}
