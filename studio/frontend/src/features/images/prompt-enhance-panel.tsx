// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Cancel01Icon, Copy01Icon, Download04Icon, SparklesIcon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { ChevronDown } from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";

import { Button } from "@/components/ui/button";
import { InfoHint } from "@/components/ui/info-hint";
import { Progress } from "@/components/ui/progress";
import { Spinner } from "@/components/ui/spinner";
import { Switch } from "@/components/ui/switch";
import { formatBytes, getHfToken, hfApiToken } from "@/features/hub";
import { copyToClipboard } from "@/lib/copy-to-clipboard";
import { toast } from "@/lib/toast";
import { cn } from "@/lib/utils";

import {
  cancelPromptUpsamplerDownload,
  getPromptUpsamplerStatus,
  previewEnhancedPrompt,
  startPromptUpsamplerDownload,
} from "./api";
import {
  ENHANCE_PROMPT_HINT,
  ENHANCE_PROMPT_LABEL,
  type PromptUpsamplerPreviewRequest,
  type PromptUpsamplerStatus,
  upsamplerDownloadBytes,
  upsamplerNeedsDownload,
  upsamplerDownloadFraction,
} from "./lib/prompt-enhance";

/** Upsampler status, polled every second while a download runs and once otherwise. */
function usePromptUpsamplerStatus(active: boolean) {
  const [status, setStatus] = useState<PromptUpsamplerStatus | null>(null);
  const [tick, setTick] = useState(0);
  const refresh = useCallback(() => setTick((t) => t + 1), []);
  const downloading = status?.download.downloading === true;
  useEffect(() => {
    if (!active) return;
    const controller = new AbortController();
    getPromptUpsamplerStatus(controller.signal)
      .then(setStatus)
      .catch(() => {});
    return () => controller.abort();
  }, [active, tick]);
  useEffect(() => {
    if (!active || !downloading) return;
    const id = window.setInterval(refresh, 1000);
    return () => window.clearInterval(id);
  }, [active, downloading, refresh]);
  return { status, refresh };
}

function CopyButton({ text, label }: { text: string; label: string }) {
  return (
    <Button
      type="button"
      size="sm"
      variant="ghost"
      className="h-7 gap-1 px-2 text-xs"
      aria-label={label}
      onClick={() => {
        void copyToClipboard(text).then((ok) =>
          ok ? toast.success("Copied") : toast.error("Could not copy"),
        );
      }}
    >
      <HugeiconsIcon icon={Copy01Icon} className="size-3.5" />
      Copy
    </Button>
  );
}

function UpsamplerDownloadNotice({
  status,
  onChanged,
}: {
  status: PromptUpsamplerStatus;
  onChanged: () => void;
}) {
  const [busy, setBusy] = useState(false);
  const { download } = status;
  const fraction = upsamplerDownloadFraction(download);
  const downloadBytes = upsamplerDownloadBytes(status);
  const size = downloadBytes > 0 ? formatBytes(downloadBytes) : null;
  const start = async () => {
    setBusy(true);
    try {
      await startPromptUpsamplerDownload(hfApiToken(getHfToken()));
    } catch (err) {
      toast.error("Could not download the prompt upsampler", {
        description: err instanceof Error ? err.message : undefined,
      });
    } finally {
      setBusy(false);
      onChanged();
    }
  };
  const cancel = async () => {
    try {
      await cancelPromptUpsamplerDownload();
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Could not cancel the download");
    } finally {
      onChanged();
    }
  };
  return (
    <div
      data-testid="prompt-upsampler-download"
      className="flex flex-col gap-1.5 rounded-md border border-border/60 bg-muted/30 px-2.5 py-2 text-ui-11"
    >
      {download.downloading ? (
        <>
          <div className="flex items-center justify-between gap-2">
            <span className="text-muted-foreground">
              {download.bytes_total
                ? `Downloading prompt upsampler: ${formatBytes(download.bytes_done ?? 0)} of ${formatBytes(download.bytes_total)}`
                : "Downloading prompt upsampler..."}
            </span>
            <Button
              type="button"
              size="sm"
              variant="ghost"
              className="h-6 gap-1 px-1.5 text-xs"
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
            {status.downloaded
              ? `The prompt upsampler's vision projector${size ? ` (${size})` : ""} is not downloaded. Until it is, edits are enhanced from the text alone.`
              : `The prompt upsampler${size ? ` (${size})` : ""} is not downloaded. Until it is, prompts run as typed.`}
          </span>
          {download.error && !download.cancelled ? (
            <span className="text-destructive">{download.error}</span>
          ) : null}
          <Button
            type="button"
            size="sm"
            variant="secondary"
            className="h-7 gap-1.5 self-start text-xs"
            disabled={busy}
            onClick={() => void start()}
          >
            {busy ? <Spinner className="size-3.5" /> : <HugeiconsIcon icon={Download04Icon} className="size-3.5" />}
            {size ? `Download (${size})` : "Download"}
          </Button>
        </>
      )}
    </div>
  );
}

/** The Enhance prompt toggle, the download notice, and the preview panel. Rendered only for a supporting family. */
export function EnhancePromptControls({
  enabled,
  onEnabledChange,
  prompt,
  previewRequest,
  onUseCaption,
}: {
  enabled: boolean;
  onEnabledChange: (on: boolean) => void;
  prompt: string;
  /** The preview body for the current form (prompt, reference images, variant). */
  previewRequest: () => PromptUpsamplerPreviewRequest;
  /** Put the caption in the prompt box and turn enhancement off. */
  onUseCaption: (caption: string) => void;
}) {
  const { status, refresh } = usePromptUpsamplerStatus(true);
  // Re-read once a download ends, so the notice clears without a reload.
  const wasDownloading = useRef(false);
  useEffect(() => {
    const now = status?.download.downloading === true;
    if (wasDownloading.current && !now) refresh();
    wasDownloading.current = now;
  }, [status?.download.downloading, refresh]);

  const [preview, setPreview] = useState<{ source: string; caption: string } | null>(null);
  const [previewOpen, setPreviewOpen] = useState(true);
  const [previewing, setPreviewing] = useState(false);
  const previewSeq = useRef(0);
  const runPreview = async () => {
    const body = previewRequest();
    if (!body.prompt.trim()) {
      toast.error("Prompt is empty");
      return;
    }
    const seq = ++previewSeq.current;
    setPreviewing(true);
    try {
      const res = await previewEnhancedPrompt(body);
      if (seq !== previewSeq.current) return;
      setPreview({ source: body.prompt, caption: res.enhanced_prompt });
      setPreviewOpen(true);
    } catch (err) {
      if (seq !== previewSeq.current) return;
      toast.error("Could not enhance the prompt", {
        description: err instanceof Error ? err.message : undefined,
      });
    } finally {
      if (seq === previewSeq.current) setPreviewing(false);
    }
  };

  const needsDownload = enabled && status !== null && upsamplerNeedsDownload(status);
  const canPreview = status === null || status.downloaded;
  // A caption for an older prompt stays readable but is marked stale.
  const stale = preview !== null && preview.source.trim() !== prompt.trim();

  return (
    <div className="flex flex-col gap-1.5" data-testid="enhance-prompt-controls">
      <div className="flex items-center justify-between gap-2">
        <span className="flex items-center gap-1 text-xs font-medium text-muted-foreground">
          {ENHANCE_PROMPT_LABEL}
          <InfoHint>{ENHANCE_PROMPT_HINT}</InfoHint>
        </span>
        <div className="flex items-center gap-1.5">
          <Button
            type="button"
            size="sm"
            variant="ghost"
            className="h-7 gap-1 px-2 text-xs"
            disabled={previewing || !canPreview || !prompt.trim()}
            title={canPreview ? "Rewrite the prompt now and show the caption" : "Download the prompt upsampler first"}
            onClick={() => void runPreview()}
          >
            {previewing ? <Spinner className="size-3.5" /> : <HugeiconsIcon icon={SparklesIcon} className="size-3.5" />}
            {previewing ? "Enhancing..." : "Preview"}
          </Button>
          <Switch checked={enabled} onCheckedChange={onEnabledChange} aria-label={ENHANCE_PROMPT_LABEL} />
        </div>
      </div>
      {needsDownload && status ? <UpsamplerDownloadNotice status={status} onChanged={refresh} /> : null}
      {preview ? (
        <div
          data-testid="enhanced-prompt-preview"
          className="flex flex-col gap-1 rounded-md border border-border/60 px-2.5 py-2 text-xs"
        >
          <div className="flex items-center justify-between gap-2">
            <button
              type="button"
              aria-expanded={previewOpen}
              onClick={() => setPreviewOpen((o) => !o)}
              className="flex items-center gap-1 font-medium text-muted-foreground transition-colors hover:text-foreground"
            >
              Enhanced prompt{stale ? " (for an earlier prompt)" : ""}
              <ChevronDown className={cn("size-3 transition-transform", previewOpen && "rotate-180")} />
            </button>
            <button
              type="button"
              aria-label="Dismiss enhanced prompt"
              className="text-muted-foreground hover:text-foreground"
              onClick={() => setPreview(null)}
            >
              <HugeiconsIcon icon={Cancel01Icon} className="size-3.5" />
            </button>
          </div>
          {previewOpen ? (
            <>
              <p className="max-h-48 overflow-y-auto whitespace-pre-wrap break-words text-foreground">
                {preview.caption}
              </p>
              <div className="flex items-center justify-end gap-1">
                <CopyButton text={preview.caption} label="Copy enhanced prompt" />
                <Button
                  type="button"
                  size="sm"
                  variant="secondary"
                  className="h-7 px-2 text-xs"
                  onClick={() => {
                    onUseCaption(preview.caption);
                    setPreview(null);
                  }}
                >
                  Use as prompt
                </Button>
              </div>
            </>
          ) : null}
        </div>
      ) : null}
    </div>
  );
}

/** The recipe's "Enhanced prompt" row: collapsible, with copy. */
export function EnhancedPromptRecipeRow({ caption }: { caption: string }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="grid grid-cols-[72px_1fr] items-start gap-2">
      <button
        type="button"
        aria-expanded={open}
        onClick={() => setOpen((o) => !o)}
        className="flex items-start gap-0.5 text-left text-muted-foreground hover:text-foreground"
      >
        Enhanced prompt
        <ChevronDown className={cn("mt-0.5 size-3 shrink-0 transition-transform", open && "rotate-180")} />
      </button>
      <div className="flex min-w-0 flex-col gap-1">
        <span
          className={cn(
            "min-w-0 text-foreground",
            open ? "whitespace-pre-wrap break-words" : "truncate",
          )}
        >
          {caption}
        </span>
        {open ? (
          <div className="flex justify-end">
            <CopyButton text={caption} label="Copy enhanced prompt" />
          </div>
        ) : null}
      </div>
    </div>
  );
}
