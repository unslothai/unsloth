// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import { Spinner } from "@/components/ui/spinner";
import { getLocale } from "@/i18n";
import { toast } from "@/lib/toast";
import { useAui } from "@assistant-ui/react";
import { YoutubeIcon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useCallback, useEffect, useRef, useState } from "react";
import {
  type YoutubeTranscript,
  fetchYoutubeTranscript,
} from "../api/youtube-api";

// The route caps the list at 8; the UI locale leads, then whatever the browser reports.
function preferredCaptionLanguages(): string[] {
  const tags = [getLocale(), ...(globalThis.navigator?.languages ?? []), "en"];
  return [
    ...new Set(tags.filter((tag) => typeof tag === "string" && tag)),
  ].slice(0, 8);
}

function formatDuration(seconds: number): string {
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  const rest = seconds % 60;
  const padded = `${minutes.toString().padStart(hours > 0 ? 2 : 1, "0")}:${rest
    .toString()
    .padStart(2, "0")}`;
  return hours > 0 ? `${hours}:${padded}` : padded;
}

function transcriptFileName(transcript: YoutubeTranscript): string {
  const cleaned = transcript.title
    // \p{C} covers the control characters a path separator check would miss.
    .replace(/[\p{C}\\/:*?"<>|]+/gu, " ")
    .replace(/\s+/g, " ")
    .trim()
    .slice(0, 80)
    .trim();
  return `${cleaned || `youtube-${transcript.videoId}`} transcript.txt`;
}

function transcriptFileText(transcript: YoutubeTranscript): string {
  const header = [
    transcript.title ? `Title: ${transcript.title}` : null,
    transcript.author ? `Channel: ${transcript.author}` : null,
    transcript.lengthSeconds > 0
      ? `Duration: ${formatDuration(transcript.lengthSeconds)}`
      : null,
    `URL: ${transcript.url}`,
    transcript.language ? `Captions: ${transcript.language}` : null,
    transcript.truncated
      ? "Note: this transcript was truncated and stops before the video ends."
      : null,
  ].filter((line): line is string => line !== null);
  return `${header.join("\n")}\n\n${transcript.text}\n`;
}

type Composer = ReturnType<ReturnType<typeof useAui>["composer"]>;

async function detachFile(composer: Composer, file: File): Promise<void> {
  const added = composer
    .getState()
    .attachments.find((attachment) => attachment.file === file);
  if (added) await composer.attachment({ id: added.id }).remove();
}

/** Offer to turn a pasted YouTube link into a transcript attachment. Sits above the composer
 *  surface, matching the prompt-queue stack, and closes once the user picks. */
export function YoutubeTranscriptPrompt({
  url,
  onClose,
}: {
  url: string;
  onClose: () => void;
}) {
  const aui = useAui();
  const [fetching, setFetching] = useState(false);
  const abortRef = useRef<AbortController | null>(null);

  // A link removed from the draft unmounts this mid-fetch; drop the in-flight request so the
  // attachment does not land in a composer that moved on.
  useEffect(() => () => abortRef.current?.abort(), []);

  const attach = useCallback(async () => {
    if (fetching) return;
    const controller = new AbortController();
    abortRef.current = controller;
    setFetching(true);
    const composer = aui.composer();
    try {
      const transcript = await fetchYoutubeTranscript(
        url,
        preferredCaptionLanguages(),
        controller.signal,
      );
      const file = new File(
        [transcriptFileText(transcript)],
        transcriptFileName(transcript),
        { type: "text/plain" },
      );
      if (controller.signal.aborted) return;
      await composer.addAttachment(file);
      // A send lands while addAttachment is still pending, and abort cannot recall it, so drop the file
      // rather than let it ride along on the next message.
      if (controller.signal.aborted) {
        await detachFile(composer, file);
        return;
      }
      onClose();
    } catch (error) {
      if (controller.signal.aborted) return;
      setFetching(false);
      toast.error("Could not attach the transcript", {
        description: error instanceof Error ? error.message : "Unknown error",
      });
    }
  }, [aui, fetching, onClose, url]);

  return (
    <div
      className="relative z-0 mx-7 mb-[-8px] rounded-t-[18px] rounded-b-none border border-border/45 bg-background/90 px-5 py-2 text-muted-foreground shadow-none backdrop-blur-md dark:bg-card/85"
      aria-label="Attach YouTube transcript"
    >
      <div className="grid h-10 grid-cols-[minmax(0,1fr)_auto] items-center gap-2.5">
        <div className="flex min-w-0 items-center gap-2.5">
          <HugeiconsIcon
            icon={YoutubeIcon}
            strokeWidth={2}
            className="size-4 shrink-0 text-muted-foreground/70"
          />
          <div className="truncate text-sm">
            {fetching
              ? "Fetching the transcript…"
              : "Attach this video's transcript?"}
          </div>
        </div>
        {fetching ? (
          <Spinner className="size-4 justify-self-end text-muted-foreground/70" />
        ) : (
          <div className="flex items-center gap-1.5">
            <Button
              type="button"
              variant="ghost"
              size="sm"
              className="h-7 px-2 text-xs text-muted-foreground"
              onClick={onClose}
            >
              Dismiss
            </Button>
            <Button
              type="button"
              size="sm"
              className="h-7 px-2 text-xs"
              onClick={() => void attach()}
            >
              Attach transcript
            </Button>
          </div>
        )}
      </div>
    </div>
  );
}
