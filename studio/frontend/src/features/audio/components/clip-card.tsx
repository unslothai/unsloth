// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Download01Icon } from "@hugeicons/core-free-icons";
import { PauseIcon, StopIcon } from "@/lib/media-control-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { type ReactNode, useCallback, useRef, useState } from "react";
import { Button } from "@/components/ui/button";
import { Spinner } from "@/components/ui/spinner";
import { cn } from "@/lib/utils";
import {
  DropdownMenuItem,
  DropdownMenuSub,
  DropdownMenuSubContent,
  DropdownMenuSubTrigger,
} from "@/components/ui/dropdown-menu";
import { formatRelativeShort } from "@/features/hub/lib/format";
import { audioModelLabel, formatClipDuration } from "../audio-workspace-utils";
import { AUDIO_WORKFLOWS, type AudioWorkflowId } from "../workflows";
import { Waveform, type WaveformControl } from "./waveform";
import { ArrowUpIcon, PlayIcon } from "lucide-react";

export type ClipSendHandlers = Partial<Record<AudioWorkflowId, () => void>>;

export function SendToItems({ handlers }: { handlers: ClipSendHandlers }) {
  return AUDIO_WORKFLOWS.filter((tab) => handlers[tab.id]).map((tab) => (
    <DropdownMenuItem key={tab.id} onClick={handlers[tab.id]}>
      <HugeiconsIcon icon={tab.icon} strokeWidth={1.75} className="size-icon" />
      {tab.label}
    </DropdownMenuItem>
  ));
}

export function ClipSendToMenu({ handlers }: { handlers: ClipSendHandlers }) {
  if (!Object.values(handlers).some(Boolean)) return null;
  return (
    <DropdownMenuSub>
      <DropdownMenuSubTrigger>
        <ArrowUpIcon
          strokeWidth={1.75}
          className="size-icon"
        />
        Send to
      </DropdownMenuSubTrigger>
      <DropdownMenuSubContent>
        <SendToItems handlers={handlers} />
      </DropdownMenuSubContent>
    </DropdownMenuSub>
  );
}

/** The expanded top row of the Clips card: it shares the rows' 40px icon column and their title edge. */
const PLAYER_CLASS =
  "grid w-full gap-2 rounded-xl bg-wash-6 px-3 py-3";

/** The 40px column every Clips row starts with, so play buttons and icons line up down the card. */
export const CLIP_ICON_COLUMN = "size-[calc(40px*var(--ui-space-scale,1))]";

/** never autoplays; a new clip focuses its play button once. */
export function ClipCard({
  title,
  model,
  createdAt,
  durationS,
  src,
  peaks,
  onDownload,
  menu,
  status,
  focusOnMount = false,
  onFocused,
  player,
}: {
  title: string;
  model: string;
  createdAt?: string;
  durationS: number | null;
  src: string | null;
  peaks: readonly number[] | null;
  onDownload: (() => void) | null;
  menu?: ReactNode;
  status?: string;
  focusOnMount?: boolean;
  onFocused?: () => void;
  player?: ReactNode;
}) {
  const control = useRef<WaveformControl | null>(null);
  const [playing, setPlaying] = useState(false);
  const onPosition = useCallback(
    (_seconds: number, isPlaying: boolean) => setPlaying(isPlaying),
    [],
  );
  const focusPlay = (element: HTMLDivElement | null) => {
    if (!(element && focusOnMount && src)) return;
    element.querySelector<HTMLButtonElement>("button")?.focus();
    onFocused?.();
  };
  return (
    <div ref={focusPlay} className={PLAYER_CLASS}>
      <div className="flex items-center gap-3">
        {player ? null : (
          <Button
            type="button"
            variant="dark"
            size="icon"
            className={cn("shrink-0 rounded-full", CLIP_ICON_COLUMN)}
            aria-label={playing ? `Pause ${title}` : `Play ${title}`}
            disabled={!src}
            onClick={() => control.current?.toggle()}
          >
            {playing ? (
              <HugeiconsIcon icon={PauseIcon} className="size-4" />
            ) : (
              <PlayIcon className="size-4" />
            )}
          </Button>
        )}
        <div className="grid min-w-0 flex-1 gap-0.5">
          <p className="line-clamp-2 text-ui-15 font-medium leading-snug text-foreground">
            {title}
          </p>
          <p className="truncate text-ui-12 text-muted-foreground">
            <span title={model}>{audioModelLabel(model)}</span>
            {createdAt ? ` · ${formatRelativeShort(createdAt)}` : ""}
            {status ? ` · ${status}` : ""}
          </p>
        </div>
        <Button
          variant="ghost"
          size="icon-sm"
          aria-label="Download audio clip"
          disabled={!onDownload}
          onClick={onDownload ?? undefined}
          className="shrink-0 rounded-full"
        >
          <HugeiconsIcon icon={Download01Icon} className="size-3.5" />
        </Button>
        {menu}
      </div>
      {player ?? (
        // Starts at the title's edge: 40px icon column + 12px gap.
        <div className="ps-[calc(52px*var(--ui-space-scale,1))]">
          <Waveform
            peaks={peaks}
            durationS={durationS}
            src={src}
            label={title || "audio clip"}
            size="lg"
            hidePlay={true}
            controlRef={control}
            onPositionChange={onPosition}
          />
        </div>
      )}
    </div>
  );
}

export function PendingClipCard({
  title,
  status,
  elapsedSeconds,
  canStop,
  onStop,
}: {
  title: string;
  status: string;
  elapsedSeconds: number | null;
  canStop: boolean;
  onStop: () => void;
}) {
  return (
    <div className={PLAYER_CLASS} aria-busy="true">
      <div className="flex items-center gap-3">
        <span
          className={cn(
            "grid shrink-0 place-items-center rounded-full bg-wash-6 text-muted-foreground",
            CLIP_ICON_COLUMN,
          )}
        >
          <Spinner className="size-4" />
        </span>
        <div className="grid min-w-0 flex-1 gap-0.5">
          <p className="line-clamp-2 text-ui-14 font-medium leading-snug text-muted-foreground">
            {title}
          </p>
          {/* No live region: the footer already announces progress. */}
          <p className="text-ui-12 text-muted-foreground">
            {status}
            {elapsedSeconds !== null ? (
              <span className="ml-1.5 font-mono tabular-nums">
                {formatClipDuration(elapsedSeconds)}
              </span>
            ) : null}
          </p>
        </div>
        {canStop ? (
          <Button
            variant="outline"
            size="sm"
            onClick={onStop}
            className="shrink-0 hover:bg-muted"
          >
            <HugeiconsIcon icon={StopIcon} className="size-3.5" />
            Stop
          </Button>
        ) : null}
      </div>
      <div className="ps-[calc(52px*var(--ui-space-scale,1))]">
        <Waveform
          peaks={null}
          durationS={null}
          src={null}
          label="clip in progress"
          size="lg"
          hidePlay={true}
        />
      </div>
    </div>
  );
}
