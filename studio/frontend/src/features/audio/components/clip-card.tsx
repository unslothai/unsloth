// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Download01Icon, SentIcon, StopIcon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import type { ReactNode } from "react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import {
  DropdownMenuItem,
  DropdownMenuSub,
  DropdownMenuSubContent,
  DropdownMenuSubTrigger,
} from "@/components/ui/dropdown-menu";
import { formatRelativeShort } from "@/features/hub/lib/format";
import { audioModelLabel, formatClipDuration } from "../audio-workspace-utils";
import { AUDIO_WORKFLOWS, type AudioWorkflowId } from "../workflows";
import { Waveform } from "./waveform";

/** Pages a clip can be sent to, keyed by workflow; only pages with a handler are offered. */
export type ClipSendHandlers = Partial<Record<AudioWorkflowId, () => void>>;

/**
 * "Send to" for a clip's menu, listing the Audio pages from the shared workflow list that can take it.
 * Pages added later appear here once they register a handler.
 */
export function ClipSendToMenu({
  current,
  handlers,
}: {
  current: AudioWorkflowId;
  handlers: ClipSendHandlers;
}) {
  const targets = AUDIO_WORKFLOWS.filter(
    (tab) => tab.id !== current && handlers[tab.id],
  );
  if (targets.length === 0) return null;
  return (
    <DropdownMenuSub>
      <DropdownMenuSubTrigger>
        <HugeiconsIcon
          icon={SentIcon}
          strokeWidth={1.75}
          className="size-icon"
        />
        Send to
      </DropdownMenuSubTrigger>
      <DropdownMenuSubContent>
        {targets.map((tab) => (
          <DropdownMenuItem key={tab.id} onClick={handlers[tab.id]}>
            <HugeiconsIcon
              icon={tab.icon}
              strokeWidth={1.75}
              className="size-icon"
            />
            {tab.label}
          </DropdownMenuItem>
        ))}
      </DropdownMenuSubContent>
    </DropdownMenuSub>
  );
}

const CARD_CLASS = "w-full max-w-xl gap-3 px-5 py-4";

/**
 * The selected clip: its text, model and age, then a waveform player with download and the clip menu.
 * It never autoplays. A clip a run just made takes focus on its play button once.
 */
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
}: {
  title: string;
  model: string;
  createdAt?: string;
  durationS: number | null;
  /** Object URL of the clip's bytes; until it exists the bars are drawn but cannot play. */
  src: string | null;
  peaks: readonly number[] | null;
  onDownload: (() => void) | null;
  /** The clip's ⋯ menu. */
  menu?: ReactNode;
  /** Extra state after the model line, such as "not saved to the gallery". */
  status?: string;
  focusOnMount?: boolean;
  onFocused?: () => void;
}) {
  const focusPlay = (element: HTMLDivElement | null) => {
    if (!(element && focusOnMount && src)) return;
    element.querySelector<HTMLButtonElement>("button")?.focus();
    onFocused?.();
  };
  return (
    <Card size="sm" className={CARD_CLASS}>
      <div className="flex items-start gap-2">
        <div className="grid min-w-0 flex-1 gap-1">
          <p className="line-clamp-2 text-ui-13 text-foreground">{title}</p>
          <p className="truncate text-ui-11p5 text-muted-foreground">
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
      <div ref={focusPlay}>
        <Waveform
          peaks={peaks}
          durationS={durationS}
          src={src}
          label={title || "audio clip"}
        />
      </div>
    </Card>
  );
}

/** Stands where the result will appear while a run works: its text, the phase, elapsed time and Stop. */
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
    <Card size="sm" className={CARD_CLASS} aria-busy="true">
      <div className="flex items-start gap-2">
        <div className="grid min-w-0 flex-1 gap-1">
          <p className="line-clamp-2 text-ui-13 text-muted-foreground">
            {title}
          </p>
          {/* The footer announces progress; this mirrors it without a second live region. */}
          <p className="text-ui-11p5 text-muted-foreground">
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
      <Waveform
        peaks={null}
        durationS={null}
        src={null}
        label="clip in progress"
      />
    </Card>
  );
}
