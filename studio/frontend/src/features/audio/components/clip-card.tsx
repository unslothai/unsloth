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
        <HugeiconsIcon
          icon={SentIcon}
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

const CARD_CLASS = "w-full max-w-xl gap-3 px-5 py-4";

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
      {player ?? (
        <div ref={focusPlay}>
          <Waveform
            peaks={peaks}
            durationS={durationS}
            src={src}
            label={title || "audio clip"}
          />
        </div>
      )}
    </Card>
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
    <Card size="sm" className={CARD_CLASS} aria-busy="true">
      <div className="flex items-start gap-2">
        <div className="grid min-w-0 flex-1 gap-1">
          <p className="line-clamp-2 text-ui-13 text-muted-foreground">
            {title}
          </p>
          {/* No live region: the footer already announces progress. */}
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
