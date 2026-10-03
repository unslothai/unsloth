// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import { DropdownMenuItem } from "@/components/ui/dropdown-menu";
import {
  PencilEdit02Icon,
  TimeQuarterPassIcon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import type { AudioGalleryClip } from "../api";
import { type SendAction, sendActionsFor } from "../music/music-edit-rules";
import { useAudioMusicStore } from "../stores/audio-music-store";
import { useAudioWorkspaceStore } from "../stores/audio-workspace-store";
import { clipWorkflow } from "../workflows";

// History is per page, so these only ever show on Music itself.
const SEND_LABEL: Record<SendAction, string> = {
  edit: "Edit",
  extend: "Extend",
};

const SEND_ICON = {
  edit: PencilEdit02Icon,
  extend: TimeQuarterPassIcon,
} as const;

/** Opens Music's Edit mode on a history clip ("Extend in Music" also picks Extend), then moves
 *  the page to Music through the same gate as its tabs. */
export function sendClipToMusic(
  clip: AudioGalleryClip,
  action: SendAction,
): void {
  useAudioMusicStore.getState().pushClipToEdit(
    {
      kind: "clip",
      id: clip.id,
      name: clip.prompt || "Generated clip",
      durationS: clip.duration_s,
      transcript: null,
      language: null,
    },
    action === "extend" ? "extend" : null,
  );
  const workspace = useAudioWorkspaceStore.getState();
  if (workspace.workflow !== "music") workspace.requestWorkflow("music");
}

/** Only music clips can be edited as music. */
export function canSendToMusic(clip: AudioGalleryClip): boolean {
  return clipWorkflow(clip) === "music";
}

function useSendActions(clip: AudioGalleryClip): readonly SendAction[] {
  const editActions = useAudioMusicStore((state) => state.loadedEditActions);
  return canSendToMusic(clip) ? sendActionsFor(editActions) : [];
}

/** "Edit" and "Extend" for a history row's menu, when the loaded model can. */
export function MusicSendToMenuItems({ clip }: { clip: AudioGalleryClip }) {
  const actions = useSendActions(clip);
  if (actions.length === 0) return null;
  return (
    <>
      {actions.map((action) => (
        <DropdownMenuItem
          key={action}
          onClick={() => sendClipToMusic(clip, action)}
        >
          <HugeiconsIcon
            icon={SEND_ICON[action]}
            strokeWidth={1.75}
            className="size-icon"
          />
          {SEND_LABEL[action]}
        </DropdownMenuItem>
      ))}
    </>
  );
}

/** The same two actions on the selected clip's toolbar. */
export function MusicSendToButtons({ clip }: { clip: AudioGalleryClip }) {
  const actions = useSendActions(clip);
  if (actions.length === 0) return null;
  return (
    <>
      {actions.map((action) => (
        <Button
          key={action}
          variant="ghost"
          size="sm"
          className="h-auto shrink-0 px-2 py-1 text-ui-11p5"
          onClick={() => sendClipToMusic(clip, action)}
        >
          <HugeiconsIcon icon={SEND_ICON[action]} className="size-3.5" />
          {SEND_LABEL[action]}
        </Button>
      ))}
    </>
  );
}
