// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

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

// history is scoped per page, so these actions appear only on Music.
const SEND_LABEL: Record<SendAction, string> = {
  edit: "Edit",
  extend: "Extend",
};

const SEND_ICON = {
  edit: PencilEdit02Icon,
  extend: TimeQuarterPassIcon,
} as const;

export function sendClipToMusic(
  clip: AudioGalleryClip,
  action: SendAction,
  name = clip.prompt || "Generated clip",
): void {
  useAudioMusicStore.getState().pushClipToEdit(
    {
      kind: "clip",
      id: clip.id,
      name,
      durationS: clip.duration_s,
      transcript: null,
      language: null,
    },
    action === "extend" ? "extend" : null,
  );
  const workspace = useAudioWorkspaceStore.getState();
  if (workspace.workflow !== "music") workspace.requestWorkflow("music");
}

export function canSendToMusic(clip: AudioGalleryClip): boolean {
  return clipWorkflow(clip) === "music";
}

function useSendActions(clip: AudioGalleryClip): readonly SendAction[] {
  const editActions = useAudioMusicStore((state) => state.loadedEditActions);
  return canSendToMusic(clip) ? sendActionsFor(editActions) : [];
}

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
