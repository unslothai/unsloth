// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Where a clip can go next. Free of app imports so the node test runner can load it directly.
// Each page adds its own target here; a target whose page does not exist yet is never offered.

import type { AudioSourceSelection } from "./audio-run-request";
import type { SendTarget } from "./components/stem-mixer-types";
import { isAudioWorkflowId } from "./workflows";

export const SEND_TARGETS: readonly SendTarget[] = [
  { id: "transcribe", workflow: "transcribe", label: "Transcribe" },
  { id: "clone", workflow: "clone", label: "Clone (as reference)" },
];

/** The targets whose page exists in this build. */
export function availableSendTargets(
  targets: readonly SendTarget[] = SEND_TARGETS,
): SendTarget[] {
  return targets.filter((target) => isAudioWorkflowId(target.workflow));
}

/** A history clip as a page's audio input. */
export function clipAsSource({
  clipId,
  name,
  durationS,
}: {
  clipId: string;
  name: string;
  durationS: number | null;
}): AudioSourceSelection {
  return { kind: "clip", id: clipId, name, durationS };
}
