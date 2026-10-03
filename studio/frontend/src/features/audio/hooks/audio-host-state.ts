// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { Dispatch, RefObject, SetStateAction } from "react";
import type { InferenceStatusResponse } from "@/features/chat";
import type { AudioBusy, AudioGenerationPhase } from "../audio-page-policy";
import type { CreateMode } from "../audio-workspace-utils";

/** The page-wide state the Audio page owns and hands to its hooks. */
export interface AudioHostState {
  active: boolean;
  activeRef: RefObject<boolean>;
  busy: AudioBusy;
  busyRef: RefObject<AudioBusy>;
  setBusy: Dispatch<SetStateAction<AudioBusy>>;
  mode: CreateMode;
  setMode: Dispatch<SetStateAction<CreateMode>>;
  modeRef: RefObject<CreateMode>;
  generationPhaseRef: RefObject<AudioGenerationPhase>;
  updateGenerationPhase: (nextPhase: AudioGenerationPhase) => void;
  generateAbort: RefObject<AbortController | null>;
  handleStopGeneration: () => void;
  status: InferenceStatusResponse | null;
  refreshStatus: () => Promise<void>;
  audioDevice: string;
  isMac: boolean;
  setAdvancedOpen: (next: boolean) => void;
}
