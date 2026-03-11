// SPDX-License-Identifier: AGPL-3.0-only - See /studio/LICENSE.AGPL-3.0
// Copyright © 2025 Unsloth AI

import type { ReactNode } from "react";

export type TourStep = {
  id: string;
  target: string; // data-tour="<target>"
  title: string;
  body: ReactNode;
  onEnter?: () => void | Promise<void>;
  onExit?: () => void | Promise<void>;
};

export type Rect = { x: number; y: number; w: number; h: number };

export type Placement = "right" | "left" | "top" | "bottom";
