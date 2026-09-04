// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  ArrowExpand01Icon,
  Edit03Icon,
  ImageUpload01Icon,
  MagicWand01Icon,
  PaintBrush02Icon,
  SparklesIcon,
  ZoomInAreaIcon,
} from "@hugeicons/core-free-icons";
import type { IconSvgElement } from "@hugeicons/react";

export type WorkflowId =
  | "create"
  | "transform"
  | "inpaint"
  | "extend"
  | "upscale"
  | "reference"
  | "edit";

/** The Images workflows, shared by the page and the sidebar submenu. `requires` is the backend
 *  workflow id (status.workflows) the loaded model must support; null = always available. */
export const WORKFLOW_TABS: Array<{
  id: WorkflowId;
  label: string;
  /** Page heading, when the sidebar's short label would read oddly on its own. Falls back to `label`. */
  heading?: string;
  requires: string | null;
  icon: IconSvgElement;
  hint: string;
}> = [
  {
    id: "create",
    label: "Create",
    // The sidebar nests this under Images, so "Create" alone is clear there.
    heading: "Create images",
    requires: null,
    // Not the pencil: that is the sidebar's New chat icon.
    icon: SparklesIcon,
    hint: "Generate a new image from a prompt",
  },
  {
    id: "transform",
    label: "Transform",
    icon: MagicWand01Icon,
    requires: "img2img",
    hint: "Redraw an image from your prompt",
  },
  {
    id: "inpaint",
    label: "Inpaint",
    icon: PaintBrush02Icon,
    requires: "inpaint",
    hint: "Regenerate a painted region",
  },
  {
    id: "extend",
    label: "Extend",
    icon: ArrowExpand01Icon,
    requires: "outpaint",
    hint: "Grow the canvas and fill the edges",
  },
  {
    id: "upscale",
    label: "Upscale",
    icon: ZoomInAreaIcon,
    requires: "upscale",
    hint: "Enlarge and re-detail an image",
  },
  {
    id: "reference",
    label: "Reference",
    icon: ImageUpload01Icon,
    requires: "reference",
    hint: "Generate guided by a reference image",
  },
  {
    id: "edit",
    label: "Edit",
    icon: Edit03Icon,
    requires: "edit",
    hint: "Change an image with an instruction",
  },
];
