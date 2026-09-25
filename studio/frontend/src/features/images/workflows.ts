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
    id: "edit",
    label: "Edit",
    icon: Edit03Icon,
    requires: "edit",
    hint: "Change an image with an instruction",
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
];

/** Starter prompt per workflow, showing what each one is for. */
export const WORKFLOW_EXAMPLE_PROMPTS: Record<WorkflowId, string> = {
  create:
    "A cozy wooden cabin on a snowy mountain at dusk, warm light glowing from the windows, pine trees and gently falling snow. Cinematic photo, soft golden light.",
  edit: "Make the sky a bright sunset orange and add a red kite flying above the trees.",
  transform:
    "Turn this into a watercolor painting with soft pastel colors and loose, visible brush strokes.",
  inpaint:
    "A fluffy orange cat curled up asleep in the painted area, soft window light, realistic fur.",
  extend:
    "Continue the scene outward with more sandy beach, gentle waves and palm trees under a clear blue sky.",
  upscale: "Sharp, highly detailed photo with crisp textures, clean edges and natural colors.",
  reference:
    "The character from the reference exploring a neon lit city street at night, cinematic lighting.",
};

