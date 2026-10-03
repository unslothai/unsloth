// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { AudioOptionSpec } from "../audio-options";
import type { AudioWorkflowId } from "../workflows";
import { audioToolPanelsFor } from "./registry";
import { panelValue } from "./select";
import type { AudioModelContext, CoreInputs } from "./types";

/** Renders the loaded model's tools for this page, where the rail always showed them. Each panel
 *  gets its own value by id, or its defaults when nothing was kept for it yet. */
export function AudioToolPanels({
  workflow,
  ctx,
  values,
  onChange,
  specs,
  disabled,
  core,
}: {
  workflow: AudioWorkflowId;
  ctx: AudioModelContext;
  values: Readonly<Record<string, unknown>>;
  onChange: (panelId: string, value: unknown) => void;
  specs: AudioOptionSpec[];
  disabled: boolean;
  /** The page's own inputs, for panels that preview their effect on them. */
  core?: CoreInputs;
}) {
  return (
    <>
      {audioToolPanelsFor(workflow, ctx).map((panel) => (
        <panel.Component
          key={panel.id}
          value={panelValue(panel, values, specs)}
          onChange={(value: unknown) => onChange(panel.id, value)}
          specs={specs}
          disabled={disabled}
          ctx={ctx}
          core={core}
        />
      ))}
    </>
  );
}
