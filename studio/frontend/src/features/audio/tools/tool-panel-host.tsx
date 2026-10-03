// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { AudioOptionSpec } from "../audio-options";
import type { AudioWorkflowId } from "../workflows";
import { type InstructionsValue, audioToolPanelsFor } from "./registry";
import type { AudioModelContext } from "./types";

/** Renders the loaded model's tools for this page, where the rail always showed them. */
export function AudioToolPanels({
  workflow,
  ctx,
  value,
  onChange,
  specs,
  disabled,
}: {
  workflow: AudioWorkflowId;
  ctx: AudioModelContext;
  value: InstructionsValue;
  onChange: (value: InstructionsValue) => void;
  specs: AudioOptionSpec[];
  disabled: boolean;
}) {
  return (
    <>
      {audioToolPanelsFor(workflow, ctx).map(({ id, Component }) => (
        <Component
          key={id}
          value={value}
          onChange={onChange}
          specs={specs}
          disabled={disabled}
          ctx={ctx}
        />
      ))}
    </>
  );
}
