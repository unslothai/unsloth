// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { deliveryInstructions } from "../edit-adapters";
import { countChanges } from "../edit-diff";
import { EDIT_COPY } from "../edit-policy";
import { EDIT_PANEL_LOGIC } from "./edit-panel-logic";
import type { AudioToolPanel } from "./types";

export const EDIT_TOOL_PANELS: readonly AudioToolPanel<null>[] =
  EDIT_PANEL_LOGIC.map((logic) => ({
    ...logic,
    Component: ({ core }) => {
      const edit = core?.edit;
      const changes = !edit
        ? 0
        : edit.mode === "delivery" && logic.adapter.delivery
          ? deliveryInstructions(edit.speed, edit.pitchSteps).length
          : (countChanges(edit.transcript, edit.edited) ?? 0);
      return (
        <section
          className="grid gap-0.5"
          aria-label={EDIT_COPY.howItEditsTitle}
        >
          <h3 className="text-ui-13 font-medium text-foreground">
            {EDIT_COPY.howItEditsTitle}
          </h3>
          <p className="text-ui-11p5 leading-snug text-muted-foreground">
            {logic.adapter.howItEdits(changes)}
          </p>
        </section>
      );
    },
  }));
