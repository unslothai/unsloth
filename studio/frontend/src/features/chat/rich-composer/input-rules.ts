// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { InputRule } from "../../../vendor/prosemirror-inputrules/index.js";
import type { MarkType } from "../../../vendor/prosemirror-model/index.js";

const escapeMarker = (text: string) => text.replace(/[*~`]/g, "\\$&");

// `**text**` and the like, typed after a space or at a line start: the markers go and the text
// takes the mark. The text may not start or end with a space or cross a line break (U+FFFC in
// the matched text), so it serialises back to exactly what was typed.
export function markerRule(marker: string, type: MarkType): InputRule {
  const delimiter = escapeMarker(marker);
  const char = escapeMarker(marker[0]);
  const edge = `[^${char}\\s\\ufffc]`;
  const pattern = new RegExp(
    `(^|[\\s\\ufffc])${delimiter}(${edge}(?:[^${char}\\ufffc]*${edge})?)${delimiter}$`,
  );
  return new InputRule(
    pattern,
    (state, [, lead = "", text = ""], start, end) => {
      const from = start + lead.length;
      // The closing marker's last character is the one being typed: it is not in the document.
      return state.tr
        .delete(from + marker.length + text.length, end)
        .delete(from, from + marker.length)
        .addMark(from, from + text.length, type.create())
        .removeStoredMark(type);
    },
    { inCodeMark: false },
  );
}
