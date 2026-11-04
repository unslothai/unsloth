// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { InputRule } from "../../../vendor/prosemirror-inputrules/index.js";
import type {
  Mark,
  MarkType,
} from "../../../vendor/prosemirror-model/index.js";
import { sliceToMarkdown } from "./markdown.ts";

const escapeMarker = (text: string) => text.replace(/[*~`]/g, "\\$&");

// No edge spaces and no line break (U+FFFC), so the result serialises back to what was typed.
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
      // The closing marker's last character is still being typed, not yet in the document.
      const tr = state.tr
        .delete(from + marker.length + text.length, end)
        .delete(from, from + marker.length);
      const to = from + text.length;
      if (type.spec.code) {
        // Marks applied inside the span were typed as literal markers.
        const runs: (readonly Mark[])[] = [];
        tr.doc.nodesBetween(from, to, (node) => {
          if (node.isText) runs.push(node.marks);
        });
        const keep = runs[0].filter((m) => runs.every((r) => m.isInSet(r)));
        if (runs.some((r) => r.length > keep.length)) {
          for (const mark of keep) tr.removeMark(from, to, mark);
          const literal = sliceToMarkdown(tr.doc.slice(from, to));
          const marks = type.create().addToSet(keep as Mark[]);
          tr.replaceWith(from, to, type.schema.text(literal, marks));
          return tr.removeStoredMark(type);
        }
      }
      return tr.addMark(from, to, type.create()).removeStoredMark(type);
    },
    { inCodeMark: false },
  );
}
