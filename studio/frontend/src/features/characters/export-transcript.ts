// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { hiddenRoleplayTurnKind, roleplayMetaOf } from "./meta.ts";
import { stripSpeakerPrefix } from "./prompt/build.ts";

type ExportRecord = { role?: string; metadata?: unknown; content?: unknown };

/** A character chat without its turn markers, each reply opening with one "Name: " (exports label
 *  every reply Assistant, so the name is what says who spoke); null for an ordinary chat. */
export function characterChatTranscript<R extends ExportRecord>(records: R[]): R[] | null {
  if (!records.some((record) => roleplayMetaOf(record) !== null)) return null;
  return records
    .filter((record) => hiddenRoleplayTurnKind(record) === null)
    .map((record) => {
      const meta = roleplayMetaOf(record);
      if (meta?.kind !== "reply" || !meta.speakerName || !Array.isArray(record.content)) return record;
      let named = false;
      const content = record.content.map((part: unknown) => {
        const text = part as { type?: unknown; text?: unknown } | null;
        if (!(text && text.type === "text" && typeof text.text === "string")) return part;
        const body = stripSpeakerPrefix(text.text, meta.speakerName);
        if (named || !body.trim()) return { ...text, text: body };
        named = true;
        return { ...text, text: `${meta.speakerName}: ${body}` };
      });
      return { ...record, content };
    });
}
