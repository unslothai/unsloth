// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

const PREFIX = "chat-query-draft:";

export type QueryDraft = {
  id: string;
  nonce: string;
  text: string;
  pastes: string[];
  temporary: boolean;
  save: () => void;
};

function makeDraft(
  nonce: string,
  text: string,
  pastes: string[],
  temporary: boolean,
): QueryDraft {
  let savedText: string | undefined;
  let savedPastes: string[] | undefined;
  const draft: QueryDraft = {
    id: PREFIX + nonce,
    nonce,
    text,
    pastes,
    temporary,
    save() {
      if (draft.temporary) return;
      try {
        const storage = window.sessionStorage;
        if (draft.text !== savedText) {
          storage.setItem(draft.id, draft.text);
          savedText = draft.text;
        }
        // Typing beside a large paste must not serialize that file on every keystroke.
        if (draft.pastes !== savedPastes) {
          if (draft.pastes.length)
            storage.setItem(`${draft.id}:pastes`, JSON.stringify(draft.pastes));
          else storage.removeItem(`${draft.id}:pastes`);
          savedPastes = draft.pastes;
        }
      } catch {
        // The in-memory draft still works when storage is unavailable or full.
      }
    },
  };
  return draft;
}

export function createQueryDraft(text: string, temporary: boolean): QueryDraft {
  const draft = makeDraft(`query-${crypto.randomUUID()}`, text, [], temporary);
  draft.save();
  return draft;
}

export function loadQueryDraft(nonce: string): QueryDraft | null {
  try {
    const text = window.sessionStorage.getItem(PREFIX + nonce);
    if (text === null) return null;
    const pastes: unknown = JSON.parse(
      window.sessionStorage.getItem(`${PREFIX}${nonce}:pastes`) ?? "[]",
    );
    return makeDraft(
      nonce,
      text,
      Array.isArray(pastes)
        ? pastes.filter(
            (value: unknown): value is string => typeof value === "string",
          )
        : [],
      false,
    );
  } catch {
    return null;
  }
}
