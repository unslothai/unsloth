// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useT } from "@/i18n";
import { toast } from "@/lib/toast";
import { useBlocker, useNavigate } from "@tanstack/react-router";
import { useEffect, useMemo, useRef, useState } from "react";
import { startCharacterChat } from "../actions";
import {
  createCharacter,
  errorText,
  getCharacter,
  removeCharacterAvatar,
  updateCharacter,
  uploadCharacterAvatar,
} from "../api";
import { blankDraft, draftOf, toSavePayload } from "../draft";
import { useCharactersStore } from "../store";
import type { Character, CharacterDraft } from "../types";

export type PictureChoice = File | "remove" | null;

/** Loads a character (or a blank one), tracks unsaved edits, and saves the fields and portrait together. */
export function useCharacterDraft(characterId: string | null) {
  const t = useT();
  const navigate = useNavigate();
  const [original, setOriginal] = useState<Character | null>(null);
  const [draft, setDraft] = useState<CharacterDraft | null>(characterId ? null : blankDraft());
  const [baseline, setBaseline] = useState<string>(() => JSON.stringify(blankDraft()));
  const [picture, setPicture] = useState<PictureChoice>(null);
  const [saving, setSaving] = useState(false);
  const [showErrors, setShowErrors] = useState(false);
  const leaving = useRef(false);

  useEffect(() => {
    if (!characterId) return;
    let live = true;
    getCharacter(characterId)
      .then((character) => {
        if (!live) return;
        setOriginal(character);
        const next = draftOf(character);
        setDraft(next);
        setBaseline(JSON.stringify(next));
      })
      .catch((error) => {
        if (!live) return;
        toast.error(errorText(error));
        void navigate({ to: "/characters" });
      });
    return () => {
      live = false;
    };
  }, [characterId, navigate]);

  const previewUrl = useMemo(() => (picture instanceof File ? URL.createObjectURL(picture) : null), [picture]);
  useEffect(() => () => {
    if (previewUrl) URL.revokeObjectURL(previewUrl);
  }, [previewUrl]);

  const dirty = draft !== null && (JSON.stringify(draft) !== baseline || picture !== null);
  const blocker = useBlocker({
    shouldBlockFn: () => dirty && !leaving.current,
    enableBeforeUnload: () => dirty,
    withResolver: true,
  });

  function set<K extends keyof CharacterDraft>(key: K, value: CharacterDraft[K]) {
    setDraft((current) => (current ? { ...current, [key]: value } : current));
  }

  async function save(thenChat: boolean) {
    if (!draft) return;
    if (!draft.name.trim()) {
      setShowErrors(true);
      return;
    }
    setSaving(true);
    let saved: Character;
    try {
      const { fields, clear } = toSavePayload(draft);
      saved = original ? await updateCharacter(original.id, { ...fields, clear }) : await createCharacter(fields);
      // Kept before the portrait upload, so a retry after a failed upload updates this character.
      setOriginal(saved);
      useCharactersStore.getState().upsert(saved);
      if (picture instanceof File) saved = await uploadCharacterAvatar(saved.id, picture);
      else if (picture === "remove" && saved.hasAvatar) saved = await removeCharacterAvatar(saved.id);
      useCharactersStore.getState().upsert(saved);
      setOriginal(saved);
      setPicture(null);
      setBaseline(JSON.stringify(draft));
      toast.success(t("characters.editor.saved", { name: saved.name }));
    } catch (error) {
      toast.error(t("characters.editor.saveFailed"), { description: errorText(error) });
      setSaving(false);
      return;
    }
    leaving.current = true;
    const profile = () => void navigate({ to: "/characters/$characterId", params: { characterId: saved.id } });
    if (!thenChat) {
      setSaving(false);
      profile();
      return;
    }
    try {
      const threadId = await startCharacterChat([saved.id]);
      void navigate({ to: "/chat", search: { thread: threadId } });
    } catch (error) {
      // The character is saved; only the chat failed to open.
      toast.error(t("characters.chat.startFailed"), { description: errorText(error) });
      profile();
    } finally {
      setSaving(false);
    }
  }

  return { original, draft, set, picture, setPicture, previewUrl, saving, showErrors, blocker, save };
}
