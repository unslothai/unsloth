// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { ChipInput } from "@/components/chip-input";
import { ConfirmDeleteDialog } from "@/components/confirm-delete-dialog";
import { useAppShellReadySignal } from "@/components/app-readiness";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Skeleton } from "@/components/ui/skeleton";
import { Slider } from "@/components/ui/slider";
import { Textarea } from "@/components/ui/textarea";
import { getInferenceStatus } from "@/features/chat/api/chat-api";
import { useChatRuntimeStore } from "@/features/chat/stores/chat-runtime-store";
import { loadedContextFields } from "@/features/model-picker/model-config/per-model-config";
import { useLocale, useT } from "@/i18n";
import { formatNumber } from "@/lib/intl-cache";
import { ArrowLeft01Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useNavigate } from "@tanstack/react-router";
import { useEffect, useState } from "react";
import { ClampedNumberInput } from "../components/number-input";
import { Card, Field, Section } from "../components/editor/editor-section";
import { GreetingsEditor } from "../components/editor/greetings-editor";
import { PortraitPicker } from "../components/editor/portrait-picker";
import { LorebookEditor } from "../components/lorebook-editor";
import { addTags, MAX_TAG_CHARS, permanentPromptText } from "../draft";
import { useCharacterDraft } from "../hooks/use-character-draft";
import { estimateTokenCount } from "@/lib/estimate-tokens";

/** The loaded model's context length; read from the server when Chat has not loaded it into the store yet. */
function useContextLength(): number | null {
  const storeContextLength = useChatRuntimeStore((s) => s.loadedContextLength);
  const [statusContextLength, setStatusContextLength] = useState<number | null>(null);
  useEffect(() => {
    if (storeContextLength != null) return;
    const controller = new AbortController();
    getInferenceStatus(controller.signal)
      .then((status) => setStatusContextLength(loadedContextFields(status).loadedContextLength))
      .catch(() => undefined);
    return () => controller.abort();
  }, [storeContextLength]);
  return storeContextLength ?? statusContextLength;
}

export function CharacterEditorPage({ characterId }: { characterId: string | null }) {
  const t = useT();
  const locale = useLocale();
  const navigate = useNavigate();
  const { original, draft, set, picture, setPicture, previewUrl, saving, showErrors, blocker, save } =
    useCharacterDraft(characterId);
  const contextLength = useContextLength();
  useAppShellReadySignal();

  if (!draft) {
    return (
      <main className="mx-auto w-full max-w-6xl px-6 pt-8 font-heading sm:px-10">
        <Skeleton className="h-9 w-56 rounded-xl" />
        <div className="mt-8 grid gap-8 md:grid-cols-[260px_1fr]">
          <Skeleton className="aspect-[2/3] w-full rounded-[24px]" />
          <Skeleton className="h-96 w-full rounded-[24px]" />
        </div>
      </main>
    );
  }

  const permanent = estimateTokenCount(permanentPromptText(draft)) ?? 0;
  const nameMissing = !draft.name.trim();
  const title = original ? t("characters.editor.editTitle", { name: original.name }) : t("characters.editor.newTitle");
  return (
    <main className="mx-auto w-full max-w-6xl 4xl:max-w-7xl px-6 pb-6 pt-6 font-heading max-sm:px-4 sm:px-10">
      <div className="flex flex-wrap items-center gap-3">
        <Button
          variant="ghost"
          size="icon-sm"
          className="rounded-full"
          aria-label={t("characters.editor.back")}
          onClick={() => window.history.length > 1 ? window.history.back() : void navigate({ to: "/characters" })}
        >
          <HugeiconsIcon icon={ArrowLeft01Icon} strokeWidth={1.75} className="size-icon" />
        </Button>
        <h1 className="min-w-0 flex-1 truncate text-ui-28 font-semibold tracking-[-0.025em] text-foreground">{title}</h1>
      </div>
      {original?.cardSpec ? (
        <p className="mt-2 pl-11 text-ui-13 text-muted-foreground">
          {t("characters.editor.importedNote", { spec: original.cardSpec.replace("chara_card_", "").toUpperCase() })}
        </p>
      ) : null}

      <div className="mt-7 grid gap-8 md:grid-cols-[260px_minmax(0,1fr)] lg:gap-12">
        <aside className="flex flex-col gap-5 md:sticky md:top-6 md:self-start">
          <PortraitPicker original={original} name={draft.name} picture={picture} previewUrl={previewUrl} onChange={setPicture} />
          <Field label={t("characters.editor.name")} htmlFor="character-name" error={showErrors && nameMissing ? t("characters.editor.nameRequired") : undefined}>
            <Input
              id="character-name"
              value={draft.name}
              maxLength={200}
              autoFocus={!characterId}
              placeholder={t("characters.editor.namePlaceholder")}
              aria-invalid={showErrors && nameMissing}
              onChange={(event) => set("name", event.target.value)}
            />
          </Field>
          <Field label={t("characters.editor.tags")}>
            <ChipInput
              values={draft.tags}
              onAdd={(value) => set("tags", addTags(draft.tags, value))}
              onRemove={(index) => set("tags", draft.tags.filter((_, i) => i !== index))}
              placeholder={t("characters.editor.tagsPlaceholder")}
              maxLength={MAX_TAG_CHARS}
              ariaLabel={t("characters.editor.tags")}
              removeLabel={(tag) => t("characters.editor.removeTag", { tag })}
            />
          </Field>
        </aside>

        <div className="flex min-w-0 flex-col gap-6">
          <Card>
            <Field label={t("characters.editor.description")} htmlFor="character-description" hint={t("characters.editor.descriptionHint")}>
              <Textarea
                id="character-description"
                value={draft.description}
                placeholder={t("characters.editor.descriptionPlaceholder")}
                className="max-h-[55vh] min-h-40 font-sans text-ui-15 leading-relaxed"
                onChange={(event) => set("description", event.target.value)}
              />
            </Field>
            <p className="text-ui-12 text-muted-foreground">{t("characters.editor.macrosHint")}</p>
          </Card>

          <GreetingsEditor
            firstMes={draft.firstMes}
            alternates={draft.alternateGreetings}
            onFirstMes={(text) => set("firstMes", text)}
            onAlternates={(alternates) => set("alternateGreetings", alternates)}
          />

          <h2 className="mt-2 px-1 text-ui-13 font-semibold uppercase tracking-[0.06em] text-muted-foreground">
            {t("characters.editor.advanced")}
          </h2>

          <Section title={t("characters.editor.personalitySection")} defaultOpen={Boolean(draft.personality || draft.scenario)}>
            <Field label={t("characters.editor.personality")} htmlFor="character-personality" hint={t("characters.editor.personalityHint")}>
              <Textarea id="character-personality" value={draft.personality} className="max-h-72 min-h-20 font-sans" onChange={(event) => set("personality", event.target.value)} />
            </Field>
            <Field label={t("characters.editor.scenario")} htmlFor="character-scenario" hint={t("characters.editor.scenarioHint")}>
              <Textarea id="character-scenario" value={draft.scenario} className="max-h-72 min-h-20 font-sans" onChange={(event) => set("scenario", event.target.value)} />
            </Field>
          </Section>

          <Section title={t("characters.editor.examplesSection")} defaultOpen={Boolean(draft.mesExample.trim())}>
            <Field label={t("characters.editor.examplesSection")} htmlFor="character-examples" hint={t("characters.editor.examplesHint")}>
              <Textarea
                id="character-examples"
                value={draft.mesExample}
                placeholder={"<START>\n{{user}}: ...\n{{char}}: ..."}
                className="max-h-[55vh] min-h-28 font-mono text-ui-13"
                onChange={(event) => set("mesExample", event.target.value)}
              />
            </Field>
          </Section>

          <Section
            title={t("characters.editor.promptSection")}
            defaultOpen={Boolean(draft.systemPrompt || draft.postHistoryInstructions || draft.depthPrompt?.prompt)}
          >
            <Field label={t("characters.editor.systemPrompt")} htmlFor="character-system-prompt" hint={t("characters.editor.systemPromptHint")}>
              <Textarea id="character-system-prompt" value={draft.systemPrompt} className="max-h-72 min-h-20 font-sans" onChange={(event) => set("systemPrompt", event.target.value)} />
            </Field>
            <Field label={t("characters.editor.postHistory")} htmlFor="character-post-history" hint={t("characters.editor.postHistoryHint")}>
              <Textarea
                id="character-post-history"
                value={draft.postHistoryInstructions}
                className="max-h-72 min-h-16 font-sans"
                onChange={(event) => set("postHistoryInstructions", event.target.value)}
              />
            </Field>
            <Field label={t("characters.editor.characterNote")} htmlFor="character-note" hint={t("characters.editor.characterNoteHint")}>
              <div className="flex flex-col gap-2 sm:flex-row sm:items-start">
                <Textarea
                  id="character-note"
                  value={draft.depthPrompt?.prompt ?? ""}
                  className="max-h-60 min-h-16 flex-1 font-sans"
                  onChange={(event) =>
                    set("depthPrompt", { prompt: event.target.value, depth: draft.depthPrompt?.depth ?? 4, role: draft.depthPrompt?.role ?? "system" })
                  }
                />
                <label className="flex shrink-0 flex-col gap-1 text-ui-12 text-muted-foreground" title={t("characters.editor.depthHint")}>
                  {t("characters.editor.depth")}
                  <ClampedNumberInput
                    min={0}
                    max={100}
                    value={draft.depthPrompt?.depth ?? 4}
                    className="h-9 w-20"
                    onChange={(depth) =>
                      set("depthPrompt", { prompt: draft.depthPrompt?.prompt ?? "", depth, role: draft.depthPrompt?.role ?? "system" })
                    }
                  />
                </label>
              </div>
            </Field>
          </Section>

          <Section title={t("characters.editor.lorebookSection")} defaultOpen={Boolean(draft.characterBook?.entries.length)}
            badge={draft.characterBook?.entries.length
              ? draft.characterBook.entries.length === 1
                ? t("characters.lorebook.entriesOne")
                : t("characters.lorebook.entriesMany", { count: draft.characterBook.entries.length })
              : undefined}
          >
            <LorebookEditor book={draft.characterBook} onChange={(book) => set("characterBook", book)} />
          </Section>

          <Section title={t("characters.editor.groupSection")}>
            <Field label={t("characters.editor.talkativeness")} hint={t("characters.editor.talkativenessHint")}>
              <div className="flex items-center gap-3">
                <span className="text-ui-12 text-muted-foreground">{t("characters.editor.quiet")}</span>
                <Slider
                  value={[draft.talkativeness]}
                  min={0}
                  max={1}
                  step={0.05}
                  className="flex-1"
                  aria-label={t("characters.editor.talkativeness")}
                  onValueChange={([value]) => set("talkativeness", value)}
                />
                <span className="text-ui-12 text-muted-foreground">{t("characters.editor.chatty")}</span>
              </div>
            </Field>
          </Section>

          <Section title={t("characters.editor.creatorSection")} defaultOpen={Boolean(draft.creatorNotes || draft.creator)}>
            <Field label={t("characters.editor.creator")} htmlFor="character-creator">
              <Input id="character-creator" value={draft.creator} maxLength={200} onChange={(event) => set("creator", event.target.value)} />
            </Field>
            <Field label={t("characters.editor.creatorNotes")} htmlFor="character-creator-notes" hint={t("characters.editor.creatorNotesHint")}>
              <Textarea id="character-creator-notes" value={draft.creatorNotes} className="max-h-72 min-h-20 font-sans" onChange={(event) => set("creatorNotes", event.target.value)} />
            </Field>
          </Section>
        </div>
      </div>

      <div className="sticky bottom-4 z-30 mt-10">
        <div className="flex w-full items-center gap-3 rounded-full bg-popover/95 py-2 pl-5 pr-2 shadow-[0_16px_40px_-18px_rgb(0_0_0/0.45)] ring-1 ring-border backdrop-blur">
          <span className="min-w-0 flex-1 truncate text-ui-13 text-muted-foreground" data-testid="editor-token-estimate">
            {permanent > 0 ? t("characters.editor.tokens", { count: formatNumber(permanent, locale, {}) }) : null}
            {permanent > 0 && contextLength
              ? ` · ${t("characters.editor.tokensOfContext", { percent: Math.max(1, Math.round((permanent / contextLength) * 100)) })}`
              : ""}
          </span>
          <Button variant="ghost" className="rounded-full" disabled={saving} onClick={() => void save(true)}>
            {t("characters.editor.saveAndChat")}
          </Button>
          <Button variant="dark" className="rounded-full px-5" disabled={saving} onClick={() => void save(false)}>
            {saving ? t("common.saving") : t("common.save")}
          </Button>
        </div>
      </div>

      <ConfirmDeleteDialog
        open={blocker.status === "blocked"}
        title={t("characters.editor.unsavedTitle")}
        description={t("characters.editor.unsavedBody", { name: draft.name.trim() || t("characters.editor.newTitle") })}
        confirmLabel={t("characters.editor.discard")}
        cancelLabel={t("characters.editor.keepEditing")}
        onOpenChange={(open) => !open && blocker.reset?.()}
        onConfirm={() => blocker.proceed?.()}
      />
    </main>
  );
}
