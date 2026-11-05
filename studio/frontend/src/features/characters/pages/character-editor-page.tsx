// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useAppShellReadySignal } from "@/components/app-readiness";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Skeleton } from "@/components/ui/skeleton";
import { Slider } from "@/components/ui/slider";
import { Textarea } from "@/components/ui/textarea";
import { getInferenceStatus } from "@/features/chat/api/chat-api";
import { useChatRuntimeStore } from "@/features/chat/stores/chat-runtime-store";
import { loadedContextFields } from "@/features/model-picker/model-config/per-model-config";
import { useNativeFileDrop } from "@/features/native-intents";
import { useT } from "@/i18n";
import { toast } from "@/lib/toast";
import { cn } from "@/lib/utils";
import {
  ArrowDown01Icon,
  ArrowLeft01Icon,
  Cancel01Icon,
  ImageUpload01Icon,
  PlusSignIcon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useBlocker, useNavigate } from "@tanstack/react-router";
import { useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { startCharacterChat } from "../actions";
import {
  createCharacter,
  getCharacter,
  removeCharacterAvatar,
  updateCharacter,
  uploadCharacterAvatar,
  type CharacterPatch,
} from "../api";
import { ConfirmDialog } from "../components/confirm-dialog";
import { LorebookEditor } from "../components/lorebook-editor";
import { Portrait } from "../components/portrait";
import { estimateTokens } from "../prompt/lorebook";
import { useCharactersStore } from "../store";
import type { Character, CharacterDraft } from "../types";

const MAX_PICTURE_BYTES = 20 * 1024 * 1024;

function blankDraft(): CharacterDraft {
  return {
    name: "",
    description: "",
    personality: "",
    scenario: "",
    firstMes: "",
    alternateGreetings: [],
    mesExample: "",
    systemPrompt: "",
    postHistoryInstructions: "",
    creatorNotes: "",
    creator: "",
    characterVersion: "",
    tags: [],
    talkativeness: 0.5,
    favorite: false,
    depthPrompt: null,
    characterBook: null,
  };
}

function draftOf(character: Character): CharacterDraft {
  const { name, description, personality, scenario, firstMes, alternateGreetings, mesExample, systemPrompt,
    postHistoryInstructions, creatorNotes, creator, characterVersion, tags, talkativeness, favorite, depthPrompt,
    characterBook } = character;
  return {
    name, description, personality, scenario, firstMes, alternateGreetings: [...alternateGreetings], mesExample,
    systemPrompt, postHistoryInstructions, creatorNotes, creator, characterVersion, tags: [...tags], talkativeness,
    favorite, depthPrompt, characterBook,
  };
}

function errorText(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

export function CharacterEditorPage({ characterId }: { characterId: string | null }) {
  const t = useT();
  const navigate = useNavigate();
  const [original, setOriginal] = useState<Character | null>(null);
  const [draft, setDraft] = useState<CharacterDraft | null>(characterId ? null : blankDraft());
  const [baseline, setBaseline] = useState<string>(() => JSON.stringify(blankDraft()));
  const [picture, setPicture] = useState<File | "remove" | null>(null);
  const [greeting, setGreeting] = useState(0);
  const [saving, setSaving] = useState(false);
  const [showErrors, setShowErrors] = useState(false);
  const fileRef = useRef<HTMLInputElement>(null);
  const storeContextLength = useChatRuntimeStore((s) => s.loadedContextLength);
  const [statusContextLength, setStatusContextLength] = useState<number | null>(null);
  useEffect(() => {
    // Opened from a link before Chat has read the server's status, the store does not know the loaded model yet.
    if (storeContextLength != null) return;
    const controller = new AbortController();
    getInferenceStatus(controller.signal)
      .then((status) => setStatusContextLength(loadedContextFields(status).loadedContextLength))
      .catch(() => undefined);
    return () => controller.abort();
  }, [storeContextLength]);
  const contextLength = storeContextLength ?? statusContextLength;
  const choosePictureRef = useRef<(file: File | undefined) => void>(() => {});
  // The portrait takes a dropped image too, the desktop app included.
  const pictureDrop = useNativeFileDrop({
    onFiles: (files) => choosePictureRef.current(files[0]),
    accept: ".png,.jpg,.jpeg,.webp,.gif",
    multiple: false,
  });
  const leaving = useRef(false);
  useAppShellReadySignal();

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

  const set = <K extends keyof CharacterDraft>(key: K, value: CharacterDraft[K]) =>
    setDraft((current) => (current ? { ...current, [key]: value } : current));
  const greetings = [draft.firstMes, ...draft.alternateGreetings];
  const permanent = estimateTokens(
    [draft.description, draft.personality, draft.scenario, draft.mesExample, draft.systemPrompt]
      .map((field) => field.trim())
      .filter(Boolean)
      .join("\n"),
  );
  const nameMissing = !draft.name.trim();
  const hasImage = picture === "remove" ? false : Boolean(previewUrl) || Boolean(original?.hasAvatar);

  function setGreetingText(index: number, text: string) {
    if (index === 0) set("firstMes", text);
    else set("alternateGreetings", draft!.alternateGreetings.map((g, i) => (i === index - 1 ? text : g)));
  }

  function choosePicture(file: File | undefined) {
    if (!file) return;
    if (file.size > MAX_PICTURE_BYTES) {
      toast.error(t("characters.editor.pictureTooLarge"));
      return;
    }
    setPicture(file);
  }

  choosePictureRef.current = choosePicture;

  async function save(thenChat: boolean) {
    if (!draft) return;
    if (nameMissing) {
      setShowErrors(true);
      return;
    }
    setSaving(true);
    try {
      const { depthPrompt, characterBook, ...rest } = draft;
      const keepNote = Boolean(depthPrompt?.prompt.trim());
      const keepBook = Boolean(characterBook?.entries.length);
      // An empty note or book is cleared rather than saved empty.
      const fields: CharacterPatch & { name: string } = {
        ...rest,
        name: draft.name.trim(),
        alternateGreetings: draft.alternateGreetings.filter((g) => g.trim()),
        ...(keepNote ? { depthPrompt } : {}),
        ...(keepBook ? { characterBook } : {}),
      };
      const clear: NonNullable<CharacterPatch["clear"]> = [
        ...(keepNote ? [] : (["depthPrompt"] as const)),
        ...(keepBook ? [] : (["characterBook"] as const)),
      ];
      let saved = original
        ? await updateCharacter(original.id, { ...fields, clear })
        : await createCharacter(fields);
      if (!original) {
        // Kept before the portrait upload, so a retry after a failed upload updates this character.
        setOriginal(saved);
        useCharactersStore.getState().upsert(saved);
      }
      if (picture instanceof File) saved = await uploadCharacterAvatar(saved.id, picture);
      else if (picture === "remove" && saved.hasAvatar) saved = await removeCharacterAvatar(saved.id);
      useCharactersStore.getState().upsert(saved);
      leaving.current = true;
      toast.success(t("characters.editor.saved", { name: saved.name }));
      if (thenChat) {
        const threadId = await startCharacterChat([saved.id]);
        void navigate({ to: "/chat", search: { thread: threadId } });
      } else {
        void navigate({ to: "/characters/$characterId", params: { characterId: saved.id } });
      }
    } catch (error) {
      leaving.current = false;
      toast.error(t("characters.editor.saveFailed"), { description: errorText(error) });
    } finally {
      setSaving(false);
    }
  }

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
          <div
            className={cn(
              "group/picture relative overflow-hidden rounded-[24px] ring-1 ring-black/[0.05] max-md:mx-auto max-md:w-52 dark:ring-white/[0.06]",
              pictureDrop.dragging && "ring-2 ring-primary",
            )}
            ref={pictureDrop.ref}
            {...pictureDrop.dragHandlers}
          >
            <Portrait
              id={original?.id}
              name={draft.name || "?"}
              hasImage={hasImage}
              rev={original?.avatarRev ?? 0}
              imageUrl={previewUrl ?? (picture === "remove" ? null : undefined)}
              variant="card"
              className="aspect-[2/3] w-full"
            />
            <button
              type="button"
              onClick={() => fileRef.current?.click()}
              className="absolute inset-0 flex cursor-pointer flex-col items-center justify-end gap-1 bg-gradient-to-t from-black/60 via-black/0 to-transparent pb-4 text-white opacity-0 transition-opacity focus-visible:opacity-100 focus-visible:outline-none group-hover/picture:opacity-100"
            >
              <HugeiconsIcon icon={ImageUpload01Icon} strokeWidth={1.6} className="size-6" />
              <span className="text-ui-13 font-medium">
                {pictureDrop.dragging
                  ? t("characters.editor.dropPicture")
                  : hasImage
                    ? t("characters.editor.changePicture")
                    : t("characters.editor.uploadPicture")}
              </span>
            </button>
          </div>
          <input
            ref={fileRef}
            type="file"
            accept="image/png,image/jpeg,image/webp,image/gif"
            className="hidden"
            onChange={(event) => {
              choosePicture(event.target.files?.[0]);
              event.target.value = "";
            }}
          />
          {hasImage ? (
            <Button variant="ghost" size="sm" className="-mt-2 self-center" onClick={() => setPicture("remove")}>
              {t("characters.editor.removePicture")}
            </Button>
          ) : null}

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
            <TagsInput tags={draft.tags} onChange={(tags) => set("tags", tags)} />
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

          <Card>
            <div className="flex flex-col gap-1">
              <h2 className="text-ui-15 font-semibold text-foreground">{t("characters.editor.greetings")}</h2>
              <p className="text-ui-13 text-muted-foreground">{t("characters.editor.greetingsHint")}</p>
            </div>
            <div className="flex flex-wrap items-center gap-1.5" role="tablist" aria-label={t("characters.editor.greetings")}>
              {greetings.map((_, index) => (
                <button
                  key={index}
                  type="button"
                  role="tab"
                  aria-selected={greeting === index}
                  onClick={() => setGreeting(index)}
                  className={cn(
                    "h-8 cursor-pointer rounded-full px-3.5 text-ui-13 font-medium transition-colors",
                    greeting === index ? "bg-foreground text-background" : "bg-muted text-muted-foreground hover:text-foreground dark:bg-card",
                  )}
                >
                  {index === 0 ? t("characters.editor.mainGreeting") : t("characters.editor.alternateGreeting", { number: index })}
                </button>
              ))}
              <Button
                type="button"
                variant="ghost"
                size="sm"
                className="rounded-full"
                onClick={() => {
                  set("alternateGreetings", [...draft.alternateGreetings, ""]);
                  setGreeting(greetings.length);
                }}
              >
                <HugeiconsIcon icon={PlusSignIcon} strokeWidth={1.75} className="size-icon" />
                {t("characters.editor.addGreeting")}
              </Button>
            </div>
            <div className="relative">
              <Textarea
                value={greetings[greeting] ?? ""}
                placeholder={t("characters.editor.greetingPlaceholder")}
                aria-label={t("characters.editor.greetings")}
                className="max-h-[55vh] min-h-36 font-sans text-ui-15 leading-relaxed"
                onChange={(event) => setGreetingText(greeting, event.target.value)}
              />
              {greeting > 0 ? (
                <Button
                  type="button"
                  variant="ghost"
                  size="sm"
                  className="mt-2"
                  onClick={() => {
                    set("alternateGreetings", draft.alternateGreetings.filter((_, i) => i !== greeting - 1));
                    setGreeting(Math.max(0, greeting - 1));
                  }}
                >
                  {t("characters.editor.removeGreeting")}
                </Button>
              ) : null}
            </div>
          </Card>

          <h2 className="mt-2 px-1 text-ui-13 font-semibold uppercase tracking-[0.06em] text-muted-foreground">
            {t("characters.editor.advanced")}
          </h2>

          <Section title={t("characters.editor.personalitySection")} defaultOpen={Boolean(draft.personality || draft.scenario)}>
            <Field label={t("characters.editor.personality")} hint={t("characters.editor.personalityHint")}>
              <Textarea value={draft.personality} className="max-h-72 min-h-20 font-sans" onChange={(event) => set("personality", event.target.value)} />
            </Field>
            <Field label={t("characters.editor.scenario")} hint={t("characters.editor.scenarioHint")}>
              <Textarea value={draft.scenario} className="max-h-72 min-h-20 font-sans" onChange={(event) => set("scenario", event.target.value)} />
            </Field>
          </Section>

          <Section title={t("characters.editor.examplesSection")} defaultOpen={Boolean(draft.mesExample.trim())}>
            <Field label={t("characters.editor.examplesSection")} hint={t("characters.editor.examplesHint")}>
              <Textarea
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
            <Field label={t("characters.editor.systemPrompt")} hint={t("characters.editor.systemPromptHint")}>
              <Textarea value={draft.systemPrompt} className="max-h-72 min-h-20 font-sans" onChange={(event) => set("systemPrompt", event.target.value)} />
            </Field>
            <Field label={t("characters.editor.postHistory")} hint={t("characters.editor.postHistoryHint")}>
              <Textarea
                value={draft.postHistoryInstructions}
                className="max-h-72 min-h-16 font-sans"
                onChange={(event) => set("postHistoryInstructions", event.target.value)}
              />
            </Field>
            <Field label={t("characters.editor.characterNote")} hint={t("characters.editor.characterNoteHint")}>
              <div className="flex flex-col gap-2 sm:flex-row sm:items-start">
                <Textarea
                  value={draft.depthPrompt?.prompt ?? ""}
                  className="max-h-60 min-h-16 flex-1 font-sans"
                  onChange={(event) =>
                    set("depthPrompt", { prompt: event.target.value, depth: draft.depthPrompt?.depth ?? 4, role: draft.depthPrompt?.role ?? "system" })
                  }
                />
                <label className="flex shrink-0 flex-col gap-1 text-ui-12 text-muted-foreground" title={t("characters.editor.depthHint")}>
                  {t("characters.editor.depth")}
                  <Input
                    type="number"
                    min={0}
                    max={100}
                    value={draft.depthPrompt?.depth ?? 4}
                    className="h-9 w-20"
                    onChange={(event) =>
                      set("depthPrompt", {
                        prompt: draft.depthPrompt?.prompt ?? "",
                        depth: Math.max(0, Math.min(100, Number(event.target.value) || 0)),
                        role: draft.depthPrompt?.role ?? "system",
                      })
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
            <Field label={t("characters.editor.creator")}>
              <Input value={draft.creator} maxLength={200} onChange={(event) => set("creator", event.target.value)} />
            </Field>
            <Field label={t("characters.editor.creatorNotes")} hint={t("characters.editor.creatorNotesHint")}>
              <Textarea value={draft.creatorNotes} className="max-h-72 min-h-20 font-sans" onChange={(event) => set("creatorNotes", event.target.value)} />
            </Field>
          </Section>
        </div>
      </div>

      <div className="sticky bottom-4 z-30 mt-10">
        <div className="flex w-full items-center gap-3 rounded-full bg-popover/95 py-2 pl-5 pr-2 shadow-[0_16px_40px_-18px_rgb(0_0_0/0.45)] ring-1 ring-border backdrop-blur">
          <span className="min-w-0 flex-1 truncate text-ui-13 text-muted-foreground" data-testid="editor-token-estimate">
            {permanent > 0 ? t("characters.editor.tokens", { count: permanent.toLocaleString() }) : null}
            {permanent > 0 && contextLength
              ? ` · ${t("characters.editor.tokensOfContext", { percent: Math.max(1, Math.round((permanent / contextLength) * 100)) })}`
              : ""}
          </span>
          <Button variant="ghost" className="rounded-full" disabled={saving} onClick={() => void save(true)}>
            {t("characters.editor.saveAndChat")}
          </Button>
          <Button variant="dark" className="rounded-full px-5" disabled={saving} onClick={() => void save(false)}>
            {saving ? t("characters.editor.saving") : t("characters.editor.save")}
          </Button>
        </div>
      </div>

      <ConfirmDialog
        open={blocker.status === "blocked"}
        title={t("characters.editor.unsavedTitle")}
        body={t("characters.editor.unsavedBody", { name: draft.name.trim() || t("characters.editor.newTitle") })}
        confirm={t("characters.editor.discard")}
        cancel={t("characters.editor.keepEditing")}
        onClose={() => blocker.reset?.()}
        onConfirm={() => blocker.proceed?.()}
      />
    </main>
  );
}

function Card({ children }: { children: ReactNode }) {
  return <section className="flex flex-col gap-4 rounded-[24px] bg-muted/45 p-5 dark:bg-card">{children}</section>;
}

function Field({
  label,
  hint,
  htmlFor,
  error,
  children,
}: {
  label: string;
  hint?: string;
  htmlFor?: string;
  error?: string;
  children: ReactNode;
}) {
  return (
    <div className="flex flex-col gap-1.5">
      <label htmlFor={htmlFor} className="text-ui-14 font-semibold text-foreground">
        {label}
      </label>
      {hint ? <p className="-mt-0.5 text-ui-13 text-muted-foreground">{hint}</p> : null}
      {children}
      {error ? <p className="text-ui-13 text-destructive">{error}</p> : null}
    </div>
  );
}

function Section({
  title,
  badge,
  defaultOpen = false,
  children,
}: {
  title: string;
  badge?: string;
  defaultOpen?: boolean;
  children: ReactNode;
}) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <section className="rounded-[24px] bg-muted/45 dark:bg-card">
      <button
        type="button"
        aria-expanded={open}
        onClick={() => setOpen((value) => !value)}
        className="flex w-full cursor-pointer items-center gap-3 px-5 py-4 text-left"
      >
        <span className="min-w-0 flex-1 text-ui-15 font-semibold text-foreground">{title}</span>
        {badge ? <span className="text-ui-12 text-muted-foreground">{badge}</span> : null}
        <HugeiconsIcon
          icon={ArrowDown01Icon}
          strokeWidth={1.75}
          className={cn("size-4 text-muted-foreground transition-transform", open && "rotate-180")}
        />
      </button>
      {open ? <div className="flex flex-col gap-4 px-5 pb-5">{children}</div> : null}
    </section>
  );
}

function TagsInput({ tags, onChange }: { tags: string[]; onChange: (tags: string[]) => void }) {
  const t = useT();
  const [value, setValue] = useState("");
  function commit() {
    const next = value
      .split(",")
      .map((tag) => tag.trim())
      .filter((tag) => tag && !tags.includes(tag));
    if (next.length) onChange([...tags, ...next].slice(0, 64));
    setValue("");
  }
  return (
    <div className="flex min-h-10 flex-wrap items-center gap-1.5 rounded-xl border border-border bg-background px-2 py-1.5 focus-within:border-ring dark:border-transparent dark:bg-[rgb(255_255_255_/_0.06)]">
      {tags.map((tag) => (
        <span key={tag} className="inline-flex items-center gap-1 rounded-full bg-muted py-0.5 pl-2.5 pr-1 text-ui-12 font-medium dark:bg-card">
          {tag}
          <button
            type="button"
            aria-label={t("characters.editor.removeTag", { tag })}
            className="flex size-4 cursor-pointer items-center justify-center rounded-full text-muted-foreground hover:bg-accent hover:text-foreground"
            onClick={() => onChange(tags.filter((other) => other !== tag))}
          >
            <HugeiconsIcon icon={Cancel01Icon} strokeWidth={2} className="size-3" />
          </button>
        </span>
      ))}
      <input
        value={value}
        placeholder={tags.length ? "" : t("characters.editor.tagsPlaceholder")}
        className="min-w-24 flex-1 bg-transparent px-1 text-ui-14 outline-none placeholder:text-muted-foreground"
        onChange={(event) => setValue(event.target.value)}
        onKeyDown={(event) => {
          if (event.key === "Enter" || event.key === ",") {
            event.preventDefault();
            commit();
          } else if (event.key === "Backspace" && !value && tags.length) {
            onChange(tags.slice(0, -1));
          }
        }}
        onBlur={commit}
      />
    </div>
  );
}
