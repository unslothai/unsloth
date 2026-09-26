// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Spinner } from "@/components/ui/spinner";
import { Switch } from "@/components/ui/switch";
import { Textarea } from "@/components/ui/textarea";
import { useCopyFeedback } from "@/features/hub";
import { useT } from "@/i18n";
import { toast } from "@/lib/toast";
import { cn } from "@/lib/utils";
import {
  ArrowLeft01Icon,
  ArrowRight01Icon,
  Cancel01Icon,
  Copy01Icon,
  PlusSignIcon,
  RefreshIcon,
  Scroll01Icon,
  Search01Icon,
  Tick02Icon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import {
  type KeyboardEvent,
  type ReactElement,
  type ReactNode,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import {
  createSkill,
  deleteSkill,
  getSkillManifest,
  isValidSkillName,
  listSkills,
  setSkillEnabled,
  type SkillDraft,
  type SkillManifest,
  type SkillRecord,
  updateSkill,
  useSkillsCatalog,
} from "../api/skills-api";

type EditDraft = { description: string; instructions: string };
type View = { kind: "library" } | { kind: "new" } | { kind: "skill"; key: string };

const EMPTY_DRAFT: SkillDraft = { name: "", description: "", instructions: "" };
const LIBRARY: View = { kind: "library" };
// Library order: the skills this dialog writes first, then the read-only folders.
const SECTIONS: ReadonlyArray<SkillRecord["source"]> = ["agents", "claude", "bundled"];

// A shadowed row shares its name with the one that wins, so skills are keyed by source too.
const keyOf = (skill: SkillRecord) => `${skill.source}:${skill.name}`;

function describe(cause: unknown): string | undefined {
  return cause instanceof Error ? cause.message : undefined;
}

function isChord(event: KeyboardEvent): boolean {
  return event.key === "Enter" && (event.metaKey || event.ctrlKey);
}

/** Only what the create_skill tool would have written: a plain folder in ~/.agents/skills.
 *  The backend refuses the rest, so the editor stays read-only on them. */
function isEditable(skill: SkillRecord): boolean {
  return skill.valid && skill.source === "agents" && !skill.linked;
}

function monogram(name: string): string {
  const parts = name.split("-").filter(Boolean);
  return (parts.length > 1 ? parts[0][0] + parts[1][0] : name.slice(0, 2)).toUpperCase();
}

export function ChatSkillsDialog({
  open,
  onOpenChange,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}): ReactElement {
  const t = useT();
  const { skills, loading, error } = useSkillsCatalog();
  const [searchQuery, setSearchQuery] = useState("");
  const [enabledOnly, setEnabledOnly] = useState(false);
  const [view, setView] = useState<View>(LIBRARY);
  const [newDraft, setNewDraft] = useState<SkillDraft>(EMPTY_DRAFT);
  const [draft, setDraft] = useState<EditDraft | null>(null);
  const [manifests, setManifests] = useState<ReadonlyMap<string, SkillManifest>>(
    () => new Map(),
  );
  // Bodies being read right now, so opening the same skill twice does not read it twice.
  const inflight = useRef(new Set<string>());
  const [pending, setPending] = useState<string | null>(null);
  const [creating, setCreating] = useState(false);
  const [changing, setChanging] = useState<string | null>(null);
  const [confirmingDelete, setConfirmingDelete] = useState<SkillRecord | null>(null);
  // What to do once the user agrees to drop an unsaved draft.
  const [confirmingDiscard, setConfirmingDiscard] = useState<(() => void) | null>(null);
  // Each open starts on the library: search, filter and drafts belong to one sitting. Reseeded
  // on render, like the project dialog, so the first frame after opening is already reset.
  const [seenOpen, setSeenOpen] = useState(open);
  if (open !== seenOpen) {
    setSeenOpen(open);
    if (open) {
      setSearchQuery("");
      setEnabledOnly(false);
      setView(LIBRARY);
      setNewDraft(EMPTY_DRAFT);
      setDraft(null);
      setManifests(new Map());
      setConfirmingDelete(null);
      setConfirmingDiscard(null);
    }
  }
  // Skills are added by writing files, so each open re-reads the folders.
  useEffect(() => {
    if (open) void listSkills(true).catch(() => undefined);
  }, [open]);

  const sourceLabel = (source: SkillRecord["source"]) =>
    source === "agents"
      ? t("skills.sourceAgents")
      : source === "claude"
        ? t("skills.sourceClaude")
        : t("skills.sourceBundled");

  const query = searchQuery.trim().toLowerCase();
  const narrowed = query !== "" || enabledOnly;
  const filtered = useMemo(
    () =>
      skills.filter((skill) => {
        if (enabledOnly && !(skill.enabled && skill.valid && !skill.shadowed)) return false;
        return (
          !query ||
          skill.name.toLowerCase().includes(query) ||
          skill.description.toLowerCase().includes(query)
        );
      }),
    [skills, enabledOnly, query],
  );
  const sections = SECTIONS.map((source) => ({
    source,
    skills: filtered.filter((skill) => skill.source === source),
  })).filter((section) => section.skills.length > 0 || (section.source === "agents" && !narrowed));
  const enabledCount = skills.filter(
    (skill) => skill.enabled && skill.valid && !skill.shadowed,
  ).length;

  // The open skill follows the catalog: gone from the folders means back to the library.
  const selected =
    view.kind === "skill" ? (skills.find((skill) => keyOf(skill) === view.key) ?? null) : null;
  if (open && view.kind === "skill" && !loading && selected === null) {
    setView(LIBRARY);
    setDraft(null);
  }
  const readable = selected !== null && selected.valid && !selected.shadowed;

  // The body is read when a skill is opened, once per sitting; Refresh forgets them all. A late
  // result is kept rather than cancelled: it is the same file either way. A read that fails
  // lands back on the library, with the reason in a toast.
  const readManifest = (skill: SkillRecord) => {
    const name = skill.name;
    if (!skill.valid || skill.shadowed || inflight.current.has(name)) return;
    inflight.current.add(name);
    getSkillManifest(name)
      .then((manifest) => {
        setManifests((prev) => (prev.has(name) ? prev : new Map(prev).set(name, manifest)));
      })
      .catch((cause: unknown) => {
        toast.error(t("skills.openError"), { description: describe(cause) });
        setView((current) =>
          current.kind === "skill" && current.key === keyOf(skill) ? LIBRARY : current,
        );
      })
      .finally(() => {
        inflight.current.delete(name);
      });
  };

  const manifest = selected ? (manifests.get(selected.name) ?? null) : null;
  const editable = selected !== null && isEditable(selected);
  const newStarted =
    newDraft.name !== "" || newDraft.description !== "" || newDraft.instructions !== "";
  const dirty = view.kind === "new" ? newStarted : draft !== null;
  const busy = pending !== null || creating;

  const trimmedName = newDraft.name.trim();
  const nameInvalid = trimmedName.length > 0 && !isValidSkillName(trimmedName);
  const canCreate =
    trimmedName.length > 0 &&
    !nameInvalid &&
    newDraft.description.trim().length > 0 &&
    newDraft.instructions.trim().length > 0;
  const canSave =
    editable &&
    draft !== null &&
    draft.description.trim().length > 0 &&
    draft.instructions.trim().length > 0;

  // Leaving an editor with a draft asks first; everything else just goes.
  const leaveTo = (next: () => void) => {
    if (dirty) setConfirmingDiscard(() => next);
    else next();
  };
  const goLibrary = () =>
    leaveTo(() => {
      setView(LIBRARY);
      setDraft(null);
      setNewDraft(EMPTY_DRAFT);
    });
  const openSkill = (skill: SkillRecord) =>
    leaveTo(() => {
      setDraft(null);
      setNewDraft(EMPTY_DRAFT);
      setView({ kind: "skill", key: keyOf(skill) });
      if (!manifests.has(skill.name)) readManifest(skill);
    });
  const openNew = () =>
    leaveTo(() => {
      setDraft(null);
      setView({ kind: "new" });
    });

  function handleOpenChange(next: boolean) {
    // A save, create or delete in flight finishes before the dialog can go away under it.
    if (!next && busy) return;
    if (!next && dirty) {
      setConfirmingDiscard(() => () => onOpenChange(false));
      return;
    }
    onOpenChange(next);
  }

  const refresh = () => {
    setManifests(new Map());
    void listSkills(true).catch(() => undefined);
    if (selected) readManifest(selected);
  };

  const toggle = async (name: string, enabled: boolean) => {
    setChanging(name);
    try {
      await setSkillEnabled(name, enabled);
    } catch (cause) {
      toast.error(t("skills.updateError"), { description: describe(cause) });
    } finally {
      setChanging(null);
    }
  };

  const editDraft = (patch: Partial<EditDraft>) => {
    if (!manifest) return;
    setDraft((prev) => {
      const current = prev ?? {
        description: manifest.description,
        instructions: manifest.instructions,
      };
      const next = { ...current, ...patch };
      // A draft that returns to the file's content is dropped rather than kept as a no-op change.
      return next.description === manifest.description &&
        next.instructions === manifest.instructions
        ? null
        : next;
    });
  };

  async function save() {
    if (!selected || !manifest || !draft || !canSave || pending !== null) return;
    const name = selected.name;
    const description = draft.description.trim();
    const instructions = draft.instructions.trim();
    setPending(name);
    try {
      await updateSkill(name, { description, instructions });
      setManifests((prev) =>
        new Map(prev).set(name, { ...manifest, description, instructions }),
      );
      setDraft(null);
      toast.success(t("skills.saved", { name }));
    } catch (cause) {
      toast.error(t("skills.saveError"), { description: describe(cause) });
    } finally {
      setPending(null);
    }
  }

  async function create() {
    if (creating || !canCreate) return;
    setCreating(true);
    try {
      const created = await createSkill({
        name: trimmedName,
        description: newDraft.description.trim(),
        instructions: newDraft.instructions.trim(),
      });
      toast.success(t("skills.created", { name: created.name }), {
        description: created.path ?? undefined,
      });
      setNewDraft(EMPTY_DRAFT);
      setView({ kind: "skill", key: keyOf(created) });
      readManifest(created);
    } catch (cause) {
      toast.error(t("skills.saveError"), { description: describe(cause) });
    } finally {
      setCreating(false);
    }
  }

  async function remove(skill: SkillRecord) {
    if (pending !== null) return;
    setPending(skill.name);
    try {
      await deleteSkill(skill.name);
      setManifests((prev) => {
        const map = new Map(prev);
        map.delete(skill.name);
        return map;
      });
      setDraft(null);
      setView(LIBRARY);
      toast.success(t("skills.deleted", { name: skill.name }));
    } catch (cause) {
      toast.error(t("skills.deleteError"), { description: describe(cause) });
    } finally {
      setPending(null);
    }
  }

  const closeButton = (
    <Button
      type="button"
      variant="ghost"
      size="icon-sm"
      disabled={busy}
      onClick={() => handleOpenChange(false)}
      aria-label={t("common.close")}
    >
      <HugeiconsIcon icon={Cancel01Icon} className="size-4" />
    </Button>
  );
  const divider = <div className="mx-1 h-5 w-px shrink-0 bg-border/60" />;

  return (
    <Dialog open={open} onOpenChange={handleOpenChange}>
      <DialogContent
        showCloseButton={false}
        className="flex h-[min(820px,calc(100dvh-3rem))] flex-col gap-0 overflow-hidden p-0 sm:max-w-[min(1120px,92vw)]"
      >
        {view.kind === "library" ? (
          <>
            <header className="flex shrink-0 items-start gap-3 border-b border-border/50 px-6 pt-5 pb-4">
              <span className="flex size-10 shrink-0 items-center justify-center rounded-xl bg-primary/10 text-primary">
                <HugeiconsIcon icon={Scroll01Icon} strokeWidth={1.75} className="size-5" />
              </span>
              <div className="min-w-0 flex-1">
                <DialogTitle className="text-base font-semibold tracking-tight">
                  {t("skills.title")}
                </DialogTitle>
                <DialogDescription className="mt-0.5 text-xs text-muted-foreground">
                  {t("skills.description")}
                </DialogDescription>
              </div>
              <div className="flex shrink-0 items-center gap-2">
                <Button
                  type="button"
                  variant="ghost"
                  size="icon-sm"
                  disabled={loading}
                  onClick={refresh}
                  aria-label={t("skills.refresh")}
                  title={t("skills.refresh")}
                >
                  {loading ? (
                    <Spinner />
                  ) : (
                    <HugeiconsIcon icon={RefreshIcon} strokeWidth={2} className="size-4" />
                  )}
                </Button>
                <Button type="button" size="sm" onClick={openNew}>
                  <HugeiconsIcon icon={PlusSignIcon} strokeWidth={2} />
                  {t("skills.newSkill")}
                </Button>
                {divider}
                {closeButton}
              </div>
            </header>

            <div className="flex shrink-0 flex-wrap items-center gap-3 border-b border-border/50 px-6 py-3">
              <div className="relative min-w-0 flex-1 sm:max-w-xs">
                <HugeiconsIcon
                  icon={Search01Icon}
                  strokeWidth={2}
                  className="pointer-events-none absolute top-1/2 left-3 size-3.5 -translate-y-1/2 text-muted-foreground/60"
                />
                <input
                  value={searchQuery}
                  onChange={(event) => setSearchQuery(event.target.value)}
                  onKeyDown={(event) => {
                    if (event.key === "Escape" && searchQuery) {
                      event.stopPropagation();
                      setSearchQuery("");
                    }
                  }}
                  placeholder={t("skills.search")}
                  aria-label={t("skills.search")}
                  className="h-8 w-full rounded-full border-0 bg-muted/50 pl-9 pr-8 text-sm outline-none transition-shadow placeholder:text-muted-foreground/60 focus:ring-1 focus:ring-ring"
                />
                {searchQuery ? (
                  <button
                    type="button"
                    onClick={() => setSearchQuery("")}
                    aria-label={t("skills.clearSearch")}
                    className="absolute top-1/2 right-2 flex size-5 -translate-y-1/2 items-center justify-center rounded-full text-muted-foreground hover:text-foreground"
                  >
                    <HugeiconsIcon icon={Cancel01Icon} className="size-3" />
                  </button>
                ) : null}
              </div>
              <label className="ml-auto flex items-center gap-2 text-xs text-muted-foreground">
                {t("skills.enabledOnly")}
                <Switch size="sm" checked={enabledOnly} onCheckedChange={setEnabledOnly} />
              </label>
              {divider}
              <span className="text-xs tabular-nums text-muted-foreground">
                {t("skills.summary", { enabled: enabledCount, total: skills.length })}
              </span>
            </div>

            <div className="hover-scrollbar min-h-0 flex-1 overflow-y-auto p-6">
              {error ? (
                <p className="rounded-xl border border-destructive/30 bg-destructive/5 px-4 py-3 text-sm text-destructive">
                  {error}
                </p>
              ) : null}
              {sections.length > 0 ? (
                <div className="flex flex-col gap-8">
                  {sections.map((section) => (
                    <section key={section.source} className="flex flex-col gap-3">
                      <div className="flex flex-wrap items-baseline gap-x-2 gap-y-0.5">
                        <h3 className="text-sm font-semibold tracking-tight">
                          {t(`skills.section${sectionSuffix(section.source)}`)}
                        </h3>
                        <span className="text-xs tabular-nums text-muted-foreground/60">
                          {section.skills.length}
                        </span>
                        <p className="basis-full text-xs text-muted-foreground sm:ml-auto sm:basis-auto">
                          {t(`skills.section${sectionSuffix(section.source)}Hint`)}
                        </p>
                      </div>
                      <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3">
                        {section.source === "agents" && !narrowed ? (
                          <button
                            type="button"
                            onClick={openNew}
                            className="flex min-h-[calc(160px*var(--ui-space-scale,1))] flex-col items-center justify-center gap-2 rounded-2xl border border-dashed border-border/70 p-4 text-center transition-colors hover:border-primary/50 hover:bg-primary/5 focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
                          >
                            <span className="flex size-10 items-center justify-center rounded-xl bg-primary/10 text-primary">
                              <HugeiconsIcon icon={PlusSignIcon} strokeWidth={2} className="size-5" />
                            </span>
                            <span className="text-sm font-semibold tracking-tight">
                              {t("skills.newSkill")}
                            </span>
                            <span className="text-xs text-muted-foreground">
                              {t("skills.newSkillHint")}
                            </span>
                          </button>
                        ) : null}
                        {section.skills.map((skill) => (
                          <SkillCard
                            key={keyOf(skill)}
                            skill={skill}
                            changing={changing === skill.name}
                            onOpen={() => openSkill(skill)}
                            onToggle={(enabled) => void toggle(skill.name, enabled)}
                          />
                        ))}
                      </div>
                    </section>
                  ))}
                </div>
              ) : loading ? (
                <div className="flex h-full items-center justify-center">
                  <Spinner />
                </div>
              ) : (
                <div className="flex h-full flex-col items-center justify-center gap-3 text-center">
                  <span className="flex size-12 items-center justify-center rounded-2xl bg-muted/60">
                    <HugeiconsIcon
                      icon={Scroll01Icon}
                      strokeWidth={1.75}
                      className="size-5 text-muted-foreground/50"
                    />
                  </span>
                  {narrowed ? (
                    <>
                      <p className="text-sm font-medium">
                        {t("skills.noMatch", {
                          query: searchQuery.trim() || t("skills.enabledOnly"),
                        })}
                      </p>
                      <button
                        type="button"
                        onClick={() => {
                          setSearchQuery("");
                          setEnabledOnly(false);
                        }}
                        className="text-xs text-primary hover:underline"
                      >
                        {t("skills.clearSearch")}
                      </button>
                    </>
                  ) : (
                    <>
                      <p className="text-sm font-medium">{t("skills.emptyTitle")}</p>
                      <p className="max-w-sm text-xs leading-relaxed text-muted-foreground">
                        {t("skills.empty")}
                      </p>
                      <Button type="button" size="sm" onClick={openNew}>
                        <HugeiconsIcon icon={PlusSignIcon} strokeWidth={2} />
                        {t("skills.newSkill")}
                      </Button>
                    </>
                  )}
                </div>
              )}
            </div>
          </>
        ) : (
          <>
            <header className="flex shrink-0 items-center gap-3 border-b border-border/50 px-4 pt-4 pb-3 sm:px-6">
              <Button
                type="button"
                variant="ghost"
                size="sm"
                disabled={busy}
                onClick={goLibrary}
                className="-ml-2 text-muted-foreground"
              >
                <HugeiconsIcon icon={ArrowLeft01Icon} strokeWidth={2} />
                {t("skills.title")}
              </Button>
              {divider}
              <Monogram
                name={view.kind === "new" ? trimmedName : (selected?.name ?? "")}
                on={view.kind === "new" ? true : (selected?.enabled ?? false)}
              />
              <div className="min-w-0 flex-1">
                <DialogTitle className="truncate text-base font-semibold tracking-tight">
                  {view.kind === "new" ? t("skills.newSkill") : (selected?.name ?? "")}
                </DialogTitle>
                <DialogDescription className="mt-0.5 truncate text-xs text-muted-foreground">
                  {view.kind === "new"
                    ? t("skills.createDescription")
                    : (manifest?.path ?? selected?.path ?? sourceLabel(selected?.source ?? "agents"))}
                </DialogDescription>
              </div>
              {selected ? (
                <div className="flex shrink-0 items-center gap-2">
                  <SourceChip label={sourceLabel(selected.source)} source={selected.source} />
                  {dirty ? (
                    <span className="rounded-full bg-primary/10 px-2 py-0.5 text-ui-11 font-medium text-primary">
                      {t("skills.unsaved")}
                    </span>
                  ) : null}
                  <label className="flex items-center gap-2 text-xs text-muted-foreground">
                    {t("skills.enabledLabel")}
                    <Switch
                      checked={selected.valid && !selected.shadowed && selected.enabled}
                      disabled={!selected.valid || selected.shadowed || changing === selected.name}
                      aria-label={t(selected.enabled ? "skills.disable" : "skills.enable", {
                        name: selected.name,
                      })}
                      onCheckedChange={(checked) => void toggle(selected.name, checked)}
                    />
                  </label>
                </div>
              ) : null}
              {view.kind === "new" ? (
                <div className="flex shrink-0 items-center gap-2">
                  <Button type="button" size="sm" variant="ghost" disabled={creating} onClick={goLibrary}>
                    {t("common.cancel")}
                  </Button>
                  <Button
                    type="submit"
                    form="skill-new-form"
                    size="sm"
                    disabled={creating || !canCreate}
                  >
                    {creating ? <Spinner /> : <HugeiconsIcon icon={Tick02Icon} strokeWidth={2} />}
                    {t("skills.create")}
                  </Button>
                </div>
              ) : editable ? (
                <>
                  {divider}
                  <div className="flex shrink-0 items-center gap-2">
                    <Button
                      type="button"
                      size="sm"
                      variant="ghost"
                      disabled={!dirty || pending !== null}
                      onClick={() => setDraft(null)}
                    >
                      {t("skills.revert")}
                    </Button>
                    <Button
                      type="submit"
                      form="skill-edit-form"
                      size="sm"
                      disabled={pending !== null || !canSave}
                    >
                      {pending !== null ? <Spinner /> : <HugeiconsIcon icon={Tick02Icon} strokeWidth={2} />}
                      {t("skills.save")}
                    </Button>
                  </div>
                </>
              ) : null}
              {divider}
              {closeButton}
            </header>

            <div className="hover-scrollbar flex min-h-0 flex-1 flex-col overflow-y-auto">
              {view.kind === "new" ? (
                <Editor
                  formId="skill-new-form"
                  name={trimmedName}
                  nameField={
                    <Field
                      htmlFor="skill-name"
                      label={t("skills.nameLabel")}
                      hint={nameInvalid ? t("skills.nameInvalid") : t("skills.nameHint")}
                      invalid={nameInvalid}
                    >
                      <Input
                        id="skill-name"
                        value={newDraft.name}
                        autoFocus={true}
                        autoComplete="off"
                        spellCheck={false}
                        maxLength={64}
                        disabled={creating}
                        aria-label={t("skills.nameLabel")}
                        aria-invalid={nameInvalid || undefined}
                        placeholder={t("skills.namePlaceholder")}
                        onChange={(event) =>
                          setNewDraft((prev) => ({ ...prev, name: event.target.value }))
                        }
                        className="font-mono text-sm sm:max-w-sm"
                      />
                    </Field>
                  }
                  description={newDraft.description}
                  instructions={newDraft.instructions}
                  readOnly={false}
                  disabled={creating}
                  onDescription={(value) => setNewDraft((prev) => ({ ...prev, description: value }))}
                  onInstructions={(value) =>
                    setNewDraft((prev) => ({ ...prev, instructions: value }))
                  }
                  onSubmit={() => void create()}
                  aside={
                    <EditorAside
                      name={trimmedName}
                      description={newDraft.description}
                      instructions={newDraft.instructions}
                      sourceText={t("skills.fromAgents")}
                      sourceChip={
                        <SourceChip label={t("skills.sourceAgents")} source="agents" />
                      }
                    />
                  }
                />
              ) : selected ? (
                readable && !manifest ? (
                  <div className="flex h-full items-center justify-center">
                    <Spinner />
                  </div>
                ) : (
                  <Editor
                    formId="skill-edit-form"
                    name={selected.name}
                    description={draft?.description ?? manifest?.description ?? selected.description}
                    instructions={draft?.instructions ?? manifest?.instructions ?? ""}
                    readOnly={!editable}
                    disabled={pending === selected.name}
                    notice={
                      !selected.valid
                        ? { tone: "error", text: selected.error ?? t("skills.invalid") }
                        : selected.shadowed
                          ? {
                              tone: "muted",
                              text: t("skills.shadowedBy", {
                                source: selected.shadowed_by
                                  ? sourceLabel(selected.shadowed_by)
                                  : "",
                              }),
                            }
                          : editable
                            ? null
                            : {
                                tone: "muted",
                                text: selected.linked
                                  ? t("skills.readOnlyLinked")
                                  : selected.source === "claude"
                                    ? t("skills.readOnlyClaude")
                                    : t("skills.readOnlyBundled"),
                              }
                    }
                    onDescription={(value) => editDraft({ description: value })}
                    onInstructions={(value) => editDraft({ instructions: value })}
                    onSubmit={() => void save()}
                    aside={
                      <EditorAside
                        name={selected.name}
                        description={draft?.description ?? manifest?.description ?? selected.description}
                        instructions={draft?.instructions ?? manifest?.instructions ?? ""}
                        sourceText={
                          selected.source === "agents"
                            ? t("skills.fromAgents")
                            : selected.source === "claude"
                              ? t("skills.fromClaude")
                              : t("skills.fromBundled")
                        }
                        sourceChip={
                          <SourceChip label={sourceLabel(selected.source)} source={selected.source} />
                        }
                        path={manifest?.path ?? selected.path ?? null}
                        details={detailsOf(selected, t)}
                        onDelete={editable ? () => setConfirmingDelete(selected) : undefined}
                        deleting={pending === selected.name}
                      />
                    }
                  />
                )
              ) : null}
            </div>

          </>
        )}
      </DialogContent>

      <AlertDialog
        open={open && confirmingDelete !== null}
        onOpenChange={(next) => {
          if (!next) setConfirmingDelete(null);
        }}
      >
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>{t("skills.deleteTitle")}</AlertDialogTitle>
            <AlertDialogDescription>
              {t("skills.deleteDescription", { name: confirmingDelete?.name ?? "" })}
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>{t("common.cancel")}</AlertDialogCancel>
            <AlertDialogAction
              variant="destructive"
              onClick={() => {
                const skill = confirmingDelete;
                setConfirmingDelete(null);
                if (skill) void remove(skill);
              }}
            >
              {t("common.delete")}
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>

      <AlertDialog
        open={open && confirmingDiscard !== null}
        onOpenChange={(next) => {
          if (!next) setConfirmingDiscard(null);
        }}
      >
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>{t("skills.discardTitle")}</AlertDialogTitle>
            <AlertDialogDescription>
              {t("skills.discardDescription", {
                name: view.kind === "skill" ? (selected?.name ?? "") : t("skills.newSkill"),
              })}
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>{t("common.cancel")}</AlertDialogCancel>
            <AlertDialogAction
              variant="destructive"
              onClick={() => {
                const next = confirmingDiscard;
                setConfirmingDiscard(null);
                setDraft(null);
                setNewDraft(EMPTY_DRAFT);
                next?.();
              }}
            >
              {t("skills.discard")}
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </Dialog>
  );
}

function sectionSuffix(source: SkillRecord["source"]): "Agents" | "Claude" | "Bundled" {
  return source === "agents" ? "Agents" : source === "claude" ? "Claude" : "Bundled";
}

function detailsOf(
  skill: SkillRecord,
  t: ReturnType<typeof useT>,
): Array<[string, string]> {
  return [
    ...(skill.license ? [[t("skills.licenseLabel"), skill.license] as [string, string]] : []),
    ...(skill.compatibility
      ? [[t("skills.compatibilityLabel"), skill.compatibility] as [string, string]]
      : []),
    ...(skill.allowed_tools
      ? [[t("skills.allowedToolsLabel"), skill.allowed_tools] as [string, string]]
      : []),
    ...Object.entries(skill.metadata ?? {}).map(([key, value]) => [key, value] as [string, string]),
  ];
}

function Monogram({ name, on }: { name: string; on: boolean }): ReactElement {
  return (
    <span
      aria-hidden="true"
      className={cn(
        "flex size-10 shrink-0 items-center justify-center rounded-xl text-sm font-semibold tracking-wide transition-colors",
        on ? "bg-primary/10 text-primary" : "bg-muted text-muted-foreground",
      )}
    >
      {name ? (
        monogram(name)
      ) : (
        <HugeiconsIcon icon={Scroll01Icon} strokeWidth={1.75} className="size-4" />
      )}
    </span>
  );
}

function SourceChip({
  label,
  source,
}: {
  label: string;
  source: SkillRecord["source"];
}): ReactElement {
  return (
    <span
      className={cn(
        "inline-flex h-5 items-center rounded-full px-2 text-ui-11 font-medium",
        source === "agents"
          ? "bg-primary/10 text-primary"
          : "bg-muted text-muted-foreground",
      )}
    >
      {label}
    </span>
  );
}

function SkillCard({
  skill,
  changing,
  onOpen,
  onToggle,
}: {
  skill: SkillRecord;
  changing: boolean;
  onOpen: () => void;
  onToggle: (enabled: boolean) => void;
}): ReactElement {
  const t = useT();
  const usable = skill.valid && !skill.shadowed;
  const on = usable && skill.enabled;
  return (
    <div
      role="button"
      tabIndex={0}
      aria-label={skill.name}
      onClick={onOpen}
      onKeyDown={(event) => {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          onOpen();
        }
      }}
      className={cn(
        "group flex cursor-pointer flex-col gap-3 rounded-2xl border bg-card p-4 text-left transition-colors focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring",
        skill.valid ? "border-border/60 hover:border-border hover:bg-muted/30" : "border-destructive/30",
        skill.shadowed && "opacity-60",
      )}
    >
      <div className="flex items-start gap-3">
        <Monogram name={skill.name} on={on} />
        <div className="min-w-0 flex-1">
          <p className="truncate text-sm font-semibold tracking-tight">{skill.name}</p>
          {/* The section already says where a card comes from; chips are for what is unusual. */}
          {skill.shadowed || skill.linked || !skill.valid ? (
          <div className="mt-1 flex flex-wrap items-center gap-1.5">
            {skill.shadowed ? (
              <span className="inline-flex h-5 items-center rounded-full bg-muted px-2 text-ui-11 font-medium text-muted-foreground">
                {t("skills.shadowed")}
              </span>
            ) : null}
            {skill.linked ? (
              <span className="inline-flex h-5 items-center rounded-full bg-muted px-2 text-ui-11 font-medium text-muted-foreground">
                {t("skills.linked")}
              </span>
            ) : null}
            {skill.valid ? null : (
              <span className="inline-flex h-5 items-center rounded-full bg-destructive/10 px-2 text-ui-11 font-medium text-destructive">
                {t("skills.invalid")}
              </span>
            )}
          </div>
          ) : null}
        </div>
        <span
          className="shrink-0"
          onClick={(event) => event.stopPropagation()}
          onKeyDown={(event) => event.stopPropagation()}
        >
          <Switch
            checked={on}
            disabled={!usable || changing}
            aria-label={t(skill.enabled ? "skills.disable" : "skills.enable", { name: skill.name })}
            onCheckedChange={onToggle}
          />
        </span>
      </div>
      <p
        className={cn(
          "line-clamp-3 text-xs leading-relaxed",
          skill.valid ? "text-muted-foreground" : "text-destructive",
        )}
      >
        {skill.valid ? skill.description : skill.error}
      </p>
      <div className="mt-auto flex items-center gap-2 text-ui-11 text-muted-foreground/70">
        <span className="truncate font-mono">@{skill.name}</span>
        <span className="ml-auto inline-flex shrink-0 items-center gap-0.5 text-muted-foreground/60 transition-colors group-hover:text-foreground group-focus-visible:text-foreground">
          {isEditable(skill) ? t("skills.edit") : t("skills.view")}
          <HugeiconsIcon icon={ArrowRight01Icon} strokeWidth={2} className="size-3" />
        </span>
      </div>
    </div>
  );
}

function Field({
  htmlFor,
  label,
  hint,
  trailing,
  invalid,
  className,
  children,
}: {
  htmlFor: string;
  label: string;
  hint?: string;
  trailing?: ReactNode;
  invalid?: boolean;
  className?: string;
  children: ReactNode;
}): ReactElement {
  return (
    <div className={cn("flex flex-col gap-1.5", className)}>
      <div className="flex items-baseline justify-between gap-2">
        <label htmlFor={htmlFor} className="text-sm font-medium">
          {label}
        </label>
        {trailing}
      </div>
      {children}
      {hint ? (
        <p className={cn("text-xs", invalid ? "text-destructive" : "text-muted-foreground")}>
          {hint}
        </p>
      ) : null}
    </div>
  );
}

function Editor({
  formId,
  nameField,
  description,
  instructions,
  readOnly,
  disabled,
  notice,
  onDescription,
  onInstructions,
  onSubmit,
  aside,
}: {
  formId: string;
  name: string;
  nameField?: ReactNode;
  description: string;
  instructions: string;
  readOnly: boolean;
  disabled: boolean;
  notice?: { tone: "muted" | "error"; text: string } | null;
  onDescription: (value: string) => void;
  onInstructions: (value: string) => void;
  onSubmit: () => void;
  aside: ReactNode;
}): ReactElement {
  const t = useT();
  return (
    <div className="flex flex-col gap-6 p-4 sm:p-6 lg:min-h-0 lg:flex-1 lg:flex-row">
      <form
        id={formId}
        className="flex min-w-0 flex-1 flex-col gap-4 lg:min-h-0"
        onSubmit={(event) => {
          event.preventDefault();
          if (!readOnly) onSubmit();
        }}
        onKeyDown={(event) => {
          if (isChord(event) && !readOnly) {
            event.preventDefault();
            onSubmit();
          }
        }}
      >
        {notice ? (
          <p
            className={cn(
              "rounded-xl px-4 py-3 text-xs leading-relaxed",
              notice.tone === "error"
                ? "border border-destructive/30 bg-destructive/5 text-destructive"
                : "bg-muted/50 text-muted-foreground",
            )}
          >
            {notice.text}
          </p>
        ) : null}
        {nameField}
        <Field
          htmlFor={`${formId}-description`}
          label={t("skills.descriptionLabel")}
          hint={t("skills.descriptionHint")}
          trailing={
            <span className="text-ui-11 tabular-nums text-muted-foreground/60">
              {description.length}/1024
            </span>
          }
        >
          <Textarea
            id={`${formId}-description`}
            value={description}
            rows={2}
            maxLength={1024}
            readOnly={readOnly}
            disabled={disabled}
            aria-label={t("skills.descriptionLabel")}
            placeholder={t("skills.descriptionPlaceholder")}
            onChange={(event) => onDescription(event.target.value)}
            className={cn(
              "max-h-[min(12rem,30vh)] min-h-[calc(4.5rem*var(--ui-space-scale,1))] overflow-y-auto leading-relaxed",
              readOnly && "text-muted-foreground",
            )}
          />
        </Field>
        <Field
          htmlFor={`${formId}-instructions`}
          label={t("skills.instructionsLabel")}
          hint={t("skills.instructionsHint")}
          className="lg:min-h-0 lg:flex-1"
          trailing={
            <span className="text-ui-11 tabular-nums text-muted-foreground/60">
              {t("skills.characters", { count: instructions.length.toLocaleString() })}
            </span>
          }
        >
          <Textarea
            id={`${formId}-instructions`}
            value={instructions}
            fieldSizing="fixed"
            readOnly={readOnly}
            disabled={disabled}
            spellCheck={false}
            aria-label={t("skills.instructionsLabel")}
            placeholder={t("skills.instructionsPlaceholder")}
            onChange={(event) => onInstructions(event.target.value)}
            className={cn(
              "h-[min(26rem,50vh)] min-h-40 resize-y font-mono text-ui-13 leading-relaxed lg:h-auto lg:min-h-32 lg:flex-1 lg:resize-none",
              readOnly && "text-muted-foreground",
            )}
          />
        </Field>
      </form>
      <aside className="hover-scrollbar flex min-w-0 flex-col gap-3 lg:min-h-0 lg:w-[calc(280px*var(--ui-space-scale,1))] lg:shrink-0 lg:overflow-y-auto">
        {aside}
      </aside>
    </div>
  );
}

function EditorAside({
  name,
  description,
  instructions,
  sourceText,
  sourceChip,
  path,
  details,
  onDelete,
  deleting,
}: {
  name: string;
  description: string;
  instructions: string;
  sourceText: string;
  sourceChip: ReactNode;
  path?: string | null;
  details?: Array<[string, string]>;
  onDelete?: () => void;
  deleting?: boolean;
}): ReactElement {
  const t = useT();
  const { copied, copy } = useCopyFeedback();
  const shownName = name || t("skills.previewName");
  return (
    <>
      <section className="flex flex-col gap-3 rounded-2xl border border-border/60 bg-card p-4">
        <div>
          <p className="text-xs font-semibold">{t("skills.previewTitle")}</p>
          <p className="mt-0.5 text-ui-11 leading-relaxed text-muted-foreground">
            {t("skills.previewHint")}
          </p>
        </div>
        <div className="rounded-lg bg-muted/50 px-3 py-2.5 font-mono text-ui-11 leading-relaxed break-words">
          <span className="text-primary">{shownName}</span>
          <span className="text-muted-foreground">: </span>
          <span className={description ? "text-foreground" : "text-muted-foreground/60"}>
            {description || t("skills.descriptionPlaceholder")}
          </span>
        </div>
        <div className="flex items-center gap-2">
          <span className="text-ui-11 text-muted-foreground">{t("skills.mention")}</span>
          <button
            type="button"
            disabled={!name}
            onClick={() => void copy(`@${name}`)}
            aria-label={t("skills.copyMention", { name: shownName })}
            className="ml-auto inline-flex h-6 items-center gap-1.5 rounded-full bg-primary/10 px-2 font-mono text-ui-11 text-primary transition-colors hover:bg-primary/15 disabled:opacity-50"
          >
            @{shownName}
            <HugeiconsIcon icon={copied ? Tick02Icon : Copy01Icon} strokeWidth={2} className="size-3" />
          </button>
        </div>
      </section>
      <section className="flex flex-col gap-3 rounded-2xl border border-border/60 bg-card p-4 text-xs">
        <div className="flex items-start justify-between gap-3">
          <div className="min-w-0">
            <p className="font-semibold">{t("skills.sourceLabel")}</p>
            <p className="mt-0.5 leading-relaxed text-muted-foreground">{sourceText}</p>
          </div>
          {sourceChip}
        </div>
        {path ? (
          <div>
            <p className="font-semibold">{t("skills.locationLabel")}</p>
            <p className="mt-0.5 break-all font-mono text-ui-11 text-muted-foreground">{path}</p>
          </div>
        ) : null}
        <div>
          <p className="font-semibold">{t("skills.detailsTitle")}</p>
          {details && details.length > 0 ? (
            <dl className="mt-1 flex flex-col gap-1">
              {details.map(([label, value]) => (
                <div key={label} className="flex items-baseline justify-between gap-3">
                  <dt className="shrink-0 text-muted-foreground">{label}</dt>
                  <dd className="truncate text-right">{value}</dd>
                </div>
              ))}
            </dl>
          ) : (
            <p className="mt-0.5 text-muted-foreground">{t("skills.detailsEmpty")}</p>
          )}
        </div>
        <p className="text-ui-11 text-muted-foreground/70">
          {t("skills.characters", { count: instructions.length.toLocaleString() })}
        </p>
      </section>
      {onDelete ? (
        <section className="flex items-center justify-between gap-3 rounded-2xl border border-border/60 bg-card p-4 text-xs">
          <div className="min-w-0">
            <p className="font-semibold">{t("skills.deleteLabel")}</p>
            <p className="mt-0.5 leading-relaxed text-muted-foreground">
              {t("skills.deleteRowDescription")}
            </p>
          </div>
          <Button
            type="button"
            variant="outline"
            size="sm"
            disabled={deleting}
            onClick={onDelete}
            aria-label={t("skills.delete", { name })}
            className="shrink-0 text-destructive hover:text-destructive"
          >
            {t("common.delete")}
          </Button>
        </section>
      ) : null}
    </>
  );
}
