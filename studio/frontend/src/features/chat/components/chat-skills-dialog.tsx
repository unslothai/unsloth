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
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Input } from "@/components/ui/input";
import { Spinner } from "@/components/ui/spinner";
import { Switch } from "@/components/ui/switch";
import { Textarea } from "@/components/ui/textarea";
import { isModEnter } from "@/features/settings/lib/keyboard-shortcuts";
import { useT } from "@/i18n";
import {
  ChevronLeftStandardIcon,
  ChevronRightStandardIcon,
} from "@/lib/chevron-icons";
import { downloadFile, isDownloadCancelled } from "@/lib/native-files";
import { Tick02Icon } from "@/lib/tick-icon";
import { toast } from "@/lib/toast";
import { cn } from "@/lib/utils";
import {
  Cancel01Icon,
  Download01Icon,
  MoreHorizontalIcon,
  PlusSignIcon,
  Refresh01Icon,
  Scroll01Icon,
  Search01Icon,
  Upload01Icon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import {
  type ChangeEvent,
  type ReactElement,
  type ReactNode,
  useEffect,
  useId,
  useMemo,
  useRef,
  useState,
} from "react";
import {
  createSkill,
  deleteSkill,
  exportSkill,
  getSkillManifest,
  importSkills,
  isValidSkillName,
  listSkills,
  setAllSkillsEnabled,
  setSkillEnabled,
  type SkillDraft,
  type SkillImportResult,
  type SkillManifest,
  type SkillRecord,
  updateSkill,
  useSkillsCatalog,
} from "../api/skills-api";

type EditDraft = { description: string; instructions: string };
type View = { kind: "library" } | { kind: "new" } | { kind: "skill"; key: string };

const EMPTY_DRAFT: SkillDraft = { name: "", description: "", instructions: "" };
const LIBRARY: View = { kind: "library" };
const SECTIONS: ReadonlyArray<SkillRecord["source"]> = ["agents", "claude", "bundled"];
// `*` is the bulk-change sentinel because skill names cannot contain it.
const ALL_SKILLS = "*";

// Shadowed rows share a name, so skills are keyed by source too.
const keyOf = (skill: SkillRecord) => `${skill.source}:${skill.name}`;

function describe(cause: unknown): string | undefined {
  return cause instanceof Error ? cause.message : undefined;
}

function isEditable(skill: SkillRecord): boolean {
  return skill.valid && skill.source === "agents" && !skill.linked;
}

function ImportResultList({ imported, failed }: SkillImportResult): ReactElement {
  return (
    <ul className="flex max-h-48 flex-col gap-1 overflow-y-auto rounded-md border border-border/60 px-2.5 py-2">
      {/* Failures first, so the scroll cap never hides the rows that need attention. */}
      {failed.map((failure) => (
        <li key={`failed:${failure.name}`} className="min-w-0">
          <div className="flex min-w-0 items-center gap-1.5">
            <HugeiconsIcon
              icon={Cancel01Icon}
              strokeWidth={2}
              className="size-3.5 shrink-0 text-destructive"
            />
            <span className="truncate">{failure.name}</span>
          </div>
          <p className="pl-5 text-xs break-words text-muted-foreground">{failure.error}</p>
        </li>
      ))}
      {imported.map((skill) => (
        <li key={keyOf(skill)} className="flex min-w-0 items-center gap-1.5">
          <HugeiconsIcon
            icon={Tick02Icon}
            strokeWidth={2}
            className="size-3.5 shrink-0 text-emerald-600 dark:text-emerald-500"
          />
          <span className="truncate">{skill.name}</span>
        </li>
      ))}
    </ul>
  );
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
  const [view, setView] = useState<View>(LIBRARY);
  const [newDraft, setNewDraft] = useState<SkillDraft>(EMPTY_DRAFT);
  const [draft, setDraft] = useState<EditDraft | null>(null);
  const [manifests, setManifests] = useState<ReadonlyMap<string, SkillManifest>>(
    () => new Map(),
  );
  const inflight = useRef(new Set<string>());
  // Bumped on catalog change so a stale read cannot land after it.
  const manifestGeneration = useRef(0);
  const [pending, setPending] = useState<string | null>(null);
  const [creating, setCreating] = useState(false);
  const [importing, setImporting] = useState(false);
  const [exporting, setExporting] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [changing, setChanging] = useState<string | null>(null);
  const [confirmingDelete, setConfirmingDelete] = useState<SkillRecord | null>(null);
  const [confirmingDiscard, setConfirmingDiscard] = useState<(() => void) | null>(null);
  const [confirmingReset, setConfirmingReset] = useState(false);
  // Reseeded on render so the first frame after opening is already reset.
  const [seenOpen, setSeenOpen] = useState(open);
  if (open !== seenOpen) {
    setSeenOpen(open);
    if (open) {
      setSearchQuery("");
      setView(LIBRARY);
      setNewDraft(EMPTY_DRAFT);
      setDraft(null);
      setManifests(new Map());
      setConfirmingDelete(null);
      setConfirmingDiscard(null);
      setConfirmingReset(false);
    }
  }
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
  const narrowed = query !== "";
  const filtered = useMemo(
    () =>
      skills.filter(
        (skill) =>
          !query ||
          skill.name.toLowerCase().includes(query) ||
          skill.description.toLowerCase().includes(query),
      ),
    [skills, query],
  );
  const sections = SECTIONS.map((source) => ({
    source,
    skills: filtered.filter((skill) => skill.source === source),
  })).filter((section) => section.skills.length > 0 || (section.source === "agents" && !narrowed));

  const selected =
    view.kind === "skill" ? (skills.find((skill) => keyOf(skill) === view.key) ?? null) : null;
  if (open && view.kind === "skill" && !loading && selected === null) {
    setView(LIBRARY);
    setDraft(null);
  }
  const readable = selected !== null && selected.valid && !selected.shadowed;

  const readManifest = (skill: SkillRecord, replace = false) => {
    const name = skill.name;
    if (!skill.valid || skill.shadowed || (!replace && inflight.current.has(name))) return;
    inflight.current.add(name);
    const generation = manifestGeneration.current;
    getSkillManifest(name)
      .then((manifest) => {
        if (generation !== manifestGeneration.current) return;
        setManifests((prev) =>
          replace || !prev.has(name) ? new Map(prev).set(name, manifest) : prev,
        );
      })
      .catch((cause: unknown) => {
        if (generation !== manifestGeneration.current) return;
        toast.error(t("skills.openError"), { description: describe(cause) });
        setView((current) =>
          current.kind === "skill" && current.key === keyOf(skill) ? LIBRARY : current,
        );
      })
      .finally(() => {
        inflight.current.delete(name);
      });
  };

  // Only the winning row may use a name-keyed manifest; a shadowed copy shares the name.
  const manifest = readable ? (manifests.get(selected.name) ?? null) : null;

  const [seenSkills, setSeenSkills] = useState(skills);
  if (skills !== seenSkills) {
    setSeenSkills(skills);
    setManifests((prev) => {
      const kept = selected ? prev.get(selected.name) : undefined;
      return kept && selected ? new Map([[selected.name, kept]]) : new Map();
    });
  }
  useEffect(() => {
    manifestGeneration.current += 1;
    if (selected) readManifest(selected, true);
    // eslint-disable-next-line react-hooks/exhaustive-deps -- keyed on the catalog alone
  }, [skills]);
  const editable = selected !== null && isEditable(selected);
  const newStarted =
    newDraft.name !== "" || newDraft.description !== "" || newDraft.instructions !== "";
  const dirty = view.kind === "new" ? newStarted : draft !== null;
  const busy = pending !== null || creating || importing || exporting;

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

  const usable = skills.filter((skill) => skill.valid && !skill.shadowed);
  const toggleAll = async (enabled: boolean | null) => {
    setChanging(ALL_SKILLS);
    try {
      await setAllSkillsEnabled(enabled);
    } catch (cause) {
      toast.error(t("skills.updateError"), { description: describe(cause) });
    } finally {
      setChanging(null);
    }
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

  async function importFile(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    // Lets the same file fire change again.
    event.target.value = "";
    if (!file || importing) return;
    setImporting(true);
    try {
      const { imported, failed } = await importSkills(file);
      const count = imported.length;
      if (failed.length === 0) {
        toast.success(t(count === 1 ? "skills.importedOne" : "skills.importedMany", { count }), {
          description: <ImportResultList imported={imported} failed={failed} />,
        });
        if (count === 1) {
          // Never pull the user out of a skill they opened during the upload.
          setView((current) =>
            current.kind === "library" ? { kind: "skill", key: keyOf(imported[0]) } : current,
          );
          readManifest(imported[0]);
        }
      } else if (count === 0 && failed.length === 1) {
        toast.error(t("skills.importError"), { description: failed[0].error });
      } else {
        const notify = count > 0 ? toast.warning : toast.error;
        notify(t("skills.importPartial", { added: count, total: count + failed.length }), {
          description: <ImportResultList imported={imported} failed={failed} />,
        });
      }
    } catch (cause) {
      toast.error(t("skills.importError"), { description: describe(cause) });
    } finally {
      setImporting(false);
    }
  }

  async function exportSelected() {
    if (!selected || exporting) return;
    const name = selected.name;
    setExporting(true);
    try {
      const blob = await exportSkill(name);
      await downloadFile(blob, `${name}.zip`, "application/zip");
    } catch (cause) {
      if (!isDownloadCancelled(cause)) {
        toast.error(t("skills.exportError"), { description: describe(cause) });
      }
    } finally {
      setExporting(false);
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

  const sourceBadge = (source: SkillRecord["source"]) => (
    <Badge variant={source === "agents" ? "secondary" : "outline"}>{sourceLabel(source)}</Badge>
  );

  return (
    <Dialog open={open} onOpenChange={handleOpenChange}>
      <DialogContent className="rounded-3xl shadow-border ring-0 max-sm:flex max-sm:flex-col max-sm:overflow-hidden sm:max-w-2xl">
        {view.kind === "library" ? (
          <>
            <DialogHeader>
              <div className="flex items-center gap-2">
                <HugeiconsIcon icon={Scroll01Icon} strokeWidth={1.75} className="size-5 text-primary" />
                <DialogTitle>{t("skills.title")}</DialogTitle>
              </div>
              <DialogDescription>{t("skills.description")}</DialogDescription>
            </DialogHeader>

            {/* Phones: search takes its own row, the actions wrap under it. */}
            <div className="flex items-center gap-2 max-sm:flex-wrap max-sm:justify-end">
              <div className="relative min-w-0 flex-1 max-sm:basis-full">
                <HugeiconsIcon
                  icon={Search01Icon}
                  strokeWidth={2}
                  className="pointer-events-none absolute top-1/2 left-3 size-3.5 -translate-y-1/2 text-muted-foreground/60"
                />
                <Input
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
                  className="h-8 pl-8 text-sm"
                />
              </div>
              <Button
                type="button"
                size="icon-sm"
                variant="ghost"
                disabled={loading}
                onClick={refresh}
                aria-label={t("skills.refresh")}
                title={t("skills.refresh")}
              >
                {loading ? <Spinner /> : <HugeiconsIcon icon={Refresh01Icon} strokeWidth={2} />}
              </Button>
              <DropdownMenu>
                <DropdownMenuTrigger asChild={true}>
                  <Button
                    type="button"
                    size="icon-sm"
                    variant="ghost"
                    disabled={loading || usable.length === 0 || changing !== null}
                    aria-label={t("skills.bulkActions")}
                    title={t("skills.bulkActions")}
                  >
                    {changing === ALL_SKILLS ? (
                      <Spinner />
                    ) : (
                      <HugeiconsIcon icon={MoreHorizontalIcon} strokeWidth={2} />
                    )}
                  </Button>
                </DropdownMenuTrigger>
                <DropdownMenuContent align="end">
                  <DropdownMenuItem
                    disabled={usable.every((skill) => skill.enabled)}
                    onSelect={() => void toggleAll(true)}
                  >
                    {t("skills.enableAll")}
                  </DropdownMenuItem>
                  <DropdownMenuItem
                    disabled={!usable.some((skill) => skill.enabled)}
                    onSelect={() => void toggleAll(false)}
                  >
                    {t("skills.disableAll")}
                  </DropdownMenuItem>
                  <DropdownMenuSeparator />
                  <DropdownMenuItem onSelect={() => setConfirmingReset(true)}>
                    {t("skills.resetAll")}
                  </DropdownMenuItem>
                </DropdownMenuContent>
              </DropdownMenu>
              <input
                ref={fileInputRef}
                type="file"
                accept=".zip,application/zip"
                className="hidden"
                onChange={(event) => void importFile(event)}
              />
              <Button
                type="button"
                size="sm"
                variant="outline"
                disabled={importing}
                // Icon-only below sm, where the icon sets no height: a circle as tall as New skill.
                className="max-sm:size-[calc(1lh_+_14px*var(--ui-space-scale,1)_+_2px)] max-sm:p-0"
                aria-label={t("skills.import")}
                title={t("skills.import")}
                onClick={() => fileInputRef.current?.click()}
              >
                {importing ? <Spinner /> : <HugeiconsIcon icon={Upload01Icon} strokeWidth={2} />}
                <span className="max-sm:hidden">{t("skills.import")}</span>
              </Button>
              <Button variant="secondary" type="button" size="sm" onClick={openNew}>
                <HugeiconsIcon icon={PlusSignIcon} strokeWidth={2} />
                {t("skills.newSkill")}
              </Button>
            </div>

            {/* -mr-7 pr-7 spans the dialog's right padding, so the scrollbar sits on its edge. */}
            <div className="hover-scrollbar min-h-0 max-h-[min(58dvh,520px)] -mr-7 space-y-5 overflow-y-auto pr-7 max-sm:flex-1 max-sm:max-h-none">
              {error ? (
                <div className="rounded-xl border border-destructive/30 bg-destructive/5 p-4 text-sm text-destructive">
                  {error}
                </div>
              ) : null}
              {sections.some((section) => section.skills.length > 0) ? (
                sections
                  .filter((section) => section.skills.length > 0)
                  .map((section) => (
                    <section key={section.source} className="space-y-3">
                      <div className="flex min-w-0 items-baseline gap-2 px-1">
                        <h3 className="shrink-0 text-xs font-medium text-muted-foreground">
                          {t(`skills.section${sectionSuffix(section.source)}`)}
                        </h3>
                        <p className="truncate text-ui-11 text-muted-foreground/70">
                          {t(`skills.section${sectionSuffix(section.source)}Hint`)}
                        </p>
                      </div>
                      {section.skills.map((skill) => (
                        <SkillRow
                          key={keyOf(skill)}
                          skill={skill}
                          changing={changing !== null}
                          onOpen={() => openSkill(skill)}
                          onToggle={(enabled) => void toggle(skill.name, enabled)}
                        />
                      ))}
                    </section>
                  ))
              ) : loading ? (
                <div className="flex justify-center p-6">
                  <Spinner />
                </div>
              ) : narrowed ? (
                <div className="rounded-xl border border-dashed p-6 text-center text-sm text-muted-foreground">
                  {t("skills.noMatch", { query: searchQuery.trim() })}{" "}
                  <button
                    type="button"
                    onClick={() => setSearchQuery("")}
                    className="text-primary hover:underline"
                  >
                    {t("skills.clearSearch")}
                  </button>
                </div>
              ) : (
                <div className="rounded-xl border border-dashed p-6 text-center text-sm text-muted-foreground">
                  {t("skills.empty")}
                </div>
              )}
            </div>
          </>
        ) : (
          <>
            <DialogHeader className="pr-8">
              <div className="flex min-w-0 items-center gap-2">
                <Button
                  type="button"
                  variant="ghost"
                  size="icon-sm"
                  disabled={busy}
                  onClick={goLibrary}
                  aria-label={t("skills.title")}
                  className="-ml-2 shrink-0"
                >
                  <HugeiconsIcon icon={ChevronLeftStandardIcon} strokeWidth={2} />
                </Button>
                <DialogTitle className="truncate">
                  {view.kind === "new" ? t("skills.newSkill") : (selected?.name ?? "")}
                </DialogTitle>
                {selected ? sourceBadge(selected.source) : null}
                {dirty && view.kind === "skill" ? (
                  <span className="shrink-0 text-xs text-muted-foreground">{t("skills.unsaved")}</span>
                ) : null}
              </div>
              <DialogDescription className="truncate">
                {view.kind === "new"
                  ? t("skills.createDescription")
                  : (manifest?.path ?? selected?.path ?? sourceLabel(selected?.source ?? "agents"))}
              </DialogDescription>
            </DialogHeader>

            <div className="hover-scrollbar min-h-0 max-h-[min(62dvh,640px)] -mr-7 overflow-y-auto pr-7 max-sm:flex-1 max-sm:max-h-none">
              {view.kind === "new" ? (
                <Editor
                  formId="skill-new-form"
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
                        className="font-mono text-sm"
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
                />
              ) : selected ? (
                readable && !manifest ? (
                  <div className="flex justify-center p-10">
                    <Spinner />
                  </div>
                ) : (
                  <Editor
                    formId="skill-edit-form"
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
                    details={detailsOf(selected, t)}
                    onSubmit={() => void save()}
                  />
                )
              ) : null}
            </div>

            <div className="flex flex-wrap items-center gap-2">
              {selected ? (
                <label className="flex items-center gap-2 text-sm text-muted-foreground">
                  <Switch
                    checked={selected.valid && !selected.shadowed && selected.enabled}
                    disabled={
                      !selected.valid ||
                      selected.shadowed ||
                      changing !== null
                    }
                    aria-label={t(selected.enabled ? "skills.disable" : "skills.enable", {
                      name: selected.name,
                    })}
                    onCheckedChange={(checked) => void toggle(selected.name, checked)}
                  />
                  {t("skills.enabledLabel")}
                </label>
              ) : null}
              <div className="ml-auto flex items-center gap-2">
                {readable ? (
                  <Button
                    type="button"
                    variant="outline"
                    disabled={busy}
                    onClick={() => void exportSelected()}
                  >
                    {exporting ? <Spinner /> : <HugeiconsIcon icon={Download01Icon} strokeWidth={2} />}
                    {t("common.export")}
                  </Button>
                ) : null}
                {view.kind === "new" ? (
                  <>
                    <Button type="button" variant="ghost" disabled={creating} onClick={goLibrary}>
                      {t("common.cancel")}
                    </Button>
                    <Button variant="secondary" type="submit" form="skill-new-form" disabled={creating || !canCreate}>
                      <HugeiconsIcon icon={PlusSignIcon} data-icon="inline-start" />
                      {creating ? <Spinner /> : null}
                      {t("skills.create")}
                    </Button>
                  </>
                ) : editable && selected ? (
                  <>
                    <Button
                      type="button"
                      variant="destructive"
                      disabled={pending !== null}
                      onClick={() => setConfirmingDelete(selected)}
                      aria-label={t("skills.delete", { name: selected.name })}
                      className="text-destructive hover:text-destructive"
                    >
                      {t("common.delete")}
                    </Button>
                    <Button
                      type="button"
                      variant="outline"
                      disabled={!dirty || pending !== null}
                      onClick={() => setDraft(null)}
                    >
                      {t("skills.revert")}
                    </Button>
                    <Button type="submit" form="skill-edit-form" disabled={pending !== null || !canSave}>
                      {pending !== null ? <Spinner /> : null}
                      {t("skills.save")}
                    </Button>
                  </>
                ) : null}
              </div>
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

      <AlertDialog open={open && confirmingReset} onOpenChange={setConfirmingReset}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>{t("skills.resetTitle")}</AlertDialogTitle>
            <AlertDialogDescription>{t("skills.resetDescription")}</AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>{t("common.cancel")}</AlertDialogCancel>
            <AlertDialogAction
              variant="destructive"
              onClick={() => {
                setConfirmingReset(false);
                void toggleAll(null);
              }}
            >
              {t("skills.reset")}
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

function SkillRow({
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
  const descriptionId = useId();
  return (
    <div
      className={cn(
        "group pointer-events-none relative grid grid-cols-[minmax(0,1fr)_auto] items-center gap-x-10 gap-y-1.5 rounded-lg border border-border/60 bg-muted/20 px-5 py-4 transition-colors hover:bg-muted/50 dark:border-transparent dark:bg-[rgb(255_255_255_/_calc(0.06*var(--contrast-wash-gain,1)))]",
        skill.shadowed && "opacity-60",
      )}
    >
      <button
        type="button"
        onClick={onOpen}
        aria-label={skill.name}
        aria-describedby={descriptionId}
        title={(skill.valid ? skill.description : skill.error) ?? undefined}
        className="pointer-events-auto absolute inset-0 cursor-pointer rounded-xl focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
      />
      <div className="flex min-w-0 flex-wrap items-center gap-2">
        <span className="truncate font-medium text-ui-14">{skill.name}</span>
        <HugeiconsIcon
          icon={ChevronRightStandardIcon}
          strokeWidth={2}
          aria-hidden="true"
          className="-ml-1 size-4 shrink-0 text-muted-foreground/50 transition-colors group-hover:text-foreground"
        />
        {skill.shadowed ? <Badge variant="secondary">{t("skills.shadowed")}</Badge> : null}
        {skill.linked ? <Badge variant="outline">{t("skills.linked")}</Badge> : null}
        {skill.valid ? null : <Badge variant="destructive">{t("skills.invalid")}</Badge>}
      </div>
      <Switch
        className="pointer-events-auto self-start mt-[calc(var(--text-ui-14)*0.3865)] text-ui-14"
        checked={usable && skill.enabled}
        disabled={!usable || changing}
        aria-label={t(skill.enabled ? "skills.disable" : "skills.enable", { name: skill.name })}
        onCheckedChange={onToggle}
      />
      <p
        id={descriptionId}
        className={cn(
          "line-clamp-2 text-ui-13",
          skill.valid ? "text-muted-foreground" : "text-destructive",
        )}
      >
        {skill.valid ? skill.description : skill.error}
      </p>
    </div>
  );
}

// Fill and padding live on the wrapper so the scrollbar clears the rounded corners.
function ScrollField({ htmlFor, children }: { htmlFor: string; children: ReactNode }): ReactElement {
  return (
    <label
      htmlFor={htmlFor}
      className="block cursor-text overflow-hidden rounded-xl border border-border bg-background py-3 pr-1.5 pl-3.5 transition-colors focus-within:border-ring dark:border-transparent dark:bg-[rgb(255_255_255_/_calc(0.06*var(--contrast-wash-gain,1)))] dark:focus-within:border-transparent dark:focus-within:bg-[rgb(255_255_255_/_calc(0.12*var(--contrast-wash-gain,1)))]"
    >
      {children}
    </label>
  );
}

const SCROLL_FIELD_TEXTAREA =
  "min-h-0 resize-none rounded-none border-0 bg-transparent p-0 pr-2 focus-visible:bg-transparent dark:bg-transparent dark:focus-visible:bg-transparent";

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
  details,
  onDescription,
  onInstructions,
  onSubmit,
}: {
  formId: string;
  nameField?: ReactNode;
  description: string;
  instructions: string;
  readOnly: boolean;
  disabled: boolean;
  notice?: { tone: "muted" | "error"; text: string } | null;
  details?: Array<[string, string]>;
  onDescription: (value: string) => void;
  onInstructions: (value: string) => void;
  onSubmit: () => void;
}): ReactElement {
  const t = useT();
  return (
    <form
      id={formId}
      className="flex flex-col gap-4 py-1"
      onSubmit={(event) => {
        event.preventDefault();
        if (!readOnly) onSubmit();
      }}
      onKeyDown={(event) => {
        if (isModEnter(event) && !readOnly) {
          event.preventDefault();
          onSubmit();
        }
      }}
    >
      {notice ? (
        <p
          className={cn(
            "rounded-lg px-4 py-3 text-xs leading-relaxed",
            notice.tone === "error"
              ? "border border-destructive/30 bg-destructive/5 text-destructive"
              : "bg-muted/50 text-muted-foreground",
          )}
        >
          {notice.text}
        </p>
      ) : null}
      {details && details.length > 0 ? (
        <dl className="flex flex-wrap gap-x-6 gap-y-1 text-xs">
          {details.map(([label, value]) => (
            <div key={label} className="flex min-w-0 gap-2">
              <dt className="shrink-0 text-muted-foreground">{label}</dt>
              <dd className="truncate">{value}</dd>
            </div>
          ))}
        </dl>
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
        <ScrollField htmlFor={`${formId}-description`}>
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
              SCROLL_FIELD_TEXTAREA,
              "max-h-[min(12rem,30vh)] min-h-[calc(3rem*var(--ui-space-scale,1))] overflow-y-auto leading-relaxed",
              readOnly && "text-muted-foreground",
            )}
          />
        </ScrollField>
      </Field>
      <Field
        htmlFor={`${formId}-instructions`}
        label={t("skills.instructionsLabel")}
        hint={t("skills.instructionsHint")}
        trailing={
          <span className="text-ui-11 tabular-nums text-muted-foreground/60">
            {t("skills.characters", { count: instructions.length.toLocaleString() })}
          </span>
        }
      >
        <ScrollField htmlFor={`${formId}-instructions`}>
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
              SCROLL_FIELD_TEXTAREA,
              "h-[min(13.5rem,29dvh)] min-h-22 font-mono text-ui-12 leading-relaxed md:text-ui-12",
              readOnly && "text-muted-foreground",
            )}
          />
        </ScrollField>
      </Field>
    </form>
  );
}
