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
// SPDX-License-Identifier: AGPL-3.0-only
import { Button } from "@/components/ui/button";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Switch } from "@/components/ui/switch";
import { Textarea } from "@/components/ui/textarea";
import { authFetch } from "@/features/auth";
import { refreshContextUsage, useChatRuntimeStore } from "@/features/chat";
import { listChatProjects } from "@/features/chat/api/chat-api";
import { type ReactNode, useEffect, useId, useRef, useState } from "react";
import { useSettingsDialogStore } from "../stores/settings-dialog-store";
import { SettingsRow } from "./settings-row";
import { SettingsSection } from "./settings-section";

type Memory = {
  enabled: boolean;
  thread_enabled: boolean;
  revision: number;
  used_bytes: number;
  max_bytes: number;
  entries: Record<string, { value: string }>;
};

type MemoryDraft = { kind: "new" | "edit"; key: string; value: string };

function newMemoryDraft(): MemoryDraft {
  return {
    kind: "new",
    key: `note_${crypto.randomUUID().replaceAll("-", "").slice(0, 12)}`,
    value: "",
  };
}
function parseMemory(body: unknown): Memory {
  if (
    typeof body !== "object" ||
    body === null ||
    !("enabled" in body) ||
    typeof body.enabled !== "boolean" ||
    !("thread_enabled" in body) ||
    typeof body.thread_enabled !== "boolean" ||
    !("revision" in body) ||
    typeof body.revision !== "number" ||
    !("used_bytes" in body) ||
    typeof body.used_bytes !== "number" ||
    !("max_bytes" in body) ||
    typeof body.max_bytes !== "number" ||
    !("entries" in body) ||
    typeof body.entries !== "object" ||
    body.entries === null
  ) {
    throw new Error("Invalid memory response.");
  }
  const entries: Memory["entries"] = {};
  for (const [key, entry] of Object.entries(body.entries)) {
    if (
      typeof entry !== "object" ||
      entry === null ||
      !("value" in entry) ||
      typeof entry.value !== "string"
    )
      throw new Error("Invalid memory fact.");
    entries[key] = { value: entry.value };
  }
  return {
    enabled: body.enabled,
    thread_enabled: body.thread_enabled,
    revision: body.revision,
    used_bytes: body.used_bytes,
    max_bytes: body.max_bytes,
    entries,
  };
}

async function requestMemory(query: string, patch?: object): Promise<Memory> {
  const response = await authFetch(
    `/api/settings/learned-memory?${query}`,
    patch
      ? {
          method: "PUT",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(patch),
        }
      : undefined,
  );
  if (!response.ok) {
    const body: unknown = await response.json();
    throw new Error(
      typeof body === "object" &&
        body !== null &&
        "detail" in body &&
        typeof body.detail === "string"
        ? body.detail
        : "Could not save memory.",
    );
  }
  return parseMemory(await response.json());
}

export function LearnedMemorySection() {
  const ref = useRef<HTMLDivElement>(null);
  const scrollTarget = useSettingsDialogStore((state) => state.scrollTarget);
  useEffect(() => {
    if (scrollTarget !== "chat-memory") return;
    const frame = requestAnimationFrame(() => {
      ref.current?.scrollIntoView({ block: "start", behavior: "smooth" });
      useSettingsDialogStore.getState().consumeScrollTarget("chat-memory");
    });
    return () => cancelAnimationFrame(frame);
  }, [scrollTarget]);
  const threadId = useChatRuntimeStore((state) => state.activeThreadId);
  const incognito = useChatRuntimeStore((state) => state.incognito);
  const [projectId, setProjectId] = useState("");
  const [projects, setProjects] = useState<
    Awaited<ReturnType<typeof listChatProjects>>
  >([]);
  useEffect(() => {
    void listChatProjects()
      .then(setProjects)
      .catch(() => {});
  }, []);
  return (
    <div ref={ref}>
      <SettingsSection
        title="Memory"
        description="Remember your preferences and project details across chats."
      >
        <MemoryEditor
          key={`${projectId}:${threadId}:${incognito}`}
          projectId={projectId}
          threadId={incognito ? undefined : (threadId ?? undefined)}
          scopeControl={
            <Select
              value={projectId || "personal"}
              onValueChange={(value) =>
                setProjectId(value === "personal" ? "" : value)
              }
            >
              <SelectTrigger aria-label="Memories for" className="w-48">
                <SelectValue />
              </SelectTrigger>
              <SelectContent align="end">
                <SelectItem value="personal">Personal</SelectItem>
                {projects.map((project) => (
                  <SelectItem key={project.id} value={project.id}>
                    {project.name}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          }
        />
      </SettingsSection>
    </div>
  );
}

function MemoryEditor({
  projectId,
  threadId,
  scopeControl,
}: { projectId: string; threadId?: string; scopeControl: ReactNode }) {
  const factId = useId();
  const [memory, setMemory] = useState<Memory | null>(null);
  const [draft, setDraft] = useState(newMemoryDraft);
  const [confirmClear, setConfirmClear] = useState(false);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [omitted, setOmitted] = useState(0);
  const contextLength = useChatRuntimeStore(
    (state) => state.loadedContextLength ?? 4096,
  );
  const query = new URLSearchParams({
    ...(projectId ? { project_id: projectId } : {}),
    ...(threadId ? { thread_id: threadId } : {}),
  }).toString();
  useEffect(() => {
    let cancelled = false;
    void requestMemory(query)
      .then((result) => {
        if (!cancelled) setMemory(result);
      })
      .catch((cause: unknown) => {
        if (!cancelled) setError(String(cause));
      });
    return () => {
      cancelled = true;
    };
  }, [query]);
  useEffect(() => {
    let cancelled = false;
    setOmitted(0);
    if (threadId && memory?.enabled && memory.thread_enabled)
      void authFetch(
        `/api/chat/memory-context?${new URLSearchParams({
          thread_id: threadId,
          context_length: String(contextLength),
        })}`,
      )
        .then((response) => response.json())
        .then((body: unknown) => {
          if (
            !cancelled &&
            typeof body === "object" &&
            body !== null &&
            "omitted" in body &&
            typeof body.omitted === "number"
          )
            setOmitted(body.omitted);
        })
        .catch(() => {});
    return () => {
      cancelled = true;
    };
  }, [
    threadId,
    contextLength,
    memory?.enabled,
    memory?.thread_enabled,
    memory?.revision,
  ]);
  async function save(patch: object) {
    if (!memory || busy) return false;
    setBusy(true);
    setError("");
    try {
      setMemory(
        await requestMemory(query, { revision: memory.revision, ...patch }),
      );
      void refreshContextUsage({ invalidate: true });
      return true;
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : String(cause));
      try {
        setMemory(await requestMemory(query));
      } catch {
        /* Keep the last readable state. */
      }
      return false;
    } finally {
      setBusy(false);
    }
  }
  async function saveDraft() {
    if (await save({ key: draft.key, value: draft.value })) {
      setDraft(newMemoryDraft());
    }
  }
  async function deleteMemory(key: string) {
    if ((await save({ key, value: null })) && key === draft.key) {
      setDraft(newMemoryDraft());
    }
  }
  const count = Object.keys(memory?.entries ?? {}).length;
  return (
    <div className="flex flex-col">
      <SettingsRow
        label="Use memory"
        description="Use saved memories and remember useful details as you chat. Turning this off keeps your saved memories."
      >
        <Switch
          aria-label="Use memory"
          disabled={!memory || busy}
          checked={memory?.enabled ?? false}
          onCheckedChange={(enabled) => void save({ enabled })}
        />
      </SettingsRow>
      {threadId && (
        <SettingsRow
          label="Memory in this chat"
          description="Controls reading and saving facts in the current chat."
        >
          <Switch
            aria-label="Memory in this chat"
            disabled={!memory?.enabled || busy}
            checked={Boolean(memory?.enabled && memory.thread_enabled)}
            onCheckedChange={(threadEnabled) =>
              void save({ thread_enabled: threadEnabled })
            }
          />
        </SettingsRow>
      )}
      <SettingsRow label="Memories for">{scopeControl}</SettingsRow>
      {omitted > 0 && (
        <p aria-live="polite" className="text-xs">
          {omitted} older memories won't be included in this chat because its
          context is small.
        </p>
      )}
      {memory &&
        Object.entries(memory.entries).map(([entryKey, entry]) => (
          <div
            key={entryKey}
            className="flex flex-wrap items-start gap-2 border-b border-border/60 py-3"
          >
            <div className="min-w-0 flex-1 basis-48">
              <p className="whitespace-pre-wrap break-words text-sm">
                {entry.value}
              </p>
            </div>
            <Button
              variant="ghost"
              size="sm"
              disabled={busy}
              onClick={() => {
                setDraft({ kind: "edit", key: entryKey, value: entry.value });
                document.getElementById(factId)?.focus();
              }}
            >
              Edit
            </Button>
            <Button
              variant="ghost"
              size="sm"
              disabled={busy}
              onClick={() => void deleteMemory(entryKey)}
            >
              Delete
            </Button>
          </div>
        ))}
      {memory && count === 0 && (
        <p className="py-3 text-xs text-muted-foreground">
          No saved memories yet. Add one below, or ask the model to remember
          something when memory is on.
        </p>
      )}
      <div className="flex flex-col gap-2 py-3">
        <label htmlFor={factId} className="text-sm font-medium">
          {draft.kind === "edit"
            ? "Edit memory"
            : "What would you like remembered?"}
        </label>
        <Textarea
          id={factId}
          fieldSizing="fixed"
          rows={3}
          className="max-h-[calc(240px*var(--ui-space-scale,1))] resize-y text-sm md:text-sm"
          placeholder="e.g. I prefer concise answers with code examples."
          value={draft.value}
          disabled={!memory || busy}
          onChange={(event) =>
            setDraft({ ...draft, value: event.target.value })
          }
        />
        <div className="flex flex-wrap items-center justify-between gap-x-4 gap-y-2">
          <span className="text-xs tabular-nums text-muted-foreground">
            {count} {count === 1 ? "memory" : "memories"}
            {memory && count > 0
              ? ` · ${Math.round((memory.used_bytes / memory.max_bytes) * 100)}% full`
              : ""}
          </span>
          <div className="ml-auto flex items-center gap-2">
            {(draft.kind === "edit" || draft.value) && (
              <Button
                variant="ghost"
                disabled={busy}
                onClick={() => setDraft(newMemoryDraft())}
              >
                Cancel
              </Button>
            )}
            <Button
              variant="outline"
              disabled={!memory || busy || !draft.value.trim()}
              onClick={() => void saveDraft()}
            >
              {busy
                ? "Saving…"
                : draft.kind === "edit"
                  ? "Save changes"
                  : "Add memory"}
            </Button>
          </div>
        </div>
        <p className="text-xs text-muted-foreground">
          {memory && !memory.enabled
            ? "Memory is off. Turn it on to use these in your chats. "
            : ""}
          Deleting a chat doesn't delete its saved memories. Temporary chats
          never use memory.
        </p>
        {count > 0 && (
          <div className="flex justify-end">
            <Button
              variant="ghost"
              size="sm"
              disabled={busy}
              onClick={() => setConfirmClear(true)}
            >
              Delete all memories
            </Button>
          </div>
        )}
      </div>
      {error && (
        <p role="alert" className="text-sm text-destructive">
          {error}
        </p>
      )}
      <AlertDialog open={confirmClear} onOpenChange={setConfirmClear}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>
              Delete all {projectId ? "project" : "personal"} memories?
            </AlertDialogTitle>
            <AlertDialogDescription>
              This removes all saved memories{" "}
              {projectId ? "for this project" : "from your personal memory"}.
              Your chats and custom instructions will stay. This can't be
              undone.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Cancel</AlertDialogCancel>
            <AlertDialogAction
              variant="destructive"
              onClick={() => {
                void save({ clear: true }).then((saved) => {
                  if (saved) setDraft(newMemoryDraft());
                });
              }}
            >
              Delete memories
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </div>
  );
}
