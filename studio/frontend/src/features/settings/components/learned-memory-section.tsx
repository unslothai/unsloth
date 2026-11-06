// SPDX-License-Identifier: AGPL-3.0-only
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
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
        description="Let the model remember durable preferences and project facts using a tool during chat. Saved facts survive chat deletion; deleting a project removes its memory."
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
              <SelectTrigger aria-label="Memory scope" className="w-48">
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
  const keyId = useId();
  const factId = useId();
  const [memory, setMemory] = useState<Memory | null>(null);
  const [key, setKey] = useState("");
  const [value, setValue] = useState("");
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
    if (!memory) return;
    setBusy(true);
    setError("");
    try {
      setMemory(
        await requestMemory(query, { revision: memory.revision, ...patch }),
      );
      void refreshContextUsage({ invalidate: true });
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : String(cause));
      try {
        setMemory(await requestMemory(query));
      } catch {
        /* Keep the last readable state. */
      }
    } finally {
      setBusy(false);
    }
  }
  return (
    <div className="flex flex-col">
      <SettingsRow
        label="Use memory"
        description="Adds saved facts and the memory tool to chats. Turning this off keeps your saved facts."
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
      <SettingsRow label="Memory scope">{scopeControl}</SettingsRow>
      {omitted > 0 && (
        <p aria-live="polite" className="text-xs">
          {omitted} saved facts do not fit in the current chat memory allowance.
        </p>
      )}
      {memory &&
        Object.entries(memory.entries).map(([entryKey, entry]) => (
          <div
            key={entryKey}
            className="flex flex-wrap items-start gap-2 border-b border-border/60 py-3"
          >
            <div className="min-w-0 flex-1 basis-48">
              <p className="text-xs font-medium">{entryKey}</p>
              <p className="whitespace-pre-wrap break-words text-sm">
                {entry.value}
              </p>
            </div>
            <Button
              variant="ghost"
              size="sm"
              disabled={busy}
              onClick={() => {
                setKey(entryKey);
                setValue(entry.value);
              }}
            >
              Edit
            </Button>
            <Button
              variant="ghost"
              size="sm"
              disabled={busy}
              onClick={() => void save({ key: entryKey, value: null })}
            >
              Delete
            </Button>
          </div>
        ))}
      <SettingsRow label="Memory key">
        <label htmlFor={keyId} className="sr-only">
          Memory key
        </label>
        <Input
          id={keyId}
          className="w-64"
          placeholder="e.g. response_style"
          value={key}
          disabled={!memory || busy}
          onChange={(event) => setKey(event.target.value)}
          maxLength={64}
        />
      </SettingsRow>
      <div className="flex flex-col gap-2 pb-3">
        <label htmlFor={factId} className="text-sm font-medium">
          What to remember
        </label>
        <Textarea
          id={factId}
          aria-label="Memory fact"
          fieldSizing="fixed"
          rows={3}
          className="max-h-[calc(240px*var(--ui-space-scale,1))] resize-y text-sm md:text-sm"
          placeholder="A short, durable fact"
          value={value}
          disabled={!memory || busy}
          onChange={(event) => setValue(event.target.value)}
        />
        <div className="flex flex-wrap items-center justify-between gap-x-4 gap-y-2">
          <span className="text-xs tabular-nums text-muted-foreground">
            {memory?.used_bytes ?? 0} / {memory?.max_bytes ?? 2048} bytes
          </span>
          <div className="ml-auto flex items-center gap-2">
            <Button
              variant="ghost"
              disabled={
                !memory || busy || Object.keys(memory.entries).length === 0
              }
              onClick={() => void save({ clear: true })}
            >
              Clear this scope
            </Button>
            <Button
              variant="outline"
              disabled={!memory || busy || !key || !value.trim()}
              onClick={() => void save({ key, value })}
            >
              Save fact
            </Button>
          </div>
        </div>
        <p className="text-xs text-muted-foreground">
          Small contexts may omit older facts. Incognito chats never use memory.
        </p>
      </div>
      {error && (
        <p role="alert" className="text-sm text-destructive">
          {error}
        </p>
      )}
    </div>
  );
}
