import {
  createPlatformChat,
  createPlatformSession,
  listAllPlatformChats,
} from "@/integrations/platform-backend";
import { buildBackendChatExport } from "@/features/chat/api/chat-api";
import { ensureGeneralPlatformChat } from "@/features/chat/api/platform-chat-adapter";
import { db } from "@/features/chat/db";
import type {
  MessageRecord,
  ProjectRecord,
  ThreadRecord,
} from "@/features/chat/types";

const MIGRATION_VERSION = 1 as const;
const LEDGER_PREFIX = "rag-platform:chat-migration:v1";
const CUSTOM_SOURCE_TIMEOUT_MS = 5_000;
const PROJECT_MARKER_PREFIX = "[Rag Platform migration:v1:";

export interface LegacyChatMigrationSnapshot {
  exportedAt: string;
  projects: ProjectRecord[];
  threads: ThreadRecord[];
  messages: MessageRecord[];
  sourceWarnings: string[];
}

export interface ChatMigrationPlanRecord {
  legacyId: string;
  label: string;
  status: "pending" | "migrated";
  platformId?: string;
}

export interface ChatMigrationUnsupportedItem {
  kind: "message" | "project-field" | "thread-field";
  count: number;
  reason: string;
}

export interface PlatformChatMigrationPlan {
  version: typeof MIGRATION_VERSION;
  generatedAt: string;
  ownerId: string;
  snapshot: LegacyChatMigrationSnapshot;
  projects: ChatMigrationPlanRecord[];
  threads: ChatMigrationPlanRecord[];
  unsupported: ChatMigrationUnsupportedItem[];
  totals: {
    projects: number;
    threads: number;
    messages: number;
    alreadyMigrated: number;
    pending: number;
  };
}

export interface PlatformChatMigrationProgress {
  completed: number;
  total: number;
  current: string;
}

export interface PlatformChatMigrationResult {
  completedProjects: number;
  completedThreads: number;
  skipped: number;
  failures: Array<{ legacyId: string; label: string; reason: string }>;
  aborted: boolean;
}

interface MigrationLedger {
  version: typeof MIGRATION_VERSION;
  ownerId: string;
  projects: Record<string, string>;
  threads: Record<string, string>;
  updatedAt: string;
}

function storageKey(ownerId: string): string {
  return `${LEDGER_PREFIX}:${encodeURIComponent(ownerId || "anonymous")}`;
}

function emptyLedger(ownerId: string): MigrationLedger {
  return {
    version: MIGRATION_VERSION,
    ownerId,
    projects: {},
    threads: {},
    updatedAt: new Date(0).toISOString(),
  };
}

function loadLedger(ownerId: string): MigrationLedger {
  if (typeof localStorage === "undefined") return emptyLedger(ownerId);
  try {
    const parsed = JSON.parse(localStorage.getItem(storageKey(ownerId)) ?? "null") as
      | Partial<MigrationLedger>
      | null;
    if (parsed?.version !== MIGRATION_VERSION || parsed.ownerId !== ownerId) {
      return emptyLedger(ownerId);
    }
    return {
      ...emptyLedger(ownerId),
      ...parsed,
      projects: { ...(parsed.projects ?? {}) },
      threads: { ...(parsed.threads ?? {}) },
    };
  } catch {
    return emptyLedger(ownerId);
  }
}

function saveLedger(ledger: MigrationLedger): void {
  if (typeof localStorage === "undefined") return;
  localStorage.setItem(
    storageKey(ledger.ownerId),
    JSON.stringify({ ...ledger, updatedAt: new Date().toISOString() }),
  );
}

function mergeById<T extends { id: string }>(
  preferred: T[],
  fallback: T[],
): T[] {
  const merged = new Map<string, T>();
  for (const record of fallback) {
    if (record?.id) merged.set(record.id, record);
  }
  for (const record of preferred) {
    if (record?.id) merged.set(record.id, record);
  }
  return [...merged.values()];
}

function linkedAbortController(signal?: AbortSignal): AbortController {
  const controller = new AbortController();
  const abort = () => controller.abort(signal?.reason);
  if (signal?.aborted) abort();
  else signal?.addEventListener("abort", abort, { once: true });
  return controller;
}

export async function readLegacyChatMigrationSnapshot(
  signal?: AbortSignal,
): Promise<LegacyChatMigrationSnapshot> {
  const sourceWarnings: string[] = [];
  const [localThreads, localMessages] = await Promise.all([
    db.threads.toArray().catch(() => [] as ThreadRecord[]),
    db.messages.toArray().catch(() => [] as MessageRecord[]),
  ]);

  const customController = linkedAbortController(signal);
  const timeoutId = window.setTimeout(
    () => customController.abort(new DOMException("Timed out", "TimeoutError")),
    CUSTOM_SOURCE_TIMEOUT_MS,
  );
  let custom:
    | Awaited<ReturnType<typeof buildBackendChatExport>>
    | null = null;
  try {
    custom = await buildBackendChatExport(customController.signal);
  } catch {
    sourceWarnings.push(
      "Özel eski sohbet servisine ulaşılamadı; tarayıcıdaki yerel kayıtlar yine de tarandı.",
    );
  } finally {
    window.clearTimeout(timeoutId);
  }

  return {
    exportedAt: new Date().toISOString(),
    projects: mergeById(custom?.projects ?? [], []),
    threads: mergeById(custom?.threads ?? [], localThreads),
    messages: mergeById(custom?.messages ?? [], localMessages),
    sourceWarnings,
  };
}

function countWhen<T>(values: T[], predicate: (value: T) => boolean): number {
  return values.reduce((count, value) => count + (predicate(value) ? 1 : 0), 0);
}

export function buildPlatformChatMigrationPlan(
  snapshot: LegacyChatMigrationSnapshot,
  ownerId: string,
  ledger = loadLedger(ownerId),
): PlatformChatMigrationPlan {
  const projects = snapshot.projects.map((project) => ({
    legacyId: project.id,
    label: project.name || "Adsız proje",
    status: ledger.projects[project.id] ? "migrated" : "pending",
    ...(ledger.projects[project.id]
      ? { platformId: ledger.projects[project.id] }
      : {}),
  })) satisfies ChatMigrationPlanRecord[];
  const threads = snapshot.threads.map((thread) => ({
    legacyId: thread.id,
    label: thread.title || "Adsız sohbet",
    status: ledger.threads[thread.id] ? "migrated" : "pending",
    ...(ledger.threads[thread.id]
      ? { platformId: ledger.threads[thread.id] }
      : {}),
  })) satisfies ChatMigrationPlanRecord[];

  const unsupported: ChatMigrationUnsupportedItem[] = [];
  if (snapshot.messages.length > 0) {
    unsupported.push({
      kind: "message",
      count: snapshot.messages.length,
      reason:
        "Backend oturum sözleşmesinde geçmiş mesaj oluşturma endpoint'i yoktur; mesajlar export içinde korunur ve otomatik yazılmaz.",
    });
  }
  const projectOverlayCount = countWhen(
    snapshot.projects,
    (project) => Boolean(project.rootPath || project.sandboxPath || project.archived),
  );
  if (projectOverlayCount > 0) {
    unsupported.push({
      kind: "project-field",
      count: projectOverlayCount,
      reason:
        "Archive, yerel kök ve sandbox alanlarının backend Chat sözleşmesinde kalıcı karşılığı yoktur.",
    });
  }
  const threadOverlayCount = countWhen(
    snapshot.threads,
    (thread) =>
      Boolean(
        thread.archived ||
          thread.pairId ||
          thread.forkedFromThreadId ||
          thread.forkedFromMessageId ||
          thread.openaiCodeExecContainerId ||
          thread.anthropicCodeExecContainerId,
      ),
  );
  if (threadOverlayCount > 0) {
    unsupported.push({
      kind: "thread-field",
      count: threadOverlayCount,
      reason:
        "Archive, karşılaştırma, fork ve geçici container alanları Session sözleşmesine taşınamaz.",
    });
  }

  const alreadyMigrated = [...projects, ...threads].filter(
    (record) => record.status === "migrated",
  ).length;
  return {
    version: MIGRATION_VERSION,
    generatedAt: new Date().toISOString(),
    ownerId,
    snapshot,
    projects,
    threads,
    unsupported,
    totals: {
      projects: projects.length,
      threads: threads.length,
      messages: snapshot.messages.length,
      alreadyMigrated,
      pending: projects.length + threads.length - alreadyMigrated,
    },
  };
}

export async function dryRunPlatformChatMigration(
  ownerId: string,
  signal?: AbortSignal,
): Promise<PlatformChatMigrationPlan> {
  return buildPlatformChatMigrationPlan(
    await readLegacyChatMigrationSnapshot(signal),
    ownerId,
  );
}

function projectMarker(legacyId: string): string {
  return `${PROJECT_MARKER_PREFIX}${legacyId}]`;
}

function dtoString(value: unknown): string {
  return typeof value === "string" ? value : "";
}

export async function runPlatformChatMigration(
  plan: PlatformChatMigrationPlan,
  options: {
    signal?: AbortSignal;
    onProgress?: (progress: PlatformChatMigrationProgress) => void;
  } = {},
): Promise<PlatformChatMigrationResult> {
  const ledger = loadLedger(plan.ownerId);
  const result: PlatformChatMigrationResult = {
    completedProjects: 0,
    completedThreads: 0,
    skipped: 0,
    failures: [],
    aborted: false,
  };
  const total = plan.projects.length + plan.threads.length;
  let completed = 0;
  const progress = (current: string) => {
    completed += 1;
    options.onProgress?.({ completed, total, current });
  };

  const existingChats = await listAllPlatformChats(options.signal);
  const chatsByMarker = new Map<string, string>();
  for (const chat of existingChats) {
    const description = dtoString(chat.description);
    const id = dtoString(chat.id);
    if (description.startsWith(PROJECT_MARKER_PREFIX) && id) {
      chatsByMarker.set(description, id);
    }
  }

  for (const item of plan.projects) {
    if (options.signal?.aborted) {
      result.aborted = true;
      break;
    }
    if (ledger.projects[item.legacyId]) {
      result.skipped += 1;
      progress(item.label);
      continue;
    }
    const source = plan.snapshot.projects.find(
      (project) => project.id === item.legacyId,
    );
    if (!source) continue;
    try {
      const marker = projectMarker(source.id);
      let platformId = chatsByMarker.get(marker);
      if (!platformId) {
        const created = await createPlatformChat(
          {
            name: source.name.trim() || "Taşınan proje",
            description: marker,
            dataset_ids: [...new Set(source.datasetIds ?? [])],
            ...(source.platformLlmId ? { llm_id: source.platformLlmId } : {}),
            ...(source.instructions?.trim()
              ? { prompt_config: { system: source.instructions.trim() } }
              : {}),
          },
          options.signal,
        );
        platformId = dtoString(created.id);
        if (!platformId) throw new Error("Backend yeni Chat kimliği döndürmedi.");
        chatsByMarker.set(marker, platformId);
      }
      ledger.projects[source.id] = platformId;
      saveLedger(ledger);
      result.completedProjects += 1;
    } catch (error) {
      result.failures.push({
        legacyId: source.id,
        label: item.label,
        reason: error instanceof Error ? error.message : "Bilinmeyen hata",
      });
    }
    progress(item.label);
  }

  let generalChatId: string | null = null;
  if (!result.aborted) {
    for (const item of plan.threads) {
      if (options.signal?.aborted) {
        result.aborted = true;
        break;
      }
      if (ledger.threads[item.legacyId]) {
        result.skipped += 1;
        progress(item.label);
        continue;
      }
      const source = plan.snapshot.threads.find(
        (thread) => thread.id === item.legacyId,
      );
      if (!source) continue;
      try {
        let chatId = source.projectId
          ? ledger.projects[source.projectId] ?? null
          : null;
        if (!chatId) {
          if (!generalChatId) {
            generalChatId = (await ensureGeneralPlatformChat(options.signal)).id;
          }
          chatId = generalChatId;
        }
        const created = await createPlatformSession(
          chatId,
          { name: source.title.trim() || "Taşınan sohbet" },
          options.signal,
        );
        const platformId = dtoString(created.id);
        if (!platformId) throw new Error("Backend yeni Session kimliği döndürmedi.");
        ledger.threads[source.id] = platformId;
        saveLedger(ledger);
        result.completedThreads += 1;
      } catch (error) {
        result.failures.push({
          legacyId: source.id,
          label: item.label,
          reason: error instanceof Error ? error.message : "Bilinmeyen hata",
        });
      }
      progress(item.label);
    }
  }

  return result;
}

export function serializePlatformChatMigrationExport(
  plan: PlatformChatMigrationPlan,
): string {
  return `${JSON.stringify(
    {
      schema: "rag-platform-chat-migration-export",
      version: MIGRATION_VERSION,
      generatedAt: new Date().toISOString(),
      dryRun: plan,
      deletionPerformed: false,
    },
    null,
    2,
  )}\n`;
}
