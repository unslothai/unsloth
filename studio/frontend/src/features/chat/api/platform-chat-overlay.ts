import type { ModelType, ProjectRecord, ThreadRecord } from "../types";

const STORAGE_KEY = "rag-platform.chat-local-overlay.v1";

type ProjectOverlay = Pick<
  ProjectRecord,
  "archived" | "rootPath" | "sandboxPath"
>;

type ThreadOverlay = Pick<
  ThreadRecord,
  | "archived"
  | "modelType"
  | "modelId"
  | "pairId"
  | "openaiCodeExecContainerId"
  | "anthropicCodeExecContainerId"
  | "forkedFromThreadId"
  | "forkedFromMessageId"
>;

interface ChatOverlayState {
  projects: Record<string, Partial<ProjectOverlay>>;
  threads: Record<string, Partial<ThreadOverlay>>;
}

const emptyState = (): ChatOverlayState => ({ projects: {}, threads: {} });

function readState(): ChatOverlayState {
  if (typeof window === "undefined") return emptyState();
  try {
    const parsed = JSON.parse(localStorage.getItem(STORAGE_KEY) ?? "null") as
      | Partial<ChatOverlayState>
      | null;
    return {
      projects:
        parsed?.projects && typeof parsed.projects === "object"
          ? parsed.projects
          : {},
      threads:
        parsed?.threads && typeof parsed.threads === "object"
          ? parsed.threads
          : {},
    };
  } catch {
    return emptyState();
  }
}

function writeState(state: ChatOverlayState): void {
  if (typeof window === "undefined") return;
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
  } catch {
    // A denied/full localStorage must not make server-backed chat unusable.
  }
}

export function getPlatformProjectOverlay(
  projectId: string,
): Partial<ProjectOverlay> {
  return readState().projects[projectId] ?? {};
}

export function setPlatformProjectOverlay(
  projectId: string,
  patch: Partial<ProjectOverlay>,
): void {
  const state = readState();
  state.projects[projectId] = { ...state.projects[projectId], ...patch };
  writeState(state);
}

export function deletePlatformProjectOverlay(projectId: string): void {
  const state = readState();
  delete state.projects[projectId];
  writeState(state);
}

export function getPlatformThreadOverlay(
  threadId: string,
): Partial<ThreadOverlay> {
  return readState().threads[threadId] ?? {};
}

export function setPlatformThreadOverlay(
  threadId: string,
  patch: Partial<ThreadOverlay>,
): void {
  const state = readState();
  state.threads[threadId] = { ...state.threads[threadId], ...patch };
  writeState(state);
}

export function deletePlatformThreadOverlays(threadIds: string[]): void {
  const state = readState();
  for (const threadId of threadIds) delete state.threads[threadId];
  writeState(state);
}

export function clearPlatformChatOverlay(): void {
  if (typeof window === "undefined") return;
  try {
    localStorage.removeItem(STORAGE_KEY);
  } catch {
    // Best effort only; server deletion remains authoritative.
  }
}

export const PLATFORM_LOCAL_ONLY_FIELDS = {
  project: ["archived", "rootPath", "sandboxPath"],
  thread: [
    "archived",
    "modelType",
    "modelId",
    "pairId",
    "openaiCodeExecContainerId",
    "anthropicCodeExecContainerId",
    "forkedFromThreadId",
    "forkedFromMessageId",
  ],
} as const satisfies {
  project: readonly (keyof ProjectOverlay)[];
  thread: readonly (keyof ThreadOverlay)[];
};

export function platformThreadModelType(threadId: string): ModelType {
  return getPlatformThreadOverlay(threadId).modelType ?? "base";
}
