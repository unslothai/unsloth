import { platformRequest } from "./client";
import {
  mapCommitFile,
  mapFileCommit,
  mapFileDiff,
  mapFileVersion,
  mapWorkspaceFile,
  toPlatformFileChange,
  type PlatformCommitDetail,
  type PlatformCommitFile,
  type PlatformCommitScope,
  type PlatformCommitsPage,
  type PlatformFileChange,
  type PlatformFileCommit,
  type PlatformFileDiff,
  type PlatformFileLinkMode,
  type PlatformFilesPage,
  type PlatformFileVersion,
  type PlatformWorkspaceFile,
} from "./file-types";

function record(value: unknown): Record<string, unknown> {
  return value && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : {};
}

function array(value: unknown): unknown[] {
  return Array.isArray(value) ? value : [];
}

function numberValue(value: unknown): number {
  const parsed = typeof value === "number" ? value : Number(value);
  return Number.isFinite(parsed) ? parsed : 0;
}

export async function listPlatformFiles(
  options: {
    parentId?: string;
    keywords?: string;
    page?: number;
    pageSize?: number;
    orderBy?: string;
    descending?: boolean;
  } = {},
  signal?: AbortSignal,
): Promise<PlatformFilesPage> {
  const data = record(
    await platformRequest<unknown>("/files", {
      query: {
        parent_id: options.parentId,
        keywords: options.keywords,
        page: options.page ?? 1,
        page_size: options.pageSize ?? 50,
        orderby: options.orderBy ?? "create_time",
        desc: options.descending ?? true,
      },
      signal,
    }),
  );
  return {
    total: numberValue(data.total),
    files: array(data.files).map(mapWorkspaceFile).filter((item) => item.id),
    parentFolder: data.parent_folder
      ? mapWorkspaceFile(data.parent_folder)
      : null,
  };
}

export async function createPlatformFolder(
  name: string,
  parentId?: string,
  signal?: AbortSignal,
): Promise<PlatformWorkspaceFile> {
  const data = await platformRequest<unknown>("/files", {
    method: "POST",
    json: { name, parent_id: parentId, type: "folder" },
    signal,
  });
  return mapWorkspaceFile(data);
}

export async function uploadPlatformFiles(
  files: File[],
  parentId?: string,
  signal?: AbortSignal,
): Promise<PlatformWorkspaceFile[]> {
  const body = new FormData();
  if (parentId) body.append("parent_id", parentId);
  for (const file of files) body.append("file", file, file.name);
  const data = await platformRequest<unknown>("/files", {
    method: "POST",
    body,
    signal,
    timeoutMs: 120_000,
  });
  return array(data).map(mapWorkspaceFile);
}

export async function deletePlatformFiles(
  ids: string[],
  signal?: AbortSignal,
): Promise<{ successCount: number; errors: string[] }> {
  const data = record(
    await platformRequest<unknown>("/files", {
      method: "DELETE",
      json: { ids },
      signal,
    }),
  );
  return {
    successCount: numberValue(data.success_count),
    errors: array(data.errors).filter(
      (item): item is string => typeof item === "string",
    ),
  };
}

export async function movePlatformFiles(
  sourceIds: string[],
  input: { destinationFolderId?: string; newName?: string },
  signal?: AbortSignal,
): Promise<boolean> {
  return platformRequest<boolean>("/files/move", {
    method: "POST",
    json: {
      src_file_ids: sourceIds,
      dest_file_id: input.destinationFolderId,
      new_name: input.newName,
    },
    signal,
  });
}

export async function linkPlatformFilesToDatasets(
  fileIds: string[],
  datasetIds: string[],
  mode: PlatformFileLinkMode,
  signal?: AbortSignal,
): Promise<boolean> {
  return platformRequest<boolean>("/files/link-to-datasets", {
    method: "POST",
    query: { mode },
    json: { file_ids: fileIds, kb_ids: datasetIds },
    signal,
    timeoutMs: 30_000,
  });
}

export async function getPlatformFileParent(
  fileId: string,
  signal?: AbortSignal,
): Promise<PlatformWorkspaceFile | null> {
  const data = record(
    await platformRequest<unknown>(
      `/files/${encodeURIComponent(fileId)}/parent`,
      { signal },
    ),
  );
  return data.parent_folder ? mapWorkspaceFile(data.parent_folder) : null;
}

export async function getPlatformFileAncestors(
  fileId: string,
  signal?: AbortSignal,
): Promise<PlatformWorkspaceFile[]> {
  const data = record(
    await platformRequest<unknown>(
      `/files/${encodeURIComponent(fileId)}/ancestors`,
      { signal },
    ),
  );
  return array(data.parent_folders).map(mapWorkspaceFile);
}

export async function downloadPlatformFile(
  fileId: string,
  signal?: AbortSignal,
): Promise<Blob> {
  return platformRequest<Blob>(`/files/${encodeURIComponent(fileId)}`, {
    responseType: "blob",
    acceptJsonBlob: true,
    signal,
    timeoutMs: 120_000,
  });
}

export async function listPlatformFileVersions(
  fileId: string,
  signal?: AbortSignal,
): Promise<PlatformFileVersion[]> {
  const data = await platformRequest<unknown>(
    `/files/${encodeURIComponent(fileId)}/versions`,
    { signal },
  );
  return array(data).map(mapFileVersion);
}

function commitBase(scope: PlatformCommitScope, scopeId: string): string {
  return `/${scope}/${encodeURIComponent(scopeId)}`;
}

export async function listPlatformCommits(
  scope: PlatformCommitScope,
  scopeId: string,
  options: { page?: number; pageSize?: number; slug?: string } = {},
  signal?: AbortSignal,
): Promise<PlatformCommitsPage> {
  const data = record(
    await platformRequest<unknown>(`${commitBase(scope, scopeId)}/commits`, {
      query: {
        page: options.page ?? 1,
        page_size: options.pageSize ?? 50,
        order_by: "create_time",
        desc: true,
        slug: options.slug,
      },
      signal,
    }),
  );
  return {
    total: numberValue(data.total),
    page: numberValue(data.page) || options.page || 1,
    pageSize: numberValue(data.page_size) || options.pageSize || 50,
    commits: array(data.commits).map(mapFileCommit),
  };
}

export async function createPlatformCommit(
  scope: PlatformCommitScope,
  scopeId: string,
  message: string,
  changes: PlatformFileChange[],
  signal?: AbortSignal,
): Promise<PlatformFileCommit> {
  const data = await platformRequest<unknown>(
    `${commitBase(scope, scopeId)}/commits`,
    {
      method: "POST",
      json: { message, files: changes.map(toPlatformFileChange) },
      signal,
      timeoutMs: 120_000,
    },
  );
  return mapFileCommit(data);
}

export async function getPlatformCommit(
  scope: PlatformCommitScope,
  scopeId: string,
  commitId: string,
  signal?: AbortSignal,
): Promise<PlatformCommitDetail> {
  const data = record(
    await platformRequest<unknown>(
      `${commitBase(scope, scopeId)}/commits/${encodeURIComponent(commitId)}`,
      { signal },
    ),
  );
  return {
    ...mapFileCommit(data),
    files: array(data.files).map(mapCommitFile),
  };
}

export async function listPlatformCommitFiles(
  scope: PlatformCommitScope,
  scopeId: string,
  commitId: string,
  signal?: AbortSignal,
): Promise<PlatformCommitFile[]> {
  const data = await platformRequest<unknown>(
    `${commitBase(scope, scopeId)}/commits/${encodeURIComponent(commitId)}/files`,
    { signal },
  );
  return array(data).map(mapCommitFile);
}

export async function diffPlatformCommits(
  scope: PlatformCommitScope,
  scopeId: string,
  fromId: string,
  toId: string,
  signal?: AbortSignal,
): Promise<PlatformFileDiff[]> {
  const data = await platformRequest<unknown>(
    `${commitBase(scope, scopeId)}/commits/diff`,
    { query: { from: fromId, to: toId }, signal },
  );
  return array(data).map(mapFileDiff);
}

export async function getPlatformUncommittedChanges(
  scope: PlatformCommitScope,
  scopeId: string,
  signal?: AbortSignal,
): Promise<PlatformFileDiff[]> {
  const data = await platformRequest<unknown>(
    `${commitBase(scope, scopeId)}/changes`,
    { signal },
  );
  return array(data).map(mapFileDiff);
}

export async function getPlatformCommitTree(
  scope: PlatformCommitScope,
  scopeId: string,
  commitId: string,
  signal?: AbortSignal,
): Promise<Record<string, unknown>> {
  return record(
    await platformRequest<unknown>(
      `${commitBase(scope, scopeId)}/commits/${encodeURIComponent(commitId)}/tree`,
      { signal },
    ),
  );
}

export async function getPlatformCommitFileContent(
  scope: PlatformCommitScope,
  scopeId: string,
  commitId: string,
  fileId: string,
  signal?: AbortSignal,
): Promise<string> {
  const data = record(
    await platformRequest<unknown>(
      `${commitBase(scope, scopeId)}/commits/${encodeURIComponent(commitId)}/files/${encodeURIComponent(fileId)}/content`,
      { signal },
    ),
  );
  return typeof data.content === "string" ? data.content : "";
}
