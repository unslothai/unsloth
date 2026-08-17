export interface PlatformWorkspaceFile {
  id: string;
  parentId: string;
  name: string;
  size: number;
  type: string;
  sourceType: string;
  createdAt: number | null;
  updatedAt: number | null;
  isFolder: boolean;
}

export interface PlatformFilesPage {
  total: number;
  files: PlatformWorkspaceFile[];
  parentFolder: PlatformWorkspaceFile | null;
}

export type PlatformFileLinkMode = "add" | "replace";
export type PlatformCommitScope = "workspace" | "folders" | "datasets";
export type PlatformFileOperation =
  | "add"
  | "modify"
  | "delete"
  | "rename"
  | "move";

export interface PlatformFileChange {
  fileId: string;
  fileName: string;
  operation: PlatformFileOperation;
  content?: string;
  oldName?: string;
  newName?: string;
  oldParentId?: string;
  newParentId?: string;
}

export interface PlatformFileDiff {
  fileId: string;
  fileName: string;
  operation: string;
  oldHash: string | null;
  newHash: string | null;
  oldLocation: string | null;
  newLocation: string | null;
  oldName: string | null;
  newName: string | null;
  oldParentId: string | null;
  newParentId: string | null;
}

export interface PlatformFileCommit {
  id: string;
  folderId: string;
  parentId: string | null;
  message: string;
  authorId: string;
  fileCount: number;
  treeState: string | null;
  createdAt: number | null;
}

export interface PlatformCommitFile {
  id: string;
  fileId: string;
  operation: string;
  oldHash: string | null;
  newHash: string | null;
  oldLocation: string | null;
  newLocation: string | null;
  oldName: string | null;
  newName: string | null;
  diff: string | null;
}

export interface PlatformCommitDetail extends PlatformFileCommit {
  files: PlatformCommitFile[];
}

export interface PlatformCommitsPage {
  total: number;
  page: number;
  pageSize: number;
  commits: PlatformFileCommit[];
}

export interface PlatformFileVersion {
  commitId: string;
  operation: string;
  hash: string;
  createdAt: number | null;
  message: string;
}

function record(value: unknown): Record<string, unknown> {
  return value && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : {};
}

function text(value: unknown): string {
  return typeof value === "string" ? value : "";
}

function nullableText(value: unknown): string | null {
  const parsed = text(value);
  return parsed || null;
}

function numberValue(value: unknown): number {
  const parsed = typeof value === "number" ? value : Number(value);
  return Number.isFinite(parsed) ? parsed : 0;
}

function nullableNumber(value: unknown): number | null {
  if (value === null || value === undefined || value === "") return null;
  const parsed = numberValue(value);
  return Number.isFinite(parsed) ? parsed : null;
}

export function mapWorkspaceFile(value: unknown): PlatformWorkspaceFile {
  const dto = record(value);
  const type = text(dto.type);
  return {
    id: text(dto.id),
    parentId: text(dto.parent_id),
    name: text(dto.name),
    size: numberValue(dto.size),
    type,
    sourceType: text(dto.source_type),
    createdAt: nullableNumber(dto.create_time),
    updatedAt: nullableNumber(dto.update_time),
    isFolder: type.toLowerCase() === "folder",
  };
}

export function mapFileDiff(value: unknown): PlatformFileDiff {
  const dto = record(value);
  return {
    fileId: text(dto.file_id),
    fileName: text(dto.file_name),
    operation: text(dto.operation),
    oldHash: nullableText(dto.old_hash),
    newHash: nullableText(dto.new_hash),
    oldLocation: nullableText(dto.old_location),
    newLocation: nullableText(dto.new_location),
    oldName: nullableText(dto.old_name),
    newName: nullableText(dto.new_name),
    oldParentId: nullableText(dto.old_parent_id),
    newParentId: nullableText(dto.new_parent_id),
  };
}

export function mapFileCommit(value: unknown): PlatformFileCommit {
  const dto = record(value);
  return {
    id: text(dto.id),
    folderId: text(dto.folder_id),
    parentId: nullableText(dto.parent_id),
    message: text(dto.message),
    authorId: text(dto.author_id),
    fileCount: numberValue(dto.file_count),
    treeState: nullableText(dto.tree_state),
    createdAt: nullableNumber(dto.create_time),
  };
}

export function mapCommitFile(value: unknown): PlatformCommitFile {
  const dto = record(value);
  return {
    id: text(dto.id),
    fileId: text(dto.file_id),
    operation: text(dto.operation),
    oldHash: nullableText(dto.old_hash),
    newHash: nullableText(dto.new_hash),
    oldLocation: nullableText(dto.old_location),
    newLocation: nullableText(dto.new_location),
    oldName: nullableText(dto.old_name),
    newName: nullableText(dto.new_name),
    diff: nullableText(dto.diff),
  };
}

export function mapFileVersion(value: unknown): PlatformFileVersion {
  const dto = record(value);
  return {
    commitId: text(dto.commit_id),
    operation: text(dto.operation),
    hash: text(dto.hash),
    createdAt: nullableNumber(dto.create_time),
    message: text(dto.message),
  };
}

export function toPlatformFileChange(
  value: PlatformFileChange,
): Record<string, unknown> {
  return {
    file_id: value.fileId,
    file_name: value.fileName,
    operation: value.operation,
    ...(value.content === undefined ? {} : { content: value.content }),
    ...(value.oldName === undefined ? {} : { old_name: value.oldName }),
    ...(value.newName === undefined ? {} : { new_name: value.newName }),
    ...(value.oldParentId === undefined
      ? {}
      : { old_parent_id: value.oldParentId }),
    ...(value.newParentId === undefined
      ? {}
      : { new_parent_id: value.newParentId }),
  };
}
