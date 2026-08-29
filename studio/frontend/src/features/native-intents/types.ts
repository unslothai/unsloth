export type NativePathOperation =
  | "validate-model"
  | "load-model"
  | "dataset-preview"
  | "dataset-import"
  | "attach"
  | "open-project"
  | "set-project-workspace"
  | "reveal"
  | "open";

export type NativePathKind =
  | "model"
  | "dataset"
  | "attachment"
  | "document-folder"
  | "project-workspace"
  | "artifact";

export type NativePathSourceKind =
  | "dialog"
  | "drop"
  | "deep-link"
  | "file-association"
  | "artifact";

export interface NativePathRef {
  token: string;
  kind: NativePathKind;
  displayLabel: string;
  allowedOperations: NativePathOperation[];
  expiresAtMs: number;
  sizeBytes?: number | null;
  modifiedMs?: number | null;
}

export interface NativeIntent {
  id: string;
  kind: NativePathKind;
  sourceKind: NativePathSourceKind;
  path: NativePathRef;
  displayLabel: string;
}

export interface NativePathLeaseResponse {
  nativePathLease: string;
  displayLabel: string;
  expiresAtMs: number;
}
