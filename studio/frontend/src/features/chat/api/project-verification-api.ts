// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";
import { formatApiErrorBody } from "@/lib/format-fastapi-error";

export type ProjectVerificationSourceFreshness = "unverified";

export interface ProjectVerificationCheck {
  name: string;
  kind: string;
  command: string;
  required: boolean;
  timeoutSeconds: number;
  logLimitBytes: number;
}

export interface ProjectVerificationResult {
  name: string;
  kind: string;
  command: string;
  required: boolean;
  timeoutSeconds: number;
  logLimitBytes?: number;
  status: string;
  exitCode: number | null;
  output: string;
  outputBytes: number;
  outputTruncated: boolean;
  startedAt: number | null;
  completedAt: number | null;
  durationMs: number | null;
  error?: string | null;
}

export interface ProjectVerificationRunMetadata {
  id: string;
  projectId: string;
  status: string;
  configRevision: number;
  workspaceRevision: number;
  evidenceRevision: number;
  cancelRequested?: boolean;
  error: string | null;
  startedAt: number | null;
  updatedAt: number;
  completedAt: number | null;
  historySequence: number | null;
  sourceFreshness: ProjectVerificationSourceFreshness;
}

export interface ProjectVerificationRunSummary
  extends ProjectVerificationRunMetadata {
  evidenceStatus: "not_loaded";
}

export interface ProjectVerificationRun extends ProjectVerificationRunMetadata {
  checks: ProjectVerificationCheck[];
  results: ProjectVerificationResult[];
}

export interface ProjectVerificationConfig {
  projectId: string;
  workspaceAvailable: boolean;
  workspaceRevision: number;
  /** True only when the saved profile is bound to the current workspace. */
  active: boolean;
  /** Server-owned run recovered independently of the component that started it. */
  activeRun: ProjectVerificationRun | null;
  checks: ProjectVerificationCheck[];
  revision: number;
  updatedAt: number | null;
  sourceFreshness: ProjectVerificationSourceFreshness;
  execution: {
    available: boolean;
    backend: string | null;
    reason: string | null;
  };
}

export interface SaveProjectVerificationConfig {
  checks: ProjectVerificationCheck[];
  expectedRevision: number;
  workspaceRevision: number;
}

export interface StartProjectVerification {
  configRevision: number;
  workspaceRevision: number;
}

type ProjectVerificationConfigIdentity = Pick<
  ProjectVerificationConfig,
  "projectId" | "revision"
>;

export function projectVerificationConfigCanReplace(
  current: ProjectVerificationConfigIdentity | null,
  next: ProjectVerificationConfigIdentity,
): boolean {
  return (
    current?.projectId !== next.projectId || next.revision >= current.revision
  );
}

export const PROJECT_VERIFICATION_MAX_CHECKS = 32;
export const PROJECT_VERIFICATION_MAX_NAME_CHARACTERS = 120;
export const PROJECT_VERIFICATION_MAX_KIND_BYTES = 64;
export const PROJECT_VERIFICATION_MAX_COMMAND_BYTES = 16 * 1024;
export const PROJECT_VERIFICATION_MAX_CONFIG_BYTES = 128 * 1024;
export const PROJECT_VERIFICATION_MAX_LOG_LIMIT_BYTES = 2 * 1024 * 1024;

const PROJECT_VERIFICATION_TEXT_ENCODER = new TextEncoder();

// Keep the current Cf and Default_Ignorable_Code_Point baseline independent of
// the Unicode tables shipped by a browser or packaged WebView.
const PROJECT_VERIFICATION_UNSAFE_INVISIBLE_RANGES: ReadonlyArray<
  readonly [number, number]
> = [
  [0x00ad, 0x00ad],
  [0x034f, 0x034f],
  [0x0600, 0x0605],
  [0x061c, 0x061c],
  [0x06dd, 0x06dd],
  [0x070f, 0x070f],
  [0x0890, 0x0891],
  [0x08e2, 0x08e2],
  [0x115f, 0x1160],
  [0x17b4, 0x17b5],
  [0x180b, 0x180f],
  [0x200b, 0x200f],
  [0x202a, 0x202e],
  [0x2060, 0x206f],
  [0x3164, 0x3164],
  [0xfe00, 0xfe0f],
  [0xfeff, 0xfeff],
  [0xffa0, 0xffa0],
  [0xfff0, 0xfffb],
  [0x110bd, 0x110bd],
  [0x110cd, 0x110cd],
  [0x13430, 0x1343f],
  [0x1bca0, 0x1bca3],
  [0x1d173, 0x1d17a],
  [0xe0000, 0xe0fff],
];

const PROJECT_VERIFICATION_NON_ASCII_WHITESPACE_RANGES: ReadonlyArray<
  readonly [number, number]
> = [
  [0x0085, 0x0085],
  [0x00a0, 0x00a0],
  [0x1680, 0x1680],
  [0x2000, 0x200a],
  [0x2028, 0x2029],
  [0x202f, 0x202f],
  [0x205f, 0x205f],
  [0x3000, 0x3000],
];

function projectVerificationUnicodeProperty(pattern: string): RegExp | null {
  try {
    return new RegExp(pattern, "u");
  } catch {
    return null;
  }
}

const PROJECT_VERIFICATION_FUTURE_FORMAT_CONTROL =
  projectVerificationUnicodeProperty("\\p{Cf}");
const PROJECT_VERIFICATION_FUTURE_DEFAULT_IGNORABLE =
  projectVerificationUnicodeProperty("\\p{Default_Ignorable_Code_Point}");
const PROJECT_VERIFICATION_FUTURE_UNICODE_WHITESPACE =
  projectVerificationUnicodeProperty("\\p{White_Space}");

function projectVerificationCodePointInRanges(
  codePoint: number,
  ranges: ReadonlyArray<readonly [number, number]>,
): boolean {
  let low = 0;
  let high = ranges.length - 1;
  while (low <= high) {
    const middle = low + Math.floor((high - low) / 2);
    const [start, end] = ranges[middle];
    if (codePoint < start) {
      high = middle - 1;
    } else if (codePoint > end) {
      low = middle + 1;
    } else {
      return true;
    }
  }
  return false;
}

function projectVerificationIsUnsafeInvisible(
  character: string,
  codePoint: number,
): boolean {
  return (
    projectVerificationCodePointInRanges(
      codePoint,
      PROJECT_VERIFICATION_UNSAFE_INVISIBLE_RANGES,
    ) ||
    PROJECT_VERIFICATION_FUTURE_FORMAT_CONTROL?.test(character) === true ||
    PROJECT_VERIFICATION_FUTURE_DEFAULT_IGNORABLE?.test(character) === true
  );
}

function projectVerificationHasUnsafeInvisible(value: string): boolean {
  for (const character of value) {
    const codePoint = character.codePointAt(0);
    if (
      codePoint !== undefined &&
      projectVerificationIsUnsafeInvisible(character, codePoint)
    ) {
      return true;
    }
  }
  return false;
}

function projectVerificationHasLoneUtf16Surrogate(value: string): boolean {
  for (const character of value) {
    const codePoint = character.codePointAt(0);
    if (codePoint !== undefined && codePoint >= 0xd800 && codePoint <= 0xdfff) {
      return true;
    }
  }
  return false;
}

function projectVerificationIsUnicodeWhitespace(
  character: string,
  codePoint: number,
): boolean {
  return (
    (codePoint >= 0x09 && codePoint <= 0x0d) ||
    codePoint === 0x20 ||
    projectVerificationCodePointInRanges(
      codePoint,
      PROJECT_VERIFICATION_NON_ASCII_WHITESPACE_RANGES,
    ) ||
    PROJECT_VERIFICATION_FUTURE_UNICODE_WHITESPACE?.test(character) === true
  );
}

function projectVerificationHasRequiredText(value: string): boolean {
  for (const character of value) {
    const codePoint = character.codePointAt(0);
    if (
      codePoint !== undefined &&
      !projectVerificationIsUnicodeWhitespace(character, codePoint)
    ) {
      return true;
    }
  }
  return false;
}

function projectVerificationHasUnsafeNameOrKindControl(value: string): boolean {
  for (const character of value) {
    const codePoint = character.codePointAt(0);
    if (codePoint !== undefined && (codePoint <= 0x1f || codePoint === 0x7f)) {
      return true;
    }
  }
  return false;
}

function projectVerificationHasUnsafeCommandControl(value: string): boolean {
  for (const character of value) {
    const codePoint = character.codePointAt(0);
    if (
      codePoint !== undefined &&
      ((codePoint <= 0x1f && codePoint !== 0x09 && codePoint !== 0x0a) ||
        (codePoint >= 0x7f && codePoint <= 0x9f))
    ) {
      return true;
    }
  }
  return false;
}

function projectVerificationHasNonAsciiWhitespace(value: string): boolean {
  for (const character of value) {
    const codePoint = character.codePointAt(0);
    if (
      codePoint !== undefined &&
      codePoint > 0x7f &&
      projectVerificationIsUnicodeWhitespace(character, codePoint)
    ) {
      return true;
    }
  }
  return false;
}

export function projectVerificationUtf8ByteLength(value: string): number {
  return PROJECT_VERIFICATION_TEXT_ENCODER.encode(value).byteLength;
}

function projectVerificationCheckTextError(
  check: ProjectVerificationCheck,
): string | null {
  if (
    [check.name, check.kind, check.command].some(
      (value) => !projectVerificationHasRequiredText(value),
    )
  ) {
    return "Every check needs a name, kind, and command.";
  }
  if (
    projectVerificationHasUnsafeInvisible(check.name) ||
    projectVerificationHasUnsafeInvisible(check.kind) ||
    projectVerificationHasUnsafeInvisible(check.command)
  ) {
    return "Check names, kinds, and commands cannot contain Unicode format controls or default-ignorable code points.";
  }
  if (
    projectVerificationHasLoneUtf16Surrogate(check.name) ||
    projectVerificationHasLoneUtf16Surrogate(check.kind) ||
    projectVerificationHasLoneUtf16Surrogate(check.command)
  ) {
    return "Check names, kinds, and commands cannot contain lone UTF-16 surrogates.";
  }
  if (
    projectVerificationHasUnsafeNameOrKindControl(check.name) ||
    projectVerificationHasUnsafeNameOrKindControl(check.kind)
  ) {
    return "Check names and kinds cannot contain C0 control characters or DEL.";
  }
  if (projectVerificationHasUnsafeCommandControl(check.command)) {
    return "Check commands can contain tabs and newlines, but no other control characters.";
  }
  if (projectVerificationHasNonAsciiWhitespace(check.command)) {
    return "Check commands cannot contain non-ASCII Unicode whitespace.";
  }
  if (
    Array.from(check.name).length > PROJECT_VERIFICATION_MAX_NAME_CHARACTERS
  ) {
    return `Check names must be at most ${PROJECT_VERIFICATION_MAX_NAME_CHARACTERS} characters.`;
  }
  if (
    projectVerificationUtf8ByteLength(check.kind) >
    PROJECT_VERIFICATION_MAX_KIND_BYTES
  ) {
    return `Check kinds must be at most ${PROJECT_VERIFICATION_MAX_KIND_BYTES} UTF-8 bytes.`;
  }
  if (
    projectVerificationUtf8ByteLength(check.command) >
    PROJECT_VERIFICATION_MAX_COMMAND_BYTES
  ) {
    return `Check commands must be at most ${PROJECT_VERIFICATION_MAX_COMMAND_BYTES} UTF-8 bytes.`;
  }
  return null;
}

function projectVerificationCheckBoundsError(
  check: ProjectVerificationCheck,
): string | null {
  if (
    !Number.isInteger(check.timeoutSeconds) ||
    check.timeoutSeconds < 1 ||
    check.timeoutSeconds > 3600
  ) {
    return "Check timeouts must be whole seconds from 1 to 3600.";
  }
  if (
    !Number.isInteger(check.logLimitBytes) ||
    check.logLimitBytes < 1024 ||
    check.logLimitBytes > PROJECT_VERIFICATION_MAX_LOG_LIMIT_BYTES
  ) {
    return `Check log limits must be from 1024 to ${PROJECT_VERIFICATION_MAX_LOG_LIMIT_BYTES} bytes.`;
  }
  return null;
}

export function projectVerificationChecksError(
  checks: ProjectVerificationCheck[],
): string | null {
  if (checks.length > PROJECT_VERIFICATION_MAX_CHECKS) {
    return `At most ${PROJECT_VERIFICATION_MAX_CHECKS} checks can be configured.`;
  }
  const exactNames = new Set<string>();
  for (const check of checks) {
    const textError = projectVerificationCheckTextError(check);
    if (textError) {
      return textError;
    }
    if (exactNames.has(check.name)) {
      return "Verification check names must be unique.";
    }
    exactNames.add(check.name);
    const boundsError = projectVerificationCheckBoundsError(check);
    if (boundsError) {
      return boundsError;
    }
  }
  const normalized = checks.map((check) => ({
    name: check.name.trim(),
    kind: check.kind.trim(),
    command: check.command.trim(),
    required: check.required,
    timeoutSeconds: check.timeoutSeconds,
    logLimitBytes: check.logLimitBytes,
  }));
  if (
    projectVerificationUtf8ByteLength(JSON.stringify(normalized)) >
    PROJECT_VERIFICATION_MAX_CONFIG_BYTES
  ) {
    return "Verification configuration exceeds 128 KiB of UTF-8 JSON.";
  }
  return null;
}

export class ProjectVerificationApiError extends Error {
  readonly status: number;

  constructor(status: number, message: string) {
    super(message);
    this.name = "ProjectVerificationApiError";
    this.status = status;
  }
}

const PROJECT_VERIFICATION_REFRESH_CONFLICT_MESSAGES = new Set([
  "Project workspace changed. Refresh verification settings and retry.",
  "Project verification settings changed. Refresh and retry.",
  "Verification configuration changed concurrently.",
  "Verification configuration or workspace changed before the run began.",
]);

const PROJECT_VERIFICATION_REVISION_CONFLICT =
  /^Verification configuration revision is \d+, not \d+\.$/;

export function verificationConflictRequiresRefresh(error: unknown): boolean {
  return (
    error instanceof ProjectVerificationApiError &&
    error.status === 409 &&
    (PROJECT_VERIFICATION_REFRESH_CONFLICT_MESSAGES.has(error.message) ||
      PROJECT_VERIFICATION_REVISION_CONFLICT.test(error.message))
  );
}

const PROJECT_VERIFICATION_EVIDENCE_REVISION_REGRESSION =
  "Verification evidence revision regressed. Refresh the run.";

export function projectVerificationPollingEvidenceRegressed(
  error: unknown,
): boolean {
  return (
    error instanceof ProjectVerificationApiError &&
    error.status === 409 &&
    error.message === PROJECT_VERIFICATION_EVIDENCE_REVISION_REGRESSION
  );
}

export function projectVerificationPollingEvidenceMarker(
  current: number | undefined,
  error: unknown,
): number | undefined {
  if (projectVerificationPollingEvidenceRegressed(error)) {
    return undefined;
  }
  return current;
}

export const PROJECT_VERIFICATION_UPDATED_EVENT =
  "unsloth-project-verification-updated";

type ProjectVerificationUpdatedDetail = {
  projectId: string;
  config?: ProjectVerificationConfig;
  run?: ProjectVerificationRun;
};

function projectVerificationPath(projectId: string, suffix = ""): string {
  const base = `/api/agent/projects/${encodeURIComponent(projectId)}`;
  return suffix ? `${base}/${suffix}` : base;
}

async function responseBody<T>(response: Response): Promise<T> {
  const body = await response.json().catch(() => null);
  if (!response.ok) {
    throw new ProjectVerificationApiError(
      response.status,
      formatApiErrorBody(body) ??
        `Project verification request failed (${response.status})`,
    );
  }
  return body as T;
}

async function request<T>(input: string, init?: RequestInit): Promise<T> {
  return responseBody<T>(await authFetch(input, init));
}

function jsonRequest<T>(
  input: string,
  method: "POST" | "PUT",
  body?: unknown,
): Promise<T> {
  return request<T>(input, {
    method,
    headers:
      body === undefined ? undefined : { "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
}

function notifyProjectVerificationUpdated(
  detail: ProjectVerificationUpdatedDetail,
): void {
  if (
    typeof window === "undefined" ||
    typeof window.dispatchEvent !== "function"
  ) {
    return;
  }
  window.dispatchEvent(
    new CustomEvent(PROJECT_VERIFICATION_UPDATED_EVENT, { detail }),
  );
}

export function subscribeProjectVerificationUpdated(
  projectId: string,
  listener: (detail: ProjectVerificationUpdatedDetail) => void,
): () => void {
  if (
    typeof window === "undefined" ||
    typeof window.addEventListener !== "function"
  ) {
    return () => {
      // No browser event target is available during server rendering.
    };
  }
  const handle = (event: Event) => {
    const detail = (event as CustomEvent<ProjectVerificationUpdatedDetail>)
      .detail;
    if (detail?.projectId === projectId) {
      listener(detail);
    }
  };
  window.addEventListener(PROJECT_VERIFICATION_UPDATED_EVENT, handle);
  return () =>
    window.removeEventListener(PROJECT_VERIFICATION_UPDATED_EVENT, handle);
}

export function getProjectVerificationConfig(
  projectId: string,
): Promise<ProjectVerificationConfig> {
  return request(projectVerificationPath(projectId, "verification"), {
    cache: "no-store",
  });
}

export async function saveProjectVerificationConfig(
  projectId: string,
  payload: SaveProjectVerificationConfig,
): Promise<ProjectVerificationConfig> {
  const config = await jsonRequest<ProjectVerificationConfig>(
    projectVerificationPath(projectId, "verification"),
    "PUT",
    payload,
  );
  notifyProjectVerificationUpdated({ projectId, config });
  return config;
}

export async function startProjectVerification(
  projectId: string,
  payload: StartProjectVerification,
): Promise<ProjectVerificationRun> {
  const run = await jsonRequest<ProjectVerificationRun>(
    projectVerificationPath(projectId, "verifications"),
    "POST",
    payload,
  );
  notifyProjectVerificationUpdated({ projectId, run });
  return run;
}

export async function listProjectVerificationRuns(
  projectId: string,
  limit = 10,
): Promise<ProjectVerificationRunSummary[]> {
  const query = new URLSearchParams({ limit: String(limit) });
  const body = await request<
    | ProjectVerificationRunMetadata[]
    | { runs: ProjectVerificationRunMetadata[] }
  >(`${projectVerificationPath(projectId, "verifications")}?${query}`, {
    cache: "no-store",
  });
  return (Array.isArray(body) ? body : body.runs).map(
    projectVerificationRunSummary,
  );
}

export async function getProjectVerificationRun(
  projectId: string,
  runId: string,
  afterEvidenceRevision?: number,
): Promise<ProjectVerificationRun | null> {
  const path = projectVerificationPath(
    projectId,
    `verifications/${encodeURIComponent(runId)}`,
  );
  const query =
    afterEvidenceRevision === undefined
      ? ""
      : `?${new URLSearchParams({
          afterEvidenceRevision: String(afterEvidenceRevision),
        })}`;
  const response = await authFetch(`${path}${query}`, { cache: "no-store" });
  if (response.status === 204) {
    return null;
  }
  return responseBody<ProjectVerificationRun>(response);
}

export async function cancelProjectVerificationRun(
  projectId: string,
  runId: string,
): Promise<ProjectVerificationRun> {
  const response = await jsonRequest<{
    cancelRequested: boolean;
    run: ProjectVerificationRun;
  }>(
    projectVerificationPath(
      projectId,
      `verifications/${encodeURIComponent(runId)}/cancel`,
    ),
    "POST",
  );
  const run = {
    ...response.run,
    cancelRequested: response.run.cancelRequested ?? response.cancelRequested,
  };
  notifyProjectVerificationUpdated({ projectId, run });
  return run;
}

function visibleUnicodeEscape(codePoint: number): string {
  const hexadecimal = codePoint.toString(16).toUpperCase();
  return codePoint <= 0xffff
    ? `\\u${hexadecimal.padStart(4, "0")}`
    : `\\u{${hexadecimal}}`;
}

function projectVerificationTextCodePointNeedsEscape(
  character: string,
  codePoint: number,
): boolean {
  return (
    codePoint < 0x20 ||
    (codePoint >= 0x7f && codePoint <= 0x9f) ||
    (codePoint >= 0xd800 && codePoint <= 0xdfff) ||
    (codePoint > 0x7f &&
      projectVerificationIsUnicodeWhitespace(character, codePoint)) ||
    projectVerificationIsUnsafeInvisible(character, codePoint)
  );
}

function visibleProjectVerificationCodePoint(
  character: string,
  codePoint: number,
): string {
  if (codePoint === 0x5c) {
    return "\\\\";
  }
  if (codePoint === 0x0a) {
    return "\\n\n";
  }
  if (codePoint === 0x09) {
    return "\\t";
  }
  return projectVerificationTextCodePointNeedsEscape(character, codePoint)
    ? visibleUnicodeEscape(codePoint)
    : character;
}

/** Render untrusted commands and output without terminal or bidi controls. */
export function visibleProjectVerificationText(value: string | null): string {
  if (value === null) {
    return "";
  }
  let visible = "";
  for (let index = 0; index < value.length; ) {
    const codePoint = value.codePointAt(index);
    if (codePoint === undefined) {
      break;
    }
    const character = String.fromCodePoint(codePoint);
    index += character.length;
    if (codePoint === 0x0d && value.codePointAt(index) === 0x0a) {
      index += 1;
      visible += "\\r\\n\n";
    } else {
      visible += visibleProjectVerificationCodePoint(character, codePoint);
    }
  }
  return visible;
}

/** Render every non-ASCII command code point as an unambiguous escape. */
export function visibleProjectVerificationCommand(value: string): string {
  let visible = "";
  for (const character of visibleProjectVerificationText(value)) {
    const codePoint = character.codePointAt(0);
    visible +=
      codePoint !== undefined && codePoint > 0x7e
        ? visibleUnicodeEscape(codePoint)
        : character;
  }
  return visible;
}

export function projectVerificationCommandNeedsExactPreview(
  value: string,
): boolean {
  return Array.from(value).some((character) => {
    const codePoint = character.codePointAt(0);
    return codePoint !== undefined && codePoint > 0x7e;
  });
}

export function projectVerificationRunIsActive(
  run: Pick<ProjectVerificationRunMetadata, "status"> | null | undefined,
): boolean {
  return (
    run?.status === "queued" ||
    run?.status === "running" ||
    run?.status === "cancelling"
  );
}

export function projectVerificationRunSummary(
  run: ProjectVerificationRunMetadata,
): ProjectVerificationRunSummary {
  return {
    id: run.id,
    projectId: run.projectId,
    status: run.status,
    configRevision: run.configRevision,
    workspaceRevision: run.workspaceRevision,
    evidenceRevision: run.evidenceRevision,
    cancelRequested: run.cancelRequested,
    error: run.error,
    startedAt: run.startedAt,
    updatedAt: run.updatedAt,
    completedAt: run.completedAt,
    historySequence: run.historySequence,
    sourceFreshness: run.sourceFreshness,
    evidenceStatus: "not_loaded",
  };
}

export function projectVerificationRunHasDetails(
  run: ProjectVerificationRun | ProjectVerificationRunSummary,
): run is ProjectVerificationRun {
  return "checks" in run && "results" in run;
}

export const PROJECT_VERIFICATION_POLL_INTERVAL_MS = 750;
export const PROJECT_VERIFICATION_POLL_MAX_BACKOFF_MS = 12_000;

const PROJECT_VERIFICATION_TERMINAL_POLL_STATUSES = new Set([401, 403, 404]);

export function projectVerificationPollingMustStop(error: unknown): boolean {
  return (
    error instanceof ProjectVerificationApiError &&
    PROJECT_VERIFICATION_TERMINAL_POLL_STATUSES.has(error.status)
  );
}

/** Mount-scoped retry state for one active verification run. */
export class ProjectVerificationPollBackoff {
  private failures = 0;

  successDelay(): number {
    this.failures = 0;
    return PROJECT_VERIFICATION_POLL_INTERVAL_MS;
  }

  failureDelay(error: unknown): number | null {
    if (projectVerificationPollingMustStop(error)) {
      return null;
    }
    this.failures += 1;
    const exponent = Math.min(this.failures, 4);
    return Math.min(
      PROJECT_VERIFICATION_POLL_INTERVAL_MS * 2 ** exponent,
      PROJECT_VERIFICATION_POLL_MAX_BACKOFF_MS,
    );
  }
}

/** Mount-scoped sequencing for reads and mutations that return snapshots. */
export class ProjectVerificationRequestGuard {
  private revision = 0;
  private mounted = false;

  activate(): void {
    this.mounted = true;
    this.revision += 1;
  }

  begin(): number {
    this.revision += 1;
    return this.revision;
  }

  accepts(revision: number): boolean {
    return this.mounted && revision === this.revision;
  }

  retire(): void {
    this.mounted = false;
    this.revision += 1;
  }
}
