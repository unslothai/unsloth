import type { PlatformCode } from "./types";

export interface PlatformApiErrorOptions {
  httpStatus: number | null;
  code: PlatformCode;
  endpoint: string;
  requestId?: string;
  cause?: unknown;
}

export class PlatformApiError extends Error {
  readonly httpStatus: number | null;
  readonly code: PlatformCode;
  readonly endpoint: string;
  readonly requestId?: string;

  constructor(message: string, options: PlatformApiErrorOptions) {
    super(message, { cause: options.cause });
    this.name = "PlatformApiError";
    this.httpStatus = options.httpStatus;
    this.code = options.code;
    this.endpoint = options.endpoint;
    this.requestId = options.requestId;
  }

  get isTimeout(): boolean {
    return this.code === "CLIENT_TIMEOUT";
  }

  get isAbort(): boolean {
    return this.code === "CLIENT_ABORTED";
  }

  get isPermissionError(): boolean {
    return this.httpStatus === 401 || this.httpStatus === 403;
  }
}

export function isPlatformApiError(error: unknown): error is PlatformApiError {
  return error instanceof PlatformApiError;
}
