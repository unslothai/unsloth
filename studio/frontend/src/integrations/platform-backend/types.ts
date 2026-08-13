export type PlatformCode = number | string;

export interface PlatformEnvelope<TData> {
  code: PlatformCode;
  message?: string;
  data?: TData;
}

export type PlatformQueryValue =
  | string
  | number
  | boolean
  | null
  | undefined
  | readonly (string | number | boolean)[];

export type PlatformQuery = Record<string, PlatformQueryValue>;

export type PlatformResponseType = "envelope" | "json" | "text" | "blob";

export interface PlatformHealthMeta {
  elapsed?: string;
  error?: string;
}

export interface PlatformSystemHealth {
  db?: string;
  redis?: string;
  doc_engine?: string;
  storage?: string;
  message_queue?: string;
  status: string;
  _meta?: Record<string, PlatformHealthMeta>;
}

export interface PlatformSseEvent {
  data: string;
  event?: string;
  id?: string;
  retry?: number;
  terminal: boolean;
}
