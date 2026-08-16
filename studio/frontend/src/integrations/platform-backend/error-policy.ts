import { isPlatformApiError } from "./errors";

export type PlatformUiErrorKind =
  | "authentication"
  | "permission"
  | "rate-limit"
  | "server"
  | "network"
  | "timeout"
  | "aborted"
  | "request";

export interface PlatformUiError {
  kind: PlatformUiErrorKind;
  message: string;
  retryable: boolean;
}

export function getPlatformUiError(error: unknown): PlatformUiError {
  if (!isPlatformApiError(error)) {
    return {
      kind: "request",
      message: "İşlem tamamlanamadı. Bilgileri kontrol edip yeniden deneyin.",
      retryable: false,
    };
  }
  if (error.isAbort) {
    return {
      kind: "aborted",
      message: "İstek iptal edildi.",
      retryable: false,
    };
  }
  if (error.isTimeout) {
    return {
      kind: "timeout",
      message: "Rag Platform isteği zaman aşımına uğradı.",
      retryable: true,
    };
  }
  if (error.httpStatus === 401) {
    return {
      kind: "authentication",
      message: "Oturumunuz sona erdi. Yeniden giriş yapın.",
      retryable: false,
    };
  }
  if (error.httpStatus === 403) {
    return {
      kind: "permission",
      message: "Bu işlem için yetkiniz yok.",
      retryable: false,
    };
  }
  if (error.httpStatus === 429) {
    return {
      kind: "rate-limit",
      message:
        "Çok fazla istek gönderildi. Kısa bir süre sonra yeniden deneyin.",
      retryable: true,
    };
  }
  if (error.httpStatus !== null && error.httpStatus >= 500) {
    return {
      kind: "server",
      message: "Rag Platform şu anda isteği tamamlayamıyor.",
      retryable: true,
    };
  }
  if (error.code === "NETWORK_ERROR") {
    return {
      kind: "network",
      message: "Rag Platform bağlantısı kurulamadı.",
      retryable: true,
    };
  }
  return {
    kind: "request",
    message: "İstek reddedildi. Bilgileri kontrol edip yeniden deneyin.",
    retryable: false,
  };
}
