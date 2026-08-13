import { platformRequest } from "./client";
import { encryptPlatformPassword } from "./auth-crypto";
import { getPlatformAuthConfig, resolvePlatformUrl } from "./config";
import { PlatformApiError, isPlatformApiError } from "./errors";
import {
  clearPlatformSession,
  getPlatformSessionToken,
  storePlatformSessionToken,
  usePlatformSessionStore,
} from "./auth-session";
import {
  mapPlatformLoginChannel,
  mapPlatformTenantModels,
  mapPlatformUser,
  type PlatformAuthCapabilities,
  type PlatformAuthResult,
  type PlatformLoginChannel,
  type PlatformLoginChannelDto,
  type PlatformProfileUpdate,
  type PlatformSystemAuthConfigDto,
  type PlatformTenantModels,
  type PlatformTenantModelsDto,
  type PlatformUser,
  type PlatformUserDto,
} from "./auth-types";

const LOGIN_ENDPOINT = "/auth/login";
const REGISTER_ENDPOINT = "/users";
const OAUTH_COOKIE_NAME = "ragflow_auth";
const OAUTH_ERROR_STORAGE_KEY = "rag-platform.oauth-error";

function normalizeAuthorizationHeader(value: string | null): string {
  const token = value?.trim().replace(/^Bearer\s+/i, "") ?? "";
  if (!token) {
    throw new PlatformApiError("Rag Platform oturum anahtarı döndürmedi.", {
      httpStatus: 200,
      code: "AUTH_HEADER_MISSING",
      endpoint: LOGIN_ENDPOINT,
    });
  }
  return token;
}

async function passwordAuthRequest(
  endpoint: string,
  body: Record<string, unknown>,
  signal?: AbortSignal,
): Promise<PlatformAuthResult> {
  let authorization: string | null = null;
  const dto = await platformRequest<PlatformUserDto>(endpoint, {
    method: "POST",
    token: null,
    json: body,
    signal,
    onResponse: (response) => {
      authorization = response.headers.get("authorization");
    },
  });
  const token = normalizeAuthorizationHeader(authorization);
  const user = mapPlatformUser(dto);
  if (!user.id || !user.email) {
    throw new PlatformApiError("Rag Platform kullanıcı yanıtı geçersiz.", {
      httpStatus: 200,
      code: "INVALID_USER_RESPONSE",
      endpoint,
    });
  }
  storePlatformSessionToken(token);
  usePlatformSessionStore.getState().setUser(user);
  return { token, user };
}

export function loginPlatformUser(
  email: string,
  password: string,
  options: { publicKeyPem?: string; signal?: AbortSignal } = {},
): Promise<PlatformAuthResult> {
  return passwordAuthRequest(
    LOGIN_ENDPOINT,
    {
      email: email.trim(),
      password: encryptPlatformPassword(password, options.publicKeyPem),
    },
    options.signal,
  );
}

export function registerPlatformUser(
  input: { email: string; nickname: string; password: string },
  options: { publicKeyPem?: string; signal?: AbortSignal } = {},
): Promise<PlatformAuthResult> {
  return passwordAuthRequest(
    REGISTER_ENDPOINT,
    {
      email: input.email.trim(),
      nickname: input.nickname.trim(),
      password: encryptPlatformPassword(input.password, options.publicKeyPem),
    },
    options.signal,
  );
}

export async function getPlatformAuthCapabilities(
  signal?: AbortSignal,
): Promise<PlatformAuthCapabilities> {
  const [system, channelDtos] = await Promise.all([
    platformRequest<PlatformSystemAuthConfigDto>("/system/config", {
      token: null,
      signal,
      getRetries: 1,
    }),
    platformRequest<PlatformLoginChannelDto[]>("/auth/login/channels", {
      token: null,
      signal,
      getRetries: 1,
    }),
  ]);
  return {
    passwordLoginEnabled: system.disablePasswordLogin !== true,
    registrationEnabled:
      system.registerEnabled === true || system.registerEnabled === 1,
    loginChannels: channelDtos
      .map(mapPlatformLoginChannel)
      .filter((channel): channel is PlatformLoginChannel => channel !== null),
  };
}

export async function getCurrentPlatformUser(
  signal?: AbortSignal,
): Promise<PlatformUser> {
  const dto = await platformRequest<PlatformUserDto>("/users/me", {
    signal,
    getRetries: 0,
  });
  const user = mapPlatformUser(dto);
  if (!user.id || !user.email) {
    throw new PlatformApiError("Rag Platform kullanıcı yanıtı geçersiz.", {
      httpStatus: 200,
      code: "INVALID_USER_RESPONSE",
      endpoint: "/users/me",
    });
  }
  usePlatformSessionStore.getState().setUser(user);
  return user;
}

let hydrateRequest: Promise<PlatformUser> | null = null;

export async function hydratePlatformSession(
  signal?: AbortSignal,
): Promise<PlatformUser> {
  if (!getPlatformSessionToken()) {
    throw new PlatformApiError("Oturum bulunamadı.", {
      httpStatus: 401,
      code: "NO_SESSION",
      endpoint: "/users/me",
    });
  }
  if (hydrateRequest && !signal) return hydrateRequest;
  usePlatformSessionStore.getState().setHydrating();
  const request = getCurrentPlatformUser(signal).catch((error: unknown) => {
    if (isPlatformApiError(error) && error.httpStatus === 401) {
      clearPlatformSession();
    } else if (!isPlatformApiError(error) || !error.isAbort) {
      usePlatformSessionStore
        .getState()
        .setError(error instanceof Error ? error.message : "Oturum okunamadı.");
    }
    throw error;
  });
  if (!signal) hydrateRequest = request;
  try {
    return await request;
  } finally {
    if (hydrateRequest === request) hydrateRequest = null;
  }
}

export async function logoutPlatformUser(signal?: AbortSignal): Promise<void> {
  try {
    if (getPlatformSessionToken()) {
      await platformRequest<boolean>("/auth/logout", {
        method: "POST",
        signal,
        redirectOnUnauthorized: false,
      });
    }
  } finally {
    clearPlatformSession();
  }
}

export async function updatePlatformProfile(
  update: PlatformProfileUpdate,
  signal?: AbortSignal,
): Promise<PlatformUser> {
  await platformRequest<boolean>("/users/me", {
    method: "PATCH",
    json: {
      ...(update.nickname === undefined ? {} : { nickname: update.nickname }),
      ...(update.avatar === undefined ? {} : { avatar: update.avatar }),
      ...(update.language === undefined ? {} : { language: update.language }),
      ...(update.timezone === undefined ? {} : { timezone: update.timezone }),
      ...(update.colorScheme === undefined
        ? {}
        : { color_schema: update.colorScheme }),
    },
    signal,
  });
  return getCurrentPlatformUser(signal);
}

export async function changePlatformPassword(
  currentPassword: string,
  newPassword: string,
  options: { publicKeyPem?: string; signal?: AbortSignal } = {},
): Promise<void> {
  await platformRequest<boolean>("/users/me", {
    method: "PATCH",
    json: {
      password: encryptPlatformPassword(currentPassword, options.publicKeyPem),
      new_password: encryptPlatformPassword(newPassword, options.publicKeyPem),
    },
    signal: options.signal,
    redirectOnUnauthorized: false,
  });
  clearPlatformSession();
}

export function requestForgotPasswordCaptcha(
  email: string,
  signal?: AbortSignal,
): Promise<Blob> {
  return platformRequest<Blob>("/auth/password/forgot/captcha", {
    method: "POST",
    token: null,
    query: { email: email.trim() },
    responseType: "blob",
    signal,
  });
}

export function sendForgotPasswordOtp(
  email: string,
  captcha: string,
  signal?: AbortSignal,
): Promise<boolean> {
  return platformRequest<boolean>("/auth/password/forgot/otp", {
    method: "POST",
    token: null,
    json: { email: email.trim(), captcha: captcha.trim() },
    signal,
  });
}

export function verifyForgotPasswordOtp(
  email: string,
  otp: string,
  signal?: AbortSignal,
): Promise<boolean> {
  return platformRequest<boolean>("/auth/password/forgot/otp/verify", {
    method: "POST",
    token: null,
    json: { email: email.trim(), otp: otp.trim() },
    signal,
  });
}

export function resetForgottenPlatformPassword(
  input: { email: string; password: string },
  options: { publicKeyPem?: string; signal?: AbortSignal } = {},
): Promise<PlatformAuthResult> {
  const encrypted = encryptPlatformPassword(
    input.password,
    options.publicKeyPem,
  );
  return passwordAuthRequest(
    "/auth/password/reset",
    {
      email: input.email.trim(),
      new_password: encrypted,
      confirm_new_password: encrypted,
    },
    options.signal,
  );
}

export async function getCurrentPlatformTenantModels(
  signal?: AbortSignal,
): Promise<PlatformTenantModels> {
  const dto = await platformRequest<PlatformTenantModelsDto>(
    "/users/me/models",
    { signal, getRetries: 0 },
  );
  return mapPlatformTenantModels(dto);
}

export async function updateCurrentPlatformTenantModels(
  models: PlatformTenantModels,
  signal?: AbortSignal,
): Promise<PlatformTenantModels> {
  await platformRequest<boolean>("/users/me/models", {
    method: "PATCH",
    json: {
      tenant_id: models.tenantId,
      llm_id: models.chatModelId,
      embd_id: models.embeddingModelId,
      rerank_id: models.rerankModelId,
      asr_id: models.asrModelId,
      tts_id: models.textToSpeechModelId,
      img2txt_id: models.imageToTextModelId,
      ocr_id: models.ocrModelId,
    },
    signal,
  });
  return getCurrentPlatformTenantModels(signal);
}

export function getPlatformOAuthLoginUrl(channel: string): string {
  if (!/^[a-zA-Z0-9_-]+$/.test(channel)) {
    throw new TypeError("Geçersiz giriş kanalı.");
  }
  return resolvePlatformUrl(`/auth/login/${encodeURIComponent(channel)}`);
}

function readCookie(name: string): string {
  if (typeof document === "undefined") return "";
  const prefix = `${name}=`;
  const item = document.cookie
    .split(";")
    .map((part) => part.trim())
    .find((part) => part.startsWith(prefix));
  if (!item) return "";
  try {
    return decodeURIComponent(item.slice(prefix.length));
  } catch {
    return "";
  }
}

function clearOAuthCookie(): void {
  if (typeof document === "undefined") return;
  document.cookie = `${OAUTH_COOKIE_NAME}=; Path=/; Max-Age=0; SameSite=Lax`;
}

export type PlatformOAuthRedirectResult =
  | { handled: false }
  | { handled: true; status: "success" }
  | { handled: true; status: "error"; error: string };

export function consumePlatformOAuthRedirect(
  href = typeof window === "undefined" ? "http://localhost/" : window.location.href,
): PlatformOAuthRedirectResult {
  const url = new URL(href, "http://localhost");
  const backendError = url.searchParams.get("error")?.trim();
  const hasAuthMarker = url.searchParams.has("auth");
  if (!backendError && !hasAuthMarker) return { handled: false };

  if (backendError) {
    clearOAuthCookie();
    const allowed = new Set([
      "invalid_state",
      "missing_code",
      "token_failed",
      "email_missing",
      "user_inactive",
      "server_error",
    ]);
    const safeError = allowed.has(backendError) ? backendError : "oauth_failed";
    if (typeof sessionStorage !== "undefined") {
      sessionStorage.setItem(OAUTH_ERROR_STORAGE_KEY, safeError);
    }
    return { handled: true, status: "error", error: safeError };
  }

  const cookieToken = readCookie(OAUTH_COOKIE_NAME);
  clearOAuthCookie();
  // The active Go callback puts a user id in `auth` and the opaque token in a
  // short-lived cookie. Never accept a URL query value as a credential.
  if (!cookieToken) {
    if (typeof sessionStorage !== "undefined") {
      sessionStorage.setItem(OAUTH_ERROR_STORAGE_KEY, "oauth_session_missing");
    }
    return {
      handled: true,
      status: "error",
      error: "oauth_session_missing",
    };
  }
  storePlatformSessionToken(cookieToken);
  return { handled: true, status: "success" };
}

export function takePlatformOAuthError(): string | null {
  if (typeof sessionStorage === "undefined") return null;
  const value = sessionStorage.getItem(OAUTH_ERROR_STORAGE_KEY);
  sessionStorage.removeItem(OAUTH_ERROR_STORAGE_KEY);
  return value;
}

export function resetPlatformAuthRequestsForTests(): void {
  hydrateRequest = null;
}

export function platformAuthRollout() {
  return getPlatformAuthConfig();
}
