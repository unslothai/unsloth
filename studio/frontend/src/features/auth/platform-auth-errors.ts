import {
  PlatformAuthConfigurationError,
  isPlatformApiError,
} from "@/integrations/platform-backend";
import { translate } from "@/i18n";

export function platformAuthErrorMessage(error: unknown): string {
  if (error instanceof PlatformAuthConfigurationError) {
    return translate("platformAuth.errors.missingSession");
  }
  if (!isPlatformApiError(error)) {
    return translate("platformAuth.errors.generic");
  }
  if (error.code === "CLIENT_TIMEOUT") {
    return translate("platformAuth.errors.timeout");
  }
  if (error.code === "NETWORK_ERROR") {
    return translate("platformAuth.errors.network");
  }
  if (error.code === "AUTH_HEADER_MISSING") {
    return translate("platformAuth.errors.missingSession");
  }
  const message = error.message.toLowerCase();
  if (
    message.includes("do not match") ||
    message.includes("not registered") ||
    message.includes("unauthorized") ||
    error.httpStatus === 401
  ) {
    return translate("platformAuth.errors.invalidCredentials");
  }
  if (message.includes("already registered")) {
    return translate("platformAuth.errors.alreadyRegistered");
  }
  if (message.includes("invalid email")) {
    return translate("platformAuth.errors.invalidEmail");
  }
  if (message.includes("disabled")) {
    return translate("platformAuth.errors.disabled");
  }
  if (message.includes("captcha")) {
    return translate("platformAuth.errors.captcha");
  }
  if (message.includes("otp") || message.includes("verification")) {
    return translate("platformAuth.errors.verification");
  }
  if (message.includes("password error")) {
    return translate("platformAuth.errors.invalidPassword");
  }
  if (error.httpStatus === 403) {
    return translate("platformAuth.errors.forbidden");
  }
  return translate("platformAuth.errors.invalidInput");
}

export function platformOAuthErrorMessage(code: string): string {
  switch (code) {
    case "invalid_state":
      return translate("platformAuth.errors.oauth.invalidState");
    case "missing_code":
      return translate("platformAuth.errors.oauth.missingCode");
    case "token_failed":
      return translate("platformAuth.errors.oauth.tokenFailed");
    case "email_missing":
      return translate("platformAuth.errors.oauth.emailMissing");
    case "user_inactive":
      return translate("platformAuth.errors.oauth.inactive");
    case "oauth_session_missing":
      return translate("platformAuth.errors.oauth.sessionMissing");
    default:
      return translate("platformAuth.errors.oauth.generic");
  }
}
