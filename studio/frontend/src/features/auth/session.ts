export const AUTH_TOKEN_KEY = "unsloth_auth_token";
export const ONBOARDING_DONE_KEY = "unsloth_onboarding_done";

type PostAuthRoute = "/onboarding" | "/studio";

function canUseStorage(): boolean {
  return typeof window !== "undefined";
}

export function hasAuthToken(): boolean {
  if (!canUseStorage()) return false;
  return Boolean(localStorage.getItem(AUTH_TOKEN_KEY));
}

export function storeAuthToken(accessToken: string): void {
  if (!canUseStorage()) return;
  localStorage.setItem(AUTH_TOKEN_KEY, accessToken);
}

export function isOnboardingDone(): boolean {
  if (!canUseStorage()) return false;
  return localStorage.getItem(ONBOARDING_DONE_KEY) === "true";
}

export function markOnboardingDone(): void {
  if (!canUseStorage()) return;
  localStorage.setItem(ONBOARDING_DONE_KEY, "true");
}

export function resetOnboardingDone(): void {
  if (!canUseStorage()) return;
  localStorage.removeItem(ONBOARDING_DONE_KEY);
}

export function getPostAuthRoute(): PostAuthRoute {
  return isOnboardingDone() ? "/studio" : "/onboarding";
}
