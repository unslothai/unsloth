


const HF_AUTH_ERROR_RE =
  /\b401\b|unauthorized|invalid.*token|invalid.*credential|authentication|forbidden|\b403\b/i;

export function isHfAuthError(message: string | null | undefined): boolean {
  return !!message && HF_AUTH_ERROR_RE.test(message);
}
