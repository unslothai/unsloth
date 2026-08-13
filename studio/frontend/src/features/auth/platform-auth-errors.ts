import {
  PlatformAuthConfigurationError,
  isPlatformApiError,
} from "@/integrations/platform-backend";

export function platformAuthErrorMessage(error: unknown): string {
  if (error instanceof PlatformAuthConfigurationError) return error.message;
  if (!isPlatformApiError(error)) {
    return "İşlem tamamlanamadı. Lütfen yeniden deneyin.";
  }
  if (error.code === "CLIENT_TIMEOUT") {
    return "İstek zaman aşımına uğradı. Lütfen yeniden deneyin.";
  }
  if (error.code === "NETWORK_ERROR") {
    return "Rag Platform bağlantısı kurulamadı.";
  }
  if (error.code === "AUTH_HEADER_MISSING") {
    return "Güvenli oturum başlatılamadı. Lütfen yeniden giriş yapın.";
  }
  const message = error.message.toLowerCase();
  if (
    message.includes("do not match") ||
    message.includes("not registered") ||
    message.includes("unauthorized") ||
    error.httpStatus === 401
  ) {
    return "E-posta veya parola hatalı.";
  }
  if (message.includes("already registered")) {
    return "Bu e-posta adresiyle bir hesap zaten var.";
  }
  if (message.includes("invalid email")) return "Geçerli bir e-posta girin.";
  if (message.includes("disabled")) {
    return "Bu hesap devre dışı. Yöneticinizle iletişime geçin.";
  }
  if (message.includes("captcha")) {
    return "Güvenlik kodu hatalı veya süresi dolmuş.";
  }
  if (message.includes("otp") || message.includes("verification")) {
    return "Doğrulama kodu hatalı veya süresi dolmuş.";
  }
  if (message.includes("password error")) return "Mevcut parola hatalı.";
  if (error.httpStatus === 403) return "Bu işlem için yetkiniz yok.";
  return "İşlem tamamlanamadı. Lütfen bilgilerinizi kontrol edip yeniden deneyin.";
}

export function platformOAuthErrorMessage(code: string): string {
  switch (code) {
    case "invalid_state":
      return "Giriş doğrulaması güvenlik kontrolünü geçemedi.";
    case "missing_code":
      return "Giriş sağlayıcısı yetkilendirme kodu döndürmedi.";
    case "token_failed":
      return "Giriş sağlayıcısıyla güvenli oturum kurulamadı.";
    case "email_missing":
      return "Giriş sağlayıcısı doğrulanmış e-posta bilgisi döndürmedi.";
    case "user_inactive":
      return "Bu hesap devre dışı.";
    case "oauth_session_missing":
      return "Giriş oturumu bulunamadı veya süresi doldu.";
    default:
      return "Harici giriş tamamlanamadı.";
  }
}
