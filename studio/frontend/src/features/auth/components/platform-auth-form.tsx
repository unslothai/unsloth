import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  getPlatformAuthCapabilities,
  getPlatformAuthConfig,
  getPlatformOAuthLoginUrl,
  loginPlatformUser,
  registerPlatformUser,
  requestForgotPasswordCaptcha,
  resetForgottenPlatformPassword,
  sendForgotPasswordOtp,
  takePlatformOAuthError,
  verifyForgotPasswordOtp,
  type PlatformAuthCapabilities,
} from "@/integrations/platform-backend";
import { useNavigate } from "@tanstack/react-router";
import { Eye, EyeOff, RefreshCw } from "lucide-react";
import {
  type FormEvent,
  type ReactElement,
  useEffect,
  useRef,
  useState,
} from "react";
import {
  platformAuthErrorMessage,
  platformOAuthErrorMessage,
} from "../platform-auth-errors";

type AuthView = "login" | "register" | "forgot";
type RecoveryStep = "email" | "captcha" | "otp" | "reset";

function PasswordInput(props: {
  autoComplete: string;
  id: string;
  label: string;
  onChange: (value: string) => void;
  value: string;
}) {
  const [visible, setVisible] = useState(false);
  return (
    <div className="space-y-2">
      <Label htmlFor={props.id}>{props.label}</Label>
      <div className="relative">
        <Input
          id={props.id}
          type={visible ? "text" : "password"}
          autoComplete={props.autoComplete}
          value={props.value}
          onChange={(event) => props.onChange(event.target.value)}
          minLength={8}
          className="pr-10"
          required
        />
        <Button
          type="button"
          variant="ghost"
          size="icon"
          className="absolute right-0 top-0 h-full px-3 text-muted-foreground hover:bg-transparent"
          onClick={() => setVisible((current) => !current)}
          aria-label={visible ? "Parolayı gizle" : "Parolayı göster"}
        >
          {visible ? <EyeOff className="size-4" /> : <Eye className="size-4" />}
        </Button>
      </div>
    </div>
  );
}

export function PlatformAuthForm(): ReactElement {
  const navigate = useNavigate();
  const authConfig = getPlatformAuthConfig();
  const [view, setView] = useState<AuthView>("login");
  const [capabilities, setCapabilities] =
    useState<PlatformAuthCapabilities | null>(null);
  const [capabilityLoading, setCapabilityLoading] = useState(true);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [email, setEmail] = useState("");
  const [nickname, setNickname] = useState("");
  const [password, setPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [captcha, setCaptcha] = useState("");
  const [otp, setOtp] = useState("");
  const [recoveryStep, setRecoveryStep] = useState<RecoveryStep>("email");
  const [captchaUrl, setCaptchaUrl] = useState<string | null>(null);
  const requestRef = useRef<AbortController | null>(null);
  const captchaUrlRef = useRef<string | null>(null);

  function replaceCaptchaUrl(next: string | null) {
    if (captchaUrlRef.current) URL.revokeObjectURL(captchaUrlRef.current);
    captchaUrlRef.current = next;
    setCaptchaUrl(next);
  }

  function loadCapabilities() {
    requestRef.current?.abort();
    const controller = new AbortController();
    requestRef.current = controller;
    setCapabilityLoading(true);
    setError(null);
    void getPlatformAuthCapabilities(controller.signal)
      .then(setCapabilities)
      .catch((caught: unknown) => {
        if (!controller.signal.aborted) setError(platformAuthErrorMessage(caught));
      })
      .finally(() => {
        if (requestRef.current === controller) {
          requestRef.current = null;
          setCapabilityLoading(false);
        }
      });
  }

  useEffect(() => {
    const oauthError = takePlatformOAuthError();
    if (oauthError) setError(platformOAuthErrorMessage(oauthError));
    loadCapabilities();
    return () => {
      const request = requestRef.current;
      requestRef.current = null;
      request?.abort();
      if (captchaUrlRef.current) URL.revokeObjectURL(captchaUrlRef.current);
      captchaUrlRef.current = null;
    };
  }, []);

  function resetMessages() {
    setError(null);
    setNotice(null);
  }

  function switchView(next: AuthView) {
    requestRef.current?.abort();
    requestRef.current = null;
    setLoading(false);
    replaceCaptchaUrl(null);
    resetMessages();
    setView(next);
    setRecoveryStep("email");
    setCaptcha("");
    setOtp("");
    setPassword("");
    setConfirmPassword("");
  }

  async function runRequest(action: (signal: AbortSignal) => Promise<void>) {
    requestRef.current?.abort();
    const controller = new AbortController();
    requestRef.current = controller;
    setLoading(true);
    resetMessages();
    try {
      await action(controller.signal);
    } catch (caught) {
      if (!controller.signal.aborted) setError(platformAuthErrorMessage(caught));
    } finally {
      if (requestRef.current === controller) {
        requestRef.current = null;
        setLoading(false);
      }
    }
  }

  async function submitLogin() {
    await runRequest(async (signal) => {
      await loginPlatformUser(email, password, {
        publicKeyPem: authConfig.publicKeyPem,
        signal,
      });
      await navigate({ to: "/chat" });
    });
  }

  async function submitRegistration() {
    if (password !== confirmPassword) {
      setError("Parolalar eşleşmiyor.");
      return;
    }
    await runRequest(async (signal) => {
      await registerPlatformUser(
        { email, nickname, password },
        { publicKeyPem: authConfig.publicKeyPem, signal },
      );
      await navigate({ to: "/chat" });
    });
  }

  async function submitRecovery() {
    if (recoveryStep === "email") {
      await runRequest(async (signal) => {
        const blob = await requestForgotPasswordCaptcha(email, signal);
        replaceCaptchaUrl(URL.createObjectURL(blob));
        setRecoveryStep("captcha");
      });
    } else if (recoveryStep === "captcha") {
      await runRequest(async (signal) => {
        await sendForgotPasswordOtp(email, captcha, signal);
        replaceCaptchaUrl(null);
        setRecoveryStep("otp");
        setNotice("Doğrulama kodu e-posta adresinize gönderildi.");
      });
    } else if (recoveryStep === "otp") {
      await runRequest(async (signal) => {
        await verifyForgotPasswordOtp(email, otp, signal);
        setRecoveryStep("reset");
        setNotice("Kod doğrulandı. Yeni parolanızı belirleyin.");
      });
    } else if (password !== confirmPassword) {
      setError("Parolalar eşleşmiyor.");
    } else {
      await runRequest(async (signal) => {
        await resetForgottenPlatformPassword(
          { email, password },
          { publicKeyPem: authConfig.publicKeyPem, signal },
        );
        await navigate({ to: "/chat" });
      });
    }
  }

  function refreshCaptcha() {
    void runRequest(async (signal) => {
      const blob = await requestForgotPasswordCaptcha(email, signal);
      replaceCaptchaUrl(URL.createObjectURL(blob));
      setCaptcha("");
    });
  }

  function submit(event: FormEvent) {
    event.preventDefault();
    if (view === "login") void submitLogin();
    else if (view === "register") void submitRegistration();
    else void submitRecovery();
  }

  const registrationAvailable =
    authConfig.registrationEnabled && capabilities?.registrationEnabled === true;
  const oauthChannels = authConfig.oauthEnabled
    ? (capabilities?.loginChannels ?? [])
    : [];
  const passwordLoginAvailable = capabilities?.passwordLoginEnabled !== false;

  return (
    <div className="w-full max-w-sm space-y-6">
      <div className="space-y-1.5 text-center">
        <p className="text-xs font-semibold uppercase tracking-[0.24em] text-primary">
          Rag Platform
        </p>
        <h2 className="text-2xl font-semibold text-foreground">
          {view === "login"
            ? "Tekrar hoş geldiniz"
            : view === "register"
              ? "Hesap oluşturun"
              : "Parolanızı yenileyin"}
        </h2>
        <p className="text-sm text-muted-foreground">
          {view === "forgot"
            ? "Doğrulama adımlarını güvenle tamamlayın."
            : "E-posta adresinizle güvenli oturum açın."}
        </p>
      </div>

      {capabilityLoading ? (
        <p className="text-center text-sm text-muted-foreground" role="status">
          Giriş seçenekleri yükleniyor…
        </p>
      ) : capabilities === null ? (
        <div className="space-y-3 text-center">
          <p className="text-sm text-destructive">{error}</p>
          <Button type="button" variant="outline" onClick={loadCapabilities}>
            <RefreshCw className="mr-2 size-4" /> Yeniden dene
          </Button>
        </div>
      ) : (
        <>
          <form className="space-y-4" onSubmit={submit}>
            {(view !== "forgot" || recoveryStep === "email") && (
              <div className="space-y-2">
                <Label htmlFor="platform-email">E-posta</Label>
                <Input
                  id="platform-email"
                  type="email"
                  autoComplete="email"
                  value={email}
                  onChange={(event) => setEmail(event.target.value)}
                  required
                />
              </div>
            )}
            {view === "register" && (
              <div className="space-y-2">
                <Label htmlFor="platform-nickname">Görünen ad</Label>
                <Input
                  id="platform-nickname"
                  value={nickname}
                  onChange={(event) => setNickname(event.target.value)}
                  autoComplete="name"
                  required
                />
              </div>
            )}
            {((view === "login" && passwordLoginAvailable) ||
              view === "register" ||
              (view === "forgot" && recoveryStep === "reset")) && (
                <PasswordInput
                  id="platform-password"
                  label={recoveryStep === "reset" ? "Yeni parola" : "Parola"}
                  value={password}
                  onChange={setPassword}
                  autoComplete={view === "login" ? "current-password" : "new-password"}
                />
              )}
            {(view === "register" ||
              (view === "forgot" && recoveryStep === "reset")) && (
              <PasswordInput
                id="platform-confirm-password"
                label="Parolayı doğrula"
                value={confirmPassword}
                onChange={setConfirmPassword}
                autoComplete="new-password"
              />
            )}
            {view === "forgot" && recoveryStep !== "email" && (
              <p className="rounded-lg bg-muted px-3 py-2 text-xs text-muted-foreground">
                {email}
              </p>
            )}
            {view === "forgot" && recoveryStep === "captcha" && (
              <div className="space-y-3">
                {captchaUrl ? (
                  <img
                    src={captchaUrl}
                    alt="Güvenlik kodu"
                    className="h-24 w-full rounded-lg border bg-white object-contain"
                  />
                ) : null}
                <Button
                  type="button"
                  variant="outline"
                  size="sm"
                  onClick={refreshCaptcha}
                  disabled={loading}
                >
                  <RefreshCw className="mr-2 size-3.5" /> Yeni güvenlik kodu
                </Button>
                <div className="space-y-2">
                  <Label htmlFor="platform-captcha">Güvenlik kodu</Label>
                  <Input
                    id="platform-captcha"
                    value={captcha}
                    onChange={(event) => setCaptcha(event.target.value)}
                    autoComplete="off"
                    required
                  />
                </div>
              </div>
            )}
            {view === "forgot" && recoveryStep === "otp" && (
              <div className="space-y-2">
                <Label htmlFor="platform-otp">E-posta doğrulama kodu</Label>
                <Input
                  id="platform-otp"
                  value={otp}
                  onChange={(event) => setOtp(event.target.value.toUpperCase())}
                  autoComplete="one-time-code"
                  required
                />
              </div>
            )}
            {notice ? (
              <p className="text-center text-sm text-emerald-700" role="status">
                {notice}
              </p>
            ) : null}
            {error ? (
              <p className="text-center text-sm text-destructive" role="alert">
                {error}
              </p>
            ) : null}
            {view !== "login" || passwordLoginAvailable ? (
              <Button type="submit" className="w-full" disabled={loading}>
                {loading
                  ? "Lütfen bekleyin…"
                  : view === "login"
                    ? "Giriş yap"
                    : view === "register"
                      ? "Hesap oluştur"
                      : recoveryStep === "email"
                        ? "Güvenlik kodunu getir"
                        : recoveryStep === "captcha"
                          ? "Doğrulama kodu gönder"
                          : recoveryStep === "otp"
                            ? "Kodu doğrula"
                            : "Parolayı yenile"}
              </Button>
            ) : null}
          </form>
          {view === "login" && oauthChannels.length > 0 ? (
            <div className="space-y-2 border-t pt-4">
              <p className="text-center text-xs text-muted-foreground">Kurumsal giriş</p>
              {oauthChannels.map((channel) => (
                <Button key={channel.channel} asChild variant="outline" className="w-full">
                  <a href={getPlatformOAuthLoginUrl(channel.channel)}>
                    {channel.displayName} ile devam et
                  </a>
                </Button>
              ))}
            </div>
          ) : null}
          <div className="flex flex-wrap justify-center gap-x-4 gap-y-2 text-sm">
            {view === "login" && registrationAvailable ? (
              <button
                type="button"
                className="text-primary hover:underline"
                onClick={() => switchView("register")}
              >
                Hesap oluştur
              </button>
            ) : null}
            {view === "login" && authConfig.passwordRecoveryEnabled ? (
              <button
                type="button"
                className="text-primary hover:underline"
                onClick={() => switchView("forgot")}
              >
                Parolamı unuttum
              </button>
            ) : null}
            {view !== "login" ? (
              <button
                type="button"
                className="text-primary hover:underline"
                onClick={() => switchView("login")}
              >
                Girişe dön
              </button>
            ) : null}
          </div>
        </>
      )}
    </div>
  );
}
