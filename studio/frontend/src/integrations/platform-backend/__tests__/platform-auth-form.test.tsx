import forge from "node-forge";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { HttpResponse, http } from "msw";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { PlatformAuthForm } from "@/features/auth/components/platform-auth-form";
import { setLocale } from "@/i18n";
import { platformTestServer } from "./test-server";

const { navigate } = vi.hoisted(() => ({ navigate: vi.fn() }));

vi.mock("@tanstack/react-router", async (importOriginal) => {
  const actual =
    await importOriginal<typeof import("@tanstack/react-router")>();
  return { ...actual, useNavigate: () => navigate };
});

function capabilities(
  options: {
    channels?: Array<Record<string, unknown>>;
    password?: boolean;
    registration?: boolean;
  } = {},
) {
  platformTestServer.use(
    http.get("http://platform.test/api/v1/system/config", () =>
      HttpResponse.json({
        code: 0,
        data: {
          disablePasswordLogin: options.password === false,
          registerEnabled: options.registration === false ? 0 : 1,
        },
      }),
    ),
    http.get("http://platform.test/api/v1/auth/login/channels", () =>
      HttpResponse.json({ code: 0, data: options.channels ?? [] }),
    ),
  );
}

describe("PlatformAuthForm", () => {
  beforeEach(() => {
    setLocale("tr");
    vi.stubEnv("VITE_RAG_PLATFORM_BASE_URL", "http://platform.test");
    vi.stubEnv("VITE_RAG_PLATFORM_API_PREFIX", "/api/v1");
    localStorage.clear();
  });

  afterEach(() => vi.unstubAllEnvs());

  it("renders the complete sign-in surface in English", async () => {
    setLocale("en");
    capabilities();
    render(<PlatformAuthForm />);

    expect(screen.getByText("Loading sign-in options…")).toBeInTheDocument();
    await screen.findByRole("button", { name: "Sign in" });
    expect(screen.getByLabelText("Email")).toBeInTheDocument();
    expect(screen.getByLabelText("Password")).toBeInTheDocument();
  });

  it("waits for runtime capabilities and hides unavailable registration/OAuth choices", async () => {
    capabilities({ registration: false });
    render(<PlatformAuthForm />);

    expect(
      screen.getByText("Giriş seçenekleri yükleniyor…"),
    ).toBeInTheDocument();
    await screen.findByRole("button", { name: "Giriş yap" });
    expect(
      screen.queryByRole("button", { name: "Hesap oluştur" }),
    ).not.toBeInTheDocument();
    expect(screen.queryByText("Kurumsal giriş")).not.toBeInTheDocument();
    expect(screen.queryByText(/kayıt.*doğrulama/i)).not.toBeInTheDocument();
  });

  it("renders only returned OAuth channels and performs email login through the typed service", async () => {
    const pair = forge.pki.rsa.generateKeyPair({ bits: 1024, e: 0x10001 });
    vi.stubEnv(
      "VITE_RAG_PLATFORM_AUTH_PUBLIC_KEY_B64",
      btoa(forge.pki.publicKeyToPem(pair.publicKey)),
    );
    capabilities({
      channels: [{ channel: "github", display_name: "GitHub", icon: "github" }],
    });
    platformTestServer.use(
      http.post("http://platform.test/api/v1/auth/login", () =>
        HttpResponse.json(
          {
            code: 0,
            data: {
              id: "user-1",
              email: "user@example.test",
              nickname: "User",
              is_active: "1",
              is_superuser: false,
              login_channel: "password",
            },
          },
          { headers: { authorization: "opaque-ui-token" } },
        ),
      ),
    );
    render(<PlatformAuthForm />);

    expect(await screen.findByText("GitHub ile devam et")).toHaveAttribute(
      "href",
      "http://platform.test/api/v1/auth/login/github",
    );
    fireEvent.change(screen.getByLabelText("E-posta"), {
      target: { value: "user@example.test" },
    });
    fireEvent.change(screen.getByLabelText("Parola"), {
      target: { value: "password-1" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Giriş yap" }));

    await waitFor(() => expect(navigate).toHaveBeenCalledWith({ to: "/chat" }));
    expect(localStorage.getItem("rag-platform.auth-token")).toBe(
      "opaque-ui-token",
    );
  });

  it("keeps registration usable when password login is disabled at runtime", async () => {
    capabilities({ password: false, registration: true });
    render(<PlatformAuthForm />);

    await screen.findByRole("button", { name: "Hesap oluştur" });
    expect(screen.queryByLabelText("Parola")).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Hesap oluştur" }));

    expect(screen.getByLabelText("Parola")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Hesap oluştur" })).toBeEnabled();
  });

  it("clears loading state when an in-flight login is aborted by changing view", async () => {
    const pair = forge.pki.rsa.generateKeyPair({ bits: 1024, e: 0x10001 });
    vi.stubEnv(
      "VITE_RAG_PLATFORM_AUTH_PUBLIC_KEY_B64",
      btoa(forge.pki.publicKeyToPem(pair.publicKey)),
    );
    capabilities();
    platformTestServer.use(
      http.post("http://platform.test/api/v1/auth/login", async () => {
        await new Promise((resolve) => setTimeout(resolve, 200));
        return HttpResponse.json({ code: 100, message: "late response" });
      }),
    );
    render(<PlatformAuthForm />);

    await screen.findByRole("button", { name: "Giriş yap" });
    fireEvent.change(screen.getByLabelText("E-posta"), {
      target: { value: "user@example.test" },
    });
    fireEvent.change(screen.getByLabelText("Parola"), {
      target: { value: "password-1" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Giriş yap" }));
    expect(
      screen.getByRole("button", { name: "Lütfen bekleyin…" }),
    ).toBeDisabled();

    fireEvent.click(screen.getByRole("button", { name: "Hesap oluştur" }));
    expect(screen.getByRole("button", { name: "Hesap oluştur" })).toBeEnabled();
  });
});
