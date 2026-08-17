import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Spinner } from "@/components/ui/spinner";
import {
  CONNECTOR_OAUTH_MESSAGE,
  clearPendingConnectorOAuth,
  completeConnectorOAuthCallback,
  matchesConnectorOAuthCorrelation,
  readPendingConnectorOAuth,
  type ConnectorOAuthMessage,
  type PlatformConnectorOAuthSource,
} from "@/integrations/platform-backend";
import { useParams } from "@tanstack/react-router";
import { useEffect, useState } from "react";

function validSource(value: string): value is PlatformConnectorOAuthSource {
  return value === "google-drive" || value === "gmail" || value === "box";
}

export function ConnectorOAuthCallbackPage() {
  const { source: rawSource } = useParams({ strict: false }) as { source: string };
  const [status, setStatus] = useState<"loading" | "success" | "error">("loading");
  const [message, setMessage] = useState("Yetkilendirme sonucu doğrulanıyor.");

  useEffect(() => {
    const controller = new AbortController();
    const query = new URLSearchParams(window.location.search);
    const state = query.get("state") ?? "";
    const pending = readPendingConnectorOAuth();
    const source = validSource(rawSource) ? rawSource : null;
    const safeQuery = {
      state,
      code: query.get("code") ?? undefined,
      error: query.get("error") ?? undefined,
      errorDescription: query.get("error_description") ?? undefined,
    };
    window.history.replaceState({}, document.title, window.location.pathname);

    if (!source || !matchesConnectorOAuthCorrelation(source, state, window.name, pending)) {
      setStatus("error");
      setMessage("OAuth state doğrulaması başarısız oldu. İşlem güvenlik nedeniyle durduruldu.");
      return () => controller.abort();
    }

    void completeConnectorOAuthCallback(source, safeQuery, controller.signal)
      .then(({ success }) => {
        const nextStatus = success ? "success" : "error";
        const payload: ConnectorOAuthMessage = {
          type: CONNECTOR_OAUTH_MESSAGE,
          source,
          flowId: state,
          status: nextStatus,
        };
        setStatus(nextStatus);
        setMessage(success ? "Yetkilendirme tamamlandı." : "Yetkilendirme tamamlanamadı.");
        if (window.opener && !window.opener.closed) {
          window.opener.postMessage(payload, window.location.origin);
          window.setTimeout(() => window.close(), 250);
        } else {
          clearPendingConnectorOAuth(state);
          const params = new URLSearchParams({
            oauth_source: source,
            oauth_flow: state,
            oauth_status: nextStatus,
          });
          window.setTimeout(() => window.location.assign(`/files?${params}`), 250);
        }
      })
      .catch((error: unknown) => {
        setStatus("error");
        setMessage(error instanceof Error ? error.message : "Yetkilendirme tamamlanamadı.");
        const payload: ConnectorOAuthMessage = {
          type: CONNECTOR_OAUTH_MESSAGE,
          source,
          flowId: state,
          status: "error",
        };
        if (window.opener && !window.opener.closed) {
          window.opener.postMessage(payload, window.location.origin);
          window.setTimeout(() => window.close(), 250);
        } else {
          clearPendingConnectorOAuth(state);
          const params = new URLSearchParams({
            oauth_source: source,
            oauth_flow: state,
            oauth_status: "error",
          });
          window.setTimeout(() => window.location.assign(`/files?${params}`), 250);
        }
      });
    return () => controller.abort();
  }, [rawSource]);

  return (
    <main className="flex min-h-screen items-center justify-center bg-background p-6">
      <Alert className="max-w-lg">
        {status === "loading" ? <Spinner className="size-4" /> : null}
        <AlertTitle>{status === "error" ? "Yetkilendirme hatası" : "Rag Platform"}</AlertTitle>
        <AlertDescription>{message}</AlertDescription>
      </Alert>
    </main>
  );
}
