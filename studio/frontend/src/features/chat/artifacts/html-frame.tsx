// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"use client";

import {
  Alert,
  AlertAction,
  AlertDescription,
  AlertTitle,
} from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
// eslint-disable-next-line no-restricted-imports -- the settings barrel imports this feature back
import { useSettingsDialogStore } from "@/features/settings/stores/settings-dialog-store";
import { useLocale, useT } from "@/i18n";
import { apiUrl } from "@/lib/api-base";
import { cn } from "@/lib/utils";
import { ShieldAlertIcon, XIcon } from "lucide-react";
import {
  type RefObject,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { useChatRuntimeStore } from "../stores/chat-runtime-store";
import { hashArtifactCode } from "./types";

const HTML_FRAME_DEFAULT_HEIGHT = 400;
const HTML_FRAME_MAX_HEIGHT = 900;
const BLOCKED_HOSTS_SHOWN = 3;
// Entries are per URL, not per host, so a page pulling a whole CDN directory counts each file.
// Still far above what a real one trips.
const BLOCKED_URIS_TRACKED = 100;
// A report carries the full URL, so this is generous next to a real one, and it bounds both
// the stored string and the host derived from it.
const BLOCKED_URI_MAX_CHARS = 2048;

// The only directives the network CSP leaves at 'none'. Kept in step by
// test_the_grant_widens_everything_but_the_locked_directives.
const GRANT_CANNOT_FIX = new Set(["object-src", "base-uri", "form-action"]);

// A hostless blockedURI is a bare scheme, and the permissive policy widens every one but this:
// its worker-src is `http: https: blob:` with no data:, so a data: Worker stays blocked
// after the grant. Kept in step by test_the_permissive_policy_widens_every_hostless_scheme_but_one.
const GRANT_CANNOT_FIX_SCHEME: Record<string, string> = { "worker-src": "data" };

type BlockedState = { code: string; uris: string[]; hosts: string[] };

const NOTHING_BLOCKED: BlockedState = { code: "", uris: [], hosts: [] };

// Reports from before a swap belong to the old canvas, so start over rather than append. The
// cap is checked BEFORE the duplicate scan, so past it a canvas posting unique URIs cannot
// make the parent rescan every stored string; returning `current` also lets React bail out.
function appendBlocked(
  current: BlockedState,
  code: string,
  uri: string,
  host: string,
): BlockedState {
  const mine = current.code === code ? current : { code, uris: [], hosts: [] };
  if (mine.uris.length >= BLOCKED_URIS_TRACKED || mine.uris.includes(uri)) {
    return mine === current ? current : mine;
  }
  return {
    code,
    uris: [...mine.uris, uri],
    hosts: mine.hosts.includes(host) ? mine.hosts : [...mine.hosts, host],
  };
}

// A non-HTTP(S) violation reports a bare token ("eval", "blob"), which the permissive CSP
// widens too. Dropping those left the canvas blank with no prompt, so label them with the
// token itself.
const BLOCKED_KEYWORD = /^[a-z-]+$/;

function blockedHost(uri: string): string | null {
  if (BLOCKED_KEYWORD.test(uri)) return uri;
  try {
    return new URL(uri).host || null;
  } catch {
    return null;
  }
}

export type ArtifactViewMode = "preview" | "source";
export const ARTIFACT_VIEW_MODES: readonly ArtifactViewMode[] = [
  "preview",
  "source",
];

export function isArtifactViewMode(value: string): value is ArtifactViewMode {
  return (ARTIFACT_VIEW_MODES as readonly string[]).includes(value);
}

export function buildArtifactSrcDoc(code: string): string {
  const resizeScript = `<script>(()=>{const post=()=>parent.postMessage({chatArtifactHeight:document.documentElement.scrollHeight},"*");new ResizeObserver(post).observe(document.documentElement);window.addEventListener("load",post);post();})();</script>`;
  return `${code}\n${resizeScript}`;
}

// Preview iframes intentionally omit allow-downloads: generated canvases can offer their own
// UI, but downloads must go through Unsloth's explicit controls outside the sandbox.
export function ArtifactHtmlFrame({
  code,
  title = "HTML canvas preview",
  className,
  fill = false,
  actionFocusTargetRef,
}: {
  code: string;
  title?: string;
  className?: string;
  fill?: boolean;
  actionFocusTargetRef?: RefObject<HTMLElement | null>;
}) {
  const t = useT();
  const locale = useLocale();
  const iframeRef = useRef<HTMLIFrameElement>(null);
  // Every canvas honors this, fence or tool. Off by default; the standing half of the gate,
  // alongside the per-canvas grant below.
  const networkAccessEnabled = useChatRuntimeStore(
    (state) => state.allowArtifactNetworkAccess,
  );
  const [height, setHeight] = useState(HTML_FRAME_DEFAULT_HEIGHT);
  // Carries the code it was reported for, so a canvas swapped in place cannot inherit the
  // previous one's banner. Clearing it from the [src] effect ran a render too late, and that
  // stale render is the one carrying the button.
  const [blocked, setBlocked] = useState<BlockedState>({
    code,
    uris: [],
    hosts: [],
  });
  const blockedForCanvas = blocked.code === code ? blocked : NOTHING_BLOCKED;
  // Granted by the banner button alone, and only for the code on screen when it was clicked;
  // nothing the canvas sends may set it, or a blocked page could talk its way onto the
  // network. Compared during render rather than reset in an effect, which runs after the DOM
  // is updated and would let the first render carrying new code reuse allow_network=1.
  const [grantedCode, setGrantedCode] = useState<string | null>(null);
  const grantedForCanvas = grantedCode === code;
  const networkAllowed = networkAccessEnabled || grantedForCanvas;
  const [dismissedCode, setDismissedCode] = useState<string | null>(null);
  const dismissedForCanvas = dismissedCode === code;
  const artifactHtml = useMemo(() => buildArtifactSrcDoc(code), [code]);
  // Identifies this load to the frame, which stamps its blocked reports with it.
  const codeVersion = useMemo(() => hashArtifactCode(code), [code]);
  const src = useMemo(() => {
    const query = new URLSearchParams({ v: codeVersion });
    // Never put the auth token in the URL: in-frame code can read location.href.
    if (networkAllowed) {
      query.set("allow_network", "1");
    }
    return apiUrl(`/api/inference/artifact-preview-frame?${query.toString()}`);
  }, [networkAllowed, codeVersion]);
  // Feed only parent-initiated loads, so a self-navigated frame can't self-upgrade.
  const pendingPostRef = useRef(false);
  useEffect(() => {
    pendingPostRef.current = true;
  }, [src]);
  const postArtifactHtml = useCallback(() => {
    if (!pendingPostRef.current) return;
    pendingPostRef.current = false;
    // Sandboxed frame has an opaque origin ("null"), so a wildcard target is required; the payload
    // only reaches this iframe's contentWindow.
    iframeRef.current?.contentWindow?.postMessage(
      { type: "unsloth:artifact-html", html: artifactHtml },
      "*",
    );
  }, [artifactHtml]);

  useEffect(() => {
    const handler = (event: MessageEvent) => {
      if (event.source !== iframeRef.current?.contentWindow) return;
      if (event.origin !== "null") return;
      if (event.data?.type === "unsloth:artifact-blocked") {
        // event.source survives the swap navigation, so without the frame's stamp a report from the
        // outgoing canvas would be tagged with the incoming code and prompt a needless grant.
        if (event.data.v !== codeVersion) return;
        const uri = event.data.blockedURI;
        // A report carries the full URL, and the canvas can post these directly rather than going
        // through the CSP. The entry cap bounds how many are kept but not their size, so a handful
        // could park megabytes otherwise.
        if (typeof uri !== "string" || uri.length > BLOCKED_URI_MAX_CHARS) {
          return;
        }
        // The grant cannot fix these three, and prompting anyway widens the policy for nothing, then
        // hides the banner because the grant is on, leaving a broken canvas and no way back.
        if (GRANT_CANNOT_FIX.has(event.data.effectiveDirective)) return;
        // Same dead end one scheme down: the grant widens worker-src to blob: but not data:, so a
        // data: Worker reports under both policies.
        if (GRANT_CANNOT_FIX_SCHEME[event.data.effectiveDirective] === uri) {
          return;
        }
        const host = blockedHost(uri);
        if (!host) return;
        setBlocked((current) => appendBlocked(current, code, uri, host));
        return;
      }
      if (typeof event.data?.chatArtifactHeight !== "number") return;
      setHeight(
        Math.min(
          Math.max(event.data.chatArtifactHeight, 160),
          HTML_FRAME_MAX_HEIGHT,
        ),
      );
    };
    window.addEventListener("message", handler);
    return () => window.removeEventListener("message", handler);
    // `code`/`codeVersion` are listed so the handler always closes over the canvas on screen,
    // rather than relying on postArtifactHtml changing.
  }, [postArtifactHtml, code, codeVersion]);

  const showBlockedBanner =
    !networkAllowed && !dismissedForCanvas && blockedForCanvas.uris.length > 0;
  const shownHosts = blockedForCanvas.hosts
    .slice(0, BLOCKED_HOSTS_SHOWN)
    .join(", ");
  const blockedFrom =
    blockedForCanvas.hosts.length > BLOCKED_HOSTS_SHOWN
      ? `${shownHosts}…`
      : shownHosts;
  const focusAfterAction = () => {
    (actionFocusTargetRef?.current ?? iframeRef.current)?.focus({
      preventScroll: true,
    });
  };

  return (
    <div
      className={cn("relative", fill ? "h-full" : undefined)}
    >
      <iframe
        ref={iframeRef}
        src={src}
        sandbox="allow-scripts"
        referrerPolicy="no-referrer"
        onLoad={postArtifactHtml}
        className={cn(
          "block w-full border-0 bg-background outline-none focus-visible:outline-2 focus-visible:-outline-offset-2 focus-visible:outline-ring",
          className,
        )}
        style={{ height: fill ? "100%" : height }}
        title={title}
      />
      {showBlockedBanner ? (
        <div className="absolute inset-x-0 top-0 p-2">
          <Alert
            role="group"
            dir={locale === "ar" ? "rtl" : "ltr"}
            aria-label={t("settings.chat.artifacts.blockedTitle")}
            className="border-amber-500/40 bg-background/95 shadow-md backdrop-blur"
          >
            <ShieldAlertIcon className="text-amber-500" />
            <AlertAction>
              <Button
                size="icon-sm"
                variant="ghost"
                aria-label={t("settings.chat.artifacts.blockedDismiss")}
                onClick={() => {
                  focusAfterAction();
                  setDismissedCode(code);
                }}
              >
                <XIcon />
              </Button>
            </AlertAction>
            <AlertTitle role="alert">
              {t("settings.chat.artifacts.blockedTitle")}
              <span className="sr-only">
                {" "}
                {t("settings.chat.artifacts.blockedHint", {
                  setting: t("settings.chat.artifacts.allowNetworkAccess"),
                })}
              </span>
            </AlertTitle>
            <AlertDescription>
              <p>
                {t(
                  blockedForCanvas.uris.length === 1
                    ? "settings.chat.artifacts.blockedBanner"
                    : "settings.chat.artifacts.blockedBannerPlural",
                  { count: blockedForCanvas.uris.length, hosts: blockedFrom },
                )}{" "}
                {t("settings.chat.artifacts.blockedHint", {
                  setting: t("settings.chat.artifacts.allowNetworkAccess"),
                })}
              </p>
              <div className="flex flex-wrap gap-2">
                <Button
                  size="sm"
                  onClick={() => {
                    focusAfterAction();
                    setGrantedCode(code);
                  }}
                >
                  {t("settings.chat.artifacts.blockedBannerAction")}
                </Button>
                <Button
                  size="sm"
                  variant="outline"
                  onClick={() => {
                    useSettingsDialogStore.getState().openDialog("chat", {
                      scrollTarget: "chat-canvas-network",
                      focusFallback:
                        actionFocusTargetRef?.current ?? iframeRef.current,
                    });
                  }}
                >
                  {t("settings.chat.artifacts.blockedSettingsAction")}
                </Button>
              </div>
            </AlertDescription>
          </Alert>
        </div>
      ) : null}
    </div>
  );
}
