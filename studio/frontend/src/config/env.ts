


import { apiUrl } from "@/lib/api-base";
import {
  isDetectionDeferred,
  isProvisionalVerdict,
  resolveVerdict,
} from "@/config/hardware-verdict";
import { isPlatformAuthEnabled } from "@/integrations/platform-backend/config";
import { create } from "zustand";

export const env = {
  MODE: import.meta.env.MODE,
  DEV: import.meta.env.DEV,
  PROD: import.meta.env.PROD,
  BASE_URL: import.meta.env.BASE_URL,
} as const;

// Platform / device type

export type DeviceType = "mac" | "windows" | "linux" | string;

interface PlatformState {
  deviceType: DeviceType;
  // Unified memory: GPU and system draw on one pool, so an over-committed load has
  // nowhere to spill and takes the machine down rather than failing. Narrower than
  // deviceType === "mac", which includes Intel Macs with a discrete GPU, where spilling
  // to system RAM is exactly what happens. Mirrors the backend's is_apple_silicon gate.
  appleSilicon: boolean;
  chatOnly: boolean;
  // Why chatOnly is set (null when training is enabled), from /api/health.
  // e.g. "mlx_unavailable" on Apple Silicon -> the UI explains the greyed-out
  // Train/Export instead of silently disabling them.
  chatOnlyReason: string | null;
  // What specifically blocked that reason, when the backend can name it. Today only the
  // MLX gate does: it is all-or-nothing across mlx, mlx-lm and mlx-vlm, so without this
  // the greyed-out Train row can only repeat "run `unsloth studio update`".
  chatOnlyDetail: string | null;
  // From /api/health (authed): live tunnel URL, direct (non-tunnel) base, and
  // whether the server was launched with --secure.
  cloudflareUrl: string | null;
  serverUrl: string | null;
  secure: boolean;
  fetched: boolean;
  // Last verdict came from a deferred reply (torch-warm kill switch): nothing settles
  // until a first-use operation detects, so the sidebar polls on this.
  detectionDeferred: boolean;
  isChatOnly: () => boolean;
  // True until /api/health has answered with a server-measured verdict. Before that `chatOnly`
  // is the browser-platform seed below, not an answer, so anything that would gray a
  // capability out has to treat it as unknown: rendering the guess blacks out Train and Video
  // on every Mac from first paint, visually identical to a measured "unsupported".
  capabilitiesUnknown: () => boolean;
}

// Client-side fallback when backend isn't ready yet.
function detectLocalPlatform(): DeviceType {
  if (typeof navigator === "undefined") return "linux";
  const platform = navigator.platform.toLowerCase();
  const ua = navigator.userAgent.toLowerCase();
  if (platform.includes("mac") || ua.includes("mac")) return "mac";
  if (platform.includes("win") || ua.includes("win")) return "windows";
  return "linux";
}

const localDeviceType = detectLocalPlatform();

export const usePlatformStore = create<PlatformState>()((_, get) => ({
  deviceType: localDeviceType,
  appleSilicon: false,
  // A guess from the user agent, kept only as the pre-measurement fallback for the redirects
  // that must decide something before /api/health answers. Capability gating must read
  // capabilitiesUnknown() first and hold, not gray a tab out on this.
  chatOnly: localDeviceType === "mac",
  chatOnlyReason: null,
  chatOnlyDetail: null,
  cloudflareUrl: null,
  serverUrl: null,
  secure: false,
  fetched: false,
  detectionDeferred: false,
  isChatOnly: () => get().chatOnly,
  // `fetched` already means "a server-reported verdict is stored" (see fetchDeviceType), so it
  // is the unknown/known line; no second flag to keep in step with it. A deferred reply counts
  // as settled even though it carries no device_type: under the torch-warm kill switch nothing
  // else is coming this session, so treating it as unknown would spin the tabs forever.
  capabilitiesUnknown: () => {
    const state = get();
    return !state.fetched && !state.detectionDeferred;
  },
}));

// Once an authoritative (server-reported) platform has been fetched, a
// non-forced response must not overwrite it. The post-render fetchDeviceType()
// in main.tsx runs before auth is ready and can resolve after the authed
// root-route/provider fetches; such a late write would reset deviceType,
// cloudflareUrl/serverUrl/secure, and fetched, whether it is a browser fallback
// (unauthenticated) or an earlier authenticated request that landed after a
// later forced refresh. Forced refreshes are explicit re-reads, so they still write.
function shouldKeepAuthoritativePlatform(force?: boolean): boolean {
  return !force && usePlatformStore.getState().fetched;
}

// How long fetchDeviceType waits out a backend that is still detecting, and how often
// it re-reads. Sized from the warm's torch import (~1-2s cold), plus headroom.
const HARDWARE_DETECT_WAIT_MS = 5000;
const HARDWARE_DETECT_POLL_MS = 200;
// The bounded wait above is spent at most once per page load: see fetchDeviceType.
let hardwareWaitSpent = false;

// `force` re-reads /api/health even if cached, to pick up a late-arriving tunnel URL.
export async function fetchDeviceType(options?: {
  force?: boolean;
}): Promise<DeviceType> {
  // Rag Platform does not expose the legacy Studio hardware-capability contract,
  // and all capability-gated pages are disabled during this rollout. Mark the
  // verdict settled locally so AppSidebar never starts its 3-second /api/health
  // recovery poll. Platform readiness has its own /api/v1 system probes.
  if (isPlatformAuthEnabled()) {
    const current = usePlatformStore.getState();
    if (!current.fetched || current.chatOnly || current.detectionDeferred) {
      usePlatformStore.setState({
        chatOnly: false,
        chatOnlyReason: null,
        fetched: true,
        detectionDeferred: false,
      });
    }
    return current.deviceType;
  }

  const { fetched } = usePlatformStore.getState();
  if (fetched && !options?.force) return usePlatformStore.getState().deviceType;

  try {
    // /api/health only reports the server's device_type to authed callers.
    // Read the token from storage directly: importing features/auth here
    // would be an import cycle (auth/session imports this store).
    const token =
      typeof window === "undefined"
        ? null
        : localStorage.getItem("unsloth_auth_token");
    // Re-read while the backend is still measuring: chat_only is its pre-detection default
    // until then, and __root.tsx's beforeLoad acts on what this returns, sending a GPU host
    // to /chat with Train hidden. The window is only the torch import, so bound the re-read.
    const headers = token ? { Authorization: `Bearer ${token}` } : undefined;
    // Wait only when a measurement can arrive, and only once. /api/health reports
    // device_type to authed callers only, so an unauthenticated poll stays provisional and
    // would just hold /login (beforeLoad awaits this) behind the torch import. Claim the
    // latch after the wait, or a concurrent caller skips a window nobody has finished.
    const spendWait = Boolean(token) && !hardwareWaitSpent;
    const deadline = spendWait ? Date.now() + HARDWARE_DETECT_WAIT_MS : 0;
    let tokenRejected = false;
    let res = await fetch(apiUrl("/api/health"), { headers });
    while (res.ok && Date.now() < deadline) {
      const peek = (await res.clone().json()) as {
        hardware_detecting?: boolean;
        hardware_detection_deferred?: boolean;
        version?: string;
      };
      // Deferred is not "in progress": nothing will settle, so do not wait.
      if (!isProvisionalVerdict(peek) || isDetectionDeferred(peek)) break;
      // A rejected token gets the unauthenticated body, which never carries device_type.
      // `version` is authed-only, so its absence means this wait can only time out,
      // holding /login for the full window on a cold boot.
      if (peek.version === undefined) {
        tokenRejected = true;
        break;
      }
      await new Promise((resolve) => setTimeout(resolve, HARDWARE_DETECT_POLL_MS));
      res = await fetch(apiUrl("/api/health"), { headers });
    }
    // Not spent when the backend rejected the token: no window was actually waited
    // out, and signing in later in this same page load must still get one.
    if (spendWait && !tokenRejected) hardwareWaitSpent = true;
    if (res.ok) {
      const data = (await res.json()) as {
        device_type?: string;
        apple_silicon?: boolean;
        chat_only?: boolean;
        chat_only_reason?: string | null;
        hardware_detecting?: boolean;
        cloudflare_url?: string | null;
        server_url?: string | null;
        secure?: boolean;
      };
      // Once the store holds an authoritative (server-reported) platform, a
      // non-forced response must not overwrite it. It may be an unauthenticated
      // fallback, or an earlier authenticated request that resolved after a
      // later forced refresh already picked up device_type and the tunnel
      // fields; writing either would reset device type or null the tunnel
      // fields. Forced refreshes are explicit re-reads, so they still write.
      if (shouldKeepAuthoritativePlatform(options?.force)) {
        return usePlatformStore.getState().deviceType;
      }
      const previous = usePlatformStore.getState();
      // A provisional reply omits device_type, so a forced refresh during the warm would
      // fall back to the browser platform and relabel a WSL, SSH or remote host as local,
      // changing model filtering, paths and install commands. Keep the server's answer.
      const keepPlatform = data.device_type === undefined && previous.fetched;
      const deviceType =
        data.device_type ?? (keepPlatform ? previous.deviceType : detectLocalPlatform());
      // Rides with device_type and is kept on the same terms: a provisional or
      // unauthenticated reply carries neither, and a browser guess cannot tell Apple
      // Silicon from Intel. Absent means false, the pre-Apple-Silicon wording -- correct
      // on an Intel Mac, and on a Mac browser pointed at a Linux host.
      const appleSilicon =
        data.apple_silicon ?? (keepPlatform ? previous.appleSilicon : false);
      // A still-provisional reply keeps the stored verdict: see resolveVerdict.
      const { chatOnly, chatOnlyReason, chatOnlyDetail } = resolveVerdict(
        data,
        previous,
      );
      // Cache only a server-reported platform. Unauthenticated responses fall
      // back to the browser platform, which can differ from the host (WSL,
      // SSH); keeping fetched=false retries once a token exists.
      usePlatformStore.setState({
        deviceType,
        appleSilicon,
        chatOnly,
        chatOnlyReason,
        chatOnlyDetail,
        cloudflareUrl: data.cloudflare_url ?? null,
        serverUrl: data.server_url ?? null,
        secure: data.secure ?? false,
        fetched: data.device_type !== undefined || keepPlatform,
        detectionDeferred: isDetectionDeferred(data),
      });
      return deviceType;
    }
  } catch {
    // Backend not ready: use client-side detection so chat-only guard works
    // on initial load (important for macOS). Keep fetched=false so a later
    // call retries against the backend. But a late non-forced failure must not
    // wipe an authoritative platform that already resolved.
    if (shouldKeepAuthoritativePlatform(options?.force)) {
      return usePlatformStore.getState().deviceType;
    }
    const deviceType = detectLocalPlatform();
    const chatOnly = deviceType === "mac";
    usePlatformStore.setState({ deviceType, chatOnly, fetched: false });
    return deviceType;
  }

  return usePlatformStore.getState().deviceType;
}
