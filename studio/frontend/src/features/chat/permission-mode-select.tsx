// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  ChevronDown,
  CircleAlert,
  Globe,
  Hand,
  Info,
  RefreshCw,
  ShieldCheck,
} from "lucide-react";
import type { ComponentType } from "react";
import { useEffect, useRef, useState } from "react";

import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { ChevronDownStandardIcon } from "@/lib/chevron-icons";
import { SparklesGlyph } from "@/lib/sparkles-icon";
import { Tick02Icon } from "@/lib/tick-icon";
import { cn } from "@/lib/utils";
import { HugeiconsIcon } from "@hugeicons/react";
import {
  type PermissionMode,
  useChatRuntimeStore,
} from "./stores/chat-runtime-store";
import {
  type ToolIsolationCapability,
  toolIsolationPresentation,
} from "./tool-isolation";
import {
  TOOL_ISOLATION_LIMITATION_TEXT,
  limitedBackendLabel,
  networkAllowlistSummary,
} from "./tool-isolation-labels";
import { capabilityOffersNetworkAllowlist } from "./utils/tool-network-policy";

/** Permission levels for tool calls. Full access stays last because it disables both approval
 *  prompts and the code sandbox. */
export const PERMISSION_MODE_OPTIONS: readonly {
  value: PermissionMode;
  label: string;
  description: string;
  menuDescription: string;
  icon: ComponentType<{ className?: string; strokeWidth?: number }>;
}[] = [
  {
    value: "ask",
    label: "Ask for approval",
    description:
      "Always ask before tool calls, editing files or using the internet",
    menuDescription: "Ask before each tool call",
    icon: Hand,
  },
  {
    value: "auto",
    label: "Approve for me",
    description:
      "Run tool calls, but ask before high-risk actions like credential access, privilege escalation, or destructive commands",
    menuDescription: "Ask only before risky actions",
    icon: ShieldCheck,
  },
  {
    value: "off",
    label: "Run automatically",
    description: "Run tool calls without approval prompts inside the sandbox",
    menuDescription: "Skip approvals; keep sandbox settings",
    icon: SparklesGlyph,
  },
  {
    value: "full",
    label: "Full access",
    description:
      "Unrestricted: no approval prompts and the code sandbox is disabled",
    menuDescription: "No approvals or sandbox",
    icon: CircleAlert,
  },
] as const;

export const FULL_ACCESS_WARNING =
  "Tools can change or delete your files, run commands, and access the network without asking. The sandbox is off. Only enable this for tasks you trust.";

export const TOOL_ISOLATION_UNAVAILABLE_WARNING =
  "The OS sandbox is unavailable. Software checks still run, but Python and Terminal may access your files, credentials, and network.";

const TOOL_ISOLATION_RESTRICTED_TOKEN_NOTE =
  "On Windows, a restricted token limits writes. It does not isolate reads, network access, or other processes.";

/** The Limited consent text for this host: the generic warning, plus what the Windows
 *  restricted token adds when the backend reports it. */
function limitedModeWarning(
  capability: Pick<
    ToolIsolationCapability,
    "limited_backend" | "limited_disclosure"
  > | null,
): string {
  if (capability?.limited_disclosure) return capability.limited_disclosure;
  if (capability?.limited_backend === "windows-restricted-token") {
    return `${TOOL_ISOLATION_UNAVAILABLE_WARNING} ${TOOL_ISOLATION_RESTRICTED_TOKEN_NOTE}`;
  }
  return TOOL_ISOLATION_UNAVAILABLE_WARNING;
}

export function permissionModeOption(mode: PermissionMode) {
  return (
    PERMISSION_MODE_OPTIONS.find((option) => option.value === mode) ??
    // Unknown values fall back to the default ("Approve for me"), not row 0 ("Ask").
    PERMISSION_MODE_OPTIONS.find((option) => option.value === "auto") ??
    PERMISSION_MODE_OPTIONS[0]
  );
}

function useToolIsolationCapabilityRefresh() {
  const capability = useChatRuntimeStore((s) => s.toolIsolationCapability);
  const loading = useChatRuntimeStore((s) => s.toolIsolationCapabilityLoading);
  const error = useChatRuntimeStore((s) => s.toolIsolationError);
  const refresh = useChatRuntimeStore((s) => s.refreshToolIsolationCapability);

  useEffect(() => {
    if (!capability && !loading && !error) {
      refresh().catch(() => undefined);
    }
  }, [capability, error, loading, refresh]);
}

/** The option rows shared by every permission dropdown/submenu. Non-full
 *  levels apply directly; picking Full access must go through the caller's
 *  danger confirmation, so it's a separate callback. */
export function PermissionModeMenuItems({
  onRequestFullAccess,
}: {
  onRequestFullAccess: () => void;
}) {
  const permissionMode = useChatRuntimeStore((s) => s.permissionMode);
  const setPermissionMode = useChatRuntimeStore((s) => s.setPermissionMode);

  return (
    <>
      {PERMISSION_MODE_OPTIONS.map((option) => (
        <DropdownMenuItem
          key={option.value}
          onSelect={() => {
            if (option.value === permissionMode) {
              return;
            }
            if (option.value === "full") {
              onRequestFullAccess();
            } else {
              setPermissionMode(option.value);
            }
          }}
          className={cn(
            "items-start gap-2 py-2",
            permissionMode === option.value && "font-medium",
            option.value === "full" &&
              permissionMode === "full" &&
              "text-bypass",
          )}
        >
          <option.icon className="size-4 shrink-0" strokeWidth={2} />
          <span className="flex min-w-0 flex-1 flex-col gap-1">
            <span className="text-sm leading-tight">{option.label}</span>
            <span className="text-xs font-normal leading-snug text-muted-foreground">
              {option.menuDescription}
            </span>
          </span>
          {permissionMode === option.value ? (
            <HugeiconsIcon
              icon={Tick02Icon}
              strokeWidth={2}
              className="ml-auto size-4 shrink-0"
            />
          ) : null}
        </DropdownMenuItem>
      ))}
    </>
  );
}

/** Danger confirmation shown before Full access turns on. Self-contained so the dropdown works
 *  outside the chat page (e.g. the Settings dialog). */
export function FullAccessConfirmDialog({
  open,
  onOpenChange,
  restoreFocus,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  restoreFocus?: () => void;
}) {
  const setPermissionMode = useChatRuntimeStore((s) => s.setPermissionMode);

  return (
    <AlertDialog open={open} onOpenChange={onOpenChange}>
      <AlertDialogContent
        onCloseAutoFocus={(event) => {
          if (restoreFocus) {
            event.preventDefault();
            restoreFocus();
          }
        }}
      >
        <AlertDialogHeader className="gap-2">
          <AlertDialogTitle>Enable Full access?</AlertDialogTitle>
          <AlertDialogDescription>{FULL_ACCESS_WARNING}</AlertDialogDescription>
        </AlertDialogHeader>
        <AlertDialogFooter>
          <AlertDialogCancel>Cancel</AlertDialogCancel>
          <AlertDialogAction
            variant="destructive"
            onClick={() => {
              setPermissionMode("full");
              onOpenChange(false);
            }}
          >
            Enable Full access
          </AlertDialogAction>
        </AlertDialogFooter>
      </AlertDialogContent>
    </AlertDialog>
  );
}

function isolationSummary(
  state: ReturnType<typeof toolIsolationPresentation>["state"],
  capability: ToolIsolationCapability | null,
): string {
  if (state === "full") return "Sandbox off";
  if (state === "limited") return "Limited · no OS isolation";
  if (state === "container") return "Container-compatible isolation";
  if (!capability) return "Checking sandbox…";
  if (state === "preview") return "Sandbox · Preview";
  if (state === "protected") return "Sandbox on";
  return "Sandbox unavailable";
}

function ToolIsolationDetailsDialog({
  open,
  onOpenChange,
  restoreFocus,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  restoreFocus?: () => void;
}) {
  const capability = useChatRuntimeStore((s) => s.toolIsolationCapability);
  const mode = useChatRuntimeStore((s) => s.toolExecutionMode);
  const grant = useChatRuntimeStore((s) => s.limitedToolGrant);
  const nestedGrant = useChatRuntimeStore((s) => s.nestedToolGrant);
  const networkPolicy = useChatRuntimeStore((s) => s.toolNetworkPolicy);
  const presentation = toolIsolationPresentation(
    mode,
    capability,
    grant,
    nestedGrant,
  );
  const isolated =
    presentation.state === "protected" || presentation.state === "preview";
  const limitedBackend = limitedBackendLabel(
    capability?.limited_backend ?? null,
  );

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent
        className="gap-6 p-6 max-sm:content-start"
        onCloseAutoFocus={(event) => {
          if (restoreFocus) {
            event.preventDefault();
            restoreFocus();
          }
        }}
      >
        <DialogHeader className="pe-8">
          <DialogTitle>Sandbox details</DialogTitle>
          <DialogDescription>{presentation.label}</DialogDescription>
        </DialogHeader>
        {capability ? (
          <>
            <dl className="grid grid-cols-[auto_minmax(0,1fr)] gap-x-4 gap-y-2 text-sm">
              <dt className="text-muted-foreground">Environment</dt>
              <dd className="break-words text-end">{capability.environment}</dd>
              {presentation.state === "container" ? (
                <>
                  <dt className="text-muted-foreground">Runtime</dt>
                  <dd className="text-end">Sandbox Runtime (SRT)</dd>
                  <dt className="text-muted-foreground">Profile</dt>
                  <dd className="break-all text-end">
                    {capability.nested_profile_id}
                  </dd>
                  <dt className="text-muted-foreground">Network</dt>
                  <dd className="text-end">Off</dd>
                </>
              ) : null}
              {isolated ? (
                <>
                  <dt className="text-muted-foreground">Runtime</dt>
                  <dd className="break-words text-end">{capability.backend}</dd>
                </>
              ) : null}
              {isolated && capability.profile_id ? (
                <>
                  <dt className="text-muted-foreground">Profile</dt>
                  <dd className="break-all text-end">
                    {capability.profile_id}
                  </dd>
                </>
              ) : null}
              {isolated ? (
                <>
                  <dt className="text-muted-foreground">Network</dt>
                  <dd className="text-end">
                    {capabilityOffersNetworkAllowlist(capability) &&
                    networkPolicy === "allowlist"
                      ? "Allowed hosts only"
                      : "Off"}
                  </dd>
                </>
              ) : null}
            </dl>
            {!isolated ? (
              <p className="text-sm">{presentation.description}</p>
            ) : null}
            {presentation.state === "unavailable" && capability.reason ? (
              <p className="whitespace-pre-wrap break-words text-sm">
                {capability.reason}
              </p>
            ) : null}
            {presentation.state === "unavailable" && capability.diagnostic ? (
              <details className="text-sm">
                <summary>Diagnostic details</summary>
                <p>Code: {capability.diagnostic.code}</p>
                <p>Stage: {capability.diagnostic.stage}</p>
                {capability.diagnostic.field ? (
                  <p>
                    {capability.diagnostic.field}: {capability.diagnostic.count}{" "}
                    entries; limit {capability.diagnostic.limit}.
                  </p>
                ) : null}
                {capability.diagnostic.dependency ? (
                  <p>Dependency: {capability.diagnostic.dependency}</p>
                ) : null}
              </details>
            ) : null}
            {(isolated || presentation.state === "unavailable") &&
            capability.limitations.length > 0 ? (
              <section className="space-y-2">
                <h3 className="text-sm font-medium">Limitations</h3>
                <ul className="list-disc space-y-2 ps-4 text-sm leading-normal">
                  {capability.limitations.map((code) => (
                    <li key={code} className="break-words">
                      {TOOL_ISOLATION_LIMITATION_TEXT[code] ?? code}
                    </li>
                  ))}
                </ul>
              </section>
            ) : null}
            {presentation.state === "unavailable" && capability.remediation ? (
              <p className="whitespace-pre-wrap break-words text-sm">
                {capability.remediation}
              </p>
            ) : null}
            {presentation.state !== "full" &&
            (limitedBackend ||
              presentation.state === "limited" ||
              presentation.state === "unavailable") ? (
              <section className="space-y-2 border-t border-border pt-4">
                <h3 className="text-sm font-medium">
                  {isolated || presentation.state === "container"
                    ? "About Limited mode"
                    : "Limited mode"}
                </h3>
                <p className="text-sm">{limitedModeWarning(capability)}</p>
                {limitedBackend ? (
                  <p className="text-sm text-muted-foreground">
                    {limitedBackend}
                  </p>
                ) : null}
                {capability.limited_limitations.length > 0 ? (
                  <ul className="list-disc space-y-2 ps-4 text-sm leading-normal">
                    {capability.limited_limitations.map((code) => (
                      <li key={code} className="break-words">
                        {TOOL_ISOLATION_LIMITATION_TEXT[code] ?? code}
                      </li>
                    ))}
                  </ul>
                ) : null}
              </section>
            ) : null}
          </>
        ) : (
          <p className="text-sm">Checking this environment…</p>
        )}
      </DialogContent>
    </Dialog>
  );
}

function ToolIsolationMenuSection({
  onRequestLimited,
  onRequestNested,
  onRequestDetails,
}: {
  onRequestLimited: () => void;
  onRequestNested: () => void;
  onRequestDetails: () => void;
}) {
  useToolIsolationCapabilityRefresh();
  const mode = useChatRuntimeStore((s) => s.toolExecutionMode);
  const capability = useChatRuntimeStore((s) => s.toolIsolationCapability);
  const grant = useChatRuntimeStore((s) => s.limitedToolGrant);
  const nestedGrant = useChatRuntimeStore((s) => s.nestedToolGrant);
  const loading = useChatRuntimeStore((s) => s.toolIsolationCapabilityLoading);
  const error = useChatRuntimeStore((s) => s.toolIsolationError);
  const refresh = useChatRuntimeStore((s) => s.refreshToolIsolationCapability);
  const setMode = useChatRuntimeStore((s) => s.setToolExecutionMode);
  const networkPolicy = useChatRuntimeStore((s) => s.toolNetworkPolicy);
  const setNetworkPolicy = useChatRuntimeStore((s) => s.setToolNetworkPolicy);
  const presentation = toolIsolationPresentation(
    mode,
    capability,
    grant,
    nestedGrant,
  );

  const unavailable =
    presentation.state === "unavailable" &&
    capability?.protection_state === "unavailable";
  const osIsolated =
    presentation.state === "protected" || presentation.state === "preview";
  // Offer the toggle only when this backend advertises allowlist enforcement.
  const offersAllowlist =
    osIsolated && capabilityOffersNetworkAllowlist(capability);
  return (
    <>
      <DropdownMenuSeparator />
      <div className="space-y-2 px-3 py-3" aria-live="polite">
        <div className="flex items-center gap-2">
          <ShieldCheck
            className={cn(
              "size-4 shrink-0",
              presentation.state === "unavailable" &&
                !loading &&
                capability &&
                "text-destructive",
              (presentation.state === "limited" ||
                presentation.state === "full") &&
                "text-bypass",
            )}
            strokeWidth={2}
          />
          <p className="text-sm font-medium">
            {loading
              ? "Checking sandbox…"
              : isolationSummary(presentation.state, capability)}
          </p>
        </div>
        <p className="text-xs leading-normal text-foreground">
          {presentation.description}
        </p>
        {error ? (
          <p className="break-words text-xs text-destructive">{error}</p>
        ) : null}
      </div>
      <DropdownMenuItem
        onSelect={() => setTimeout(onRequestDetails, 0)}
        className="gap-2 text-sm"
      >
        <Info className="size-4" strokeWidth={2} />
        Sandbox details
      </DropdownMenuItem>
      {offersAllowlist && capability ? (
        <DropdownMenuItem
          onSelect={(event) => {
            // Keep the menu open so the Network row above reflects the change at once.
            event.preventDefault();
            setNetworkPolicy(
              networkPolicy === "allowlist" ? "deny" : "allowlist",
            );
          }}
          className={cn(
            "items-start gap-2 py-2",
            networkPolicy === "allowlist" && "font-medium",
          )}
          aria-checked={networkPolicy === "allowlist"}
          role="menuitemcheckbox"
        >
          <Globe className="size-4 shrink-0" strokeWidth={2} />
          <span className="flex min-w-0 flex-1 flex-col gap-1">
            <span className="text-sm leading-tight">
              Allow package and model downloads
            </span>
            <span className="text-xs font-normal leading-snug text-muted-foreground">
              {networkAllowlistSummary(capability.network_allowlist)}
            </span>
          </span>
          {networkPolicy === "allowlist" ? (
            <HugeiconsIcon
              icon={Tick02Icon}
              strokeWidth={2}
              className="ml-auto size-4 shrink-0"
            />
          ) : null}
        </DropdownMenuItem>
      ) : null}
      {unavailable && capability?.nested_eligible ? (
        <DropdownMenuItem
          onSelect={() => setTimeout(onRequestNested, 0)}
          className="text-sm"
        >
          <ShieldCheck className="size-4" strokeWidth={2} />
          Try container-compatible isolation…
        </DropdownMenuItem>
      ) : null}
      {unavailable ? (
        <DropdownMenuItem
          onSelect={() => setTimeout(onRequestLimited, 0)}
          className="text-sm"
        >
          <CircleAlert className="size-4" strokeWidth={2} />
          Use Limited mode…
        </DropdownMenuItem>
      ) : null}
      {presentation.state === "limited" ||
      presentation.state === "container" ? (
        <DropdownMenuItem
          onSelect={() => setMode("os_isolation_required")}
          className="text-sm"
        >
          <ShieldCheck className="size-4" strokeWidth={2} />
          Require sandbox
        </DropdownMenuItem>
      ) : null}
      {!loading &&
      (!capability || capability.retryable || !capability.available) ? (
        <DropdownMenuItem
          onSelect={(event) => {
            event.preventDefault();
            refresh().catch(() => undefined);
          }}
          className="text-sm"
        >
          <RefreshCw className="size-4" strokeWidth={2} />
          Check again
        </DropdownMenuItem>
      ) : null}
    </>
  );
}

export function LimitedModeConfirmDialog({
  open,
  onOpenChange,
  restoreFocus,
  variant = "limited",
  onRequestNested,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  restoreFocus?: () => void;
  variant?: "limited" | "nested";
  onRequestNested?: () => void;
}) {
  const requestGrant = useChatRuntimeStore((s) =>
    variant === "nested" ? s.requestNestedToolGrant : s.requestLimitedToolGrant,
  );
  const clearGrant = useChatRuntimeStore((s) =>
    variant === "nested" ? s.clearNestedToolGrant : s.clearLimitedToolGrant,
  );
  const loading = useChatRuntimeStore((s) => s.toolIsolationGrantLoading);
  const error = useChatRuntimeStore((s) => s.toolIsolationError);
  const diagnostic = useChatRuntimeStore((s) => s.toolIsolationErrorDiagnostic);
  const capability = useChatRuntimeStore((s) => s.toolIsolationCapability);

  return (
    <AlertDialog
      open={open}
      onOpenChange={(nextOpen) => {
        if (!nextOpen && loading) clearGrant();
        onOpenChange(nextOpen);
      }}
    >
      <AlertDialogContent
        onCloseAutoFocus={(event) => {
          if (restoreFocus) {
            event.preventDefault();
            restoreFocus();
          }
        }}
      >
        <AlertDialogHeader className="gap-2">
          <AlertDialogTitle>
            {variant === "nested"
              ? "Use container-compatible tool isolation?"
              : "Use Limited mode?"}
          </AlertDialogTitle>
          <AlertDialogDescription>
            {variant === "nested"
              ? capability?.nested_disclosure ||
                "Applies only to Python and Terminal tool calls. Keeps file and network restrictions, but shares the container's process information and relies partly on its isolation."
              : limitedModeWarning(capability)}
          </AlertDialogDescription>
        </AlertDialogHeader>
        {error ? (
          <p className="text-center text-xs text-destructive">{error}</p>
        ) : null}
        {error && diagnostic ? (
          <details className="text-sm">
            <summary>Diagnostic details</summary>
            <p>Code: {diagnostic.code}</p>
            <p>Stage: {diagnostic.stage}</p>
            {diagnostic.dependency ? (
              <p>Dependency: {diagnostic.dependency}</p>
            ) : null}
            {diagnostic.field ? (
              <p>
                {diagnostic.field}: {diagnostic.count} entries; limit{" "}
                {diagnostic.limit}.
              </p>
            ) : null}
          </details>
        ) : null}
        {variant === "limited" &&
        capability?.nested_eligible &&
        onRequestNested ? (
          <button
            type="button"
            className="text-sm underline"
            disabled={loading}
            onClick={onRequestNested}
          >
            Try container-compatible isolation instead…
          </button>
        ) : null}
        <AlertDialogFooter className="sm:items-center">
          <AlertDialogCancel>Cancel</AlertDialogCancel>
          <AlertDialogAction
            className="whitespace-normal"
            disabled={
              loading || (variant === "nested" && !capability?.nested_eligible)
            }
            onClick={(event) => {
              event.preventDefault();
              requestGrant()
                .then(() => onOpenChange(false))
                .catch(() => undefined);
            }}
          >
            {loading
              ? "Checking…"
              : variant === "nested"
                ? "Check and enable"
                : "Use Limited mode"}
          </AlertDialogAction>
        </AlertDialogFooter>
      </AlertDialogContent>
    </AlertDialog>
  );
}

/** Page-root dialog used by the pre-send gate when Required cannot launch. */
export function ToolIsolationConsentDialog() {
  const open = useChatRuntimeStore((s) => s.toolIsolationConsentOpen);
  const setOpen = useChatRuntimeStore((s) => s.setToolIsolationConsentOpen);
  return open ? <ActiveToolIsolationConsentDialog setOpen={setOpen} /> : null;
}

function ActiveToolIsolationConsentDialog({
  setOpen,
}: {
  setOpen: (open: boolean) => void;
}) {
  const [variant, setVariant] = useState<"limited" | "nested">("limited");
  return (
    <LimitedModeConfirmDialog
      open={true}
      variant={variant}
      onRequestNested={() => setVariant("nested")}
      onOpenChange={(next) => {
        setOpen(next);
      }}
    />
  );
}

/**
 * Select-style dropdown (like the MCP composer menu) for picking the
 * permission level. Used in General settings and the chat settings sheet.
 */
export function PermissionModeDropdown({
  side = "bottom",
  align = "end",
  triggerClassName,
}: {
  side?: "top" | "bottom";
  align?: "start" | "end";
  triggerClassName?: string;
} = {}) {
  useToolIsolationCapabilityRefresh();
  const permissionMode = useChatRuntimeStore((s) => s.permissionMode);
  const toolExecutionMode = useChatRuntimeStore((s) => s.toolExecutionMode);
  const capability = useChatRuntimeStore((s) => s.toolIsolationCapability);
  const grant = useChatRuntimeStore((s) => s.limitedToolGrant);
  const nestedGrant = useChatRuntimeStore((s) => s.nestedToolGrant);
  const [confirmOpen, setConfirmOpen] = useState(false);
  const [limitedConfirmOpen, setLimitedConfirmOpen] = useState(false);
  const [nestedConfirmOpen, setNestedConfirmOpen] = useState(false);
  const [detailsOpen, setDetailsOpen] = useState(false);
  const triggerRef = useRef<HTMLButtonElement>(null);
  const restoreFocus = () => triggerRef.current?.focus();
  const active = permissionModeOption(permissionMode);
  const isolation = toolIsolationPresentation(
    toolExecutionMode,
    capability,
    grant,
    nestedGrant,
  );
  const ActiveIcon = active.icon;

  return (
    <>
      <DropdownMenu>
        <DropdownMenuTrigger asChild={true} ref={triggerRef}>
          <Button
            variant="outline"
            size="sm"
            className={cn(
              "max-w-full gap-2",
              triggerClassName,
              // Last so a text color in triggerClassName cannot override it.
              permissionMode === "full" &&
                "text-bypass hover:text-bypass border-bypass/50",
            )}
            aria-label="Permission level for tool calls"
          >
            <ActiveIcon className="size-4 shrink-0" strokeWidth={2} />
            <span className="min-w-0 flex-1 truncate text-left">
              {active.label}
            </span>
            <span className="min-w-0 truncate text-xs text-muted-foreground">
              {isolationSummary(isolation.state, capability)}
            </span>
            <ChevronDown className="size-4 shrink-0 opacity-60" />
          </Button>
        </DropdownMenuTrigger>
        <DropdownMenuContent
          side={side}
          align={align}
          className="w-80 max-w-[calc(100vw-2rem)]"
          avoidCollisions={true}
        >
          <DropdownMenuLabel>Tool permissions</DropdownMenuLabel>
          <PermissionModeMenuItems
            // Defer past the menu-close focus restoration so the dialog's focus trap is not broken by the
            // dropdown grabbing focus back.
            onRequestFullAccess={() =>
              setTimeout(() => setConfirmOpen(true), 0)
            }
          />
          <ToolIsolationMenuSection
            onRequestLimited={() => setLimitedConfirmOpen(true)}
            onRequestNested={() => setNestedConfirmOpen(true)}
            onRequestDetails={() => setDetailsOpen(true)}
          />
        </DropdownMenuContent>
      </DropdownMenu>
      <FullAccessConfirmDialog
        restoreFocus={restoreFocus}
        open={confirmOpen}
        onOpenChange={setConfirmOpen}
      />
      <ToolIsolationDetailsDialog
        restoreFocus={restoreFocus}
        open={detailsOpen}
        onOpenChange={setDetailsOpen}
      />
      <LimitedModeConfirmDialog
        variant="nested"
        restoreFocus={restoreFocus}
        open={nestedConfirmOpen}
        onOpenChange={setNestedConfirmOpen}
      />
      <LimitedModeConfirmDialog
        restoreFocus={restoreFocus}
        open={limitedConfirmOpen}
        onOpenChange={setLimitedConfirmOpen}
      />
    </>
  );
}

/** Composer pill showing the current permission level in the chat box; clicking opens the level
 *  dropdown. Danger-styled while Full access is on. The Full access pick routes through the
 *  store-driven confirm dialog mounted at the chat-page root, so the warning survives this
 *  menu unmounting. */
export function PermissionModeComposerPill({
  side = "bottom",
}: {
  side?: "top" | "bottom";
} = {}) {
  useToolIsolationCapabilityRefresh();
  const permissionMode = useChatRuntimeStore((s) => s.permissionMode);
  const toolExecutionMode = useChatRuntimeStore((s) => s.toolExecutionMode);
  const capability = useChatRuntimeStore((s) => s.toolIsolationCapability);
  const grant = useChatRuntimeStore((s) => s.limitedToolGrant);
  const nestedGrant = useChatRuntimeStore((s) => s.nestedToolGrant);
  const setBypassConfirmOpen = useChatRuntimeStore(
    (s) => s.setBypassConfirmOpen,
  );
  const [limitedConfirmOpen, setLimitedConfirmOpen] = useState(false);
  const [nestedConfirmOpen, setNestedConfirmOpen] = useState(false);
  const [detailsOpen, setDetailsOpen] = useState(false);
  const triggerRef = useRef<HTMLButtonElement>(null);
  const restoreFocus = () => triggerRef.current?.focus();
  const active = permissionModeOption(permissionMode);
  const isolation = toolIsolationPresentation(
    toolExecutionMode,
    capability,
    grant,
    nestedGrant,
  );
  const ActiveIcon = active.icon;
  const fullAccess = permissionMode === "full";
  const codeToolsEnabled = useChatRuntimeStore((s) => s.codeToolsEnabled);
  const showIsolation =
    codeToolsEnabled ||
    fullAccess ||
    toolExecutionMode === "limited" ||
    toolExecutionMode === "container_isolation";

  return (
    <>
      <DropdownMenu>
        <DropdownMenuTrigger asChild={true} ref={triggerRef}>
          <button
            type="button"
            className="composer-pill-btn composer-pill-permissions"
            data-pill-label={active.label}
            data-active={fullAccess ? "true" : "false"}
            data-variant={fullAccess ? "danger" : undefined}
            aria-label="Permission level for tool calls"
            title={
              showIsolation
                ? `${active.label}: ${active.menuDescription}. ${isolationSummary(isolation.state, capability)}.`
                : `${active.label}: ${active.description}`
            }
          >
            <span className="composer-pill-glyph">
              <ActiveIcon className="size-[15px]" strokeWidth={2} />
            </span>
            <span>{active.label}</span>
            {showIsolation ? (
              <span className="truncate text-xs font-normal">
                {isolationSummary(isolation.state, capability)}
              </span>
            ) : null}
            <HugeiconsIcon
              icon={ChevronDownStandardIcon}
              strokeWidth={1.5}
              className="composer-pill-caret size-[15px]"
            />
          </button>
        </DropdownMenuTrigger>
        <DropdownMenuContent
          side={side}
          align="start"
          sideOffset={0}
          avoidCollisions={true}
          className="unsloth-plus-menu w-80 max-w-[calc(100vw-2rem)]"
        >
          <DropdownMenuLabel>Tool permissions</DropdownMenuLabel>
          <PermissionModeMenuItems
            // Defer past the menu-close focus restoration (see PermissionModeDropdown).
            onRequestFullAccess={() =>
              setTimeout(() => setBypassConfirmOpen(true), 0)
            }
          />
          <ToolIsolationMenuSection
            onRequestLimited={() => setLimitedConfirmOpen(true)}
            onRequestNested={() => setNestedConfirmOpen(true)}
            onRequestDetails={() => setDetailsOpen(true)}
          />
        </DropdownMenuContent>
      </DropdownMenu>
      <ToolIsolationDetailsDialog
        restoreFocus={restoreFocus}
        open={detailsOpen}
        onOpenChange={setDetailsOpen}
      />
      <LimitedModeConfirmDialog
        variant="nested"
        restoreFocus={restoreFocus}
        open={nestedConfirmOpen}
        onOpenChange={setNestedConfirmOpen}
      />
      <LimitedModeConfirmDialog
        restoreFocus={restoreFocus}
        open={limitedConfirmOpen}
        onOpenChange={setLimitedConfirmOpen}
      />
    </>
  );
}
