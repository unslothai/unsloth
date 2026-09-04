// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  ChevronDown,
  CircleAlert,
  Hand,
  RefreshCw,
  ShieldCheck,
} from "lucide-react";
import type { ComponentType } from "react";
import { useEffect, useState } from "react";

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
import { toolIsolationPresentation } from "./tool-isolation";

const TOOL_ISOLATION_LIMITATION_TEXT: Readonly<Record<string, string>> = {
  deprecated_undocumented_sbpl:
    "Apple deprecates sandbox-exec and does not document SBPL for third-party products.",
  detached_descendant_cleanup_unverified:
    "Cleanup of descendants that create a new session or double-fork is unverified.",
};

/**
 * Permission levels for tool calls. Full access stays last because it disables
 * both approval prompts and the code sandbox.
 */
export const PERMISSION_MODE_OPTIONS: readonly {
  value: PermissionMode;
  label: string;
  description: string;
  icon: ComponentType<{ className?: string; strokeWidth?: number }>;
}[] = [
  {
    value: "ask",
    label: "Ask for approval",
    description:
      "Always ask before tool calls, editing files or using the internet",
    icon: Hand,
  },
  {
    value: "auto",
    label: "Approve for me",
    description:
      "Run tool calls, but ask before high-risk actions like credential access, privilege escalation, or destructive commands",
    icon: ShieldCheck,
  },
  {
    value: "off",
    label: "Run automatically",
    description: "Run tool calls without approval prompts inside the sandbox",
    icon: SparklesGlyph,
  },
  {
    value: "full",
    label: "Full access",
    description:
      "Unrestricted: no approval prompts and the code sandbox is disabled",
    icon: CircleAlert,
  },
] as const;

export const FULL_ACCESS_WARNING =
  "Full access lets tool calls run without approval prompts or the code sandbox. They can modify or delete files, run commands, and make network requests. Enable it only when you trust the current task.";

export const TOOL_ISOLATION_UNAVAILABLE_WARNING =
  "OS isolation isn’t available in this environment. Python and Terminal can run with Unsloth’s software safeguards, but they may access anything available to the Studio process.";

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
          <option.icon className="mt-0.5 size-4 shrink-0" strokeWidth={2} />
          <span className="flex min-w-0 flex-1 flex-col gap-0.5">
            <span className="text-ui-13 leading-tight">{option.label}</span>
            <span className="text-xs font-normal leading-snug text-muted-foreground">
              {option.description}
            </span>
          </span>
          {permissionMode === option.value ? (
            <HugeiconsIcon
              icon={Tick02Icon}
              strokeWidth={2}
              className="ml-auto mt-0.5 size-4 shrink-0"
            />
          ) : null}
        </DropdownMenuItem>
      ))}
    </>
  );
}

/** Danger confirmation shown before Full access turns on. Self-contained so
 *  the dropdown works outside the chat page (e.g. the Settings dialog). */
export function FullAccessConfirmDialog({
  open,
  onOpenChange,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}) {
  const setPermissionMode = useChatRuntimeStore((s) => s.setPermissionMode);

  return (
    <AlertDialog open={open} onOpenChange={onOpenChange}>
      <AlertDialogContent size="sm">
        <AlertDialogHeader>
          <AlertDialogTitle>Enable Full access?</AlertDialogTitle>
          <AlertDialogDescription>{FULL_ACCESS_WARNING}</AlertDialogDescription>
        </AlertDialogHeader>
        <AlertDialogFooter>
          <AlertDialogCancel>Cancel</AlertDialogCancel>
          <AlertDialogAction
            variant="destructive"
            className="!bg-destructive !text-destructive-foreground hover:!bg-destructive/90"
            onClick={() => {
              setPermissionMode("full");
              onOpenChange(false);
            }}
          >
            I understand
          </AlertDialogAction>
        </AlertDialogFooter>
      </AlertDialogContent>
    </AlertDialog>
  );
}

function ToolIsolationMenuSection({
  onRequestLimited,
}: {
  onRequestLimited: () => void;
}) {
  useToolIsolationCapabilityRefresh();
  const mode = useChatRuntimeStore((s) => s.toolExecutionMode);
  const capability = useChatRuntimeStore((s) => s.toolIsolationCapability);
  const grant = useChatRuntimeStore((s) => s.limitedToolGrant);
  const loading = useChatRuntimeStore((s) => s.toolIsolationCapabilityLoading);
  const error = useChatRuntimeStore((s) => s.toolIsolationError);
  const refresh = useChatRuntimeStore((s) => s.refreshToolIsolationCapability);
  const setMode = useChatRuntimeStore((s) => s.setToolExecutionMode);
  const presentation = toolIsolationPresentation(mode, capability, grant);

  const unavailable =
    presentation.state === "unavailable" &&
    capability?.protection_state === "unavailable";

  return (
    <>
      <DropdownMenuSeparator />
      <div className="space-y-2 px-3 py-2.5" aria-live="polite">
        <div className="flex items-start gap-2">
          <ShieldCheck
            className={cn(
              "mt-0.5 size-4 shrink-0",
              presentation.state === "unavailable" && "text-destructive",
              presentation.state === "limited" && "text-amber-600",
              presentation.state === "full" && "text-bypass",
            )}
            strokeWidth={2}
          />
          <div className="min-w-0 space-y-1">
            <p className="text-ui-13 font-medium leading-tight">
              {loading ? "Checking OS isolation…" : presentation.label}
            </p>
            <p className="text-xs leading-snug text-muted-foreground">
              {presentation.description}
            </p>
          </div>
        </div>
        {capability ? (
          <dl className="grid grid-cols-[auto_minmax(0,1fr)] gap-x-2 gap-y-1 text-xs text-muted-foreground">
            <dt>Environment</dt>
            <dd className="truncate text-right text-foreground/80">
              {capability.environment}
            </dd>
            <dt>Backend</dt>
            <dd className="truncate text-right text-foreground/80">
              {capability.backend ?? "None qualified"}
            </dd>
            {capability.profile_id ? (
              <>
                <dt>Profile</dt>
                <dd className="truncate text-right text-foreground/80">
                  {capability.profile_id}
                </dd>
              </>
            ) : null}
          </dl>
        ) : null}
        {capability?.reason ? (
          <p className="text-xs leading-snug text-muted-foreground">
            {capability.reason}
          </p>
        ) : null}
        {capability?.limitations.map((limitation) => (
          <p
            key={limitation}
            className="text-xs leading-snug text-amber-700 dark:text-amber-400"
          >
            {TOOL_ISOLATION_LIMITATION_TEXT[limitation] ?? limitation}
          </p>
        ))}
        {capability?.remediation ? (
          <p className="text-xs leading-snug text-muted-foreground">
            {capability.remediation}
          </p>
        ) : null}
        {presentation.state === "limited" || unavailable ? (
          <p className="text-xs leading-snug text-muted-foreground">
            Process Guard, sanitized environment, resource limits, descriptor
            closure, workdir policy, timeout, cancellation, and cleanup remain
            active. Limited is not an OS sandbox.
          </p>
        ) : null}
        {error ? (
          <p className="text-xs leading-snug text-destructive">{error}</p>
        ) : null}
      </div>
      {unavailable ? (
        <DropdownMenuItem
          onSelect={() => setTimeout(onRequestLimited, 0)}
          className="text-ui-13"
        >
          <CircleAlert className="size-4" strokeWidth={2} />
          Use Limited mode for this session
        </DropdownMenuItem>
      ) : null}
      {presentation.state === "limited" ? (
        <DropdownMenuItem
          onSelect={() => setMode("os_isolation_required")}
          className="text-ui-13"
        >
          <ShieldCheck className="size-4" strokeWidth={2} />
          Require OS isolation
        </DropdownMenuItem>
      ) : null}
      {!loading && (!capability || capability.retryable) ? (
        <DropdownMenuItem
          onSelect={(event) => {
            event.preventDefault();
            refresh().catch(() => undefined);
          }}
          className="text-ui-13"
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
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}) {
  const requestGrant = useChatRuntimeStore((s) => s.requestLimitedToolGrant);
  const loading = useChatRuntimeStore((s) => s.toolIsolationGrantLoading);
  const error = useChatRuntimeStore((s) => s.toolIsolationError);

  return (
    <AlertDialog open={open} onOpenChange={onOpenChange}>
      <AlertDialogContent size="sm">
        <AlertDialogHeader>
          <AlertDialogTitle>Use Limited mode?</AlertDialogTitle>
          <AlertDialogDescription>
            {TOOL_ISOLATION_UNAVAILABLE_WARNING}
          </AlertDialogDescription>
        </AlertDialogHeader>
        {error ? (
          <p className="text-center text-xs text-destructive">{error}</p>
        ) : null}
        <AlertDialogFooter>
          <AlertDialogCancel disabled={loading}>Cancel</AlertDialogCancel>
          <AlertDialogAction
            disabled={loading}
            onClick={(event) => {
              event.preventDefault();
              requestGrant()
                .then(() => onOpenChange(false))
                .catch(() => undefined);
            }}
          >
            {loading ? "Enabling…" : "Use Limited mode for this session"}
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

  return <LimitedModeConfirmDialog open={open} onOpenChange={setOpen} />;
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
  const [confirmOpen, setConfirmOpen] = useState(false);
  const [limitedConfirmOpen, setLimitedConfirmOpen] = useState(false);
  const active = permissionModeOption(permissionMode);
  const isolation = toolIsolationPresentation(
    toolExecutionMode,
    capability,
    grant,
  );
  const ActiveIcon = active.icon;

  return (
    <>
      <DropdownMenu>
        <DropdownMenuTrigger asChild={true}>
          <Button
            variant="outline"
            size="sm"
            className={cn(
              "gap-1.5",
              triggerClassName,
              // Last so a text color in triggerClassName cannot override it.
              permissionMode === "full" &&
                "text-bypass hover:text-bypass border-bypass/50",
            )}
            aria-label={`Tool permissions: ${active.label}. ${isolation.label}`}
          >
            <ActiveIcon className="size-3.5 shrink-0" strokeWidth={2} />
            <span className="min-w-0 flex-1 truncate text-left">
              {active.label}
            </span>
            <span className="truncate text-xs text-muted-foreground">
              {isolation.label}
            </span>
            <ChevronDown className="size-3.5 shrink-0 opacity-60" />
          </Button>
        </DropdownMenuTrigger>
        <DropdownMenuContent
          side={side}
          align={align}
          className="w-[300px]"
          avoidCollisions={true}
        >
          <DropdownMenuLabel>
            How should tool calls be approved?
          </DropdownMenuLabel>
          <PermissionModeMenuItems
            // Defer past the menu-close focus restoration so the dialog's
            // focus trap isn't broken by the dropdown grabbing focus back.
            onRequestFullAccess={() =>
              setTimeout(() => setConfirmOpen(true), 0)
            }
          />
          <ToolIsolationMenuSection
            onRequestLimited={() => setLimitedConfirmOpen(true)}
          />
        </DropdownMenuContent>
      </DropdownMenu>
      <FullAccessConfirmDialog
        open={confirmOpen}
        onOpenChange={setConfirmOpen}
      />
      <LimitedModeConfirmDialog
        open={limitedConfirmOpen}
        onOpenChange={setLimitedConfirmOpen}
      />
    </>
  );
}

/**
 * Composer pill (mirrors the MCP pill) showing the current permission level
 * in the chat box; clicking opens the level dropdown. Danger-styled while
 * Full access is on. The Full access pick routes through the store-driven
 * BypassPermissionsConfirmDialog mounted at the chat-page root, so the
 * warning survives this menu unmounting.
 */
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
  const setBypassConfirmOpen = useChatRuntimeStore(
    (s) => s.setBypassConfirmOpen,
  );
  const [limitedConfirmOpen, setLimitedConfirmOpen] = useState(false);
  const active = permissionModeOption(permissionMode);
  const isolation = toolIsolationPresentation(
    toolExecutionMode,
    capability,
    grant,
  );
  const ActiveIcon = active.icon;
  const fullAccess = permissionMode === "full";

  return (
    <>
      <DropdownMenu>
        <DropdownMenuTrigger asChild={true}>
          <button
            type="button"
            className="composer-pill-btn composer-pill-permissions"
            data-pill-label={`${active.label} · ${isolation.label}`}
            data-active={fullAccess ? "true" : "false"}
            data-variant={fullAccess ? "danger" : undefined}
            aria-label={`Tool permissions: ${active.label}. ${isolation.label}`}
            title={`${active.label}: ${active.description}. ${isolation.label}.`}
          >
            <span className="composer-pill-glyph">
              <ActiveIcon className="size-[15px]" strokeWidth={2} />
            </span>
            <span>{active.label}</span>
            <span className="max-w-[190px] truncate text-ui-11 font-normal opacity-75">
              {isolation.label}
            </span>
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
          className="unsloth-plus-menu w-[300px]"
        >
          <DropdownMenuLabel>
            How should tool calls be approved?
          </DropdownMenuLabel>
          <PermissionModeMenuItems
            // Defer past the menu-close focus restoration (see PermissionModeDropdown).
            onRequestFullAccess={() =>
              setTimeout(() => setBypassConfirmOpen(true), 0)
            }
          />
          <ToolIsolationMenuSection
            onRequestLimited={() => setLimitedConfirmOpen(true)}
          />
        </DropdownMenuContent>
      </DropdownMenu>
      <LimitedModeConfirmDialog
        open={limitedConfirmOpen}
        onOpenChange={setLimitedConfirmOpen}
      />
    </>
  );
}
