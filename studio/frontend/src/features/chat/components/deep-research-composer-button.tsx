// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import { Telescope02Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { cn } from "@/lib/utils";
import { ChevronDownStandardIcon } from "@/lib/chevron-icons";
import { XIcon } from "lucide-react";
import { type KeyboardEvent, useState } from "react";
import {
  DEFAULT_RESEARCH_MODEL_TIMEOUT_SECONDS,
  useChatRuntimeStore,
} from "../stores/chat-runtime-store";
import { MAX_RESEARCH_MODEL_TIMEOUT_SECONDS } from "../utils/mirrored-chat-settings";
import type { ResearchWebsitePolicy } from "../types/research";

// The field is in minutes; its ceiling is the seconds cap the backend enforces.
const MAX_RESEARCH_MODEL_TIMEOUT_MINUTES = Math.floor(
  MAX_RESEARCH_MODEL_TIMEOUT_SECONDS / 60,
);

function normalizeDomain(raw: string): string | null {
  const value = raw.trim();
  if (!value || /[\\\s]/.test(value)) return null;
  try {
    const url = new URL(value.includes("://") ? value : `https://${value}`);
    if (
      !/^https?:$/.test(url.protocol) ||
      url.username ||
      url.password ||
      url.port
    ) {
      return null;
    }
    return url.hostname
      .toLowerCase()
      .replace(/^\[|\]$/g, "")
      .replace(/\.$/, "");
  } catch {
    return null;
  }
}

function DomainList({
  label,
  description,
  values,
  onChange,
}: {
  label: string;
  description: string;
  values: string[];
  onChange: (values: string[]) => void;
}) {
  const [draft, setDraft] = useState("");
  const [error, setError] = useState("");

  const addDraft = () => {
    if (!draft.trim()) return;
    const domain = normalizeDomain(draft);
    if (!domain) {
      setError("Enter a domain without a port, such as arxiv.org.");
      return;
    }
    if (values.length >= 100 && !values.includes(domain)) {
      setError("You can add up to 100 domains to each list.");
      return;
    }
    if (!values.includes(domain)) onChange([...values, domain]);
    setDraft("");
    setError("");
  };

  const handleKeyDown = (event: KeyboardEvent<HTMLInputElement>) => {
    if (event.key === "Enter" || event.key === ",") {
      event.preventDefault();
      addDraft();
    } else if (event.key === "Backspace" && !draft && values.length) {
      onChange(values.slice(0, -1));
    }
  };

  return (
    <div className="space-y-2">
      <div>
        <div className="text-sm font-medium">{label}</div>
        <p className="mt-0.5 text-xs leading-relaxed text-muted-foreground">
          {description}
        </p>
      </div>
      <div
        className={cn(
          "flex min-h-10 flex-wrap items-center gap-1.5 rounded-2xl border border-input bg-input/20 p-1.5 transition-colors focus-within:border-ring focus-within:ring-3 focus-within:ring-ring/50",
          error && "border-destructive/70",
        )}
      >
        {values.map((domain) => (
          <span
            key={domain}
            className="flex h-6 items-center gap-1 rounded-full bg-muted px-2 text-xs font-medium"
          >
            {domain}
            <button
              type="button"
              className="text-muted-foreground transition-colors hover:text-foreground"
              aria-label={`Remove ${domain}`}
              onClick={() =>
                onChange(values.filter((value) => value !== domain))
              }
            >
              <XIcon className="size-3" />
            </button>
          </span>
        ))}
        <Input
          value={draft}
          onChange={(event) => {
            setDraft(event.target.value);
            setError("");
          }}
          onBlur={addDraft}
          onKeyDown={handleKeyDown}
          placeholder={values.length ? "Add another domain" : "example.com"}
          aria-invalid={Boolean(error)}
          className="h-7 min-w-36 flex-1 border-0 bg-transparent px-1 shadow-none focus-visible:ring-0"
        />
      </div>
      {error ? <p className="text-xs text-destructive">{error}</p> : null}
    </div>
  );
}

export function DeepResearchComposerButton({
  onConfigure,
}: {
  onConfigure: () => void;
}) {
  const enabled = useChatRuntimeStore((state) => state.deepResearchEnabled);
  const setEnabled = useChatRuntimeStore(
    (state) => state.setDeepResearchEnabled,
  );

  if (!enabled) return null;

  return (
    <button
      type="button"
      onClick={onConfigure}
      className="composer-pill-btn"
      data-pill-label="Deep research"
      data-active="true"
      aria-label="Configure Deep Research website access"
      title="Configure website access"
    >
      <span
        role="button"
        aria-label="Disable deep research"
        tabIndex={-1}
        onPointerDown={(event) => event.stopPropagation()}
        onClick={(event) => {
          event.stopPropagation();
          setEnabled(false);
        }}
        className="composer-pill-glyph cursor-pointer"
      >
        <HugeiconsIcon icon={Telescope02Icon} className="size-[15px]" />
        <XIcon className="composer-pill-x" />
      </span>
      <span>Deep research</span>
      {/* Same caret as the other composer pills, so the arrows match. */}
      <HugeiconsIcon
        icon={ChevronDownStandardIcon}
        strokeWidth={1.5}
        className="composer-pill-caret size-[15px] text-primary/70"
      />
    </button>
  );
}

export function DeepResearchWebsiteAccessDialog({
  open,
  onOpenChange,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}) {
  const policy = useChatRuntimeStore((state) => state.researchWebsitePolicy);
  const setPolicy = useChatRuntimeStore(
    (state) => state.setResearchWebsitePolicy,
  );
  const modelTimeoutSeconds = useChatRuntimeStore(
    (state) => state.researchModelTimeoutSeconds,
  );
  const setModelTimeoutSeconds = useChatRuntimeStore(
    (state) => state.setResearchModelTimeoutSeconds,
  );

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      {open ? (
        <DeepResearchWebsiteAccessContent
          policy={policy}
          setPolicy={setPolicy}
          modelTimeoutSeconds={modelTimeoutSeconds}
          setModelTimeoutSeconds={setModelTimeoutSeconds}
          onClose={() => onOpenChange(false)}
        />
      ) : null}
    </Dialog>
  );
}

function DeepResearchWebsiteAccessContent({
  policy,
  setPolicy,
  modelTimeoutSeconds,
  setModelTimeoutSeconds,
  onClose,
}: {
  policy: ResearchWebsitePolicy;
  setPolicy: (policy: ResearchWebsitePolicy) => void;
  modelTimeoutSeconds: number;
  setModelTimeoutSeconds: (seconds: number) => void;
  onClose: () => void;
}) {
  const [draft, setDraft] = useState<ResearchWebsitePolicy>(policy);
  const [unlimited, setUnlimited] = useState(modelTimeoutSeconds === 0);
  // Unlimited has no minutes of its own, so turning the limit back on offers the default.
  const [timeoutMinutes, setTimeoutMinutes] = useState(
    String(
      Math.ceil(
        (modelTimeoutSeconds || DEFAULT_RESEARCH_MODEL_TIMEOUT_SECONDS) / 60,
      ),
    ),
  );
  // The API accepts second-level values the minutes field cannot spell, so saving an untouched
  // control must replay the stored seconds rather than the rounded minutes.
  const [timeoutEdited, setTimeoutEdited] = useState(false);

  return (
    <DialogContent className="sm:max-w-lg">
      <DialogHeader>
        <DialogTitle>Deep research</DialogTitle>
        <DialogDescription>
          Control website access and model request time for the next Deep
          Research run.
        </DialogDescription>
      </DialogHeader>
      <div className="space-y-6">
        <div className="space-y-2">
          <div>
            {/* A run makes many model requests, so this bounds each one, not the run. */}
            <div className="text-sm font-medium">Time per model request</div>
            <p className="mt-0.5 text-xs leading-relaxed text-muted-foreground">
              {unlimited
                ? "No limit on a single model request. Slow models can continue while they keep producing output."
                : "Maximum time for each model request, so a run of many requests can take longer. Output stall safeguards stay active."}
            </p>
          </div>
          <div className="flex gap-2">
            <Input
              type="number"
              min="1"
              max={MAX_RESEARCH_MODEL_TIMEOUT_MINUTES}
              step="1"
              value={timeoutMinutes}
              disabled={unlimited}
              onChange={(event) => {
                setTimeoutEdited(true);
                setTimeoutMinutes(event.target.value);
              }}
              aria-label="Deep Research time per model request in minutes"
              className="w-28"
            />
            <span className="self-center text-sm text-muted-foreground">
              minutes
            </span>
            <Button
              type="button"
              variant="outline"
              onClick={() => {
                setTimeoutEdited(true);
                setUnlimited((value) => !value);
              }}
            >
              {unlimited ? "Use a limit" : "No limit"}
            </Button>
          </div>
        </div>
        <DomainList
          label="Allow only"
          description="When set, research can access only these domains and their subdomains."
          values={draft.allowedDomains}
          onChange={(allowedDomains) => setDraft({ ...draft, allowedDomains })}
        />
        <DomainList
          label="Always block"
          description="These domains and their subdomains stay blocked. Blocking takes precedence."
          values={draft.blockedDomains}
          onChange={(blockedDomains) => setDraft({ ...draft, blockedDomains })}
        />
      </div>
      <DialogFooter>
        <Button variant="ghost" onClick={onClose}>
          Cancel
        </Button>
        <Button
          onClick={() => {
            setPolicy(draft);
            const minutes = Number(timeoutMinutes);
            // The max attribute does not stop a typed value reaching here, and falling through to the default
            // would hand someone asking for a long run a short one.
            setModelTimeoutSeconds(
              unlimited
                ? 0
                : !timeoutEdited
                  ? modelTimeoutSeconds
                  : Number.isSafeInteger(minutes) && minutes >= 1
                    ? Math.min(minutes, MAX_RESEARCH_MODEL_TIMEOUT_MINUTES) * 60
                    : DEFAULT_RESEARCH_MODEL_TIMEOUT_SECONDS,
            );
            onClose();
          }}
        >
          Save limits
        </Button>
      </DialogFooter>
    </DialogContent>
  );
}
