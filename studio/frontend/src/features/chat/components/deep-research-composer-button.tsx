// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import { Telescope01Icon } from "@hugeicons/core-free-icons";
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
import { ChevronDownIcon, XIcon } from "lucide-react";
import { type KeyboardEvent, useState } from "react";
import { useChatRuntimeStore } from "../stores/chat-runtime-store";
import type { ResearchWebsitePolicy } from "../types/research";

function normalizeDomain(raw: string): string | null {
  const value = raw.trim();
  if (!value || /[\\\s]/.test(value)) return null;
  try {
    const url = new URL(value.includes("://") ? value : `https://${value}`);
    if (!/^https?:$/.test(url.protocol) || url.username || url.password || url.port) {
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
              onClick={() => onChange(values.filter((value) => value !== domain))}
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
  const setEnabled = useChatRuntimeStore((state) => state.setDeepResearchEnabled);

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
        <HugeiconsIcon icon={Telescope01Icon} className="size-[15px]" />
        <XIcon className="composer-pill-x" />
      </span>
      <span>Deep research</span>
      <span className="composer-pill-caret flex items-center gap-0.5 text-primary/70">
        <ChevronDownIcon className="size-3" />
      </span>
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
  const setPolicy = useChatRuntimeStore((state) => state.setResearchWebsitePolicy);

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      {open ? (
        <DeepResearchWebsiteAccessContent
          policy={policy}
          setPolicy={setPolicy}
          onClose={() => onOpenChange(false)}
        />
      ) : null}
    </Dialog>
  );
}

function DeepResearchWebsiteAccessContent({
  policy,
  setPolicy,
  onClose,
}: {
  policy: ResearchWebsitePolicy;
  setPolicy: (policy: ResearchWebsitePolicy) => void;
  onClose: () => void;
}) {
  const [draft, setDraft] = useState<ResearchWebsitePolicy>(policy);

  return (
    <DialogContent className="sm:max-w-lg">
        <DialogHeader>
          <DialogTitle>Website access</DialogTitle>
          <DialogDescription>
            Control which websites the next Deep Research run can search and
            read. Limits are enforced by the server and shared with the research
            model.
          </DialogDescription>
        </DialogHeader>
        <div className="space-y-6">
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
              onClose();
            }}
          >
            Save limits
          </Button>
        </DialogFooter>
    </DialogContent>
  );
}
