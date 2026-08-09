// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * Optimized-kernel health, for the About tab and the Settings banner.
 *
 * NVIDIA QA P0-1: the managed Windows xFormers was built for PyTorch 2.10.0+cu128 and
 * Python 3.10.11 while the app ran cu130 and Python 3.13.2, so its CUDA extensions never
 * loaded and memory-efficient attention silently went missing. Nothing in the UI said so
 * -- the About tab showed a version string, which a mismatched wheel reports just as
 * happily as a working one.
 *
 * So the version alone is never the status here. The status is whether the kernels load,
 * and when they do not, what the wheel was built for versus what is running -- the pair
 * that names the fix.
 */

import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import {
  type AcceleratorPackage,
  type AcceleratorReport,
  type Health,
  acceleratorHealth,
  acceleratorShowsReason,
  hasDeadAccelerator,
} from "@/hooks/accelerator-report";
import { useAcceleratorReport } from "@/hooks/use-accelerator-report";
import { type TranslationKey, useT } from "@/i18n";
import { cn } from "@/lib/utils";
import { Alert01Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { SettingsRow } from "./settings-row";
import { SettingsSection } from "./settings-section";

// Import name -> display label. Not translated: these are package names.
const PACKAGE_LABELS: Record<string, string> = {
  xformers: "xFormers",
  flash_attn: "FlashAttention",
  torchao: "torchao",
  bitsandbytes: "bitsandbytes",
};

function packageLabel(name: string): string {
  return PACKAGE_LABELS[name] ?? name;
}

const HEALTH_LABEL_KEYS: Record<Health, TranslationKey> = {
  working: "settings.about.accelerator.working",
  broken: "settings.about.accelerator.notLoading",
  absent: "settings.about.accelerator.notInstalled",
  unknown: "settings.about.accelerator.notChecked",
};

const HEALTH_CLASSES: Record<Health, string> = {
  // Deliberately not colour-only: each badge carries its own word, so the state
  // survives a screenshot in greyscale and a colour-vision difference.
  working: "text-muted-foreground",
  broken: "text-destructive font-medium",
  absent: "text-muted-foreground/70",
  unknown: "text-muted-foreground/70",
};

/** "torch 2.10.0+cu128 / Python 3.10.11" from a wheel's recorded build. */
function describeBuild(pkg: AcceleratorPackage): string | null {
  const build = pkg.builtFor;
  if (!build) return null;
  const parts: string[] = [];
  if (build.torch) {
    parts.push(`torch ${build.torch}`);
  } else if (build.cuda) {
    parts.push(`CUDA ${build.cuda}`);
  } else if (build.hip) {
    parts.push(`ROCm ${build.hip}`);
  }
  if (build.python) parts.push(`Python ${build.python}`);
  return parts.length > 0 ? parts.join(" / ") : null;
}

function AcceleratorRow({
  pkg,
  probed,
}: {
  pkg: AcceleratorPackage;
  probed: boolean;
}) {
  const t = useT();
  const health = acceleratorHealth(pkg, probed);
  const build = describeBuild(pkg);
  // Explain a broken one, and an unknown one that came with a reason. On a healthy machine
  // the build detail is noise, and the raw exception text is never the first thing to show.
  //
  // The unknown arm matters because several probes return `runs: null` DELIBERATELY, with an
  // explanation: flash-attn imported but no kernel was launched, torchao registered no native
  // operator. Without it the row read "Not checked" and threw the reason away, so a skipped
  // native extension was indistinguishable from a probe that never ran at all.
  const explained = acceleratorShowsReason(health, pkg.reason);
  const detail = !explained
    ? null
    : health === "broken" && build
      ? t("settings.about.accelerator.builtFor", { build })
      : pkg.reason;

  return (
    <SettingsRow
      label={packageLabel(pkg.name)}
      description={detail}
      alignTop={detail != null}
      // The full reason stays reachable without dominating the row.
      hint={explained && pkg.reason ? pkg.reason : undefined}
    >
      <span className="flex items-baseline gap-2">
        <code className="font-mono text-xs text-muted-foreground">
          {pkg.version ?? "—"}
        </code>
        <span className={cn("text-xs", HEALTH_CLASSES[health])}>
          {t(HEALTH_LABEL_KEYS[health])}
        </span>
      </span>
    </SettingsRow>
  );
}

export function AcceleratorSection() {
  const t = useT();
  const report: AcceleratorReport | null = useAcceleratorReport();
  // Older backends do not send this at all; render nothing rather than an empty section.
  if (!report || report.packages.length === 0) return null;

  return (
    <SettingsSection
      title={t("settings.about.accelerator.sectionTitle")}
      description={t("settings.about.accelerator.sectionDescription")}
    >
      {report.packages.map((pkg) => (
        <AcceleratorRow key={pkg.name} pkg={pkg} probed={report.probed} />
      ))}
    </SettingsSection>
  );
}

/**
 * Shown on every Settings tab, not only the one a user would have to think to open. A
 * dead kernel is silent by nature; this is the thing that surfaces it.
 *
 * Reads the hook itself rather than taking a prop so it can be mounted inside
 * DialogContent, which Radix only renders while the dialog is open. That keeps the
 * detail fetch -- and the native imports behind it -- off app startup.
 */
export function AcceleratorBanner() {
  const t = useT();
  const report = useAcceleratorReport();
  if (!hasDeadAccelerator(report)) return null;

  const packages = (report?.degraded ?? []).map(packageLabel).join(", ");
  return (
    <Alert variant="destructive" className="mb-4">
      <HugeiconsIcon icon={Alert01Icon} />
      <AlertTitle>{t("settings.about.accelerator.bannerTitle")}</AlertTitle>
      <AlertDescription>
        {t("settings.about.accelerator.bannerBody", { packages })}
      </AlertDescription>
    </Alert>
  );
}
