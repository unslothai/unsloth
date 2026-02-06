// src/hooks/ledger-loader.ts
import type { PluginInput } from "@opencode-ai/plugin";
import { readFile, readdir } from "node:fs/promises";
import { join } from "node:path";
import { config } from "../utils/config";

export interface LedgerInfo {
  sessionName: string;
  filePath: string;
  content: string;
}

export async function findCurrentLedger(directory: string): Promise<LedgerInfo | null> {
  const ledgerDir = join(directory, config.paths.ledgerDir);

  try {
    const files = await readdir(ledgerDir);
    const ledgerFiles = files.filter((f) => f.startsWith(config.paths.ledgerPrefix) && f.endsWith(".md"));

    if (ledgerFiles.length === 0) return null;

    // Get most recently modified ledger
    let latestFile = ledgerFiles[0];
    let latestMtime = 0;

    for (const file of ledgerFiles) {
      const filePath = join(ledgerDir, file);
      try {
        const stat = await Bun.file(filePath).stat();
        if (stat && stat.mtime.getTime() > latestMtime) {
          latestMtime = stat.mtime.getTime();
          latestFile = file;
        }
      } catch {
        // Skip files we can't stat
      }
    }

    const filePath = join(ledgerDir, latestFile);
    const content = await readFile(filePath, "utf-8");
    const sessionName = latestFile.replace(config.paths.ledgerPrefix, "").replace(".md", "");

    return { sessionName, filePath, content };
  } catch {
    return null;
  }
}

export function formatLedgerInjection(ledger: LedgerInfo): string {
  return `<continuity-ledger session="${ledger.sessionName}">
${ledger.content}
</continuity-ledger>

You are resuming work from a previous context clear. The ledger above contains your session state.
Review it and continue from where you left off. The "Now" item is your current focus.`;
}

export function createLedgerLoaderHook(ctx: PluginInput) {
  return {
    "chat.params": async (
      _input: { sessionID: string },
      output: { options?: Record<string, unknown>; system?: string },
    ) => {
      const ledger = await findCurrentLedger(ctx.directory);
      if (!ledger) return;

      const injection = formatLedgerInjection(ledger);

      if (output.system) {
        output.system = `${injection}\n\n${output.system}`;
      } else {
        output.system = injection;
      }
    },
  };
}
