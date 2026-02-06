import * as fs from "node:fs/promises";
import * as os from "node:os";
import * as path from "node:path";

import yaml from "js-yaml";
import { z } from "zod";

import { getDefaultDescription } from "./letta";

export type MemoryScope = "global" | "project";

export type MemoryBlock = {
  scope: MemoryScope;
  label: string;
  description: string;
  limit: number;
  readOnly: boolean;
  value: string;
  filePath: string;
  lastModified: Date;
};


const FrontmatterSchema = z.looseObject({
  label: z.string().min(1).optional(),
  description: z.string().optional(),
  limit: z.number().int().positive().optional(),
  read_only: z.boolean().optional(),
});

type ParsedFrontmatter = z.infer<typeof FrontmatterSchema>;

function splitFrontmatter(text: string): {
  frontmatterText: string | undefined;
  body: string;
} {
  if (!text.startsWith("---\n")) {
    return { frontmatterText: undefined, body: text };
  }

  const endIndex = text.indexOf("\n---\n", 4);
  if (endIndex === -1) {
    return { frontmatterText: undefined, body: text };
  }

  const frontmatterText = text.slice(4, endIndex);
  const body = text.slice(endIndex + "\n---\n".length);
  return { frontmatterText, body };
}

function parseFrontmatter(frontmatterText: string | undefined): ParsedFrontmatter {
  if (!frontmatterText) {
    return {};
  }

  const loaded = yaml.load(frontmatterText);
  const parsed = FrontmatterSchema.safeParse(loaded);
  if (!parsed.success) {
    throw new Error(`Invalid frontmatter: ${parsed.error.message}`);
  }

  return parsed.data;
}

const DEFAULT_LIMIT = 5000;

async function readBlockFile(
  scope: MemoryScope,
  filePath: string,
): Promise<MemoryBlock> {
  const [raw, stats] = await Promise.all([
    fs.readFile(filePath, "utf-8"),
    fs.stat(filePath),
  ]);
  const { frontmatterText, body } = splitFrontmatter(raw);
  const fm = parseFrontmatter(frontmatterText);

  const label = (fm.label ?? path.basename(filePath, path.extname(filePath))).trim();
  const description = (fm.description && fm.description.trim().length > 0
    ? fm.description
    : getDefaultDescription(label)).trim();
  const limit = fm.limit ?? DEFAULT_LIMIT;
  const readOnly = (fm.read_only ?? false) === true;

  return {
    scope,
    label,
    description,
    limit,
    readOnly,
    value: body.trim(),
    filePath,
    lastModified: stats.mtime,
  };
}

async function writeBlockFile(
  filePath: string,
  block: Pick<MemoryBlock, "label" | "description" | "limit" | "readOnly" | "value">,
): Promise<void> {
  const frontmatter = {
    label: block.label,
    description: block.description,
    limit: block.limit,
    read_only: block.readOnly,
  };

  const frontmatterYaml = yaml.dump(frontmatter, {
    lineWidth: 120,
    noRefs: true,
    sortKeys: true,
  });

  const content = `---\n${frontmatterYaml}---\n${block.value.trim()}\n`;

  // Atomic write: write to temp file then rename
  const tempPath = path.join(path.dirname(filePath), `.${path.basename(filePath)}.tmp`);
  await fs.writeFile(tempPath, content, "utf-8");
  await fs.rename(tempPath, filePath);
}

function validateLabel(label: string): string {
  const trimmed = label.trim();
  if (!/^[a-z0-9][a-z0-9-_]{1,60}$/i.test(trimmed)) {
    throw new Error(
      `Invalid label "${label}". Use letters/numbers/dash/underscore (2-61 chars).`,
    );
  }
  return trimmed;
}

export type MemoryStore = {
  ensureSeed(): Promise<void>;
  listBlocks(scope: MemoryScope | "all"): Promise<MemoryBlock[]>;
  getBlock(scope: MemoryScope, label: string): Promise<MemoryBlock>;
  setBlock(
    scope: MemoryScope,
    label: string,
    value: string,
    opts?: { description?: string; limit?: number },
  ): Promise<void>;
  replaceInBlock(scope: MemoryScope, label: string, oldText: string, newText: string): Promise<void>;
};

const SEED_BLOCKS: Array<{ scope: MemoryScope; label: string }> = [
  { scope: "global", label: "persona" },
  { scope: "global", label: "human" },
  { scope: "project", label: "project" },
];

function scopeDir(projectDirectory: string, scope: MemoryScope): string {
  return scope === "global"
    ? path.join(os.homedir(), ".config", "opencode", "memory")
    : path.join(projectDirectory, ".opencode", "memory");
}

async function exists(filePath: string): Promise<boolean> {
  try {
    await fs.access(filePath);
    return true;
  } catch {
    return false;
  }
}

async function ensureGitignore(projectDirectory: string): Promise<void> {
  const memoryDir = path.join(projectDirectory, ".opencode", "memory");
  const gitignorePath = path.join(memoryDir, ".gitignore");

  await fs.mkdir(memoryDir, { recursive: true });

  if (await exists(gitignorePath)) {
    return;
  }

  await fs.writeFile(gitignorePath, "*\n", "utf-8");
}

function stableSortBlocks(blocks: MemoryBlock[]): MemoryBlock[] {
  // Stable ordering for prompt caching (if provider supported).
  // Prefer a small set of canonical blocks first.
  const priority = (block: MemoryBlock): [number, string] => {
    if (block.scope === "global" && block.label === "persona") return [0, block.label];
    if (block.scope === "global" && block.label === "human") return [1, block.label];
    if (block.scope === "project" && block.label === "project") return [2, block.label];

    const scopeBase = block.scope === "global" ? 10 : 20;
    return [scopeBase, block.label];
  };

  blocks.sort((a, b) => {
    const [pa, la] = priority(a);
    const [pb, lb] = priority(b);
    if (pa !== pb) return pa - pb;
    return la.localeCompare(lb);
  });

  return blocks;
}

export function createMemoryStore(projectDirectory: string): MemoryStore {
  return {
    async ensureSeed() {
      await ensureGitignore(projectDirectory);

      for (const seed of SEED_BLOCKS) {
        const dir = scopeDir(projectDirectory, seed.scope);
        await fs.mkdir(dir, { recursive: true });

        const filePath = path.join(dir, `${seed.label}.md`);
        if (await exists(filePath)) {
          continue;
        }

        await writeBlockFile(filePath, {
          label: seed.label,
          description: "",
          limit: 5000,
          readOnly: false,
          value: "",
        });
      }
    },

    async listBlocks(scope) {
      const scopes: MemoryScope[] = scope === "all" ? ["global", "project"] : [scope];
      const blocks: MemoryBlock[] = [];

      for (const s of scopes) {
        const dir = scopeDir(projectDirectory, s);
        if (!(await exists(dir))) {
          continue;
        }

        const entries = await fs.readdir(dir, { withFileTypes: true });
        for (const entry of entries) {
          if (!entry.isFile()) continue;
          if (!entry.name.endsWith(".md")) continue;

          const filePath = path.join(dir, entry.name);
          try {
            blocks.push(await readBlockFile(s, filePath));
          } catch (err) {
            // Ignore invalid files silently for now, but keep going.
          }
        }
      }

      return stableSortBlocks(blocks)
    },

    async getBlock(scope, label) {
      const safeLabel = validateLabel(label);
      const dir = scopeDir(projectDirectory, scope);
      const filePath = path.join(dir, `${safeLabel}.md`);

      if (!(await exists(filePath))) {
        throw new Error(`Memory block not found: ${scope}:${safeLabel}`);
      }

      return readBlockFile(scope, filePath);
    },

    async setBlock(scope, label, value, opts) {
      const safeLabel = validateLabel(label);
      const dir = scopeDir(projectDirectory, scope);
      await fs.mkdir(dir, { recursive: true });

      const filePath = path.join(dir, `${safeLabel}.md`);
      const existing = (await exists(filePath)) ? await readBlockFile(scope, filePath) : undefined;

      if (existing?.readOnly) {
        throw new Error(`Memory block is read-only: ${scope}:${safeLabel}`);
      }

      const description = (opts?.description ?? existing?.description ?? "").trim();
      const limit = opts?.limit ?? existing?.limit ?? 5000;

      if (value.length > limit) {
        throw new Error(
          `Value too large for ${scope}:${safeLabel} (chars=${value.length}, limit=${limit}).`,
        );
      }

      await writeBlockFile(filePath, {
        label: safeLabel,
        description,
        limit,
        readOnly: existing?.readOnly ?? false,
        value,
      });
    },

    async replaceInBlock(scope, label, oldText, newText) {
      const block = await this.getBlock(scope, label);
      if (block.readOnly) {
        throw new Error(`Memory block is read-only: ${scope}:${block.label}`);
      }

      if (!block.value.includes(oldText)) {
        throw new Error(`Old text not found in ${scope}:${block.label}.`);
      }

      const next = block.value.replace(oldText, newText);
      if (next.length > block.limit) {
        throw new Error(
          `Value too large for ${scope}:${block.label} after replace (chars=${next.length}, limit=${block.limit}).`,
        );
      }

      await writeBlockFile(block.filePath, {
        label: block.label,
        description: block.description,
        limit: block.limit,
        readOnly: block.readOnly,
        value: next,
      });
    },
  };
}
