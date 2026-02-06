import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import type { ReactElement } from "react";
import type { LlmConfig } from "../../types";

type InlineLlmProps = {
  config: LlmConfig;
  onUpdate: (patch: Partial<LlmConfig>) => void;
};

const CODE_LANG_OPTIONS = [
  "python",
  "javascript",
  "typescript",
  "java",
  "kotlin",
  "go",
  "rust",
  "ruby",
  "scala",
  "swift",
  "sql:sqlite",
  "sql:postgres",
  "sql:mysql",
  "sql:tsql",
  "sql:bigquery",
  "sql:ansi",
] as const;

export function InlineLlm({ config, onUpdate }: InlineLlmProps): ReactElement {
  const isCode = config.llm_type === "code";

  return (
    <div className="grid gap-2">
      <Input
        className="nodrag h-7 text-xs"
        placeholder="Model alias"
        value={config.model_alias}
        onChange={(event) =>
          onUpdate({
            // biome-ignore lint/style/useNamingConvention: api schema
            model_alias: event.target.value,
          })
        }
      />
      {isCode && (
        <Select
          value={config.code_lang?.trim() || "python"}
          onValueChange={(value) =>
            onUpdate({
              // biome-ignore lint/style/useNamingConvention: api schema
              code_lang: value,
            })
          }
        >
          <SelectTrigger className="nodrag h-7 text-xs">
            <SelectValue placeholder="Language" />
          </SelectTrigger>
          <SelectContent>
            {CODE_LANG_OPTIONS.map((lang) => (
              <SelectItem key={lang} value={lang}>
                {lang}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      )}
      <p className="text-[11px] text-muted-foreground">
        Prompt/System are edited in dialog or linked input nodes.
      </p>
    </div>
  );
}
