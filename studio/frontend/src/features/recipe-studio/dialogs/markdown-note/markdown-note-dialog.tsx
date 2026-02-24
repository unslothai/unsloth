import { Textarea } from "@/components/ui/textarea";
import type { ReactElement } from "react";
import type { MarkdownNoteConfig } from "../../types";
import { FieldLabel } from "../shared/field-label";
import { NameField } from "../shared/name-field";

type MarkdownNoteDialogProps = {
  config: MarkdownNoteConfig;
  onUpdate: (patch: Partial<MarkdownNoteConfig>) => void;
};

export function MarkdownNoteDialog({
  config,
  onUpdate,
}: MarkdownNoteDialogProps): ReactElement {
  const markdownId = `${config.id}-markdown`;

  return (
    <div className="space-y-4">
      <NameField value={config.name} onChange={(value) => onUpdate({ name: value })} />
      <div className="grid gap-2">
        <FieldLabel
          label="Markdown"
          htmlFor={markdownId}
          hint="UI-only note. Not sent to backend payload recipe."
        />
        <Textarea
          id={markdownId}
          className="corner-squircle nodrag min-h-[180px]"
          placeholder="## Note"
          value={config.markdown}
          onChange={(event) => onUpdate({ markdown: event.target.value })}
        />
      </div>
    </div>
  );
}
