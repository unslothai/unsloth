import { Button } from "@/components/ui/button";
import { Cancel01Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { type KeyboardEvent, type ReactElement, useState } from "react";

type ChipInputProps = {
  values: string[];
  onAdd: (value: string) => void;
  onRemove: (index: number) => void;
  placeholder?: string;
};

export function ChipInput({
  values,
  onAdd,
  onRemove,
  placeholder = "Type and press Enter",
}: ChipInputProps): ReactElement {
  const [draft, setDraft] = useState("");

  const handleKeyDown = (event: KeyboardEvent<HTMLInputElement>) => {
    if (event.key === "Enter") {
      event.preventDefault();
      const trimmed = draft.trim();
      if (trimmed) {
        onAdd(trimmed);
        setDraft("");
      }
    }
    if (event.key === "Backspace" && !draft && values.length > 0) {
      onRemove(values.length - 1);
    }
  };

  return (
    <div className="bg-input/30 border-input focus-within:border-ring focus-within:ring-ring/50 flex min-h-9 flex-wrap items-center gap-1.5 rounded-4xl border bg-clip-padding px-1.5 py-1.5 text-sm transition-colors focus-within:ring-[3px]">
      {values.map((value, index) => (
        <span
          key={`${value}-${index}`}
          className="bg-muted-foreground/10 text-foreground flex h-[calc(--spacing(5.5))] w-fit items-center justify-center gap-1 rounded-4xl pr-0 pl-2 text-xs font-medium whitespace-nowrap"
        >
          {value}
          <Button
            type="button"
            variant="ghost"
            size="icon-xs"
            className="-ml-1 opacity-50 hover:opacity-100"
            onClick={() => onRemove(index)}
          >
            <HugeiconsIcon
              icon={Cancel01Icon}
              strokeWidth={2}
              className="pointer-events-none"
            />
          </Button>
        </span>
      ))}
      <input
        className="nodrag min-w-16 flex-1 bg-transparent text-sm outline-none placeholder:text-muted-foreground"
        placeholder={values.length === 0 ? placeholder : ""}
        value={draft}
        onChange={(event) => setDraft(event.target.value)}
        onKeyDown={handleKeyDown}
      />
    </div>
  );
}
