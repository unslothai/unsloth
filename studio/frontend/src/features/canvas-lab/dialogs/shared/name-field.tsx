import { Input } from "@/components/ui/input";
import { type ReactElement, useId } from "react";

type NameFieldProps = {
  id?: string;
  value: string;
  onChange: (value: string) => void;
};

export function NameField({
  id,
  value,
  onChange,
}: NameFieldProps): ReactElement {
  const fallbackId = useId();
  const inputId = id ?? fallbackId;
  return (
    <div className="grid gap-2">
      <label
        className="text-xs font-semibold uppercase text-muted-foreground"
        htmlFor={inputId}
      >
        Column name
      </label>
      <Input
        id={inputId}
        className="nodrag"
        value={value}
        onChange={(event) => onChange(event.target.value)}
      />
    </div>
  );
}
