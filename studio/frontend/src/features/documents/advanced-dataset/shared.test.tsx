import { PlatformApiError } from "@/integrations/platform-backend";
import { act, render, screen } from "@testing-library/react";
import { useCallback } from "react";
import { describe, expect, it } from "vitest";
import {
  Field,
  SectionCard,
  errorState,
  inputClass,
  textareaClass,
  useAbortableLoad,
} from "./shared";

function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((done) => {
    resolve = done;
  });
  return { promise, resolve };
}

function Harness({
  id,
  loads,
}: { id: string; loads: Map<string, ReturnType<typeof deferred<string>>> }) {
  const loader = useCallback(
    () => loads.get(id)?.promise ?? Promise.resolve(id),
    [id, loads],
  );
  const state = useAbortableLoad(loader);
  return <div>{state.data ?? state.state}</div>;
}

describe("Phase 10 async state boundary", () => {
  it("keeps cards and form controls shrinkable on narrow viewports", () => {
    const { container } = render(
      <SectionCard
        title="Responsive kart"
        actions={<button type="button">İşlem</button>}
      >
        <Field label="Alan">
          <input className={inputClass} />
        </Field>
        <textarea aria-label="Uzun içerik" className={textareaClass} />
      </SectionCard>,
    );

    expect(container.querySelector("section")).toHaveClass(
      "min-w-0",
      "overflow-hidden",
    );
    expect(screen.getByLabelText("Alan")).toHaveClass("min-w-0", "w-full");
    expect(screen.getByLabelText("Uzun içerik")).toHaveClass(
      "min-w-0",
      "w-full",
    );
  });

  it("classifies business-code permission failures", () => {
    expect(
      errorState(
        new PlatformApiError("denied", {
          httpStatus: 200,
          code: 109,
          endpoint: "/datasets/x",
        }),
      ),
    ).toBe("permission");
  });

  it("ignores a stale response after the dataset changes", async () => {
    const oldLoad = deferred<string>();
    const newLoad = deferred<string>();
    const loads = new Map([
      ["old", oldLoad],
      ["new", newLoad],
    ]);
    const view = render(<Harness id="old" loads={loads} />);
    view.rerender(<Harness id="new" loads={loads} />);
    await act(async () => newLoad.resolve("new-value"));
    expect(await screen.findByText("new-value")).toBeInTheDocument();
    await act(async () => oldLoad.resolve("old-value"));
    expect(screen.queryByText("old-value")).not.toBeInTheDocument();
  });
});
