import { describe, expect, it } from "vitest";

import { translate } from "./messages";

describe("Turkish locale", () => {
  it("translates shared navigation and profile labels", () => {
    expect(translate("shell.navigation.newChat", undefined, "tr")).toBe(
      "Yeni sohbet",
    );
    expect(translate("settings.profile.email", undefined, "en")).toBe("Email");
    expect(translate("settings.profile.email", undefined, "tr")).toBe(
      "E-posta",
    );
  });

  it("preserves interpolation values in Turkish messages", () => {
    expect(
      translate("platformAuth.continueWith", { provider: "GitHub" }, "tr"),
    ).toBe("GitHub ile devam et");
  });
});
