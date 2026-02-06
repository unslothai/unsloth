// bin/platform.test.ts
import { describe, expect, test } from "bun:test";
import { getPlatformPackage, getBinaryPath } from "./platform.js";

describe("getPlatformPackage", () => {
  // #region Darwin platforms
  test("returns darwin-arm64 for macOS ARM64", () => {
    // #given macOS ARM64 platform
    const input = { platform: "darwin", arch: "arm64" };

    // #when getting platform package
    const result = getPlatformPackage(input);

    // #then returns correct package name
    expect(result).toBe("oh-my-opencode-darwin-arm64");
  });

  test("returns darwin-x64 for macOS Intel", () => {
    // #given macOS x64 platform
    const input = { platform: "darwin", arch: "x64" };

    // #when getting platform package
    const result = getPlatformPackage(input);

    // #then returns correct package name
    expect(result).toBe("oh-my-opencode-darwin-x64");
  });
  // #endregion

  // #region Linux glibc platforms
  test("returns linux-x64 for Linux x64 with glibc", () => {
    // #given Linux x64 with glibc
    const input = { platform: "linux", arch: "x64", libcFamily: "glibc" };

    // #when getting platform package
    const result = getPlatformPackage(input);

    // #then returns correct package name
    expect(result).toBe("oh-my-opencode-linux-x64");
  });

  test("returns linux-arm64 for Linux ARM64 with glibc", () => {
    // #given Linux ARM64 with glibc
    const input = { platform: "linux", arch: "arm64", libcFamily: "glibc" };

    // #when getting platform package
    const result = getPlatformPackage(input);

    // #then returns correct package name
    expect(result).toBe("oh-my-opencode-linux-arm64");
  });
  // #endregion

  // #region Linux musl platforms
  test("returns linux-x64-musl for Alpine x64", () => {
    // #given Linux x64 with musl (Alpine)
    const input = { platform: "linux", arch: "x64", libcFamily: "musl" };

    // #when getting platform package
    const result = getPlatformPackage(input);

    // #then returns correct package name with musl suffix
    expect(result).toBe("oh-my-opencode-linux-x64-musl");
  });

  test("returns linux-arm64-musl for Alpine ARM64", () => {
    // #given Linux ARM64 with musl (Alpine)
    const input = { platform: "linux", arch: "arm64", libcFamily: "musl" };

    // #when getting platform package
    const result = getPlatformPackage(input);

    // #then returns correct package name with musl suffix
    expect(result).toBe("oh-my-opencode-linux-arm64-musl");
  });
  // #endregion

  // #region Windows platform
  test("returns windows-x64 for Windows", () => {
    // #given Windows x64 platform (win32 is Node's platform name)
    const input = { platform: "win32", arch: "x64" };

    // #when getting platform package
    const result = getPlatformPackage(input);

    // #then returns correct package name with 'windows' not 'win32'
    expect(result).toBe("oh-my-opencode-windows-x64");
  });
  // #endregion

  // #region Error cases
  test("throws error for Linux with null libcFamily", () => {
    // #given Linux platform with null libc detection
    const input = { platform: "linux", arch: "x64", libcFamily: null };

    // #when getting platform package
    // #then throws descriptive error
    expect(() => getPlatformPackage(input)).toThrow("Could not detect libc");
  });

  test("throws error for Linux with undefined libcFamily", () => {
    // #given Linux platform with undefined libc
    const input = { platform: "linux", arch: "x64", libcFamily: undefined };

    // #when getting platform package
    // #then throws descriptive error
    expect(() => getPlatformPackage(input)).toThrow("Could not detect libc");
  });
  // #endregion
});

describe("getBinaryPath", () => {
  test("returns path without .exe for Unix platforms", () => {
    // #given Unix platform package
    const pkg = "oh-my-opencode-darwin-arm64";
    const platform = "darwin";

    // #when getting binary path
    const result = getBinaryPath(pkg, platform);

    // #then returns path without extension
    expect(result).toBe("oh-my-opencode-darwin-arm64/bin/oh-my-opencode");
  });

  test("returns path with .exe for Windows", () => {
    // #given Windows platform package
    const pkg = "oh-my-opencode-windows-x64";
    const platform = "win32";

    // #when getting binary path
    const result = getBinaryPath(pkg, platform);

    // #then returns path with .exe extension
    expect(result).toBe("oh-my-opencode-windows-x64/bin/oh-my-opencode.exe");
  });

  test("returns path without .exe for Linux", () => {
    // #given Linux platform package
    const pkg = "oh-my-opencode-linux-x64";
    const platform = "linux";

    // #when getting binary path
    const result = getBinaryPath(pkg, platform);

    // #then returns path without extension
    expect(result).toBe("oh-my-opencode-linux-x64/bin/oh-my-opencode");
  });
});
