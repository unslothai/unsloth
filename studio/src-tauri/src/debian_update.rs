// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

use std::fs;
use std::io::{Read, Write};
use std::os::unix::fs::{MetadataExt, PermissionsExt};
use std::path::Path;
use std::process::{Command, Output, Stdio};

const INSTALL_ARGUMENT: &str = "--install-debian-update";
const INSTALLED_BINARY: &str = "/usr/bin/unsloth-studio";
const PACKAGE_NAME: &str = "unsloth";
const MAX_PACKAGE_BYTES: u64 = 512 * 1024 * 1024;

pub(crate) fn is_debian_bundle() -> bool {
    matches!(
        tauri::utils::platform::bundle_type(),
        Some(tauri::utils::config::BundleType::Deb)
    )
}

pub(crate) fn is_supported_install() -> bool {
    is_debian_bundle()
        && tauri::utils::platform::current_exe()
            .is_ok_and(|path| path == Path::new(INSTALLED_BINARY))
        && fs::metadata(INSTALLED_BINARY)
            .is_ok_and(|metadata| metadata.uid() == 0 && metadata.mode() & 0o022 == 0)
        && Path::new("/usr/bin/pkexec").is_file()
}

pub(crate) fn install(bytes: &[u8], signature: &str, version: &str) -> Result<(), String> {
    if !is_supported_install() {
        return Err("The installed Debian updater or system authentication is unavailable. Install the package from the release page.".into());
    }
    let mut command = Command::new("/usr/bin/pkexec");
    command.args([
        "--disable-internal-agent",
        INSTALLED_BINARY,
        INSTALL_ARGUMENT,
        version,
        signature,
    ]);
    send_package(command, bytes)
}

fn send_package(mut command: Command, bytes: &[u8]) -> Result<(), String> {
    let mut child = command
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|error| format!("Could not request administrator authentication: {error}"))?;
    let write_result = child
        .stdin
        .take()
        .ok_or("The installer input is unavailable.")?
        .write_all(bytes);
    let output = child
        .wait_with_output()
        .map_err(|error| format!("Could not wait for the Debian installer: {error}"))?;
    match output.status.code() {
        Some(126) => return Err("Update cancelled at the administrator prompt.".into()),
        Some(127) => return Err("Administrator authentication was unavailable or denied. Use the release page to install the Debian package manually.".into()),
        _ => {}
    }
    require_success(output, "Debian package installation")?;
    write_result.map_err(|error| format!("Could not send the update to the installer: {error}"))
}

pub(crate) fn run_installer() -> Option<Result<(), String>> {
    let mut args = std::env::args_os().skip(1);
    if args.next().as_deref() != Some(std::ffi::OsStr::new(INSTALL_ARGUMENT)) {
        return None;
    }
    Some((|| {
        // this entry point must run before logging, shell discovery, or gui initialization.
        if unsafe { libc::geteuid() } != 0 {
            return Err("Debian updates require system administrator authentication.".into());
        }
        let version = args
            .next()
            .and_then(|value| value.into_string().ok())
            .ok_or("The update version is missing.")?;
        let signature = args
            .next()
            .and_then(|value| value.into_string().ok())
            .ok_or("The update signature is missing.")?;
        if args.next().is_some() || version.len() > 128 || signature.len() > 4096 {
            return Err("Invalid Debian installer arguments.".into());
        }
        let bytes = read_package(std::io::stdin().lock())?;
        install_verified_package(bytes, &signature, &version)
    })())
}

fn install_verified_package(bytes: Vec<u8>, signature: &str, version: &str) -> Result<(), String> {
    crate::desktop_updater::verify_bundle_signature(&bytes, signature)?;
    // root owns both the verified bytes and the directory used by the package manager,
    // and 0700 is stated rather than inherited: tempfile creates directories 0777 masked
    // by the umask, which pkexec passes through from the calling session untouched. At
    // umask 000 that is a world-writable directory holding the package between this
    // verification and apt reading it back, so another local user can unlink the verified
    // file and put their own there. Same reasoning for the file itself, which would
    // otherwise be 0666 masked.
    let directory = tempfile::Builder::new()
        .prefix("unsloth-update-")
        .permissions(fs::Permissions::from_mode(0o700))
        .tempdir_in("/var/tmp")
        .map_err(|error| error.to_string())?;
    let package = directory.path().join("unsloth.deb");
    fs::write(&package, bytes).map_err(|error| error.to_string())?;
    fs::set_permissions(&package, fs::Permissions::from_mode(0o600))
        .map_err(|error| error.to_string())?;
    let (debian_version_override, installed_version) = validate_package(&package, version)?;
    let mut command = system_command("/usr/bin/apt-get");
    if debian_version_override {
        // tauri writes raw semver, so debian sorts a stable release below its own prerelease.
        command.arg("--allow-downgrades");
    }
    // apt holds its frontend lock when this hook checks the previously validated version.
    command.arg("-o").arg(format!(
        "DPkg::Pre-Invoke::={}",
        installed_version_guard(&installed_version)
    ));
    let output = command
        .args([
            "--assume-yes",
            "--no-remove",
            "-o",
            "DPkg::Lock::Timeout=0",
            "install",
        ])
        .arg(&package)
        .output()
        .map_err(|error| format!("Could not start the Debian package manager: {error}"))?;
    require_success(output, "Debian package installation")?;
    let installed = installed_package()?;
    if installed
        != [
            "install ok installed",
            version,
            &package_field(&package, "Architecture")?,
        ]
    {
        return Err(
            "The package manager did not finish installing the requested Unsloth version.".into(),
        );
    }
    Ok(())
}

fn read_package(input: impl Read) -> Result<Vec<u8>, String> {
    let mut bytes = Vec::new();
    input
        .take(MAX_PACKAGE_BYTES + 1)
        .read_to_end(&mut bytes)
        .map_err(|error| format!("Could not read the Debian update: {error}"))?;
    if bytes.is_empty() || bytes.len() as u64 > MAX_PACKAGE_BYTES {
        return Err("The Debian update is empty or exceeds the 512 MiB limit.".into());
    }
    Ok(bytes)
}

fn system_command(program: &str) -> Command {
    let mut command = Command::new(program);
    command
        .env_clear()
        .env("PATH", "/usr/sbin:/usr/bin:/sbin:/bin")
        .env("LANG", "C")
        .env("DEBIAN_FRONTEND", "noninteractive")
        .current_dir("/")
        .stdin(Stdio::null());
    command
}

fn require_success(output: Output, operation: &str) -> Result<String, String> {
    if !output.status.success() {
        return Err(format!(
            "{operation} failed ({}): {}",
            output.status,
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }
    String::from_utf8(output.stdout)
        .map(|value| value.trim().to_owned())
        .map_err(|error| format!("Invalid {operation} output: {error}"))
}

fn package_field(package: &Path, field: &str) -> Result<String, String> {
    let output = system_command("/usr/bin/dpkg-deb")
        .arg("--field")
        .arg(package)
        .arg(field)
        .output()
        .map_err(|error| error.to_string())?;
    require_success(output, "Debian package inspection")
}

fn installed_package() -> Result<Vec<String>, String> {
    let output = system_command("/usr/bin/dpkg-query")
        .args([
            "--show",
            "--showformat=${Status}\n${Version}\n${Architecture}",
            PACKAGE_NAME,
        ])
        .output()
        .map_err(|error| error.to_string())?;
    Ok(require_success(output, "Installed package inspection")?
        .lines()
        .map(str::to_owned)
        .collect())
}

fn validate_identity(
    name: &str,
    version: &str,
    architecture: &str,
    expected_version: &str,
    installed: &[String],
) -> Result<(), String> {
    if name != PACKAGE_NAME
        || version != expected_version
        || installed.len() != 3
        || architecture != installed[2]
    {
        return Err("The signed update does not match the installed Unsloth package, version, or architecture.".into());
    }
    if installed[0] != "install ok installed" {
        return Err(
            "Repair the existing Unsloth package with the system package manager before updating."
                .into(),
        );
    }
    Ok(())
}

fn validate_package(package: &Path, expected_version: &str) -> Result<(bool, String), String> {
    let installed = installed_package()?;
    let version = package_field(package, "Version")?;
    validate_identity(
        &package_field(package, "Package")?,
        &version,
        &package_field(package, "Architecture")?,
        expected_version,
        &installed,
    )?;
    let override_version = needs_debian_version_override(&version, &installed[1])?;
    Ok((override_version, installed[1].clone()))
}

fn installed_version_guard(installed_version: &str) -> String {
    // the version has passed semver parsing, which excludes shell metacharacters.
    format!(
        "test \"$(/usr/bin/dpkg-query --show --showformat='${{Version}}' unsloth)\" = '{installed_version}' || \
         {{ echo 'Installed Unsloth changed during the update. Retry the update.' >&2; exit 1; }}"
    )
}

fn needs_debian_version_override(version: &str, installed: &str) -> Result<bool, String> {
    let next = semver::Version::parse(version).map_err(|error| error.to_string())?;
    let current = semver::Version::parse(installed).map_err(|error| error.to_string())?;
    if !next.cmp_precedence(&current).is_gt() {
        return Err("The Debian update must be newer than the installed Unsloth package.".into());
    }
    let status = system_command("/usr/bin/dpkg")
        .args(["--compare-versions", version, "lt", installed])
        .status()
        .map_err(|error| error.to_string())?;
    match status.code() {
        Some(0) => Ok(true),
        Some(1) => Ok(false),
        _ => Err("Could not compare Debian package versions.".into()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_package_manager_guard_rejects_a_concurrently_changed_version() {
        let directory = tempfile::tempdir().unwrap();
        let status = directory.path().join("status");
        let guard = installed_version_guard("0.1.811-beta").replacen(
            "/usr/bin/dpkg-query",
            &format!(
                "/usr/bin/dpkg-query --admindir='{}'",
                directory.path().display()
            ),
            1,
        );
        for (version, accepted) in [("0.1.811-beta", true), ("0.1.812-beta", false)] {
            fs::write(
                &status,
                format!(
                    "Package: unsloth\nStatus: install ok installed\nVersion: {version}\n\
                     Architecture: amd64\nMaintainer: Test <test@example.invalid>\n\
                     Description: Update fixture\n\n"
                ),
            )
            .unwrap();
            let output = Command::new("/bin/sh")
                .args(["-c", &guard])
                .output()
                .unwrap();
            assert_eq!(output.status.success(), accepted);
            if !accepted {
                assert!(String::from_utf8_lossy(&output.stderr)
                    .contains("Installed Unsloth changed during the update"));
            }
        }
    }

    #[test]
    fn semver_upgrades_allow_stable_promotion_but_never_a_rollback() {
        assert!(needs_debian_version_override("0.1.811", "0.1.811-beta").unwrap());
        assert!(!needs_debian_version_override("0.1.811-beta", "0.1.810-beta").unwrap());
        assert!(needs_debian_version_override("0.1.811-beta", "0.1.811").is_err());
        assert!(needs_debian_version_override("0.1.810", "0.1.811").is_err());
        assert!(needs_debian_version_override("0.1.811", "0.1.811").is_err());
    }

    #[test]
    fn authentication_failures_take_precedence_over_a_closed_input_pipe() {
        for (code, message) in [
            (126, "Update cancelled at the administrator prompt."),
            (
                127,
                "Administrator authentication was unavailable or denied.",
            ),
        ] {
            let mut command = Command::new("/bin/sh");
            command.args(["-c", &format!("exit {code}")]);
            let error = send_package(command, &vec![0; 131072]).unwrap_err();
            assert!(error.starts_with(message), "{error}");
        }
    }
}
