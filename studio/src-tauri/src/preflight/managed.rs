use super::types::ManagedProbe;
use super::version::{
    backend_version_stale_reason, DESKTOP_MANAGEABILITY_VERSION, DESKTOP_PROTOCOL_VERSION,
};
use serde::Deserialize;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::Duration;
use tokio::io::AsyncReadExt;
use tokio::process::Command;

#[derive(Debug, Deserialize)]
struct DesktopCapability {
    desktop_protocol_version: Option<u16>,
    desktop_manageability_version: Option<u16>,
    supports_api_only: Option<bool>,
    supports_provision_desktop_auth: Option<bool>,
    supports_desktop_backend_ownership: Option<bool>,
    desktop_auth_stale_reason: Option<String>,
    version: Option<String>,
}

async fn run_cli_probe(bin: &Path, args: &[&str]) -> bool {
    let mut cmd = Command::new(bin);
    cmd.args(args).stdout(Stdio::null()).stderr(Stdio::null());

    #[cfg(target_os = "linux")]
    if std::env::var_os("APPIMAGE").is_some() {
        cmd.env_remove("LD_LIBRARY_PATH");
        cmd.env_remove("PYTHONHOME");
        cmd.env_remove("PYTHONPATH");
    }

    // Tauri uses the legacy root regardless of UNSLOTH_STUDIO_HOME / STUDIO_HOME;
    // probe subprocesses must follow the same isolation as process.rs.
    cmd.env_remove("UNSLOTH_STUDIO_HOME");
    cmd.env_remove("STUDIO_HOME");

    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        cmd.creation_flags(crate::process::CREATE_NO_WINDOW);
    }

    let Ok(mut child) = cmd.spawn() else {
        return false;
    };

    match tokio::time::timeout(Duration::from_secs(10), child.wait()).await {
        Ok(Ok(status)) => status.success(),
        _ => {
            let _ = child.kill().await;
            let _ = child.wait().await;
            false
        }
    }
}

async fn probe_cli_capability(bin: &Path) -> Option<DesktopCapability> {
    let mut cmd = Command::new(bin);
    cmd.args(["studio", "desktop-capabilities", "--json"])
        .stdout(Stdio::piped())
        .stderr(Stdio::null());

    #[cfg(target_os = "linux")]
    if std::env::var_os("APPIMAGE").is_some() {
        cmd.env_remove("LD_LIBRARY_PATH");
        cmd.env_remove("PYTHONHOME");
        cmd.env_remove("PYTHONPATH");
    }

    // Tauri uses the legacy root regardless of UNSLOTH_STUDIO_HOME / STUDIO_HOME;
    // probe subprocesses must follow the same isolation as process.rs.
    cmd.env_remove("UNSLOTH_STUDIO_HOME");
    cmd.env_remove("STUDIO_HOME");

    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        cmd.creation_flags(crate::process::CREATE_NO_WINDOW);
    }

    let Ok(mut child) = cmd.spawn() else {
        return None;
    };
    let Some(mut stdout) = child.stdout.take() else {
        return None;
    };

    match tokio::time::timeout(Duration::from_secs(10), child.wait()).await {
        Ok(Ok(status)) if status.success() => {}
        Err(_) => {
            let _ = child.kill().await;
            let _ = child.wait().await;
            return None;
        }
        _ => return None,
    }

    let mut output = Vec::new();
    if stdout.read_to_end(&mut output).await.is_err() {
        return None;
    }

    serde_json::from_slice::<DesktopCapability>(&output).ok()
}

fn desktop_capability_stale_reason(capability: &DesktopCapability) -> Option<String> {
    if capability.desktop_protocol_version != Some(DESKTOP_PROTOCOL_VERSION) {
        return Some("desktop_protocol_incompatible".to_string());
    }
    if capability.supports_api_only != Some(true) {
        return Some("desktop_api_only_unsupported".to_string());
    }
    if capability.supports_provision_desktop_auth != Some(true) {
        return capability
            .desktop_auth_stale_reason
            .clone()
            .or_else(|| Some("desktop_auth_unsupported".to_string()));
    }
    if capability.desktop_manageability_version.unwrap_or(0) < DESKTOP_MANAGEABILITY_VERSION {
        return Some("desktop_manageability_unsupported".to_string());
    }
    if capability.supports_desktop_backend_ownership != Some(true) {
        return Some("desktop_backend_ownership_unsupported".to_string());
    }
    backend_version_stale_reason(capability.version.as_deref())
}

fn desktop_capability_ready(capability: &DesktopCapability) -> bool {
    desktop_capability_stale_reason(capability).is_none()
}

pub(super) async fn probe_managed_bin(bin: PathBuf) -> ManagedProbe {
    if !run_cli_probe(&bin, &["-h"]).await {
        return ManagedProbe::Stale {
            bin,
            reason: "cli_unusable".to_string(),
        };
    }

    let capability = probe_cli_capability(&bin).await;
    if let Some(capability) = capability {
        if desktop_capability_ready(&capability) {
            return ManagedProbe::Ready { bin };
        }
        return ManagedProbe::Stale {
            bin,
            reason: desktop_capability_stale_reason(&capability)
                .unwrap_or_else(|| "desktop_capability_incompatible".to_string()),
        };
    }

    ManagedProbe::Stale {
        bin,
        reason: "desktop_capability_probe_failed".to_string(),
    }
}

pub(super) async fn probe_managed_install() -> ManagedProbe {
    match crate::process::find_unsloth_binary() {
        Some(bin) => probe_managed_bin(bin).await,
        None => ManagedProbe::Missing,
    }
}

pub async fn managed_install_ready() -> bool {
    matches!(probe_managed_install().await, ManagedProbe::Ready { .. })
}
