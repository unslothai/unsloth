use serde::Serialize;
use std::collections::HashMap;

const DESKTOP_RELEASE_PAGE_BASE_URL: &str = "https://github.com/unslothai/unsloth/releases/tag/";
const DESKTOP_RELEASE_TAG_PREFIX: &str = "desktop-v";
const DESKTOP_UPDATER_CHANNEL_URL: &str =
    "https://github.com/unslothai/unsloth/releases/download/desktop-latest/latest.json";

#[allow(dead_code)]
#[derive(Debug, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum DesktopUpdateMode {
    InApp,
    ManualLinuxPackage,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct DesktopUpdatePolicy {
    mode: DesktopUpdateMode,
    release_page_base_url: &'static str,
    release_tag_prefix: &'static str,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct ManualUpdateInfo {
    version: String,
    current_version: String,
    body: Option<String>,
    date: Option<String>,
}

#[derive(Debug, serde::Deserialize)]
struct ChannelMetadata {
    version: String,
    body: Option<String>,
    date: Option<String>,
    platforms: HashMap<String, ChannelPlatform>,
}

#[derive(Debug, serde::Deserialize)]
struct ChannelPlatform {
    url: String,
    signature: String,
}

#[tauri::command]
pub(crate) fn desktop_update_policy() -> DesktopUpdatePolicy {
    DesktopUpdatePolicy {
        mode: desktop_update_mode(),
        release_page_base_url: DESKTOP_RELEASE_PAGE_BASE_URL,
        release_tag_prefix: DESKTOP_RELEASE_TAG_PREFIX,
    }
}

#[tauri::command]
pub(crate) async fn check_desktop_manual_update() -> Result<Option<ManualUpdateInfo>, String> {
    if !matches!(desktop_update_mode(), DesktopUpdateMode::ManualLinuxPackage) {
        return Ok(None);
    }

    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(10))
        .build()
        .map_err(|e| e.to_string())?;
    let response = match client.get(DESKTOP_UPDATER_CHANNEL_URL).send().await {
        Ok(response) => response,
        Err(error) => {
            log::warn!("Manual update metadata check failed: {}", error);
            return Ok(None);
        }
    };
    if !response.status().is_success() {
        log::warn!(
            "Manual update metadata check returned HTTP {}",
            response.status()
        );
        return Ok(None);
    }

    let metadata = response
        .json::<ChannelMetadata>()
        .await
        .map_err(|e| format!("Invalid desktop updater metadata: {e}"))?;
    let current_version = env!("CARGO_PKG_VERSION");
    let Some(latest_version) = normalize_version(&metadata.version) else {
        return Err(format!(
            "Desktop updater metadata has invalid version: {}",
            metadata.version
        ));
    };
    validate_channel_metadata(&metadata, &latest_version)?;

    if compare_versions(&latest_version, current_version) <= 0 {
        return Ok(None);
    }

    Ok(Some(ManualUpdateInfo {
        version: latest_version,
        current_version: current_version.to_string(),
        body: metadata.body,
        date: metadata.date,
    }))
}

fn validate_channel_metadata(
    metadata: &ChannelMetadata,
    normalized_version: &str,
) -> Result<(), String> {
    if metadata.platforms.is_empty() {
        return Err("Desktop updater metadata has no platforms".to_string());
    }

    let expected_prefix = format!(
        "https://github.com/unslothai/unsloth/releases/download/desktop-v{normalized_version}/"
    );
    for (platform, entry) in &metadata.platforms {
        if entry.url.trim().is_empty() {
            return Err(format!(
                "Desktop updater metadata missing URL for {platform}"
            ));
        }
        if entry.signature.trim().is_empty() {
            return Err(format!(
                "Desktop updater metadata missing signature for {platform}"
            ));
        }
        if entry.url.contains("/releases/latest/")
            || entry.url.contains("/releases/download/desktop-latest/")
            || !entry.url.starts_with(&expected_prefix)
        {
            return Err(format!(
                "Desktop updater metadata has untrusted URL for {platform}: {}",
                entry.url
            ));
        }
    }
    Ok(())
}

fn desktop_update_mode() -> DesktopUpdateMode {
    #[cfg(target_os = "linux")]
    {
        if std::env::var_os("APPIMAGE").is_some() {
            DesktopUpdateMode::InApp
        } else {
            DesktopUpdateMode::ManualLinuxPackage
        }
    }

    #[cfg(not(target_os = "linux"))]
    {
        DesktopUpdateMode::InApp
    }
}

fn normalize_version(version: &str) -> Option<String> {
    let trimmed = version.trim();
    let without_v = trimmed.strip_prefix('v').unwrap_or(trimmed);
    if parse_version_tuple(without_v).is_some() {
        Some(without_v.to_string())
    } else {
        None
    }
}

fn compare_versions(left: &str, right: &str) -> i8 {
    let Some(left) = parse_version_tuple(left) else {
        return 0;
    };
    let Some(right) = parse_version_tuple(right) else {
        return 0;
    };
    match left.cmp(&right) {
        std::cmp::Ordering::Greater => 1,
        std::cmp::Ordering::Equal => 0,
        std::cmp::Ordering::Less => -1,
    }
}

fn parse_version_tuple(version: &str) -> Option<[u64; 3]> {
    let mut parts = version.splitn(3, '.');
    let major = parts.next()?.parse().ok()?;
    let minor = parts.next()?.parse().ok()?;
    let patch_and_suffix = parts.next()?;
    let patch_len = patch_and_suffix
        .find(|c: char| !c.is_ascii_digit())
        .unwrap_or(patch_and_suffix.len());
    if patch_len == 0 {
        return None;
    }
    let suffix = &patch_and_suffix[patch_len..];
    if !version_suffix_allowed(suffix) {
        return None;
    }
    Some([major, minor, patch_and_suffix[..patch_len].parse().ok()?])
}

fn version_suffix_allowed(suffix: &str) -> bool {
    suffix.is_empty()
        || suffix.starts_with('-')
        || suffix.starts_with('+')
        || suffix.starts_with(".post")
        || suffix.starts_with(".dev")
        || suffix.starts_with("rc")
        || suffix.starts_with('a')
        || suffix.starts_with('b')
}
