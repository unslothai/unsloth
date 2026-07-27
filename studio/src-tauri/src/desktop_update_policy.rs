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
    if parse_version(without_v).is_some() {
        Some(without_v.to_string())
    } else {
        None
    }
}

fn compare_versions(left: &str, right: &str) -> i8 {
    let Some(left) = parse_version(left) else {
        return 0;
    };
    let Some(right) = parse_version(right) else {
        return 0;
    };
    match compare_parsed_versions(&left, &right) {
        std::cmp::Ordering::Greater => 1,
        std::cmp::Ordering::Equal => 0,
        std::cmp::Ordering::Less => -1,
    }
}

#[derive(Debug, Eq, PartialEq)]
struct ParsedVersion {
    release: [u64; 3],
    suffix: VersionSuffix,
}

#[derive(Debug, Eq, PartialEq)]
enum VersionSuffix {
    Dev(u64),
    Alpha(u64),
    Beta(u64),
    Rc(u64),
    PreRelease(String),
    Stable,
    Build,
    Post(u64),
}

fn parse_version(version: &str) -> Option<ParsedVersion> {
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
    let suffix = parse_version_suffix(&patch_and_suffix[patch_len..])?;
    Some(ParsedVersion {
        release: [major, minor, patch_and_suffix[..patch_len].parse().ok()?],
        suffix,
    })
}

fn parse_version_suffix(suffix: &str) -> Option<VersionSuffix> {
    if suffix.is_empty() {
        return Some(VersionSuffix::Stable);
    }

    if let Some(value) = suffix.strip_prefix('+') {
        if suffix_part_valid(value) {
            return Some(VersionSuffix::Build);
        }
        return None;
    }

    if let Some(value) = suffix.strip_prefix('-') {
        if suffix_part_valid(value) {
            return Some(VersionSuffix::PreRelease(value.to_ascii_lowercase()));
        }
        return None;
    }

    if let Some(number) = parse_numbered_suffix(suffix, &[".post", "post"]) {
        return number.map(VersionSuffix::Post);
    }
    if let Some(number) = parse_numbered_suffix(suffix, &[".dev", "dev"]) {
        return number.map(VersionSuffix::Dev);
    }
    if let Some(number) = parse_numbered_suffix(suffix, &[".rc", "rc"]) {
        return number.map(VersionSuffix::Rc);
    }
    if let Some(number) = parse_numbered_suffix(suffix, &["a"]) {
        return number.map(VersionSuffix::Alpha);
    }
    if let Some(number) = parse_numbered_suffix(suffix, &["b"]) {
        return number.map(VersionSuffix::Beta);
    }

    None
}

fn parse_numbered_suffix(suffix: &str, prefixes: &[&str]) -> Option<Option<u64>> {
    for prefix in prefixes {
        let Some(number) = suffix.strip_prefix(prefix) else {
            continue;
        };
        if number.is_empty() {
            return Some(Some(0));
        }
        return Some(number.parse().ok());
    }
    None
}

fn suffix_part_valid(value: &str) -> bool {
    !value.is_empty()
        && value.split('.').all(|part| {
            !part.is_empty() && part.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'-')
        })
}

fn compare_parsed_versions(left: &ParsedVersion, right: &ParsedVersion) -> std::cmp::Ordering {
    match left.release.cmp(&right.release) {
        std::cmp::Ordering::Equal => compare_suffixes(&left.suffix, &right.suffix),
        ordering => ordering,
    }
}

fn compare_suffixes(left: &VersionSuffix, right: &VersionSuffix) -> std::cmp::Ordering {
    let left_precedence = suffix_precedence(left);
    let right_precedence = suffix_precedence(right);
    if left_precedence != right_precedence {
        return left_precedence.cmp(&right_precedence);
    }

    match (left, right) {
        (VersionSuffix::Dev(left), VersionSuffix::Dev(right))
        | (VersionSuffix::Alpha(left), VersionSuffix::Alpha(right))
        | (VersionSuffix::Beta(left), VersionSuffix::Beta(right))
        | (VersionSuffix::Rc(left), VersionSuffix::Rc(right))
        | (VersionSuffix::Post(left), VersionSuffix::Post(right)) => left.cmp(right),
        (VersionSuffix::Rc(left), VersionSuffix::PreRelease(right)) => {
            compare_numbered_prefix_to_prerelease("rc", *left, right)
                .unwrap_or(std::cmp::Ordering::Equal)
        }
        (VersionSuffix::PreRelease(left), VersionSuffix::Rc(right)) => {
            compare_numbered_prefix_to_prerelease("rc", *right, left)
                .map(std::cmp::Ordering::reverse)
                .unwrap_or(std::cmp::Ordering::Equal)
        }
        (VersionSuffix::PreRelease(left), VersionSuffix::PreRelease(right)) => {
            compare_prerelease(left, right)
        }
        _ => std::cmp::Ordering::Equal,
    }
}

fn suffix_precedence(suffix: &VersionSuffix) -> u8 {
    match suffix {
        VersionSuffix::Dev(_) => 0,
        VersionSuffix::Alpha(_) => 1,
        VersionSuffix::Beta(_) => 2,
        VersionSuffix::Rc(_) | VersionSuffix::PreRelease(_) => 3,
        VersionSuffix::Stable | VersionSuffix::Build => 4,
        VersionSuffix::Post(_) => 5,
    }
}

fn compare_prerelease(left: &str, right: &str) -> std::cmp::Ordering {
    for (left_part, right_part) in left.split('.').zip(right.split('.')) {
        let ordering = compare_prerelease_part(left_part, right_part);
        if !ordering.is_eq() {
            return ordering;
        }
    }
    left.split('.').count().cmp(&right.split('.').count())
}

fn compare_prerelease_part(left: &str, right: &str) -> std::cmp::Ordering {
    let left_number = left.parse::<u64>();
    let right_number = right.parse::<u64>();
    match (left_number, right_number) {
        (Ok(left), Ok(right)) => return left.cmp(&right),
        (Ok(_), Err(_)) => return std::cmp::Ordering::Less,
        (Err(_), Ok(_)) => return std::cmp::Ordering::Greater,
        (Err(_), Err(_)) => {}
    }

    match (split_alpha_numeric(left), split_alpha_numeric(right)) {
        (Some((left_prefix, left_number)), Some((right_prefix, right_number)))
            if left_prefix == right_prefix =>
        {
            return left_number.cmp(&right_number);
        }
        _ => {}
    }

    left.cmp(right)
}

fn compare_numbered_prefix_to_prerelease(
    prefix: &str,
    number: u64,
    prerelease: &str,
) -> Option<std::cmp::Ordering> {
    let (other_prefix, other_number) = split_alpha_numeric(prerelease)?;
    (other_prefix == prefix).then(|| number.cmp(&other_number))
}

fn split_alpha_numeric(value: &str) -> Option<(&str, u64)> {
    let digit_start = value.find(|c: char| c.is_ascii_digit())?;
    if digit_start == 0 || value[digit_start..].bytes().any(|b| !b.is_ascii_digit()) {
        return None;
    }
    Some((&value[..digit_start], value[digit_start..].parse().ok()?))
}

#[cfg(test)]
mod tests {
    #[test]
    fn compare_versions_orders_supported_suffixes() {
        assert!(super::compare_versions("2026.5.3", "2026.5.3-rc1") > 0);
        assert!(super::compare_versions("2026.5.3", "2026.5.3rc1") > 0);
        assert!(super::compare_versions("2026.5.3", "2026.5.3.dev1") > 0);
        assert!(super::compare_versions("2026.5.3.post1", "2026.5.3") > 0);
        assert!(super::compare_versions("2026.5.3.post1", "2026.5.3+build1") > 0);
        assert!(super::compare_versions("2026.5.3+build1", "2026.5.3-beta.1") > 0);
        assert!(super::compare_versions("2026.5.3-rc10", "2026.5.3-rc2") > 0);
        assert!(super::compare_versions("2026.5.3-rc10", "2026.5.3rc2") > 0);
        assert!(super::compare_versions("2026.5.3rc2", "2026.5.3-rc10") < 0);
        assert!(super::compare_versions("2026.5.3-beta10", "2026.5.3-beta2") > 0);
        assert!(super::compare_versions("2026.5.3rc2", "2026.5.3rc1") > 0);
        assert!(super::compare_versions("2026.5.3b1", "2026.5.3a1") > 0);
    }

    #[test]
    fn normalize_version_accepts_backend_suffix_formats() {
        for version in [
            "v2026.5.4.post1",
            "2026.5.4post1",
            "2026.5.4.dev1",
            "2026.5.4dev1",
            "2026.5.4.rc1",
            "2026.5.4rc1",
            "2026.5.4a1",
            "2026.5.4b1",
            "2026.5.4-rc1",
            "2026.5.4+build1",
        ] {
            assert!(super::normalize_version(version).is_some(), "{version}");
        }
    }

    #[test]
    fn normalize_version_rejects_invalid_suffixes() {
        for version in [
            "2026.5.4garbage",
            "2026.5.4-",
            "2026.5.4+",
            "2026.5.4-rc..1",
            "2026.5.4.devx",
        ] {
            assert!(super::normalize_version(version).is_none(), "{version}");
        }
    }
}
