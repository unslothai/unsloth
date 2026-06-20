use regex::Regex;
use std::path::Path;
use std::sync::OnceLock;

use super::studio_dir;

#[derive(Default, Debug)]
pub(crate) struct RedactionReport {
    pub(crate) replacements: usize,
}

pub(crate) fn redact_text(text: &str, report: &mut RedactionReport) -> String {
    let mut out = ansi_re().replace_all(text, "").to_string();
    if out.len() != text.len() {
        report.replacements += 1;
    }
    out = replace_regex(private_key_re(), &out, "<redacted private key>", report);
    out = replace_regex(url_credentials_re(), &out, "$1<redacted>@", report);
    out = replace_regex(auth_header_re(), &out, "$1: <redacted>", report);
    out = replace_regex(cookie_re(), &out, "$1: <redacted>", report);
    out = replace_regex(token_re(), &out, "<redacted token>", report);
    out = replace_regex(env_secret_re(), &out, "$1=<redacted>", report);
    out = replace_regex(
        native_path_lease_re(),
        &out,
        "$1<redacted native path lease>",
        report,
    );
    out = replace_known_paths(&out, report);
    out = replace_regex(windows_studio_re(), &out, "<studio_home>", report);
    out = replace_regex(windows_home_re(), &out, "%USERPROFILE%", report);
    out = replace_regex(unix_studio_re(), &out, "<studio_home>", report);
    out = replace_regex(unix_home_re(), &out, "$HOME", report);
    out = replace_regex(email_re(), &out, "<redacted email>", report);
    out
}

fn replace_regex(
    regex: &'static Regex,
    input: &str,
    replacement: &str,
    report: &mut RedactionReport,
) -> String {
    let count = regex.find_iter(input).count();
    if count > 0 {
        report.replacements += count;
        regex.replace_all(input, replacement).to_string()
    } else {
        input.to_string()
    }
}

fn replace_known_paths(input: &str, report: &mut RedactionReport) -> String {
    let mut out = input.to_string();
    let studio = studio_dir();
    out = replace_path_literal(&out, &studio, "<studio_home>", report);
    if let Some(home) = dirs::home_dir() {
        out = replace_path_literal(&out, &home, "$HOME", report);
    }
    out
}

fn replace_path_literal(
    input: &str,
    path: &Path,
    replacement: &str,
    report: &mut RedactionReport,
) -> String {
    let path_str = path.display().to_string();
    let mut out = input.to_string();
    for needle in path_variants(&path_str) {
        if needle.is_empty() {
            continue;
        }
        let count = out.matches(&needle).count();
        if count > 0 {
            report.replacements += count;
            out = out.replace(&needle, replacement);
        }
    }
    out
}

fn path_variants(path: &str) -> Vec<String> {
    let mut variants = vec![path.to_string()];
    variants.push(path.replace('/', "\\"));
    variants.push(path.replace('\\', "/"));
    variants.sort();
    variants.dedup();
    variants
}

fn ansi_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"\x1b\[[0-9;?]*[ -/]*[@-~]").unwrap())
}

fn private_key_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(
            r"(?is)-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----.*?-----END [A-Z0-9 ]*PRIVATE KEY-----",
        )
        .unwrap()
    })
}

fn url_credentials_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"(?i)\b([a-z][a-z0-9+.-]*://)[^/\s:@]+(:[^/\s@]*)?@").unwrap())
}

fn auth_header_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"(?i)\b(authorization|proxy-authorization)\s*[:=]\s*(bearer|basic)?\s*[A-Za-z0-9._~+/=-]+").unwrap())
}

fn cookie_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"(?i)\b(cookie|set-cookie)\s*[:=]\s*[^\r\n]+").unwrap())
}

fn token_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"(?i)\b(hf_[A-Za-z0-9]{16,}|github_pat_[A-Za-z0-9_]{20,}|ghp_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9]{20,})\b").unwrap())
}

fn env_secret_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(r"(?i)\b((?:[A-Z0-9]+_)*(?:TOKEN|SECRET|PASSWORD|API_KEY|ACCESS_KEY|KEY)(?:_[A-Z0-9]+)*)\s*=\s*[^\s;&]+").unwrap()
    })
}

fn native_path_lease_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(
            r#"(?i)(\b(?:native_path_lease|nativePathLease)[\"']?\s*[:=]\s*[\"']?)[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+"#,
        )
        .unwrap()
    })
}

fn windows_studio_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(r"(?i)\b[A-Z]:[\\/]Users[\\/][^\\/\r\n\s]+[\\/]\.unsloth[\\/]studio").unwrap()
    })
}

fn windows_home_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"(?i)\b[A-Z]:[\\/]Users[\\/][^\\/\r\n\s]+").unwrap())
}

fn unix_studio_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"(?i)(?:/Users|/home)/[A-Za-z0-9._-]+/\.unsloth/studio").unwrap())
}

fn unix_home_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"(?i)(?:/Users|/home)/[A-Za-z0-9._-]+").unwrap())
}

fn email_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b").unwrap())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn redacts_common_secret_and_path_patterns() {
        let input = concat!(
            "\u{1b}[31mred\u{1b}[0m\n",
            "Authorization: Bearer abcdefghijklmnop\n",
            "Cookie: session=abcdef\n",
            "HF_TOKEN=hf_abcdefghijklmnopqrstuvwxyz\n",
            "API_KEY=secret123\n",
            "native_path_lease=abc.DEF_123\n",
            "url=https://user:pass@example.com/path\n",
            "email=alex@example.com\n",
            "path=/Users/alex/.unsloth/studio/logs/install.log\n",
            "win=C:\\Users\\Alex\\.unsloth\\studio\\logs\\install.log\n",
            "-----BEGIN PRIVATE KEY-----\nabc\n-----END PRIVATE KEY-----\n"
        );
        let mut report = RedactionReport::default();
        let redacted = redact_text(input, &mut report);
        assert!(!redacted.contains("\u{1b}"));
        assert!(!redacted.contains("abcdefghijklmnop"));
        assert!(!redacted.contains("session=abcdef"));
        assert!(!redacted.contains("hf_abcdefghijklmnopqrstuvwxyz"));
        assert!(!redacted.contains("secret123"));
        assert!(!redacted.contains("abc.DEF_123"));
        assert!(redacted.contains("native_path_lease=<redacted native path lease>"));
        assert!(redacted.contains("https://<redacted>@example.com/path"));
        assert!(!redacted.contains("alex@example.com"));
        assert!(redacted.contains("<studio_home>"));
        assert!(!redacted.contains("PRIVATE KEY-----\nabc"));
    }

    #[test]
    fn redaction_avoids_keyboard_monkey_false_positives() {
        let input = "keyboard=present monkey=banana MONKEY=banana KEYBOARD=present API_KEY=secret";
        let mut report = RedactionReport::default();
        let redacted = redact_text(input, &mut report);
        assert!(redacted.contains("keyboard=present"));
        assert!(redacted.contains("monkey=banana"));
        assert!(redacted.contains("MONKEY=banana"));
        assert!(redacted.contains("KEYBOARD=present"));
        assert!(redacted.contains("API_KEY=<redacted>"));
    }
}
