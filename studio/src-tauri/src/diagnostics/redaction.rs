use regex::{Captures, Regex};
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
    out = replace_regex(url_query_value_re(), &out, "$1=<redacted>", report);
    out = replace_regex(url_fragment_re(), &out, "$1#<redacted>", report);
    out = replace_regex(auth_scheme_re(), &out, "$1<redacted>", report);
    out = replace_with(secret_kv_re(), &out, redact_kv, report);
    out = replace_with(weak_secret_kv_re(), &out, redact_weak_kv, report);
    out = replace_with(secret_flag_re(), &out, redact_kv, report);
    out = replace_with(bearer_re(), &out, redact_bearer, report);
    out = replace_regex(cookie_re(), &out, "$1<redacted>", report);
    out = replace_regex(sk_key_re(), &out, "$1<redacted token>", report);
    out = replace_regex(token_re(), &out, "$1<redacted token>", report);
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

// Like replace_regex, but the closure can decline a match (returns it unchanged) and
// only real changes are counted.
fn replace_with(
    regex: &'static Regex,
    input: &str,
    redact: fn(&Captures) -> Option<String>,
    report: &mut RedactionReport,
) -> String {
    let mut count = 0;
    let out = regex.replace_all(input, |caps: &Captures| match redact(caps) {
        Some(replacement) => {
            count += 1;
            replacement
        }
        None => caps[0].to_string(),
    });
    report.replacements += count;
    out.into_owned()
}

const REDACTED: &str = "<redacted>";
const AUTH_SCHEMES: [&str; 6] = ["bearer", "basic", "digest", "token", "apikey", "negotiate"];

fn redact_kv(caps: &Captures) -> Option<String> {
    let whole = caps.get(0)?;
    let value = ["ev2", "ev", "dv", "sv", "uv"]
        .iter()
        .find_map(|name| caps.name(name))?;
    let key = caps["key"].to_ascii_lowercase();
    let numeric_is_secret = key.starts_with("pass") || key.ends_with("secret");
    if value.as_str().bytes().all(|b| b.is_ascii_digit()) && !numeric_is_secret {
        return None;
    }
    // An Authorization value carries its scheme inside the quotes; keep the scheme readable.
    let (scheme, rest) = value
        .as_str()
        .split_once(' ')
        .unwrap_or((value.as_str(), ""));
    let masked = if AUTH_SCHEMES.contains(&scheme.to_ascii_lowercase().as_str()) {
        let rest = rest.trim_start();
        if rest.is_empty() || rest == REDACTED {
            return None;
        }
        format!("{scheme} {REDACTED}")
    } else {
        REDACTED.to_string()
    };
    let text = whole.as_str();
    let start = whole.start();
    Some(format!(
        "{}{masked}{}",
        &text[..value.start() - start],
        &text[value.end() - start..]
    ))
}

// A bare "token" key also names tokenizer settings ("eos_token": "</s>"), so only a
// credential-shaped value counts.
fn redact_weak_kv(caps: &Captures) -> Option<String> {
    let value = ["ev2", "ev", "dv", "sv", "uv"]
        .iter()
        .find_map(|name| caps.name(name))?;
    if value.as_str().starts_with('[') || !looks_like_credential(value.as_str()) {
        return None;
    }
    redact_kv(caps)
}

fn redact_bearer(caps: &Captures) -> Option<String> {
    let credential = &caps[2];
    if !looks_like_credential(credential) {
        return None;
    }
    Some(format!("{}{REDACTED}", &caps[1]))
}

// Spares prose such as "Bearer credentials were not accepted".
fn looks_like_credential(value: &str) -> bool {
    if value.len() < 8 || value.starts_with('<') {
        return false;
    }
    value.len() >= 20
        || value
            .bytes()
            .any(|b| b.is_ascii_digit() || b"._-+/=~".contains(&b))
        || (value.bytes().any(|b| b.is_ascii_uppercase())
            && value.bytes().any(|b| b.is_ascii_lowercase()))
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

fn url_query_value_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"([?&][^=\s&`]+)=[^&#\s`]+").unwrap())
}

fn url_fragment_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"(?i)(https?://[^\s`#]+)#[^\s`]+").unwrap())
}

// Key names whose value is a secret. "token" alone is absent so max_tokens and eos_token
// survive; this mirrors studio/backend/utils/log_redaction.py.
const SECRET_KEYS: &str = r"proxy-authorization|authorization|x-api-key|api[-_]?key|apikey|(?:hf|access|refresh|auth|bearer|id|session|wandb|hub)[-_]?token|aws_session_token|client[-_]?secret|aws_secret_access_key|secret[-_]?access[-_]?key|secret[-_]?key|private[-_]?key|password|passwd|secret";

// Checked against the value's shape instead, see redact_weak_kv.
const WEAK_SECRET_KEYS: &str = r"(?:[A-Za-z0-9]+[-_]?)?token|credentials?";

// Quoted values come in three shapes: "plain", 'single', and \"escaped\" (a dict logged
// inside a JSON log line, which is how the backend's server-*.log files are written; a
// body logged as a string inside that is escaped twice, \\\"like this\\\"). Each escaped
// shape ends only at a quote with its own backslash count; a quote escaped deeper than that
// is part of the value.
const SECRET_VALUE: &str = r#"(?:\\\\\\"(?P<ev2>(?:[^"\\\r\n]|\\{4,}"|\\+[^"\\\r\n])+)(?:\\\\\\")?|\\"(?P<ev>(?:[^"\\\r\n]|\\{2,}"|\\+[^"\\\r\n])+)(?:\\")?|"(?P<dv>(?:[^"\\\r\n]|\\.)+)"?|'(?P<sv>(?:[^'\\\r\n]|\\.)+)'?|(?P<uv>[^"'\\\s,;}\]<>]{6,}))"#;

// The leading class stands in for a lookbehind (the regex crate has none), so the key
// still matches inside OPENAI_API_KEY or x-goog-api-key.
fn kv_re(keys: &str) -> Regex {
    Regex::new(&format!(
        r#"(?i)(?:^|[^A-Za-z0-9]|\\[nrt])(?P<key>{keys})\b(?:\\*["'])?[ \t]*[:=][ \t]*{SECRET_VALUE}"#
    ))
    .unwrap()
}

fn secret_kv_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| kv_re(SECRET_KEYS))
}

fn weak_secret_kv_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| kv_re(WEAK_SECRET_KEYS))
}

fn secret_flag_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(&format!(
            r"(?i)(?:^|\s|\\[nrt])--(?P<key>{SECRET_KEYS})[ \t]+{SECRET_VALUE}"
        ))
        .unwrap()
    })
}

fn auth_scheme_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r#"(?i)((?:\\[nrt]|\b)(?:proxy-)?authorization(?:\\*["'])?[ \t]*[:=][ \t]*(?:\\*["'])?(?:bearer|basic|digest|token|apikey|negotiate)[ \t]+)[^\s"'\\,}\]<]+"#).unwrap())
}

fn bearer_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r#"(?i)((?:\\[nrt]|\b)bearer[ \t]+)([^\s"'\\,}\]<]+)"#).unwrap())
}

fn cookie_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(r#"(?i)((?:\\[nrt]|\b)(?:set-)?cookie(?:\\*["'])?[ \t]*[:=][ \t]*)(?:[^\r\n\\]|\\[^rnt])+"#).unwrap()
    })
}

// Not after a hyphen, so a name like checkpoint-sk-9f8a... is left alone.
fn sk_key_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"(^|[^A-Za-z0-9-]|\\[nrt])sk-[A-Za-z0-9_-]{20,}").unwrap())
}

fn token_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"(\\[nrt]|\b)(?:hf_(?:oauth_[A-Za-z0-9._~+/=-]{20,}|[A-Za-z0-9]{16,})|github_pat_[A-Za-z0-9_]{20,}|(?:gh[pousr]_|gsk_|xai-|glpat-|xox[abprs]-|ya29\.)[A-Za-z0-9_.-]{16,}|AIza[0-9A-Za-z_-]{30,}|(?:AKIA|ASIA)[0-9A-Z]{16}\b|eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{5,})").unwrap())
}

fn env_secret_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(r#"(?i)\b((?:[A-Z0-9]+_)*(?:TOKEN|SECRET|PASSWORD|API_KEY|ACCESS_KEY|KEY)(?:_[A-Z0-9]+)*)\s*=\s*(?:["'][^"'<\r\n][^"'\r\n]*["']?|[^\s;&<"'][^\s;&]*)"#).unwrap()
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
        Regex::new(r"(?i)\b[A-Z]:[\\/]Users[\\/][^\\/:\r\n\s]+[\\/]\.unsloth[\\/]studio").unwrap()
    })
}

fn windows_home_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"(?i)\b[A-Z]:[\\/]Users[\\/][^\\/:\r\n\s]+").unwrap())
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
    fn redacts_hugging_face_oauth_tokens_whole() {
        let token = "hf_oauth_A1b2C3d4E5A1b2C3d4E5A1b2C3d4E5A1b2C3d4E5";
        let input = format!("download failed while using {token} for org/model\n");
        let mut report = RedactionReport::default();
        let redacted = redact_text(&input, &mut report);
        assert!(!redacted.contains("A1b2C3d4E5"));
        assert!(redacted.contains("for org/model"));
    }

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
            "signed=https://example.com/object?X-Amz-Signature=presignedvalue987&version=1#fragmentsecret\n",
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
        assert!(!redacted.contains("presignedvalue987"));
        assert!(!redacted.contains("fragmentsecret"));
        assert!(redacted.contains("?X-Amz-Signature=<redacted>&version=<redacted>#<redacted>"));
        assert!(!redacted.contains("alex@example.com"));
        assert!(redacted.contains("<studio_home>"));
        assert!(!redacted.contains("PRIVATE KEY-----\nabc"));
    }

    #[test]
    fn windows_home_redaction_keeps_the_line_and_column_suffix() {
        // A colon cannot appear inside a Windows path segment, but the Windows classes used to
        // allow one, so a single-segment path swallowed whatever followed it. uv reports a bad
        // requirements path as `<path>:<line>:<col>`, and issue #11012 arrived with the position
        // and the file name already eaten, which is most of what made it expensive to diagnose.
        // The Unix rules never had this because their class is [A-Za-z0-9._-].
        let mut report = RedactionReport::default();
        let redacted = redact_text(
            "error: Unexpected '[' at C:\\Users\\Alex:1:1\n",
            &mut report,
        );
        assert!(
            !redacted.contains("Alex"),
            "user name still present: {redacted}"
        );
        assert!(
            redacted.contains("%USERPROFILE%:1:1"),
            "line and column were redacted away: {redacted}"
        );

        // A deeper path was never affected, and must stay that way.
        let mut report = RedactionReport::default();
        let deep = redact_text(
            "at C:\\Users\\Alex\\AppData\\Local\\Temp\\tmp1.tmp:1:1\n",
            &mut report,
        );
        assert!(!deep.contains("Alex"), "user name still present: {deep}");
        assert!(
            deep.contains("\\AppData\\Local\\Temp\\tmp1.tmp:1:1"),
            "{deep}"
        );
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

    #[test]
    fn redacts_credentials_in_every_shape_a_log_carries() {
        // (input, secret that must not survive, text that must)
        // Vendor-prefixed fakes are split with concat! so secret scanners don't flag them.
        let cases: &[(&str, &str, &str)] = &[
            (
                "key sk-unsloth-0123456789abcdef0123456789abcdef ok",
                "0123456789abcdef",
                "key <redacted token> ok",
            ),
            (
                "sk-proj-AbCdEf0123456789_AbCdEf0123456789",
                "AbCdEf0123456789",
                "",
            ),
            (
                concat!("sk-ant", "-api03-AbCdEf0123456789AbCdEf0123"),
                "AbCdEf0123456789",
                "",
            ),
            (
                "key_sk-unsloth-0123456789abcdef0123456789abcdef",
                "0123456789abcdef",
                "",
            ),
            (
                "token-gho_AbCdEf0123456789AbCdEf0123",
                "AbCdEf0123456789",
                "",
            ),
            (
                concat!(
                    "gsk",
                    "_AbCdEf0123456789AbCdEf xai",
                    "-AbCdEf0123456789AbCdEf"
                ),
                "AbCdEf0123456789",
                "",
            ),
            (
                concat!(
                    "glpat",
                    "-AbCdEf0123456789AbCd xoxb",
                    "-1234567890-AbCdEf012345"
                ),
                "AbCdEf",
                "",
            ),
            (
                concat!(
                    "AIza",
                    "SyAbCdEf0123456789AbCdEf0123456789 AKIA",
                    "ABCDEF0123456789"
                ),
                "AbCdEf0123456789",
                "",
            ),
            ("hf_oauth_AbCdEf.0123~456789AbCdEf0123", "456789AbCdEf", ""),
            (
                "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJvd25lciJ9.c2lnbmF0dXJlc2lnbmF0dXJl",
                "c2lnbmF0dXJl",
                "",
            ),
            (
                r#"{"Authorization": "Bearer opaque0123456789"}"#,
                "opaque0123456789",
                r#"{"Authorization": "Bearer <redacted>"}"#,
            ),
            (
                "Authorization: Token opaque0123456789",
                "opaque0123456789",
                "Authorization: Token <redacted>",
            ),
            (
                r#"{"Authorization": ["Bearer opaque0123456789"]}"#,
                "opaque0123456789",
                "",
            ),
            (
                "retrying with Bearer opaque0123456789",
                "opaque0123456789",
                "",
            ),
            (
                "X-Api-Key: opaque0123456789",
                "opaque0123456789",
                "X-Api-Key: <redacted>",
            ),
            ("api-key: opaque0123456789", "opaque0123456789", ""),
            ("x-goog-api-key: opaque0123456789", "opaque0123456789", ""),
            (
                r#"{"refresh_token":"opaque0123456789","token_type":"bearer"}"#,
                "opaque0123456789",
                r#""token_type":"bearer""#,
            ),
            (
                r#"{"apiKey": "opaque0123456789", 'password': 'opaque0123456789'}"#,
                "opaque0123456789",
                r#""apiKey": "<redacted>""#,
            ),
            (
                r#"{"openai_api_key": "opaque0123456789", "OPENAI_API_KEY": "opaque0123456789"}"#,
                "opaque0123456789",
                "",
            ),
            (
                r#"{"hf_token": "opaque0123456789", "wandb_token": "opaque0123456789"}"#,
                "opaque0123456789",
                "",
            ),
            (
                r#"{"aws_secret_access_key": "opaque0123456789", "private_key": "opaque0123456789"}"#,
                "opaque0123456789",
                "",
            ),
            (
                r#"{"client_secret": "opaque0123456789", "db_password": "opaque0123456789"}"#,
                "opaque0123456789",
                "",
            ),
            (
                r#"{"password": "ab\"opaque0123456789"}"#,
                "opaque0123456789",
                "",
            ),
            (
                "password='correct horse battery staple'",
                "horse battery",
                "password='<redacted>'",
            ),
            (
                "api_key: opaque0123456789",
                "opaque0123456789",
                "api_key: <redacted>",
            ),
            (
                "unsloth studio --api-key opaque0123456789 --port 8888",
                "opaque0123456789",
                "--port 8888",
            ),
            (
                r#"{"set-cookie": "session=opaque0123456789"}"#,
                "opaque0123456789",
                "",
            ),
            // A dict logged inside a JSON log line arrives with escaped quotes.
            (
                r#"{"event": "h {\"Authorization\": \"Bearer opaque0123456789\"}"}"#,
                "opaque0123456789",
                r#"\"Authorization\": \"Bearer <redacted>\"}"#,
            ),
            (
                r#"{"event": "r {\"access_token\": \"opaque0123456789\", \"n\": 1}"}"#,
                "opaque0123456789",
                r#"\"access_token\": \"<redacted>\", \"n\": 1}"#,
            ),
            // A response body logged as a string inside that is escaped twice.
            (
                r#"{"event": "body='{\\\"access_token\\\": \\\"opaque0123456789\\\"}'"}"#,
                "opaque0123456789",
                r#"\\\"access_token\\\": \\\"<redacted>\\\"}'"#,
            ),
            // A quote inside the value is escaped one level deeper than the delimiters.
            (
                r#"{"event": "body {\"password\": \"ab\\\"opaque0123456789\", \"n\": 1}"}"#,
                "opaque0123456789",
                r#"\"password\": \"<redacted>\", \"n\": 1}"#,
            ),
            (
                r#"{"event": "body='{\\\"password\\\": \\\"ab\\\\\\\"opaque0123456789\\\"}'"}"#,
                "opaque0123456789",
                r#"\\\"password\\\": \\\"<redacted>\\\"}'"#,
            ),
            (
                r#"{"apiToken": "opaque0123456789", "cf_tunnel_token": "opaque0123456789"}"#,
                "opaque0123456789",
                r#""apiToken": "<redacted>""#,
            ),
            (
                r#"{"token": "opaque0123456789", "credentials": "opaque0123456789"}"#,
                "opaque0123456789",
                "",
            ),
        ];
        // Failures name the case by index: printing a fixture secret trips CodeQL's
        // cleartext-logging rule.
        for (i, (input, secret, kept)) in cases.iter().enumerate() {
            let mut report = RedactionReport::default();
            let redacted = redact_text(input, &mut report);
            assert!(!redacted.contains(secret), "case {i}: secret survived redaction");
            assert!(redacted.contains(kept), "case {i}: expected {kept:?} to survive");
            assert!(report.replacements > 0, "case {i}: nothing was redacted");
        }
    }

    #[test]
    fn credential_rules_leave_ordinary_log_text_alone() {
        let input = concat!(
            "pip install scikit-learn (sk-learn) and a bare sk- in prose\n",
            "saved checkpoint-sk-9f8a7b6c5d4e3f2a1b0c9d8e7f6a5b4c\n",
            "the payload starts with eyJ but is not a token\n",
            "Authorization header missing; n_tokens=4096\n",
            "Bearer credentials were not accepted\n",
            "{\"token_type\":\"bearer\",\"max_tokens\": 4096,\"eos_token\": \"</s>\"}\n",
            "{\"secret_sauce_path\": \"/x\", \"password_required\": true, \"api_key_count\": 2}\n",
            "{\"event\": \"loaded {\\\"pad_token\\\": \\\"<pad>\\\"}\"}\n",
            "{\"unk_token\": \"[UNK]\", \"eos_token\": \"<|im_end|>\", \"next_token\": \"Hello\"}\n",
            "{\"num_tokens\": 512, \"token_count\": 12, \"tokenizer\": \"fast\", \"credentials\": null}\n",
            "sha256:9f8a7b6c5d4e3f2a1b0c9d8e7f6a5b4c3d2e1f0a\n",
        );
        let mut report = RedactionReport::default();
        let redacted = redact_text(input, &mut report);
        assert_eq!(redacted, input);
        assert_eq!(report.replacements, 0);
    }

    #[test]
    fn an_escaped_newline_counts_as_a_boundary() {
        // Multi-line text inside a JSON log event (a traceback, a config dump) keeps its
        // newlines as a literal \n, which word boundaries do not see.
        for body in [
            r"api_key: opaque0123456789",
            r"Authorization: Basic opaque0123456789",
            r"hf_opaque0123456789abcd",
            r"sk-unsloth-opaque0123456789abcdef0123",
            r"eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJ4In0.opaque0123456789",
            r"Cookie: s=opaque0123456789",
            r"--api-key opaque0123456789",
            r"Bearer opaque0123456789",
        ] {
            for sep in [r"\n", r"\r\n", r"\t"] {
                let input = format!(r#"{{"event": "dump:{sep}{body}{sep}next line"}}"#);
                let redacted = redact_text(&input, &mut RedactionReport::default());
                assert!(!redacted.contains("opaque"), "leaked: {redacted}");
                assert!(redacted.contains(&format!("{sep}next line")), "{redacted}");
            }
        }
    }

    #[test]
    fn a_bare_scheme_never_swallows_the_next_line() {
        let mut report = RedactionReport::default();
        let redacted = redact_text("Authorization: Bearer\nnextline stays\n", &mut report);
        assert!(redacted.contains("nextline stays"), "{redacted}");
    }

    #[test]
    fn already_redacted_values_are_not_counted_twice() {
        let mut report = RedactionReport::default();
        let redacted = redact_text(
            "OPENAI_API_KEY=sk-proj-AbCdEf0123456789_AbCdEf0123456789\n",
            &mut report,
        );
        assert_eq!(redacted, "OPENAI_API_KEY=<redacted>\n");
        assert_eq!(report.replacements, 1);
    }
}
