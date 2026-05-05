use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine as _};
use hmac::{Hmac, Mac};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

type HmacSha256 = Hmac<Sha256>;

pub const LEASE_SECRET_ENV: &str = "UNSLOTH_STUDIO_NATIVE_PATH_LEASE_SECRET";
const LEASE_VERSION: u8 = 1;
const LEASE_TTL: Duration = Duration::from_secs(2 * 60);

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum NativePathOperation {
    ValidateModel,
    LoadModel,
    DatasetPreview,
    DatasetImport,
    Attach,
    Reveal,
    Open,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum NativePathKind {
    Model,
    Dataset,
    Attachment,
    Artifact,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum NativePathSourceKind {
    Dialog,
    Drop,
    DeepLink,
    FileAssociation,
    Artifact,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum NativePathType {
    File,
    Directory,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "snake_case")]
pub struct NativePathLeasePayload {
    pub version: u8,
    pub operation: NativePathOperation,
    pub canonical_path: String,
    pub path_kind: NativePathKind,
    pub path_type: NativePathType,
    pub source_kind: NativePathSourceKind,
    pub token_id_hash: String,
    pub issued_at_ms: u64,
    pub expires_at_ms: u64,
    pub nonce: String,
    pub display_label: String,
    pub size_bytes: Option<u64>,
    pub modified_ms: Option<u64>,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct NativePathLeaseResponse {
    pub native_path_lease: String,
    pub display_label: String,
    pub expires_at_ms: u64,
}

pub fn new_lease_secret() -> Vec<u8> {
    rand::random::<[u8; 32]>().to_vec()
}

pub fn encode_secret_env(secret: &[u8]) -> String {
    URL_SAFE_NO_PAD.encode(secret)
}

pub fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_else(|_| Duration::from_secs(0))
        .as_millis() as u64
}

pub fn token_hash(token: &str) -> String {
    hex_bytes(&Sha256::digest(token.as_bytes()))
}

pub fn random_token(prefix: &str) -> String {
    format!("{}{}", prefix, hex_bytes(&rand::random::<[u8; 24]>()))
}

pub fn random_nonce() -> String {
    hex_bytes(&rand::random::<[u8; 16]>())
}

pub fn sign_path_lease(
    secret: &[u8],
    operation: NativePathOperation,
    canonical_path: String,
    path_kind: NativePathKind,
    path_type: NativePathType,
    source_kind: NativePathSourceKind,
    token: &str,
    display_label: String,
    size_bytes: Option<u64>,
    modified_ms: Option<u64>,
) -> Result<NativePathLeaseResponse, String> {
    let issued_at_ms = now_ms();
    let expires_at_ms = issued_at_ms + LEASE_TTL.as_millis() as u64;
    let payload = NativePathLeasePayload {
        version: LEASE_VERSION,
        operation,
        canonical_path,
        path_kind,
        path_type,
        source_kind,
        token_id_hash: token_hash(token),
        issued_at_ms,
        expires_at_ms,
        nonce: random_nonce(),
        display_label: display_label.clone(),
        size_bytes,
        modified_ms,
    };
    sign_payload(secret, &payload).map(|native_path_lease| NativePathLeaseResponse {
        native_path_lease,
        display_label,
        expires_at_ms,
    })
}

fn sign_payload(secret: &[u8], payload: &NativePathLeasePayload) -> Result<String, String> {
    let payload_json = serde_json::to_vec(payload).map_err(|e| e.to_string())?;
    let payload_b64 = URL_SAFE_NO_PAD.encode(payload_json);
    let signature = sign_bytes(secret, payload_b64.as_bytes())?;
    Ok(format!(
        "{}.{}",
        payload_b64,
        URL_SAFE_NO_PAD.encode(signature)
    ))
}

fn sign_bytes(secret: &[u8], bytes: &[u8]) -> Result<Vec<u8>, String> {
    let mut mac = HmacSha256::new_from_slice(secret).map_err(|e| e.to_string())?;
    mac.update(bytes);
    Ok(mac.finalize().into_bytes().to_vec())
}

pub fn hex_bytes(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for &byte in bytes {
        out.push(HEX[(byte >> 4) as usize] as char);
        out.push(HEX[(byte & 0x0f) as usize] as char);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn token_hash_is_stable_hex_sha256() {
        assert_eq!(
            token_hash("native-token"),
            "d0c16f641bc0a0ee6b63ff88cec29756638d19590893c340a1ae36c9fae7b07f"
        );
    }

    #[test]
    fn signed_lease_has_two_base64url_parts() {
        let lease = sign_path_lease(
            b"01234567890123456789012345678901",
            NativePathOperation::ValidateModel,
            "/tmp/model.gguf".to_string(),
            NativePathKind::Model,
            NativePathType::File,
            NativePathSourceKind::Dialog,
            "token",
            "model.gguf".to_string(),
            Some(123),
            Some(456),
        )
        .unwrap();
        let parts: Vec<&str> = lease.native_path_lease.split('.').collect();
        assert_eq!(parts.len(), 2);
        assert!(!parts[0].contains('='));
        assert!(!parts[1].contains('='));
    }
}
