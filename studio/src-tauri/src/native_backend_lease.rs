use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine as _};
use hmac::{Hmac, KeyInit, Mac};
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
    LinkDocuments,
    Reveal,
    Open,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum NativePathKind {
    Model,
    Dataset,
    Attachment,
    DocumentFolder,
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

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
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

pub struct NativePathLeaseRequest {
    pub operation: NativePathOperation,
    pub canonical_path: String,
    pub path_kind: NativePathKind,
    pub path_type: NativePathType,
    pub source_kind: NativePathSourceKind,
    pub token: String,
    pub display_label: String,
    pub size_bytes: Option<u64>,
    pub modified_ms: Option<u64>,
}

pub fn sign_path_lease(
    secret: &[u8],
    request: NativePathLeaseRequest,
) -> Result<NativePathLeaseResponse, String> {
    let issued_at_ms = now_ms();
    let expires_at_ms = issued_at_ms + LEASE_TTL.as_millis() as u64;
    let payload = NativePathLeasePayload {
        version: LEASE_VERSION,
        operation: request.operation,
        canonical_path: request.canonical_path,
        path_kind: request.path_kind,
        path_type: request.path_type,
        source_kind: request.source_kind,
        token_id_hash: token_hash(&request.token),
        issued_at_ms,
        expires_at_ms,
        nonce: random_nonce(),
        display_label: request.display_label.clone(),
        size_bytes: request.size_bytes,
        modified_ms: request.modified_ms,
    };
    sign_payload(secret, &payload).map(|native_path_lease| NativePathLeaseResponse {
        native_path_lease,
        display_label: request.display_label,
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

    fn decode_payload(lease: &str) -> serde_json::Value {
        let payload = lease.split('.').next().unwrap();
        serde_json::from_slice(&URL_SAFE_NO_PAD.decode(payload).unwrap()).unwrap()
    }

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
            NativePathLeaseRequest {
                operation: NativePathOperation::ValidateModel,
                canonical_path: "/tmp/model.gguf".to_string(),
                path_kind: NativePathKind::Model,
                path_type: NativePathType::File,
                source_kind: NativePathSourceKind::Dialog,
                token: "token".to_string(),
                display_label: "model.gguf".to_string(),
                size_bytes: Some(123),
                modified_ms: Some(456),
            },
        )
        .unwrap();
        let parts: Vec<&str> = lease.native_path_lease.split('.').collect();
        assert_eq!(parts.len(), 2);
        assert!(!parts[0].contains('='));
        assert!(!parts[1].contains('='));
    }

    #[test]
    fn document_folder_lease_payload_has_backend_contract_values() {
        let lease = sign_path_lease(
            b"01234567890123456789012345678901",
            NativePathOperation::LinkDocuments,
            "/tmp/knowledge".to_string(),
            NativePathKind::DocumentFolder,
            NativePathType::Directory,
            NativePathSourceKind::Dialog,
            "path_token",
            "knowledge".to_string(),
            None,
            Some(456),
        )
        .unwrap();
        let payload = decode_payload(&lease.native_path_lease);

        assert_eq!(payload["operation"], "link-documents");
        assert_eq!(payload["path_kind"], "document-folder");
        assert_eq!(payload["path_type"], "directory");
        assert_eq!(payload["source_kind"], "dialog");
        assert_eq!(payload["token_id_hash"], token_hash("path_token"));
        let issued_at_ms = payload["issued_at_ms"].as_u64().unwrap();
        let expires_at_ms = payload["expires_at_ms"].as_u64().unwrap();
        assert!(expires_at_ms > issued_at_ms);
        assert_eq!(expires_at_ms - issued_at_ms, 120_000);
    }
}
