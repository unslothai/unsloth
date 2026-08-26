use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine as _};
use hmac::{Hmac, KeyInit, Mac};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

type HmacSha256 = Hmac<Sha256>;

pub const LEASE_SECRET_ENV: &str = "UNSLOTH_STUDIO_NATIVE_PATH_LEASE_SECRET";
const LEASE_VERSION: u8 = 2;
const LEASE_TTL: Duration = Duration::from_secs(2 * 60);
const LEASE_NONCE_BYTES: usize = 16;
const LEASE_ENCRYPTION_DOMAIN: &[u8] = b"unsloth-native-path-lease-v2-encryption\0";
const LEASE_AUTH_DOMAIN: &[u8] = b"unsloth-native-path-lease-v2-auth\0";

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum NativePathOperation {
    ValidateModel,
    LoadModel,
    DatasetPreview,
    DatasetImport,
    Attach,
    LinkDocuments,
    OpenProject,
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

#[derive(Clone, Debug, Deserialize, Serialize)]
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
    pub device_id: Option<String>,
    pub file_id: Option<String>,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct NativePathLeaseResponse {
    pub native_path_lease: String,
    pub display_label: String,
    pub expires_at_ms: u64,
}

/// Lockstep with `_MIN_LEASE_SECRET_BYTES` in
/// `studio/backend/utils/native_path_leases.py`: a shorter secret is refused
/// there, so a shorter one here would advertise leases the backend rejects.
pub const MIN_LEASE_SECRET_BYTES: usize = 32;

pub fn new_lease_secret() -> Vec<u8> {
    rand::random::<[u8; MIN_LEASE_SECRET_BYTES]>().to_vec()
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
    pub device_id: Option<String>,
    pub file_id: Option<String>,
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
        device_id: request.device_id,
        file_id: request.file_id,
    };
    sign_payload(secret, &payload).map(|native_path_lease| NativePathLeaseResponse {
        native_path_lease,
        display_label: request.display_label,
        expires_at_ms,
    })
}

fn sign_payload(secret: &[u8], payload: &NativePathLeasePayload) -> Result<String, String> {
    let nonce = rand::random::<[u8; LEASE_NONCE_BYTES]>();
    sign_payload_with_nonce(secret, payload, &nonce)
}

fn sign_payload_with_nonce(
    secret: &[u8],
    payload: &NativePathLeasePayload,
    nonce: &[u8],
) -> Result<String, String> {
    if nonce.len() != LEASE_NONCE_BYTES {
        return Err("native path lease envelope nonce has the wrong length".to_string());
    }
    let payload_json = serde_json::to_vec(payload).map_err(|e| e.to_string())?;
    let ciphertext = xor_lease_stream(secret, nonce, &payload_json)?;
    let mut envelope = Vec::with_capacity(LEASE_NONCE_BYTES + ciphertext.len());
    envelope.extend_from_slice(nonce);
    envelope.extend_from_slice(&ciphertext);
    let signature = sign_lease_envelope(secret, &envelope)?;
    Ok(format!(
        "{}.{}.{}",
        LEASE_VERSION,
        URL_SAFE_NO_PAD.encode(envelope),
        URL_SAFE_NO_PAD.encode(signature)
    ))
}

fn xor_lease_stream(secret: &[u8], nonce: &[u8], input: &[u8]) -> Result<Vec<u8>, String> {
    let mut output = Vec::with_capacity(input.len());
    for (block_index, chunk) in input.chunks(32).enumerate() {
        let counter = u64::try_from(block_index).map_err(|e| e.to_string())?;
        let mut seed = Vec::with_capacity(
            LEASE_ENCRYPTION_DOMAIN.len() + nonce.len() + std::mem::size_of::<u64>(),
        );
        seed.extend_from_slice(LEASE_ENCRYPTION_DOMAIN);
        seed.extend_from_slice(nonce);
        seed.extend_from_slice(&counter.to_be_bytes());
        let stream = sign_bytes(secret, &seed)?;
        output.extend(
            chunk
                .iter()
                .zip(stream.iter())
                .map(|(byte, mask)| byte ^ mask),
        );
    }
    Ok(output)
}

fn sign_lease_envelope(secret: &[u8], envelope: &[u8]) -> Result<Vec<u8>, String> {
    let mut authenticated = Vec::with_capacity(LEASE_AUTH_DOMAIN.len() + envelope.len());
    authenticated.extend_from_slice(LEASE_AUTH_DOMAIN);
    authenticated.extend_from_slice(envelope);
    sign_bytes(secret, &authenticated)
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

    #[derive(serde::Deserialize)]
    struct NativeLeaseKnownAnswer {
        secret_base64url: String,
        envelope_nonce_base64url: String,
        payload: NativePathLeasePayload,
        lease: String,
    }

    fn decode_payload(lease: &str) -> serde_json::Value {
        let parts: Vec<&str> = lease.split('.').collect();
        assert_eq!(parts[0], LEASE_VERSION.to_string());
        let envelope = URL_SAFE_NO_PAD.decode(parts[1]).unwrap();
        let (nonce, ciphertext) = envelope.split_at(LEASE_NONCE_BYTES);
        let payload =
            xor_lease_stream(b"01234567890123456789012345678901", nonce, ciphertext).unwrap();
        serde_json::from_slice(&payload).unwrap()
    }

    #[test]
    fn token_hash_is_stable_hex_sha256() {
        assert_eq!(
            token_hash("native-token"),
            "d0c16f641bc0a0ee6b63ff88cec29756638d19590893c340a1ae36c9fae7b07f"
        );
    }

    #[test]
    fn v2_known_answer_is_stable_for_python_verification() {
        let vector: NativeLeaseKnownAnswer = serde_json::from_str(include_str!(
            "../../backend/tests/fixtures/native_path_lease_v2_rust.json"
        ))
        .unwrap();
        let secret = URL_SAFE_NO_PAD
            .decode(vector.secret_base64url.as_bytes())
            .unwrap();
        let envelope_nonce = URL_SAFE_NO_PAD
            .decode(vector.envelope_nonce_base64url.as_bytes())
            .unwrap();

        let lease = sign_payload_with_nonce(&secret, &vector.payload, &envelope_nonce).unwrap();

        assert_eq!(lease, vector.lease);
    }

    #[test]
    fn signed_lease_is_an_opaque_authenticated_envelope() {
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
                device_id: Some("7".to_string()),
                file_id: Some("8".to_string()),
            },
        )
        .unwrap();
        let parts: Vec<&str> = lease.native_path_lease.split('.').collect();
        assert_eq!(parts.len(), 3);
        assert_eq!(parts[0], "2");
        assert!(!parts[1].contains('='));
        assert!(!parts[2].contains('='));
        let envelope = URL_SAFE_NO_PAD.decode(parts[1]).unwrap();
        assert!(!String::from_utf8_lossy(&envelope).contains("/tmp/model.gguf"));
        assert!(!lease.native_path_lease.contains("model.gguf"));
    }

    #[test]
    fn document_folder_lease_payload_has_backend_contract_values() {
        let lease = sign_path_lease(
            b"01234567890123456789012345678901",
            NativePathLeaseRequest {
                operation: NativePathOperation::LinkDocuments,
                canonical_path: "/tmp/knowledge".to_string(),
                path_kind: NativePathKind::DocumentFolder,
                path_type: NativePathType::Directory,
                source_kind: NativePathSourceKind::Dialog,
                token: "path_token".to_string(),
                display_label: "knowledge".to_string(),
                size_bytes: None,
                modified_ms: None,
                device_id: Some("7".to_string()),
                file_id: Some("8".to_string()),
            },
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
        assert!(payload["modified_ms"].is_null());
        assert_eq!(payload["device_id"], "7");
        assert_eq!(payload["file_id"], "8");
    }
}
