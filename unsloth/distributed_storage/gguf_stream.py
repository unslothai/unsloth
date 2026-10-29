import struct
import io
import time
from typing import Dict, Any, Optional
import numpy as np
from huggingface_hub import HfApi

# GGUF v3 Binary Constants
GGUF_MAGIC = 0x46554747 # "GGUF" in LE
GGUF_VERSION = 3

GGUF_TYPE_UINT8   = 0
GGUF_TYPE_INT8    = 1
GGUF_TYPE_UINT16  = 2
GGUF_TYPE_INT16   = 3
GGUF_TYPE_UINT32  = 4
GGUF_TYPE_INT32   = 5
GGUF_TYPE_FLOAT32 = 6
GGUF_TYPE_BOOL    = 7
GGUF_TYPE_STRING  = 8
GGUF_TYPE_ARRAY   = 9
GGUF_TYPE_UINT64  = 10
GGUF_TYPE_INT64   = 11
GGUF_TYPE_FLOAT64 = 12

GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1

def pack_str(s: str) -> bytes:
    encoded = s.encode('utf-8')
    return struct.pack('<Q', len(encoded)) + encoded

def build_in_memory_gguf(model_tensors: Dict[str, np.ndarray], metadata: Dict[str, Any]) -> bytes:
    """
    Constructs a valid binary GGUF v3 buffer directly in RAM with 32-byte alignment.
    Strictly 0 bytes written to disk.
    """
    buf = io.BytesIO()

    tensor_count = len(model_tensors)
    kv_count = len(metadata)
    buf.write(struct.pack('<IIQQ', GGUF_MAGIC, GGUF_VERSION, tensor_count, kv_count))

    for k, v in metadata.items():
        buf.write(pack_str(k))
        if isinstance(v, str):
            buf.write(struct.pack('<I', GGUF_TYPE_STRING))
            buf.write(pack_str(v))
        elif isinstance(v, int):
            buf.write(struct.pack('<I', GGUF_TYPE_UINT32))
            buf.write(struct.pack('<I', v))
        elif isinstance(v, float):
            buf.write(struct.pack('<I', GGUF_TYPE_FLOAT32))
            buf.write(struct.pack('<f', v))

    current_offset = 0
    alignment = 32
    tensor_binary_data = []

    tensor_infos = io.BytesIO()
    for name, arr in model_tensors.items():
        arr_f32 = arr.astype(np.float32)
        tensor_infos.write(pack_str(name))
        shape = list(arr_f32.shape)[::-1]
        tensor_infos.write(struct.pack('<I', len(shape)))
        for dim in shape:
            tensor_infos.write(struct.pack('<Q', dim))
        tensor_infos.write(struct.pack('<I', GGML_TYPE_F32))
        tensor_infos.write(struct.pack('<Q', current_offset))

        raw_bytes = arr_f32.tobytes()
        tensor_binary_data.append(raw_bytes)
        pad = (alignment - (len(raw_bytes) % alignment)) % alignment
        tensor_binary_data.append(b'\x00' * pad)
        current_offset += len(raw_bytes) + pad

    buf.write(tensor_infos.getvalue())

    pos = buf.tell()
    pad = (alignment - (pos % alignment)) % alignment
    buf.write(b'\x00' * pad)

    for data in tensor_binary_data:
        buf.write(data)

    return buf.getvalue()

def stream_gguf_to_hub(
    api: HfApi,
    repo_id: str,
    model_tensors: Dict[str, np.ndarray],
    metadata: Dict[str, Any],
    filename: str = "model-latest.gguf",
    commit_message: Optional[str] = None
):
    """
    Directly streams an in-memory GGUF v3 model to Hugging Face Hub.
    Eliminates the intermediate local disk export requirement.
    """
    gguf_bytes = build_in_memory_gguf(model_tensors, metadata)
    api.upload_file(
        path_or_fileobj=io.BytesIO(gguf_bytes),
        path_in_repo=filename,
        repo_id=repo_id,
        repo_type="model",
        commit_message=commit_message or "Upload GGUF model via Unsloth Zero-Disk Syncer"
    )
