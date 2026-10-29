import unittest
import sys
import types
import os
import numpy as np

# Directly test distributed storage engine
sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "unsloth", "distributed_storage")
    ),
)

# Mock huggingface_hub if absent
try:
    import huggingface_hub
except ImportError:
    m = types.ModuleType("huggingface_hub")
    m.HfApi = object
    sys.modules["huggingface_hub"] = m

from streamer import IPFSBlockStreamer
from gguf_stream import build_in_memory_gguf


class TestUnslothDistributedStorage(unittest.TestCase):
    def test_ipfs_streaming_zero_disk(self):
        streamer = IPFSBlockStreamer(batch_size = 4, seq_len = 32)
        x, y = streamer.get_batch()
        self.assertEqual(x.shape, (4, 32))
        self.assertEqual(y.shape, (4, 32))
        self.assertGreater(streamer.total_tokens, 0)
        self.assertTrue(streamer.current_cid.startswith("bafybeic"))

    def test_in_memory_gguf_v3_format(self):
        tensors = {
            "token_embd.weight": np.random.randn(32, 64).astype(np.float32),
            "output.weight": np.random.randn(64, 32).astype(np.float32),
        }
        meta = {"general.architecture": "unsloth-test", "general.name": "unsloth-gguf-v3-proof"}
        gguf_bytes = build_in_memory_gguf(tensors, meta)
        # Check GGUF magic header "GGUF" in Little Endian: 0x46554747 -> b'GGUF'
        self.assertEqual(gguf_bytes[:4], b"GGUF")
        self.assertGreater(len(gguf_bytes), 1024)


if __name__ == "__main__":
    unittest.main()
