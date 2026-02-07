# Apply Mac compatibility patches BEFORE importing unsloth
import platform
if platform.system() == "Darwin":
    from patcher import patch_for_mac
    patch_for_mac()

import torch
import unittest
from unittest.mock import MagicMock, patch
import os
import sys

# Mock DEVICE_TYPE for MPS
with patch("unsloth_zoo.device_type.DEVICE_TYPE", "mps"):
    from unsloth.device_utils import get_available_memory, get_total_memory, DEVICE_TYPE
    from unsloth.save import unsloth_save_model, _merge_lora

class TestMemorySafeSaving(unittest.TestCase):
    def test_mps_available_memory(self):
        """Test that get_available_memory on MPS respects usable limits."""
        with patch("unsloth.kernels.mps.get_apple_hardware_info") as mock_hw:
            mock_hw.return_value = {
                "total_memory_bytes": 16 * 1024**3,
                "total_memory_gb": 16.0,
                "usable_memory_gb": 11.2, # 70% of 16
                "chip_variant": "base"
            }
            with patch("psutil.virtual_memory") as mock_vm:
                mock_vm.return_value.available = 12 * 1024**3
                with patch("torch.mps.current_allocated_memory", return_value=8 * 1024**3):
                    # Usable total is 11.2GB. Allocated is 8GB. Remainder is 3.2GB.
                    # System available is 12GB. So it should return 3.2GB.
                    avail = get_available_memory()
                    self.assertLess(avail, 4 * 1024**3)
                    self.assertGreater(avail, 3 * 1024**3)

    @patch("unsloth.save._merge_lora")
    @patch("torch.save")
    @patch("os.makedirs")
    @patch("os.path.exists", return_value=True)
    def test_unsloth_save_model_disk_fallback(self, mock_exists, mock_makedirs, mock_save, mock_merge):
        """Test that unsloth_save_model falls back to disk when VRAM (Unified) is tight."""
        mock_model = MagicMock()
        mock_model.model.layers = [MagicMock()]
        mock_proj = MagicMock()
        mock_model.model.layers[0].self_attn.q_proj = mock_proj
        
        # Mock memory to be very low
        with patch("unsloth.device_utils.get_total_memory", return_value=16 * 1024**3):
            with patch("unsloth.device_utils.get_current_memory_usage", return_value=15 * 1024**3):
                # max_vram will be total * 0.9 = 14.4GB. 
                # Already used 15GB. Next weight will definitely trigger disk path.
                
                mock_weight = torch.randn(10, 10)
                mock_merge.return_value = (mock_weight, None)
                
                # Mock tqdm
                with patch("tqdm.tqdm", lambda x: x):
                    # Should call torch.save for disk fallback
                    unsloth_save_model(mock_model, None, "save_dir", save_method="merged_16bit")
                    
                    # Verify that Disk fallback was triggered (logger.warning_once called with "save to Disk")
                    # Or check that torch.save was called with a .pt filename in the temporary location
                    disk_save_called = any(".pt" in str(args[1]) for args, kwargs in mock_save.call_args_list)
                    self.assertTrue(disk_save_called, "Disk fallback should have been triggered")

if __name__ == "__main__":
    unittest.main()
