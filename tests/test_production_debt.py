import importlib.util
import os
import sys
import unittest

# Load module directly
file_path = os.path.join(
    os.path.dirname(__file__),
    "../unsloth/models/production_debt.py",
)
spec = importlib.util.spec_from_file_location("unsloth_production_debt", file_path)
production_debt_mod = importlib.util.module_from_spec(spec)
sys.modules["unsloth_production_debt"] = production_debt_mod
spec.loader.exec_module(production_debt_mod)

ProductionDebtFineTuningGate = production_debt_mod.ProductionDebtFineTuningGate
TechnicalDueDiligenceLedger = production_debt_mod.TechnicalDueDiligenceLedger
GENESIS_HASH = production_debt_mod.GENESIS_HASH


class TestProductionDebtFineTuningGate(unittest.TestCase):
    def setUp(self) -> None:
        self.gate = ProductionDebtFineTuningGate(
            never_equate_intent_to_approval = True,
            max_acceptable_ftdi = 12.0,
        )

    def test_clean_training_run_passes_readiness(self) -> None:
        report = self.gate.evaluate_training_run(
            model_name = "llama-3.3-70b-qlora-enterprise",
            lora_rank = 16,
            allocated_vram_gb = 14.2,
            peak_vram_gb = 15.1,
            step_latency_seconds = 0.85,
            checkpoint_corruption_count = 0,
            un_gated_mutations = 0,
        )
        self.assertTrue(report.is_production_ready)
        self.assertLessEqual(report.ftdi_score, 12.0)
        self.assertEqual(len(report.critical_smells), 0)
        self.assertTrue(bool(report.receipt_hash))

    def test_degraded_training_run_fails_debt(self) -> None:
        report = self.gate.evaluate_training_run(
            model_name = "uncalibrated_full_rank_run",
            lora_rank = 128,  # LoRA rank sprawl
            allocated_vram_gb = 14.0,
            peak_vram_gb = 28.5,  # High VRAM allocation sprawl (2.0x)
            step_latency_seconds = 6.5,  # High step latency
            checkpoint_corruption_count = 1,  # Checkpoint anomaly
            un_gated_mutations = 2,  # 2 un-gated mutations
        )
        self.assertFalse(report.is_production_ready)
        self.assertGreater(report.ftdi_score, 50.0)
        self.assertIn("HIGH_VRAM_ALLOCATION_SPRAWL_2.04X", report.critical_smells)
        self.assertIn("LORA_RANK_SPRAWL_R128", report.critical_smells)
        self.assertIn("HIGH_STEP_LATENCY_6.50S", report.critical_smells)
        self.assertIn("DETECTED_1_CHECKPOINT_ANOMALIES", report.critical_smells)
        self.assertIn("DETECTED_2_UNGATED_MODEL_MUTATIONS", report.critical_smells)

    def test_cryptographic_ledger_integrity(self) -> None:
        self.gate.evaluate_training_run("model-1")
        self.gate.evaluate_training_run("model-2")
        self.gate.evaluate_training_run("model-3")

        entries = self.gate.ledger.get_ledger_entries()
        self.assertEqual(len(entries), 3)
        self.assertEqual(entries[0]["prev_hash"], GENESIS_HASH)
        self.assertEqual(entries[1]["prev_hash"], entries[0]["curr_hash"])
        self.assertEqual(entries[2]["prev_hash"], entries[1]["curr_hash"])
        self.assertTrue(self.gate.ledger.verify_ledger_integrity())


if __name__ == "__main__":
    unittest.main()
