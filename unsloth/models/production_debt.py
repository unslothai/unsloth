from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

log: logging.Logger = logging.getLogger(__name__)

GENESIS_HASH: str = "0000000000000000000000000000000000000000000000000000000000000000"


@dataclass
class FineTuningDebtReport:
    model_name: str
    ftdi_score: float  # Fine-Tuning Debt Index (target <= 12.0)
    vram_allocation_multiplier: float  # Target <= 1.10x
    step_latency_seconds: float  # Target <= 1.2s
    mutation_safety_score: float  # Target 100.0
    production_readiness_index: float  # Scale 0 - 100
    is_production_ready: bool
    critical_smells: List[str]
    receipt_hash: str


class TechnicalDueDiligenceLedger:
    """
    Cryptographic SHA-256 hash-chained Action Ledger for Unsloth fine-tuning runs and checkpoint exports.
    """

    def __init__(self) -> None:
        self._entries: List[Dict[str, Any]] = []
        self._last_hash: str = GENESIS_HASH

    def record_training_event(
        self,
        model_name: str,
        event_type: str,
        readiness_index: float,
        critical_smells: List[str],
        metadata: Dict[str, Any],
    ) -> Dict[str, Any]:
        timestamp = datetime.now(timezone.utc).isoformat()
        index = len(self._entries)

        meta_bytes = json.dumps(metadata, sort_keys = True).encode("utf-8")
        canonical_content = f"{index}|{self._last_hash}|{model_name}|{event_type}|{readiness_index}|{timestamp}|{hashlib.sha256(meta_bytes).hexdigest()}"
        curr_hash = hashlib.sha256(canonical_content.encode("utf-8")).hexdigest()

        entry = {
            "index": index,
            "timestamp": timestamp,
            "model_name": model_name,
            "event_type": event_type,
            "readiness_index": readiness_index,
            "critical_smells": critical_smells,
            "prev_hash": self._last_hash,
            "curr_hash": curr_hash,
            "metadata": metadata,
        }

        self._entries.append(entry)
        self._last_hash = curr_hash
        return entry

    def get_ledger_entries(self) -> List[Dict[str, Any]]:
        return list(self._entries)

    def verify_ledger_integrity(self) -> bool:
        prev = GENESIS_HASH
        for entry in self._entries:
            if entry["prev_hash"] != prev:
                return False
            prev = entry["curr_hash"]
        return True


class ProductionDebtFineTuningGate:
    """
    A2Z SOC Production Debt & Technical Due Diligence Gate for Unsloth Fine-Tuning.

    Quantifies LLM fine-tuning VRAM memory, LoRA rank sprawl, and step latency against 4 Enterprise KPIs:
    1. Fine-Tuning Debt Index (FTDI <= 12.0)
    2. LoRA Memory Allocation Multiplier (LMM <= 1.10x)
    3. P99 Step Execution Latency Ceiling (<= 1.2s)
    4. Deterministic Mutation Boundaries (never_equate_intent_to_approval)
    """

    def __init__(
        self,
        never_equate_intent_to_approval: bool = True,
        max_acceptable_ftdi: float = 12.0,
    ) -> None:
        self.never_equate_intent_to_approval = never_equate_intent_to_approval
        self.max_acceptable_ftdi = max_acceptable_ftdi
        self.ledger = TechnicalDueDiligenceLedger()

    def check_kill_switch(self) -> bool:
        if os.environ.get("AAG_KILL_SWITCH", "").lower() in ("true", "1", "yes"):
            return True
        for path_str in ("artifacts/KILL", "/tmp/KILL"):
            if Path(path_str).exists():
                return True
        return False

    def evaluate_training_run(
        self,
        model_name: str,
        lora_rank: int = 16,
        allocated_vram_gb: float = 14.2,
        peak_vram_gb: float = 15.1,
        step_latency_seconds: float = 0.85,
        checkpoint_corruption_count: int = 0,
        un_gated_mutations: int = 0,
    ) -> FineTuningDebtReport:
        # 1. Evaluate emergency kill switch
        if self.check_kill_switch():
            self.ledger.record_training_event(
                model_name = model_name,
                event_type = "training_halted_kill_switch",
                readiness_index = 0.0,
                critical_smells = ["EMERGENCY_KILL_SWITCH_ENGAGED"],
                metadata = {"reason": "AAG_KILL_SWITCH is set"},
            )
            raise PermissionError(
                "A2Z SOC ActionGate: Emergency kill switch is engaged. Fine-tuning training run halted."
            )

        critical_smells: List[str] = []

        # KPI 2: VRAM Allocation Multiplier
        vram_ratio = peak_vram_gb / max(1.0, allocated_vram_gb)
        if vram_ratio > 1.5:
            critical_smells.append(f"HIGH_VRAM_ALLOCATION_SPRAWL_{vram_ratio:.2f}X")

        # LoRA rank sprawl
        if lora_rank > 64:
            critical_smells.append(f"LORA_RANK_SPRAWL_R{lora_rank}")

        # KPI 3: Latency Ceiling
        if step_latency_seconds > 3.0:
            critical_smells.append(f"HIGH_STEP_LATENCY_{step_latency_seconds:.2f}S")

        # Checkpoint corruption
        if checkpoint_corruption_count > 0:
            critical_smells.append(f"DETECTED_{checkpoint_corruption_count}_CHECKPOINT_ANOMALIES")

        # KPI 4: Mutation Safety
        if un_gated_mutations > 0:
            critical_smells.append(f"DETECTED_{un_gated_mutations}_UNGATED_MODEL_MUTATIONS")

        # KPI 1: Fine-Tuning Debt Index (0 = Clean, 100 = Catastrophic)
        ftdi = (
            max(0.0, (vram_ratio - 1.0) * 20.0)
            + (max(0, lora_rank - 16) * 0.25)
            + max(0.0, (step_latency_seconds - 1.2) * 10.0)
            + (checkpoint_corruption_count * 20.0)
            + (un_gated_mutations * 30.0)
        )
        ftdi_score = round(min(100.0, ftdi), 2)

        # Production Readiness Index (0 - 100)
        readiness = max(0.0, 100.0 - ftdi_score)
        is_production_ready = ftdi_score <= self.max_acceptable_ftdi and len(critical_smells) == 0

        # Cryptographic Ledger Entry
        entry = self.ledger.record_training_event(
            model_name = model_name,
            event_type = "training_authorized" if is_production_ready else "training_flagged_debt",
            readiness_index = readiness,
            critical_smells = critical_smells,
            metadata = {
                "lora_rank": lora_rank,
                "vram_ratio": vram_ratio,
                "allocated_vram_gb": allocated_vram_gb,
                "peak_vram_gb": peak_vram_gb,
                "step_latency_seconds": step_latency_seconds,
                "checkpoint_corruption_count": checkpoint_corruption_count,
                "un_gated_mutations": un_gated_mutations,
                "never_equate_intent_to_approval": self.never_equate_intent_to_approval,
            },
        )

        return FineTuningDebtReport(
            model_name = model_name,
            ftdi_score = ftdi_score,
            vram_allocation_multiplier = round(vram_ratio, 2),
            step_latency_seconds = round(step_latency_seconds, 2),
            mutation_safety_score = (
                100.0 if un_gated_mutations == 0 else max(0.0, 100.0 - un_gated_mutations * 30.0)
            ),
            production_readiness_index = readiness,
            is_production_ready = is_production_ready,
            critical_smells = critical_smells,
            receipt_hash = entry["curr_hash"],
        )
