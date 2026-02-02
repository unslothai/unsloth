# Training Pydantic Models Recommendation

## Current State

| Model | Purpose |
|-------|---------|
| `TrainingStartRequest` | Request payload to start training |
| `TrainingStartResponse` | Immediate response after `/start` |
| `TrainingStatusResponse` | Response for `/status` polling |
| `TrainingMetricsResponse` | Historical metrics |
| `TrainingProgressResponse` | Per-step progress data |

---

## Proposed Models

### 1. `TrainingStartRequest` — Keep as-is
Well-structured, no changes needed.

---

### 2. `TrainingJobResponse` (rename from `TrainingStartResponse`)
Returned **once** when `/train/start` is called.

```python
class TrainingJobResponse(BaseModel):
    """Immediate response when training is initiated"""
    job_id: str
    status: Literal["queued", "error"]
    message: str
    error: Optional[str] = None
```

---

### 3. `TrainingStatus` (unifies `TrainingStatusResponse` + `TrainingPhaseUpdate`)
Single model for **both streaming and polling**.

```python
class TrainingStatus(BaseModel):
    """Current training job status - works for streaming or polling"""
    job_id: str
    phase: Literal[
        "idle",
        "loading_model",
        "loading_dataset",
        "configuring",
        "training",
        "completed",
        "error",
        "stopped"
    ]
    is_training_running: bool    # True if training loop is actively running
    message: str
    error: Optional[str] = None
    details: Optional[dict] = None  # Phase-specific info, e.g. {"model_size": "8B"}
```

**Usage:**
- **Streaming**: Push when phase changes
- **Polling**: Return from `GET /train/status/{job_id}`

---

### 4. `TrainingProgress` (rename from `TrainingProgressResponse`)
Per-step metrics during active training.

```python
class TrainingProgress(BaseModel):
    """Training progress metrics - for streaming or polling"""
    step: int
    total_steps: int
    loss: float
    learning_rate: float
    progress_percent: float
    epoch: Optional[int] = None
    elapsed_seconds: Optional[float] = None
    eta_seconds: Optional[float] = None
```

---

### 5. `TrainingMetricsResponse` — Remove
Frontend stores history in IndexedDB.

---

## Summary

| Current | Proposed | Action |
|---------|----------|--------|
| `TrainingStartRequest` | `TrainingStartRequest` | Keep |
| `TrainingStartResponse` | `TrainingJobResponse` | Rename |
| `TrainingStatusResponse` | `TrainingStatus` | Rename + enhance |
| `TrainingMetricsResponse` | — | **Remove** |
| `TrainingProgressResponse` | `TrainingProgress` | Rename + enhance |

---

## API Flow

```
POST /train/start
  └─► TrainingJobResponse { job_id, status: "queued" }

StreamingResponse / Polling:
  └─► TrainingStatus { phase: "loading_model", is_training_running: false }
  └─► TrainingStatus { phase: "loading_dataset", is_training_running: false }
  └─► TrainingStatus { phase: "training", is_training_running: true }
  └─► TrainingProgress { step: 1, loss: 2.5, ... }
  └─► TrainingProgress { step: 2, loss: 2.3, ... }
  └─► TrainingStatus { phase: "completed", is_training_running: false }
```
