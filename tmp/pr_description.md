# Refactor Pydantic API Models

## Summary
Cleans up and restructures the Pydantic models for the training and model management APIs to reduce redundancy, improve naming clarity, and align with the frontend architecture.

---

## Training Models (`studio/backend/models/training.py`)

| Before | After | Change |
|--------|-------|--------|
| `TrainingStartResponse` | `TrainingJobResponse` | Rename - clearer that it represents a created job |
| `TrainingStatusResponse` | `TrainingStatus` | Rename + add `phase` field with explicit pipeline stages |
| `TrainingProgressResponse` | `TrainingProgress` | Rename + add `epoch`, `elapsed_seconds`, `eta_seconds` |
| `TrainingMetricsResponse` | — | **Removed** - frontend stores history in IndexedDB |

**Key improvements:**
- `TrainingStatus` unifies status polling and streaming with explicit `phase` literals: `idle`, `loading_model`, `loading_dataset`, `configuring`, `training`, `completed`, `error`, `stopped`
- Renamed `is_active` → `is_training_running` for clarity
- `TrainingProgress` now includes timing info (`elapsed_seconds`, `eta_seconds`) for better UX

---

## Model Management (`studio/backend/models/models.py`)

| Before | After | Change |
|--------|-------|--------|
| `ModelSearchRequest` | — | **Removed** - search handled via query params |
| `ModelSearchResponse` | — | **Removed** |
| `ModelListResponse` | — | **Removed** |
| `ModelInfo` | — | **Removed** - redundant with ModelDetails |
| `ModelConfigResponse` | `ModelDetails` | Rename |

**Remaining models:** `ModelDetails`, `LoRAInfo`, `LoRAScanResponse`

---

## Breaking Changes
- All renamed/removed models will require updates in any code referencing them
- Frontend TypeScript types should be regenerated
