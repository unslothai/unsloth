# Plan PR : Intégration de Muon dans Unsloth (FFT)

> Agent cible : opencode + DeepSeek V4 Pro  
> GPU requis : NON (sauf étape 6 — test de validation, ponctuel)  
> Durée estimée sans GPU : 1–2 weekends  
> Base de travail : `unsloth/unsloth` @ main (cloner en local)

---

## Contexte technique (à lire avant de commencer)

### Où sont les optimizers dans Unsloth

```
unsloth/
  optimizers/
    __init__.py          ← exporte GaLoreProjector + QGaLoreAdamW8bit
    q_galore_adamw.py    ← implémentation Q-GaLore
    q_galore_projector.py
  trainer.py             ← UnslothTrainer.create_optimizer() ← POINT D'ENTRÉE PRINCIPAL
  models/
    loader.py            ← full_finetuning=True branching
    loader_utils.py      ← validation des flags incompatibles
    _utils.py            ← prepare_model_for_training()
```

### Comment Q-GaLore est intégré (pattern à reproduire exactement)

Dans `trainer.py`, `UnslothTrainer.create_optimizer()` suit cette logique :
```
1. Si q_galore_config présent → _create_q_galore_optimizer()
2. Elif embedding_learning_rate présent → _create_unsloth_optimizer()
3. Else → super().create_optimizer()
```

Le pattern est : **un dataclass Config + une méthode `_create_X_optimizer()` + un branchement dans `create_optimizer()`**.  
C'est exactement ce qu'on va faire pour Muon.

### État de Muon dans PyTorch

`torch.optim.Muon` existe depuis **PyTorch 2.9.0** (confirmé dans le source).  
Paramètres clés :
```python
torch.optim.Muon(
    params,              # UNIQUEMENT les params 2D (matrices)
    lr=1e-3,
    momentum=0.95,
    nesterov=True,
    ns_steps=5,          # itérations Newton-Schulz
    weight_decay=0.1,
)
```

**Contrainte fondamentale** : Muon ne s'applique qu'aux tenseurs `ndim == 2`.  
Tout le reste (embeddings, layernorms, biases, 1D params) → AdamW fallback.

### Ce qui n'existe pas encore dans Unsloth

`grep -rn "muon\|Muon" unsloth/` → **zéro résultat**. Terrain complètement vierge.

---

## Phase 0 — Setup & Audit (sans GPU)

### 0.1 Fork et clone

```bash
# Forker unslothai/unsloth sur GitHub
git clone https://github.com/<ton-fork>/unsloth.git
cd unsloth
git checkout -b feat/muon-optimizer
```

### 0.2 Installer les dépendances de dev (sans CUDA)

```bash
pip install -e ".[dev]" --no-deps
pip install pytest torch  # torch CPU suffit pour les tests statiques
```

### 0.3 Audit du codebase — questions à vérifier AVANT d'écrire une ligne

L'agent doit répondre à ces questions en lisant le code :

**Q1** : Dans `loader.py`, quand `full_finetuning=True`, quels params ont `requires_grad=True` ?  
→ Chercher dans `loader_utils.py` la fonction qui set les requires_grad (autour de la ligne 379).

**Q2** : Dans `_create_unsloth_optimizer()` (trainer.py L173), comment les param groups sont construits ?  
→ Comprendre le split embeddings / non-embeddings — Muon devra faire un split 2D / non-2D similaire.

**Q3** : Est-ce que `UnslothTrainingArguments` passe `**kwargs` à `SFTConfig` ?  
→ Si oui, on peut ajouter `muon_config` sans casser la signature parente.

**Q4** : `torch.optim.Muon` est-il disponible en import sans CUDA ?  
```python
import torch
print(hasattr(torch.optim, 'Muon'))  # doit être True sur torch >= 2.9.0
```

**Q5** : Dans les tests existants (`tests/python/`), quel est le pattern de mock pour simuler un modèle sans GPU ?  
→ Lire `test_patch_trl_rl_trainers_defensive.py` — utilise `monkeypatch` de pytest, pas de GPU.

**Q6** : Est-ce que `QGaloreConfig` est exporté dans `unsloth/__init__.py` ?  
→ Si oui, `MuonConfig` doit l'être aussi. Vérifier le fichier.

---

## Phase 1 — Implémentation Core (sans GPU)

### 1.1 Créer `unsloth/optimizers/muon.py`

Ce fichier contient la logique de construction des param groups.  
**NE PAS réimplémenter Muon** — utiliser `torch.optim.Muon` directement.

```python
# unsloth/optimizers/muon.py

from __future__ import annotations
import torch
from typing import Optional, List

__all__ = ["make_muon_param_groups"]


def _is_muon_eligible(param: torch.Tensor) -> bool:
    """
    Muon (Newton-Schulz) ne s'applique qu'aux matrices 2D.
    Embeddings, layernorms, biases → AdamW fallback.
    """
    return param.ndim == 2 and param.requires_grad


def make_muon_param_groups(
    model: torch.nn.Module,
    lr: float,
    weight_decay: float,
    muon_lr_scale: float = 1.0,
    adamw_lr: Optional[float] = None,
    adamw_betas: tuple = (0.9, 0.999),
    adamw_eps: float = 1e-8,
    adamw_weight_decay: Optional[float] = None,
    target_modules: Optional[List[str]] = None,
) -> tuple[list, list]:
    """
    Sépare les paramètres en deux groupes :
    - muon_params  : tenseurs 2D → torch.optim.Muon
    - adamw_params : tout le reste → AdamW

    Retourne (muon_param_groups, adamw_param_groups).
    """
    adamw_lr = adamw_lr or lr
    adamw_weight_decay = adamw_weight_decay if adamw_weight_decay is not None else weight_decay

    muon_params = []
    adamw_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # Si target_modules est spécifié, filtrer par nom
        if target_modules is not None:
            if not any(mod in name for mod in target_modules):
                adamw_params.append(param)
                continue
        if _is_muon_eligible(param):
            muon_params.append(param)
        else:
            adamw_params.append(param)

    muon_groups = [{"params": muon_params, "lr": lr * muon_lr_scale, "weight_decay": weight_decay}]
    adamw_groups = [{"params": adamw_params, "lr": adamw_lr, "weight_decay": adamw_weight_decay}]

    return muon_groups, adamw_groups
```

> ⚠️ **Piège attendu** : en LoRA, les params trainables sont les adaptateurs A/B — ils sont 2D.  
> Muon sera donc appliqué aux LoRA layers aussi si on ne filtre pas. Décider en phase 1 si on veut :  
> - (a) Muon FFT seulement → ajouter une guard `if not hasattr(param, 'lora_...')`  
> - (b) Muon + LoRA → laisser tel quel et documenter le comportement

### 1.2 Mettre à jour `unsloth/optimizers/__init__.py`

```python
from .q_galore_projector import GaLoreProjector
from .q_galore_adamw import QGaLoreAdamW8bit
from .muon import make_muon_param_groups  # ← ajouter

__all__ = [
    "GaLoreProjector",
    "QGaLoreAdamW8bit",
    "make_muon_param_groups",  # ← ajouter
]
```

### 1.3 Ajouter `MuonConfig` dans `trainer.py`

Même pattern que `QGaloreConfig` (L137-156 dans trainer.py).

```python
@dataclass
class MuonConfig:
    """Configuration pour l'optimizer Muon.
    
    Muon utilise Newton-Schulz orthogonalization pour les params 2D.
    Les params 1D (embeddings, layernorms, biases) tombent sur AdamW.
    
    Requiert PyTorch >= 2.9.0.
    """
    momentum: float = 0.95
    nesterov: bool = True
    ns_steps: int = 5
    muon_lr_scale: float = 1.0          # multiplicateur lr pour Muon vs AdamW
    adamw_lr: Optional[float] = None    # lr spécifique pour le fallback AdamW
    adamw_betas: tuple = (0.9, 0.999)
    adamw_eps: float = 1e-8
    target_modules: Optional[List[str]] = None
```

Ajouter `MuonConfig` dans `__all__` de `trainer.py`.

### 1.4 Ajouter `muon_config` dans `UnslothTrainingArguments`

```python
class UnslothTrainingArguments(TrainingArguments):
    def __init__(
        self,
        embedding_learning_rate: float = None,
        q_galore_config: Optional[QGaloreConfig] = None,
        muon_config: Optional[MuonConfig] = None,  # ← ajouter
        *args,
        **kwargs,
    ):
        self.q_galore_config = q_galore_config
        self.muon_config = muon_config              # ← ajouter
        self.embedding_learning_rate = embedding_learning_rate
        super().__init__(*args, **kwargs)
        self.embedding_learning_rate = embedding_learning_rate
```

### 1.5 Ajouter `_create_muon_optimizer()` dans `UnslothTrainer`

```python
def _create_muon_optimizer(self, config: "MuonConfig"):
    """Build un optimizer mixte Muon + AdamW depuis un MuonConfig."""
    
    # Vérifier torch >= 2.9.0
    from packaging.version import Version as PkgVersion
    import torch
    if not hasattr(torch.optim, 'Muon'):
        raise ImportError(
            "Unsloth: torch.optim.Muon requiert PyTorch >= 2.9.0.\n"
            f"Version actuelle : {torch.__version__}\n"
            "Mettez à jour avec : pip install --upgrade torch"
        )
    
    from unsloth.optimizers.muon import make_muon_param_groups
    
    lr = self.args.learning_rate
    weight_decay = self.args.weight_decay
    
    muon_groups, adamw_groups = make_muon_param_groups(
        self.model,
        lr=lr,
        weight_decay=weight_decay,
        muon_lr_scale=config.muon_lr_scale,
        adamw_lr=config.adamw_lr,
        adamw_betas=config.adamw_betas,
        adamw_eps=config.adamw_eps,
        target_modules=config.target_modules,
    )
    
    n_muon = sum(p.numel() for g in muon_groups for p in g["params"])
    n_adamw = sum(p.numel() for g in adamw_groups for p in g["params"])
    total = n_muon + n_adamw
    
    print(
        f"🦥 Unsloth: Muon enabled — "
        f"{n_muon:,} params via Muon ({100*n_muon/total:.1f}%), "
        f"{n_adamw:,} params via AdamW fallback ({100*n_adamw/total:.1f}%)"
    )
    
    muon_optimizer = torch.optim.Muon(
        muon_groups,
        lr=lr * config.muon_lr_scale,
        momentum=config.momentum,
        nesterov=config.nesterov,
        ns_steps=config.ns_steps,
        weight_decay=weight_decay,
    )
    
    # ⚠️ PROBLÈME CRITIQUE : HuggingFace Trainer n'accepte qu'un seul optimizer.
    # Solution : wrapper les deux dans un ChainedOptimizer.
    # Voir Phase 2.1 pour l'implémentation du wrapper.
    adamw_optimizer = torch.optim.AdamW(
        adamw_groups,
        lr=config.adamw_lr or lr,
        betas=config.adamw_betas,
        eps=config.adamw_eps,
        weight_decay=weight_decay,
    )
    
    self.optimizer = _MuonAdamWChained(muon_optimizer, adamw_optimizer)
    return self.optimizer
```

### 1.6 Brancher dans `create_optimizer()`

```python
def create_optimizer(self):
    # --- Muon optimizer --- (à ajouter EN PREMIER, avant Q-GaLore)
    muon_config = getattr(self.args, "muon_config", None)
    if muon_config is not None and self.optimizer is None:
        return self._create_muon_optimizer(muon_config)
    
    # --- Q-GaLore optimizer ---
    q_galore_config = getattr(self.args, "q_galore_config", None)
    if q_galore_config is not None and self.optimizer is None:
        # ... code existant ...
```

---

## Phase 2 — Problèmes Critiques Anticipés

### 2.1 ⚠️ HuggingFace Trainer n'accepte qu'un seul `self.optimizer`

**Le problème** : Muon pour les params 2D + AdamW pour le reste = deux optimizers.  
HF Trainer fait `self.optimizer.step()` — il ne sait pas gérer deux.

**Solution : `_MuonAdamWChained`** — un wrapper minimal qui délègue à chacun :

```python
class _MuonAdamWChained:
    """
    Wrapper qui enchaîne Muon (params 2D) + AdamW (params 1D).
    Présente l'interface minimale attendue par HF Trainer.
    """
    def __init__(self, muon: torch.optim.Muon, adamw: torch.optim.AdamW):
        self.muon = muon
        self.adamw = adamw
        # HF Trainer accède à self.optimizer.param_groups pour le LR scheduler
        self.param_groups = muon.param_groups + adamw.param_groups
    
    def step(self, closure=None):
        self.muon.step(closure)
        self.adamw.step(closure)
    
    def zero_grad(self, set_to_none=True):
        self.muon.zero_grad(set_to_none=set_to_none)
        self.adamw.zero_grad(set_to_none=set_to_none)
    
    def state_dict(self):
        return {"muon": self.muon.state_dict(), "adamw": self.adamw.state_dict()}
    
    def load_state_dict(self, state_dict):
        self.muon.load_state_dict(state_dict["muon"])
        self.adamw.load_state_dict(state_dict["adamw"])
```

> **Ce qu'il faut vérifier** : HF Trainer accède-t-il à d'autres attributs de l'optimizer ?  
> Chercher `self.optimizer.` dans `transformers/trainer.py` et lister TOUS les accès.  
> Les candidats typiques : `.param_groups`, `.state`, `.state_dict()`, `.load_state_dict()`, `.step()`, `.zero_grad()`

### 2.2 ⚠️ LR Scheduler — `get_last_lr()` et param_groups

HF Trainer crée un scheduler sur `self.optimizer.param_groups`. Le chained wrapper expose `muon.param_groups + adamw.param_groups`.

**Problème** : le scheduler va appliquer le même LR decay aux deux groupes — est-ce voulu ?  
**Option A** : Oui (le plus simple, implémenter d'abord)  
**Option B** : LR séparé Muon/AdamW → nécessite un scheduler custom ou deux schedulers

Pour la v1 du PR, **implémenter Option A** et documenter la limitation.

### 2.3 ⚠️ Checkpoint save/load

HF Trainer sauvegarde `optimizer.state_dict()` dans les checkpoints.  
Le format du `_MuonAdamWChained.state_dict()` est `{"muon": ..., "adamw": ...}` — format non-standard.

**Risque** : `resume_from_checkpoint` va crasher si quelqu'un essaie de reprendre avec AdamW après avoir entraîné avec Muon.

**Mitigation** : ajouter un warning explicite dans `_create_muon_optimizer()` :
```python
print("⚠️ Unsloth Muon: le format de checkpoint est incompatible avec AdamW. "
      "Ne pas utiliser resume_from_checkpoint avec un checkpoint AdamW.")
```

### 2.4 ⚠️ Gradient accumulation

Unsloth a un fix custom de gradient accumulation (voir `unsloth_train()` dans `trainer.py`).  
Vérifier que `_MuonAdamWChained.step()` est bien appelé au bon moment avec gradient accumulation.

**Test à faire sans GPU** : mocker le step et vérifier que step() est appelé exactement une fois par `gradient_accumulation_steps` steps.

### 2.5 ⚠️ Compatibilité LoRA

Si l'utilisateur fait LoRA (pas FFT) + Muon, les params trainables sont les adaptateurs A/B (2D) → Muon va s'appliquer dessus.  
Ce n'est pas forcément incorrect, mais c'est non-testé et potentiellement instable.

**Solution** : dans `_create_muon_optimizer()`, détecter si le model a des modules PEFT et logger un warning :
```python
from peft import PeftModel
if isinstance(self.model, PeftModel):
    print("⚠️ Unsloth Muon: détecté modèle PEFT/LoRA. "
          "Muon sera appliqué aux adaptateurs 2D. "
          "Résultats non garantis — utilisez full_finetuning=True pour le comportement attendu.")
```

### 2.6 ⚠️ `torch.optim.Muon` et distributed training

`torch.optim.Muon` en PyTorch 2.9 ne supporte pas encore le training distribué.  
Ajouter une guard :

```python
import torch.distributed as dist
if dist.is_available() and dist.is_initialized():
    raise RuntimeError(
        "Unsloth Muon: torch.optim.Muon ne supporte pas encore le training distribué "
        "dans PyTorch 2.9. Utilisez AdamW ou attendez PyTorch 2.10+."
    )
```

---

## Phase 3 — Tests (sans GPU)

### Pattern de test Unsloth

Les tests dans `tests/python/` utilisent `pytest` + `monkeypatch`, jamais de vrai GPU.  
Regarder `test_patch_trl_rl_trainers_defensive.py` pour le pattern exact.

### 3.1 Créer `tests/python/test_muon_optimizer.py`

```python
"""Tests unitaires pour l'intégration Muon dans Unsloth.
Tous les tests tournent sans GPU via mocks.
"""
import pytest
import torch
from unittest.mock import MagicMock, patch


# ── Helpers ────────────────────────────────────────────────────────────────

def _make_fake_model_with_params():
    """Créer un faux modèle avec params 2D et 1D pour tester le split."""
    model = torch.nn.Module()
    model.register_parameter("weight_2d", torch.nn.Parameter(torch.randn(4, 4)))
    model.register_parameter("bias_1d", torch.nn.Parameter(torch.randn(4)))
    model.register_parameter("embedding", torch.nn.Parameter(torch.randn(10, 4)))  # 2D mais embedding
    return model


# ── Tests make_muon_param_groups ────────────────────────────────────────────

def test_make_muon_param_groups_splits_correctly():
    """Les params 2D vont à Muon, les 1D à AdamW."""
    from unsloth.optimizers.muon import make_muon_param_groups, _is_muon_eligible
    
    model = _make_fake_model_with_params()
    muon_groups, adamw_groups = make_muon_param_groups(model, lr=1e-3, weight_decay=0.1)
    
    muon_params = [p for g in muon_groups for p in g["params"]]
    adamw_params = [p for g in adamw_groups for p in g["params"]]
    
    # weight_2d et embedding sont 2D → Muon
    assert len(muon_params) == 2
    # bias_1d est 1D → AdamW
    assert len(adamw_params) == 1


def test_is_muon_eligible_rejects_1d():
    from unsloth.optimizers.muon import _is_muon_eligible
    p = torch.nn.Parameter(torch.randn(4))
    assert not _is_muon_eligible(p)


def test_is_muon_eligible_accepts_2d():
    from unsloth.optimizers.muon import _is_muon_eligible
    p = torch.nn.Parameter(torch.randn(4, 4))
    assert _is_muon_eligible(p)


def test_no_requires_grad_excluded():
    from unsloth.optimizers.muon import make_muon_param_groups
    model = torch.nn.Linear(4, 4)
    model.weight.requires_grad = False
    model.bias.requires_grad = False
    muon_groups, adamw_groups = make_muon_param_groups(model, lr=1e-3, weight_decay=0.0)
    total = sum(len(g["params"]) for g in muon_groups + adamw_groups)
    assert total == 0


# ── Tests MuonConfig ────────────────────────────────────────────────────────

def test_muon_config_defaults():
    from unsloth.trainer import MuonConfig
    cfg = MuonConfig()
    assert cfg.momentum == 0.95
    assert cfg.nesterov is True
    assert cfg.ns_steps == 5


# ── Tests _MuonAdamWChained ─────────────────────────────────────────────────

def test_chained_optimizer_step_calls_both():
    from unsloth.trainer import _MuonAdamWChained
    muon = MagicMock()
    muon.param_groups = []
    adamw = MagicMock()
    adamw.param_groups = []
    
    chained = _MuonAdamWChained(muon, adamw)
    chained.step()
    
    muon.step.assert_called_once()
    adamw.step.assert_called_once()


def test_chained_optimizer_zero_grad_calls_both():
    from unsloth.trainer import _MuonAdamWChained
    muon = MagicMock()
    muon.param_groups = []
    adamw = MagicMock()
    adamw.param_groups = []
    
    chained = _MuonAdamWChained(muon, adamw)
    chained.zero_grad()
    
    muon.zero_grad.assert_called_once()
    adamw.zero_grad.assert_called_once()


def test_chained_state_dict_roundtrip():
    from unsloth.trainer import _MuonAdamWChained
    muon = MagicMock()
    muon.param_groups = []
    muon.state_dict.return_value = {"state": "muon"}
    adamw = MagicMock()
    adamw.param_groups = []
    adamw.state_dict.return_value = {"state": "adamw"}
    
    chained = _MuonAdamWChained(muon, adamw)
    sd = chained.state_dict()
    
    assert sd == {"muon": {"state": "muon"}, "adamw": {"state": "adamw"}}
    
    chained.load_state_dict(sd)
    muon.load_state_dict.assert_called_once_with({"state": "muon"})
    adamw.load_state_dict.assert_called_once_with({"state": "adamw"})


# ── Tests compatibilité PyTorch version ─────────────────────────────────────

def test_muon_raises_on_old_torch(monkeypatch):
    """Doit raise clairement si torch < 2.9."""
    import torch
    monkeypatch.delattr(torch.optim, "Muon", raising=False)
    
    # Simuler un trainer minimal
    with pytest.raises(ImportError, match="PyTorch >= 2.9"):
        # Appeler _create_muon_optimizer avec un config mock
        # (nécessite de mocker self.args et self.model)
        pass  # TODO: implémenter le mock complet du trainer


# ── Tests surface API publique ──────────────────────────────────────────────

def test_muon_config_exported_from_trainer():
    from unsloth.trainer import MuonConfig
    assert MuonConfig is not None


def test_muon_config_in_all():
    import unsloth.trainer as t
    assert "MuonConfig" in t.__all__
```

### 3.2 Tests de régression — s'assurer de ne rien casser

```bash
# Tourner TOUS les tests existants sans GPU
python -m pytest tests/python/ -v --tb=short

# Spécifiquement le test de surface API (critique)
python -m pytest tests/test_public_api_surface.py -v
```

**Si un test casse → blocker, ne pas ouvrir le PR.**

### 3.3 Vérification manuelle de l'import

```python
# Script de smoke test — 0 GPU requis
from unsloth.trainer import MuonConfig, UnslothTrainingArguments
from unsloth.optimizers.muon import make_muon_param_groups
import torch

# Vérifier que torch.optim.Muon est dispo
assert hasattr(torch.optim, 'Muon'), "torch >= 2.9.0 requis"

# Test split basique
import torch.nn as nn
model = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 2))
muon_groups, adamw_groups = make_muon_param_groups(model, lr=1e-3, weight_decay=0.1)
print(f"Muon params: {sum(p.numel() for g in muon_groups for p in g['params'])}")
print(f"AdamW params: {sum(p.numel() for g in adamw_groups for p in g['params'])}")

cfg = MuonConfig(momentum=0.95, ns_steps=5)
print(f"MuonConfig: {cfg}")
print("✅ Smoke test passed")
```

---

## Phase 4 — Documentation & Export

### 4.1 Exporter depuis `unsloth/__init__.py`

Vérifier ce qui est exporté actuellement, puis ajouter :
```python
from .trainer import MuonConfig  # ← ajouter avec QGaloreConfig
```

### 4.2 Docstring usage dans `MuonConfig`

Ajouter un exemple d'usage complet dans la docstring (copier le style de `QGaloreConfig`) :

```python
"""
Example:
    model, tokenizer = FastLanguageModel.from_pretrained(
        "unsloth/Qwen3-8B",
        full_finetuning=True,  # Muon est pensé pour le FFT
    )
    
    trainer = UnslothTrainer(
        model=model,
        args=UnslothTrainingArguments(
            muon_config=MuonConfig(
                momentum=0.95,
                ns_steps=5,
                muon_lr_scale=1.0,
            ),
            learning_rate=1e-4,
            ...
        ),
        ...
    )
"""
```

### 4.3 Ajouter `MuonConfig` dans `__all__` de `trainer.py`

```python
__all__ = [
    "UnslothTrainingArguments",
    "UnslothTrainer",
    "unsloth_train",
    "_patch_trl_trainer",
    "UnslothVisionDataCollator",
    "QGaloreConfig",
    "MuonConfig",       # ← ajouter
    "_MuonAdamWChained", # ← ajouter (pour tests)
]
```

---

## Phase 5 — Validation GPU (ponctuelle, ~30min)

> C'est la seule étape qui nécessite un GPU. Faire tourner sur RunPod (A100 40GB) ou Vast.ai.  
> Un 2B model sur 50-100 steps suffit pour valider.

### 5.1 Script de validation minimal

```python
# validate_muon.py — à faire tourner sur GPU
from unsloth import FastLanguageModel
from unsloth.trainer import UnslothTrainer, UnslothTrainingArguments, MuonConfig
from datasets import load_dataset
from trl import SFTConfig

model, tokenizer = FastLanguageModel.from_pretrained(
    "unsloth/Qwen3-2B",
    max_seq_length=512,
    full_finetuning=True,  # FFT obligatoire pour Muon well-defined
)

dataset = load_dataset("HuggingFaceH4/alpaca_gpt4_en", split="train[:200]")

trainer = UnslothTrainer(
    model=model,
    tokenizer=tokenizer,
    args=UnslothTrainingArguments(
        muon_config=MuonConfig(momentum=0.95, ns_steps=5),
        output_dir="./muon_test",
        num_train_epochs=1,
        max_steps=50,
        per_device_train_batch_size=2,
        learning_rate=1e-4,
        logging_steps=10,
    ),
    train_dataset=dataset,
    dataset_text_field="output",
)

trainer.train()
print("✅ Muon training completed — loss devrait descendre sur 50 steps")
```

### 5.2 Ce qu'il faut vérifier

- La loss descend (pas NaN, pas plateau immédiat)
- Le ratio Muon/AdamW params est logique (s'affiche dans le print)
- Pas de crash sur `.step()`, `.zero_grad()`, checkpoint save
- `trainer.state.log_history` contient des `loss` décroissantes

---

## Phase 6 — PR

### 6.1 Checklist avant d'ouvrir le PR

- [ ] Tous les tests `tests/python/` passent sans GPU
- [ ] `test_public_api_surface.py` passe
- [ ] `MuonConfig` importable depuis `unsloth` directement
- [ ] Script de validation GPU a tourné sans crash
- [ ] Warning PEFT/LoRA ajouté
- [ ] Warning distributed training ajouté
- [ ] Warning checkpoint incompatibilité ajouté
- [ ] Docstring avec example d'usage

### 6.2 Titre du PR

```
feat: add Muon optimizer support (torch.optim.Muon, requires PyTorch ≥ 2.9)
```

### 6.3 Description du PR (template)

```markdown
## Motivation
Muon (Momentum + Newton-Schulz orthogonalization) a montré des résultats 
prometteurs sur des runs de pré-training à grande échelle (DeepSeek-V4, Kimi Moonlight). 
Cette PR l'intègre dans Unsloth pour le FFT, en suivant exactement le pattern 
de QGaloreConfig existant.

## Changements
- `unsloth/optimizers/muon.py` : `make_muon_param_groups()` — split 2D/1D params
- `unsloth/trainer.py` : `MuonConfig`, `_MuonAdamWChained`, `_create_muon_optimizer()`
- `unsloth/optimizers/__init__.py` : export
- `tests/python/test_muon_optimizer.py` : tests sans GPU

## Utilisation
[coller l'exemple de la docstring]

## Limitations connues
- torch >= 2.9.0 requis (torch.optim.Muon)
- Pas de support distributed training (limitation upstream torch.optim.Muon)
- Format checkpoint incompatible avec resume depuis AdamW
- Comportement en LoRA non garanti (warning ajouté)
```

---

## Résumé des fichiers à créer/modifier

| Fichier | Action | Lignes estimées |
|---|---|---|
| `unsloth/optimizers/muon.py` | **CRÉER** | ~60 |
| `unsloth/optimizers/__init__.py` | modifier | +3 |
| `unsloth/trainer.py` | modifier | +120 |
| `tests/python/test_muon_optimizer.py` | **CRÉER** | ~100 |

Total : ~280 lignes de code net.

---

## Pièges spécifiques à surveiller pendant le coding

1. **Ne jamais passer les MÊMES params à Muon ET AdamW** — vérifier que les deux listes sont disjointes
2. **`param_groups` du chained optimizer** — HF Trainer itère dessus pour le LR decay, vérifier que chaque group a un `lr` key
3. **`torch.optim.Muon` adjust_lr_fn** — par défaut Muon ajuste le LR selon la shape du param (`sqrt(max(1, A/B))`). Documenter que `learning_rate` dans `TrainingArguments` est le LR de base avant ajustement
4. **State dict format** — si HF Trainer appelle `optimizer.state_dict()` pour autre chose que checkpoint (ex: FSDP), le format `{"muon": ..., "adamw": ...}` peut causer des problèmes
5. **`ns_steps` et vitesse** — chaque step Muon fait `ns_steps` passes Newton-Schulz. Avec `ns_steps=5` (défaut), chaque optimizer step est ~5x plus cher en compute que SGD. Documenter l'impact sur le throughput
