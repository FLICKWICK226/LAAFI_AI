# LAAFI_AI — Drive Experiment Logger

Script de logging automatique qui écrit les résultats de chaque run de modèle directement dans le Google Doc **04_Metriques_et_Suivi_Experiments** sans saisie manuelle.

---

## Installation

```bash
pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib
```

Pour l'intégration MLflow (optionnelle) :
```bash
pip install mlflow
```

---

## Configuration (2 options)

### Option A — Service Account (recommandé pour l'automatisation)

1. Aller sur [console.cloud.google.com](https://console.cloud.google.com/)
2. Créer un projet ou sélectionner le projet existant
3. Activer l'API **Google Docs API**
4. Créer un **Service Account** → Télécharger `credentials.json`
5. Partager le Google Doc avec l'adresse e-mail du service account (accès Éditeur)

### Option B — OAuth 2.0 (pour usage interactif local)

1. Aller sur [console.cloud.google.com](https://console.cloud.google.com/)
2. Activer l'API **Google Docs API**
3. Créer des identifiants OAuth 2.0 (type "Application de bureau")
4. Télécharger `credentials.json`
5. Au premier lancement, un navigateur s'ouvrira pour authentifier — un `token.json` sera créé automatiquement

---

## Variables d'environnement

Créer un fichier `.env` à la racine du repo (ou exporter les variables) :

```env
LAAFI_DOC_ID=15-fNXMxzaHWX9jDOfTVkCtYVFCc4QZEQiikDyFN5vJY
GDOCS_CREDENTIALS_PATH=credentials.json
GDOCS_TOKEN_PATH=token.json
```

**Ne jamais committer `credentials.json` ou `token.json` dans Git.**
Ces fichiers sont déjà dans `.gitignore`.

---

## Utilisation

### En Python (intégration dans le pipeline d'entraînement)

```python
from drive_logger import ExperimentLogger, ExperimentResult

# Initialisation (lit les variables d'env automatiquement)
logger = ExperimentLogger()

# À la fin de ton run d'entraînement
result = ExperimentResult(
    exp_id       = "EXP-003",           # EXP-001 à EXP-007
    method       = "MacenkoNormalizer via torchstain",
    dataset      = "PCam val set",
    auc_roc      = 0.9581,              # float [0, 1]
    recall       = 0.812,               # Sensibilité
    precision    = 0.961,
    accuracy     = 0.889,               # optionnel
    ece          = 0.042,               # optionnel
    failure_mode = "Artéfacts bleus sur ~6% des patches éosine",
    note         = "Committer dans main — gain +3.3 pp recall",
    extra        = {
        "SSIM source":          "0.923",
        "FPS à l'inférence":    "4.0",
        "Delta recall vs. EXP-001": "+3.3 pp",
    },
)
logger.log(result)
```

### Intégration dans trainer.py

```python
# Dans Trainer.fit() ou evaluate(), après calcul des métriques :

from drive_logger import ExperimentLogger, ExperimentResult

def log_to_drive(metrics: dict, exp_id: str = "EXP-005"):
    """Appeler à la fin de chaque run significatif."""
    logger = ExperimentLogger()
    result = ExperimentResult(
        exp_id       = exp_id,
        method       = metrics.get("method", ""),
        dataset      = metrics.get("dataset", "PCam val set"),
        auc_roc      = metrics["auc_roc"],
        recall       = metrics["recall"],
        precision    = metrics["precision"],
        accuracy     = metrics.get("accuracy"),
        ece          = metrics.get("ece"),
        failure_mode = metrics.get("failure_mode", "Aucun observé"),
        note         = metrics.get("note", ""),
        mlflow_run_id= metrics.get("mlflow_run_id", ""),
    )
    logger.log(result)
```

### Intégration avec MLflow

```python
import mlflow
from drive_logger import ExperimentLogger, ExperimentResult

with mlflow.start_run() as run:
    # ... ton entraînement ...
    result = ExperimentResult(
        exp_id        = "EXP-005",
        method        = "Fine-tuning layer3+layer4 + CosineAnnealingLR",
        dataset       = "PCam train/val/test",
        auc_roc       = 0.963,
        recall        = 0.851,
        precision     = 0.964,
        mlflow_run_id = run.info.run_id,
    )
    logger = ExperimentLogger()
    logger.log(result)
    # Les métriques sont aussi écrites dans MLflow automatiquement
```

### Via CLI (sans écrire de Python)

```bash
python drive_logger.py \
    --exp EXP-003 \
    --method "MacenkoNormalizer via torchstain" \
    --dataset "PCam val set" \
    --auc 0.958 \
    --recall 0.812 \
    --precision 0.961 \
    --accuracy 0.889 \
    --failure "Artéfacts bleus sur ~6% patches eosin" \
    --note "Committer dans main — gain +3.3 pp recall" \
    --extra "SSIM source=0.923" "FPS à l'inférence=4.0"
```

#### Dry run (vérifier sans écrire dans Drive)

```bash
python drive_logger.py --exp EXP-001 --method "Youden seuil" \
    --dataset "PCam val" --auc 0.951 --recall 0.805 --precision 0.963 \
    --dry-run
```

---

## Tableau des expériences prises en charge

| ID      | Expérience                              | Table Doc |
|---------|-----------------------------------------|-----------|
| EXP-001 | Optimisation seuil (Youden's J)         | Table 2   |
| EXP-002 | Temperature Scaling (calibration)       | Table 3   |
| EXP-003 | Normalisation Macenko                   | Table 4   |
| EXP-004 | Normalisation StainNet                  | Table 5   |
| EXP-005 | Fine-tuning layer3+layer4 + LR différencié | Table 6 |
| EXP-006 | Linear probe UNI / Virchow2             | Table 7   |
| EXP-007 | WSI Inference CAMELYON16                | Table 8   |

---

## Sécurité

- `credentials.json` → ne jamais committer
- `token.json` → ne jamais committer
- `.env` → ne jamais committer
- Ces 3 fichiers sont dans `.gitignore`

---

## Dépannage

| Erreur | Cause | Solution |
|--------|-------|----------|
| `FileNotFoundError: credentials.json` | Fichier absent | Voir section Configuration |
| `HttpError 403` | Doc non partagé avec le service account | Partager le Doc avec l'email du service account |
| `Label not found in table` | Champ `extra` mal orthographié | Vérifier le nom exact dans le Doc Drive |
| `LAAFI_DOC_ID is not set` | Variable d'env manquante | Ajouter au `.env` ou passer `doc_id=` |
