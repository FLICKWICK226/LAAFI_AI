# Setup manuel Colab et portfolio

Certaines actions ne peuvent pas etre automatisees proprement depuis le code. Ce guide les liste clairement.

## 1. Activer le GPU dans Google Colab

Dans Colab :

1. Menu `Execution`.
2. `Modifier le type d'execution`.
3. Accelerateur materiel : `GPU`.
4. Verifie avec :

```python
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
```

## 2. Installer les dependances

Dans une cellule Colab execute :

```python
%pip install -q -r requirements-colab.txt
```

Si le notebook n'est pas lance depuis le dossier `LAAFI_AI`, monte Google Drive puis deplace-toi dans le bon dossier :

```python
from google.colab import drive
drive.mount("/content/drive")
%cd /content/drive/MyDrive/LAAFI_AI
```

## 3. Telecharger PatchCamelyon

Le code utilise Hugging Face datasets :

```python
from datasets import load_dataset
dataset = load_dataset("1aurent/PatchCamelyon")
print(dataset)
```

Le dataset est volumineux. Pour tester vite, commence avec `max_train_samples=512` et `max_val_samples=128`.

## 4. Configurer Weights & Biases optionnel

Weights & Biases sert a suivre les courbes d'entrainement.

```python
import wandb
wandb.login()
```

Puis active `use_wandb: true` dans `configs/default.yaml`.

## 5. Sauvegarder les poids

Ne versionne pas les fichiers lourds dans GitHub. Sauvegarde les checkpoints dans :

```text
outputs/checkpoints/
```

Pour un portfolio, tu peux publier les poids sur Hugging Face Hub ou les exclure et fournir un lien separe.

## 6. Resultats attendus

Sur un entrainement complet bien parametre, une AUC superieure a 0.92 est une cible raisonnable. Une AUC proche de 0.99 doit etre traitee comme suspecte : fuite de donnees, split incorrect ou bug d'evaluation.

## 7. Checklist portfolio

- README clair avec contexte medical.
- Courbes ROC et PR.
- Matrice de confusion.
- Grad-CAM sur exemples positifs et negatifs.
- Limites honnetes : dataset de patches, pas de validation clinique, biais de coloration H&E.
- Code reproductible avec configuration et logs.
