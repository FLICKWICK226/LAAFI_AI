# LAAFI_AI — Roadmap

## ✅ Phase 1 : Pipeline de base (terminée)

- [x] Dataset PatchCamelyon via Hugging Face
- [x] ResNet50 pré-entraîné ImageNet
- [x] Feature extraction (backbone gelé) — 5 époques
- [x] Fine-tuning partiel (layer4 dégelé) — 10 époques
- [x] Évaluation test set : AUC = 0.9513, Accuracy = 86%
- [x] Courbes ROC, PR, matrice de confusion
- [x] Grad-CAM sur TP, FP, TN, FN (5 exemples chacun)
- [x] Code modulaire, typé, testé
- [x] Configuration YAML + CLI
- [x] Notebooks reproductibles Colab
- [x] Prototype FastAPI + Gradio (démo)
- [x] Dockerfile pour déploiement

---

## 🔲 Phase 2 : Normalisation de coloration et augmentations

- [x] Implémenter la normalisation Macenko ou Vahadane pour corriger les variations de coloration H&E entre laboratoires
- [ ] Ajouter des augmentations avancées : Mixup, CutMix, stain augmentation
- [ ] Mesurer l'impact sur la sensibilité (actuellement 0.749, trop faible pour usage clinique)
- [ ] Objectif : sensibilité ≥ 0.85 sans sacrifier la spécificité

---

## 🔲 Phase 3 : Calibration et optimisation du seuil

- [x] Appliquer temperature scaling ou Platt scaling pour calibrer les probabilités
- [x] Tracer un diagramme de fiabilité (reliability diagram)
- [x] Optimiser le seuil de décision via la courbe ROC (Youden's J) plutôt que 0.5 par défaut
- [x] Rapporter les intervalles de confiance (bootstrap) sur toutes les métriques

---

## 🔲 Phase 4 : Pipeline MLOps

- [x] Intégrer MLflow ou DVC pour le suivi des expériences
- [x] Versionner les checkpoints et les données avec DVC
- [x] Automatiser l'entraînement avec des scripts reproductibles (pas seulement notebooks)
- [x] Ajouter un pipeline CI/CD : tests, lint, formatage

---

## 🔲 Phase 5 : CAMELYON16 — Passage à l'échelle WSI

- [ ] Adapter le pipeline aux Whole Slide Images (WSI) du challenge CAMELYON16
- [ ] Implémenter un sliding window + agrégation des prédictions patch-level
- [ ] Évaluer les performances au niveau slide (pas seulement au niveau patch)
- [ ] Explorer les architectures d'attention pour l'agrégation (MIL — Multiple Instance Learning)

---

## 🔲 Phase 6 : Déploiement et démo

- [ ] Finaliser l'API FastAPI avec documentation Swagger complète
- [ ] Déployer la démo Gradio sur Hugging Face Spaces
- [ ] Publier les poids du meilleur modèle sur Hugging Face Hub
- [ ] Ajouter un monitoring de performance en production

---

## Notes

> Ce projet est un travail portfolio. Il ne vise pas un usage clinique réel.
> Les extensions ci-dessus sont des axes d'amélioration identifiés, classés par priorité.
