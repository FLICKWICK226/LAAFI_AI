# LAAFI_AI — Notes pour le pitch oral (2 minutes)

## 🎯 Problème (20 secondes)

> "Les pathologistes analysent des lames histopathologiques pour détecter des métastases dans les ganglions lymphatiques. C'est un processus long, répétitif, et sujet aux erreurs humaines — surtout quand les métastases sont petites. Mon projet propose un outil d'aide au diagnostic basé sur le deep learning pour accélérer ce screening."

## 🔧 Méthode (30 secondes)

> "J'ai utilisé le dataset PatchCamelyon — 327 000 patches de 96×96 pixels extraits de lames H&E, labellisés manuellement par des pathologistes. J'ai entraîné un ResNet50 pré-entraîné ImageNet en deux temps :
> - D'abord en feature extraction, backbone gelé, pour entraîner uniquement le classifieur.
> - Puis en fine-tuning partiel, en dégelant le layer4 pour adapter les features de haut niveau au domaine médical.
>
> Le code est modulaire : configuration YAML, modules Python typés, tests unitaires, reproductible via Colab."

## 📊 Résultats (30 secondes)

> "Sur le test set officiel — jamais utilisé pendant l'entraînement :
> - AUC ROC : 0.95 — le modèle discrimine très bien positifs et négatifs.
> - Spécificité : 97% — très peu de faux positifs.
> - Mais sensibilité : 75% — ça veut dire que 25% des vraies métastases passent à travers.
> - J'ai aussi produit des cartes Grad-CAM pour vérifier que le modèle regarde les bonnes régions du tissu."

## ⚠️ Limites (20 secondes)

> "Trois limites honnêtes :
> 1. La sensibilité de 75% est insuffisante pour un usage clinique — un vrai outil ne peut pas rater 1 cancer sur 4.
> 2. Ce sont des patches isolés, pas des lames complètes (WSI) — en conditions réelles, il faut agréger les prédictions.
> 3. Pas de normalisation de coloration — les variations entre laboratoires peuvent dégrader les performances."

## 🚀 Améliorations futures (20 secondes)

> "Les prochaines étapes :
> - Normalisation Macenko pour gérer les variations de coloration H&E.
> - Calibration des probabilités et optimisation du seuil de décision.
> - Passage aux WSI avec le dataset CAMELYON16 et l'architecture MIL (Multiple Instance Learning).
> - Déploiement d'une démo Gradio sur Hugging Face Spaces."

---

## Conseils pour la présentation

- **Ne pas cacher la sensibilité faible** — c'est ce qui montre ta maturité. Un junior cacherait cette info. Un bon ingénieur la met en avant et explique comment l'améliorer.
- **Montrer les Grad-CAM** — c'est visuellement fort et ça prouve que tu comprends l'interprétabilité.
- **Dire "test set officiel"** — ça montre que tu sais que le split compte.
- **Dire "aide au diagnostic"** et jamais "diagnostic automatique" — tu sais que ce n'est pas un dispositif médical.
