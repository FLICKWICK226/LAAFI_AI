#!/bin/bash
# Setup DVC pour le versionnement des données et des modèles LAAFI_AI

set -e

echo "=== Initialisation de DVC ==="

# 1. Initialiser DVC s'il n'est pas déjà initialisé
if [ ! -d ".dvc" ]; then
    dvc init
    echo "DVC initialisé."
else
    echo "DVC est déjà initialisé."
fi

# 2. Configurer le remote (Optionnel : à décommenter et adapter)
# echo "=== Configuration du Remote DVC ==="
# dvc remote add -d myremote s3://mybucket/dvcstore
# dvc remote add -d myremote gdrive://<ID_DU_DOSSIER_GDRIVE>
# dvc remote add -d myremote /chemin/local/partagé

# 3. Ajouter les dossiers volumineux à DVC
echo "=== Ajout des données à DVC ==="
if [ -d "outputs" ]; then
    dvc add outputs
    echo "Dossier 'outputs' ajouté à DVC."
fi

if [ -d "outputs_final" ]; then
    dvc add outputs_final
    echo "Dossier 'outputs_final' ajouté à DVC."
fi

# 4. Git commit des fichiers de suivi DVC
echo "=== Commit Git ==="
git add .dvc/config .dvcignore outputs.dvc outputs_final.dvc
# git commit -m "chore: setup dvc and add output tracking"

echo "=== Terminé ! ==="
echo "Pensez à configurer un DVC remote (S3, Google Drive, etc.) si vous souhaitez partager ces fichiers."
echo "Pour pousser vos données : dvc push"
