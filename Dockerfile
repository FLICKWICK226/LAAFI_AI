# Image de base Linux avec Python et CUDA (pour GPU)
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# Dossier de travail dans le conteneur
WORKDIR /app

# Installer les dépendances système
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    python-multipart \
    gradio \
    pillow \
    torchvision

# Copier le code source
COPY src/ /app/src/

# Copier le checkpoint du meilleur modèle.
# ⚠️ Ce fichier n'est PAS versionné dans Git (trop lourd : ~90 MB).
# Avant de builder, copiez votre checkpoint ici :
#   cp outputs_feature_extraction/checkpoints/best_resnet50_pcam.pt ./best_model.pt
# Ou téléchargez-le depuis votre stockage externe.
COPY best_model.pt /app/best_model.pt

# Variables d'environnement
ENV MODEL_PATH=/app/best_model.pt

# Exposer les ports : FastAPI (8000), Gradio (7860)
EXPOSE 8000
EXPOSE 7860

# Par défaut, lancer l'API FastAPI
# Pour Gradio : docker run -p 7860:7860 laafi-ai python src/ui.py
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
