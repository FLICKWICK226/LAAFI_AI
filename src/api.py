from fastapi import FastAPI, File, UploadFile, HTTPException
import torch
from inference import load_model, predict_image

app = FastAPI(
    title="LAAFI AI - PCam API",
    description="API de détection de métastases via PatchCamelyon et ResNet50",
    version="1.0.0"
)

# Chargement global du modèle au démarrage du serveur
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "../resnet50_finetuned_weights.pth" # Assurez-vous du chemin
MODEL = load_model(MODEL_PATH, DEVICE)

@app.get("/")
def root():
    return {"message": "Bienvenue sur l'API LAAFI AI. Utilisez /docs pour voir la documentation."}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Le fichier doit être une image.")
        
    image_bytes = await file.read()
    result = predict_image(MODEL, image_bytes, DEVICE)
    return result
