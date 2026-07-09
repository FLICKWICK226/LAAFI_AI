import gradio as gr
import torch
import io
from inference import load_model, predict_image

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL = load_model("../resnet50_finetuned_weights.pth", DEVICE)

def gradio_predict(image_pil):
    # Convertir l'image PIL de Gradio en bytes pour notre fonction
    img_byte_arr = io.BytesIO()
    image_pil.save(img_byte_arr, format='PNG')
    image_bytes = img_byte_arr.getvalue()
    
    # Prédiction
    result = predict_image(MODEL, image_bytes, DEVICE)
    
    # Formater pour l'affichage Gradio
    risque = result["probabilite"]
    
    # Retourner un dictionnaire pour le composant "Label" de Gradio
    return {"Métastase (Cancer)": risque, "Tissu Sain": 1 - risque}

# Construction de l'interface
demo = gr.Interface(
    fn=gradio_predict,
    inputs=gr.Image(type="pil", label="Uploadez un patch histopathologique"),
    outputs=gr.Label(num_top_classes=2, label="Résultat du diagnostic AI"),
    title="LAAFI AI : Détection de Métastases (CDSS)",
    description="Outil d'aide au diagnostic basé sur ResNet50. Insérez un patch cellulaire pour obtenir une probabilité d'invasion métastatique.\n⚠️ Ceci est un prototype et ne remplace pas un médecin.",
    theme="soft"
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
