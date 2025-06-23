import streamlit as st
import pickle
import os
from PIL import Image
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

# === Configuration de la page
st.set_page_config(page_title="Book Genre Classifier", layout="centered")

# === Affichage du logo
logo_path = os.path.join("..","img", "Kafklassifier_logo.png")  # adapte selon ton arborescence
if os.path.exists(logo_path):
    logo = Image.open(logo_path)
    st.image(logo, width=200)
else:
    st.warning("Logo introuvable à l'emplacement spécifié.")

# === Titres
st.markdown(
    "<h2 style='text-align: center;'>Prédicteur de genre littéraire</h2>"
    "<p style='text-align: center;'>Entrez un résumé de livre et obtenez les genres prédits par l'IA</p>",
    unsafe_allow_html=True
)

# === Chemins des fichiers
MODEL_DIR = os.path.join("..", "model", "book_genre_model")
LABEL_ENCODER_PATH = os.path.join("..", "model", "label_encoder.pkl")

# === Chargement du modèle et du tokenizer
@st.cache_resource
def load_pipeline():
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    return pipeline("text-classification", model=model, tokenizer=tokenizer, return_all_scores=True)

# === Chargement du label encoder
@st.cache_data
def load_label_encoder():
    with open(LABEL_ENCODER_PATH, "rb") as f:
        return pickle.load(f)

# === Fonction de prédiction
def predict_genre(text, classifier, label_encoder, threshold=0.10):
    raw_preds = classifier(text)[0]
    decoded_preds = [
        {
            "label": label_encoder.inverse_transform([int(p["label"].split("_")[-1])])[0],
            "score": p["score"]
        }
        for p in raw_preds
    ]
    decoded_preds = sorted(decoded_preds, key=lambda x: x["score"], reverse=True)
    return [pred for pred in decoded_preds if pred["score"] > threshold]

# === Zone de saisie utilisateur
user_input = st.text_area("✏️ Résumé du livre", height=200)

# === Slider pour régler le seuil de score
threshold_percent = st.slider("🎯 Seuil minimal de score (%)", min_value=0, max_value=100, value=10)
threshold = threshold_percent / 100

# === Bouton de prédiction
if st.button("🔍 Prédire le genre") and user_input.strip():
    with st.spinner("Analyse en cours..."):
        classifier = load_pipeline()
        label_encoder = load_label_encoder()
        predictions = predict_genre(user_input, classifier, label_encoder, threshold=threshold)

    if predictions:
        st.success("Genres prédits :")
        for pred in predictions:
            st.markdown(f"<div style='background-color:#f0f2f6;padding:10px;border-radius:10px;'>"
                        f"<strong>{pred['label']}</strong> — {pred['score']:.2%}</div>",
                        unsafe_allow_html=True)
    else:
        st.warning("Aucun genre détecté avec un score supérieur au seuil choisi.")