import streamlit as st
import pickle
import os
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

MODEL_DIR = os.path.join("..", "model", "book_genre_model")
LABEL_ENCODER_PATH = os.path.join("..", "model", "label_encoder.pkl")

# --- Chargement du modèle et tokenizer
@st.cache_resource
def load_pipeline():
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    return pipeline("text-classification", model=model, tokenizer=tokenizer, return_all_scores=True)


# --- Chargement du LabelEncoder
@st.cache_data
def load_label_encoder():
    with open(LABEL_ENCODER_PATH, "rb") as f:
        return pickle.load(f)

# --- Fonction de prédiction
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

# === Streamlit App ===
st.set_page_config(page_title="Book Genre Classifier", layout="centered")
st.title("📚 Prédicteur de genre littéraire")
st.write("Entrez un résumé de livre, et l'IA vous propose les genres les plus probables.")

# Zone de saisie
user_input = st.text_area("✏️ Résumé du livre", height=200)

# Bouton de prédiction
if st.button("🔍 Prédire le genre") and user_input.strip():
    with st.spinner("Analyse en cours..."):
        classifier = load_pipeline()
        label_encoder = load_label_encoder()
        predictions = predict_genre(user_input, classifier, label_encoder)

    if predictions:
        st.success("Genres prédits :")
        for pred in predictions:
            st.markdown(f"**{pred['label']}** : {pred['score']:.2%}")
    else:
        st.warning("Aucun genre détecté avec un score significatif.")
