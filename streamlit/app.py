import os
import pickle
import requests
from PIL import Image
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
import requests
import streamlit as st

# === Configuration de la page
st.set_page_config(page_title="Book Genre Classifier", layout="centered")

if "page" not in st.session_state:
    st.session_state.page = None
    
    # Fonctions pour afficher les pages
def show_login():
    st.subheader("Connexion")
    username = st.text_input("Nom d'utilisateur", key="login_username")
    password = st.text_input("Mot de passe", type="password", key="login_password")
    
    if st.button("Se connecter"):
        if username and password:
            try:
                response = requests.post(
                    "http://localhost:8000/login/", 
                    headers={"Content-Type": "application/x-www-form-urlencoded"},
                    data={"username": username, "password": password}
                )
                if response.status_code == 200:
                    token = response.json()["access_token"]
                    st.session_state["token"] = token
                    st.success("Connexion réussie ! ✅")
                    st.write(f"Token stocké dans session_state.")
                else:
                    st.error(f"Erreur de connexion ({response.status_code}) : {response.json().get('detail', 'inconnu')}")
            except Exception as e:
                st.error(f"Erreur de connexion au serveur : {e}")
        else:
            st.warning("Veuillez remplir les deux champs.")

def show_register():
    st.subheader("Créer un compte")

    username = st.text_input("Nom d'utilisateur", key="reg_username")
    email = st.text_input("Adresse email", key="reg_email")
    password = st.text_input("Mot de passe", type="password", key="reg_password")

    if st.button("S'inscrire"):
        if username and email and password:
            payload = {
                "username": username,
                "email": email,
                "password": password
            }

            try:
                response = requests.post(
                    "http://localhost:8000/register/",  
                    headers={"Content-Type": "application/json"},
                    json=payload
                )

                if response.status_code == 200 or response.status_code == 201:
                    st.success(f"Compte créé avec succès pour {username} ✅")
                else:
                    st.error(f"Erreur ({response.status_code}) : {response.text}")
            except Exception as e:
                st.error(f"Erreur lors de l'inscription : {e}")
        else:
            st.warning("Merci de remplir tous les champs.")


# Menu de choix
col1, col2 = st.columns(2)
with col1:
    if st.button("🔓 Login"):
        st.session_state.page = "login"
with col2:
    if st.button("📝 Register"):
        st.session_state.page = "register"

# Affichage dynamique
if st.session_state.page == "login":
    show_login()
elif st.session_state.page == "register":
    show_register()


# === Affichage du logo
logo_path = os.path.join("..","img", "Kafklassifier_logo.png") 
if os.path.exists(logo_path):
    logo = Image.open(logo_path)
    st.image(logo, width=200)
else:
    st.warning("Logo introuvable à l'emplacement spécifié.")

# === Titres
st.markdown(
    "<h2 style='text-align: center;'>Book genre predictor</h2>"
    "<p style='text-align: center;'>Enter a book summary and get the genres predicted by AI</p>",
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
user_input = st.text_area("✏️ Book summary", height=200)

# === Slider pour régler le seuil de score
threshold_percent = st.slider("🎯 Minimal score threshold (%)", min_value=0, max_value=100, value=10)
threshold = threshold_percent / 100


if st.button("🔍 Predict the Genre"):
    if "token" not in st.session_state:
        st.warning("⚠️ Vous devez être connecté.")
    elif len(user_input.strip()) < 100:
        st.warning("⚠️ Résumé trop court (min 100 caractères).")
    else:
        with st.spinner("Prédiction en cours..."):
            try:
                response = requests.post(
                    "http://localhost:8000/predict/",  
                    headers={
                        "Authorization": f"Bearer {st.session_state['token']}",
                        "Content-Type": "application/json"
                    },
                    json={"text": user_input}
                )

                if response.status_code == 200:
                    predicted_genre = response.json()["genre"]
                    st.success(f"📚 Genre prédit : {predicted_genre}")
                else:
                    st.error(f"Erreur ({response.status_code}) : {response.text}")

            except Exception as e:
                st.error(f"❌ Erreur de requête : {e}")


