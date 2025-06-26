import os

import requests
from dotenv import load_dotenv

# Charger les variables d'environnement
load_dotenv()

HF_token = os.getenv("HF_token")
HF_model = os.getenv("HF_model")
API_URL = f"https://api-inference.huggingface.co/models/{HF_model}"
headers = {"Authorization": f"Bearer {HF_token}"}


def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    try:
        return response.json()
    except Exception as e:
        print("Erreur:", e)
        print("Texte brut:", response.text)
        return None


if __name__ == "__main__":
    example_text = "Un groupe d'aventuriers combat un dragon dans un royaume lointain."
    result = query({"inputs": example_text})
    print("Résultat :")
    print(result)
