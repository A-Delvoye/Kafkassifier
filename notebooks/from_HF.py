import io
import pickle

import requests
from transformers import pipeline

# Charger modèle et tokenizer depuis Hugging Face
model_name = "Coffeewaves/kafklassifier-nlp"

classifier = pipeline("text-classification", model=model_name, tokenizer=model_name, return_all_scores=True)

# Charger label_encoder.pkl depuis Hugging Face
response = requests.get(f"https://huggingface.co/{model_name}/resolve/main/label_encoder.pkl")
label_encoder = pickle.load(io.BytesIO(response.content))

# exemple :
texts = [
    "A thrilling journey of magic and dragons.",
    "A love story set in the countryside.",
    (
        "Blending archaeology, ancient texts, and modern scholarship, this book brings to life the siege "
        "that inspired poets and shaped Greek identity. Meet the real Achilles, Hektor, and Agamemnon—and "
        "discover the political ambitions, economic motives, and cultural myths that fueled one of the most "
        "famous conflicts of the ancient world. A gripping retelling and critical analysis, The Fall of Troy "
        "explores where myth ends and history begins."
    ),
]

for text in texts:
    raw_preds = classifier(text)[0]
    decoded_preds = [
        {"label": label_encoder.inverse_transform([int(p["label"].split("_")[-1])])[0], "score": p["score"]}
        for p in raw_preds
    ]
    decoded_preds = sorted(decoded_preds, key=lambda x: x["score"], reverse=True)

    print(f"\nTexte : {text}")
    for pred in decoded_preds:
        if pred["score"] > 0.1:
            print(f"→ {pred['label']} : {pred['score']:.2f}")
