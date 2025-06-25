# # FINAL TEST

# ### Step 1 : Import + data loading

import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import LabelEncoder
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    pipeline,
)

df = pd.read_csv("../data/data.csv")  # adapte le nom du fichier
df.head()


print("Dimensions du dataset :", df.shape)
print("\nColonnes :", df.columns.to_list())
print("\nTypes de données :\n", df.dtypes)
print("\nValeurs manquantes :\n", df.isnull().sum())
print("\nDistribution des genres :\n", df["genre"].value_counts())
print("\nNombre de classes de genre :", df["genre"].nunique())


genre_counts = df["genre"].value_counts()
plt.figure(figsize=(6, 6))
plt.pie(genre_counts, labels=genre_counts.index, autopct="%1.1f%%", startangle=190)
plt.title("Répartition des genres du dataset")
plt.axis("equal")
plt.show()


# ### 2.1. Label encoder (-> int)

le = LabelEncoder()
df["label"] = le.fit_transform(df["genre"])


# ### 2.2. Save label encoder for the inference stage

with open("label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)


# ### Step 3 : Convert to Huggingface dataset

dataset = Dataset.from_pandas(df[["summary", "label"]])  # Conversion en dataset Hugginface


# ### Step 4 : Tokenization

model_checkpoint = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)


def preprocess_function(examples):
    return tokenizer(examples["summary"], truncation=True, padding="max_length", max_length=256)


tokenized_dataset = dataset.map(preprocess_function, batched=True)


# ### Step 5 : Split dataset

split = tokenized_dataset.train_test_split(test_size=0.2)  # 80/20 (-> small dataset)
train_dataset = split["train"]
eval_dataset = split["test"]


# ### STEP 6 : Loading multi-label classifier MODEL

num_labels = len(le.classes_)
model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=num_labels)


# # Step 7 : Metrics

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average="weighted")
    acc = accuracy_score(labels, predictions)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}


# 8. Configurer entraînement
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
)

trainer = Trainer(
    model=model,  # DistilBERT
    args=training_args,  # config perso
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
)


# ### Step 10  : Training and Saving the model


trainer.train()

trainer.save_model("./book_genre_model")
tokenizer.save_pretrained("./book_genre_model")


# # UTILISATION

classifier = pipeline(
    "text-classification", model="./book_genre_model", tokenizer="./book_genre_model", return_all_scores=True
)

# Recharger le label encoder
with open("label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

texts = [
    "A thrilling journey of magic and dragons.",
    "A love story set in the beautiful countryside.",
    "A journey to the east riding a van on a road trip"
    " on a travel to make memories around the globe in every country",
    "A terrific music festival in Valenciennes eating some"
    " Lucullus while listening death melodic metal and headbanging",
]

# Obtenir les prédictions
for text in texts:
    raw_preds = classifier(text)[0]
    decoded_preds = [
        {"label": le.inverse_transform([int(p["label"].split("_")[-1])])[0], "score": p["score"]}
        for p in raw_preds
    ]
    decoded_preds = sorted(decoded_preds, key=lambda x: x["score"], reverse=True)

    print(f"\nTexte : {text}")
    for pred in decoded_preds:
        print(f"→ {pred['label']} : {pred['score']:.2f}")


for text in texts:
    raw_preds = classifier(text)[0]
    decoded_preds = [
        {
            "label": le.inverse_transform([int(p["label"].split("_")[-1])])[0],
            "score": p["score"],
        }
        for p in raw_preds
    ]
    decoded_preds = sorted(decoded_preds, key=lambda x: x["score"], reverse=True)

    print(f"\nTexte : {text}")
    for pred in decoded_preds:
        if pred["score"] > 0.10:
            print(f"Genre : [{pred['label']}] : {pred['score']:.2f}")
