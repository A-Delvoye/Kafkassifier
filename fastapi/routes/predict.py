from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
import book_genre_model
from transformers import AutoTokenizer , AutoModelForSequenceClassification
import torch

router = APIRouter()

tokenizer = AutoTokenizer.from_pretrained("book_genre_model")
model = AutoModelForSequenceClassification.from_pretrained("book_genre_model")

genre = {
    0 : 'crime',
    1 : 'fantasy',
    2 : 'history',
    3 : 'horror',
    4 : 'psychology',
    5 : 'romance',
    6 : 'science',
    7 : 'sports',
    8 : 'thriller',
    9 : 'travel'
}

class Item(BaseModel):
    text : str

def tokenisation(input) :
    output = tokenizer(input, return_tensors="pt") 
    # tokenizer retourne des tenseurs pytorch compatible avec le modele
    return output

def modelisation(output):
    final = model(**output)
    logits = final.logits
    result = torch.argmax(logits, dim=1)
    return result.item()

@router.post("/")
async def predict(item : Item):
    input = item.text
    encoded_input = tokenisation(input)
    modelisation_done = modelisation(encoded_input)
    return {"genre": genre[modelisation_done]}


