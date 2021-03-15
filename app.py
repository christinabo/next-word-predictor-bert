from typing import Optional
from fastapi import FastAPI, Depends
import uvicorn
from next_word_predictor import Model, get_model, predict

app = FastAPI()

tokenizer = None
model = None


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/predict/{q}")
def read_item(q: str = None,  bert_model: Model = Depends(get_model)):
    print(q)
    preditions = predict(q, bert_model.model, bert_model.tokenizer)
    return {"q": q, "predictions": preditions}

