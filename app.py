from typing import Optional
from fastapi import FastAPI, Depends
import uvicorn
from next_word_predictor import Model, get_model

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/predict/{q}")
def read_item(q: str = None,  bert_model: Model = Depends(get_model)):
    print(q)
    predictions = bert_model.predict(q)
    return {"q": q, "predictions": predictions}

