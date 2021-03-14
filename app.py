from typing import Optional
from fastapi import FastAPI
from next_word_predictor import load_models

app = FastAPI()

tokenizer = None
model = None


# @app.on_event("startup")
# async def load_models():
#     tokenizer, model = load_models()


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/items/{input_text}")
def read_item(input_text: str, q: Optional[str] = None):

    return {"item_id": item_id, "q": q}