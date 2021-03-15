import torch
from transformers import BertTokenizer, BertForMaskedLM
import string


class Model:
    def __init__(self):
        print("Called")
        self.tokenizer = BertTokenizer.from_pretrained('prajjwal1/bert-small')
        self.model = BertForMaskedLM.from_pretrained('prajjwal1/bert-small')


bert_model = Model()


def get_model():
    return bert_model


def encode_input(input_text, tokenizer, add_special_tokens=True):
    input_text = input_text + ' ' + tokenizer.mask_token
    if tokenizer.mask_token == input_text.split()[-1]:
        input_text += ' .'
    input_ids = torch.tensor([tokenizer.encode(input_text, add_special_tokens=add_special_tokens)])
    mask_idx = torch.where(input_ids == tokenizer.mask_token_id)[1].tolist()[0]

    return input_ids, mask_idx


def decode(tokenizer, pred_idx, top_clean):
    ignore_tokens = string.punctuation + '[PAD]'
    tokens = []
    for w in pred_idx:
        token = ''.join(tokenizer.decode(w).split())
        if token not in ignore_tokens:
            tokens.append(token.replace('##', ''))
    return tokens[:top_clean]


def predict(input_text, model, tokenizer, top_k=5, top_clean=5):
    input_ids, mask_idx = encode_input(input_text, tokenizer)
    with torch.no_grad():
        predictions = model(input_ids)[0]
    bert = decode(tokenizer, predictions[0, mask_idx, :].topk(top_k).indices.tolist(), top_clean)
    return bert

