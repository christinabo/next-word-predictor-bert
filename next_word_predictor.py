import torch
from transformers import BertTokenizer, BertForMaskedLM
import string


def load_models():
    tokenizer = BertTokenizer.from_pretrained('prajjwal1/bert-small')
    model = BertForMaskedLM.from_pretrained('prajjwal1/bert-small')
    return tokenizer, model


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
    return '\n'.join(tokens[:top_clean])


def predict(input_ids, model, tokenizer, top_k=5, top_clean=5):
    with torch.no_grad():
        predictions = model(input_ids)[0]
    bert = decode(tokenizer, predictions[0, mask_idx, :].topk(top_k).indices.tolist(), top_clean)
    return bert


if __name__ == '__main__':
    tokenizer, model = load_models()
    input_text = "The anarchism is relevant when"
    input_ids, mask_idx = encode_input(input_text, tokenizer)
    print(predict(input_ids, model, tokenizer))
