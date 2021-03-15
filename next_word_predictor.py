import torch
from transformers import BertTokenizer, BertForMaskedLM
import string


class Model:
    def __init__(self):
        print("Called")
        self.tokenizer = BertTokenizer.from_pretrained('prajjwal1/bert-small')
        self.model = BertForMaskedLM.from_pretrained('prajjwal1/bert-small')

    def encode_input(self, input_text, add_special_tokens=True):
        input_text = input_text + ' ' + self.tokenizer.mask_token
        if self.tokenizer.mask_token == input_text.split()[-1]:
            input_text += ' .'
        input_ids = torch.tensor([self.tokenizer.encode(input_text, add_special_tokens=add_special_tokens)])
        mask_idx = torch.where(input_ids == self.tokenizer.mask_token_id)[1].tolist()[0]

        return input_ids, mask_idx

    def decode(self, pred_idx, top_clean):
        ignore_tokens = string.punctuation + '[PAD]'
        tokens = []
        for w in pred_idx:
            token = ''.join(self.tokenizer.decode(w).split())
            if token not in ignore_tokens:
                tokens.append(token.replace('##', ''))
        return tokens[:top_clean]

    def predict(self, input_text, top_k=5, top_clean=5):
        input_ids, mask_idx = self.encode_input(input_text)
        with torch.no_grad():
            predictions = self.model(input_ids)[0]
        decoded_predictions = self.decode(predictions[0, mask_idx, :].topk(top_k).indices.tolist(), top_clean)
        return decoded_predictions


bert_model = Model()


def get_model():
    return bert_model










