import torch

class Tokenizer():
    def __init__(self):
        self.tokens = []

    def tokenize(self, text):
        for txt in text: self.tokens+=txt.split()
        for each in self.tokens: each = each.lower()
        vocabs = {word:id for id, word in enumerate(self.tokens)}
        token_ids = [vocabs.get(token, 0) for token in self.tokens]  # Handle unknown words with 0 (padding)
        input_ids = torch.tensor(token_ids, dtype=torch.long)  # Long tensor for integer IDs
        return input_ids

