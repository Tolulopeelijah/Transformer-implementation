import torch

text = open('data.txt', 'r')
text = text.readlines()
tokens = []
for txt in text: tokens+=txt.split()
for each in tokens: each = each.lower()
vocabs = {word:id for id, word in enumerate(tokens)}
token_ids = [vocabs.get(token, 0) for token in tokens]  # Handle unknown words with 0 (padding)
# # Convert token IDs to a PyTorch tensor
input_ids = torch.tensor(token_ids, dtype=torch.long)  # Long tensor for integer IDs
