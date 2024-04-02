import dataset
from transformer import Transformer
text = open('data.txt', 'r')
text = text.readlines()
tokenizer = dataset.Tokenizer()
token_ids = tokenizer.tokenize(text)
model = Transformer(layer_no=6, d_model=512, vocab_size=20000, h=8, d_k=64, d_v=64, d_ff=2048)

# Pass the processed input tensor to the model (replace with your actual tensor)
output = model.forward(token_ids)  # Output will be a tensor of probabilities or logits