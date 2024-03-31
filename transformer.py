import torch.nn as nn
import torch
import math


# 3.2
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, h, d_k, d_v):  # d_k represent the dimension of key
        super(MultiHeadAttention).__init__()
        self.d_model = d_model
        self.h = h
        self.d_k = d_k
        self.d_v = d_v

        self.W_q = nn.Linear(d_model, d_k)
        self.W_k = nn.Linear(d_model, d_k)
        self.W_v = nn.Linear(d_model, d_v)
        self.W_O = nn.Linear(h * d_v, d_model)

    @staticmethod
    def scaled_dot_attention(q, k, v):
        softmax = nn.Softmax()
        scaled_dot = torch.dot(q, k.T) / torch.sqrt(k.shape[0])
        soft = softmax(scaled_dot)
        return torch.mm(soft, v)

    def forward(self, q, k, v):
        query = self.W_q(q)
        key = self.W_q(k)
        value = self.W_v(v)
        head_lists = []
        for i in range(self.h): head_lists.append(MultiHeadAttention.scaled_dot_attention(query, key, value))
        # concatenate the heads
        heads = torch.cat(head_lists)
        return self.W_O(heads)


# 3.3
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(FeedForward).__init__()
        self.d_model = d_model
        self.d_ff = d_ff

        self.relu = nn.ReLU()
        self.W1 = nn.Linear(d_model, d_ff)
        self.W2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.W1(x)
        x = self.relu(x)
        x = self.W2(x)
        return x


# 3.5
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, vocab_size):
        super(PositionalEncoding).__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embeddings = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        embed = self.embeddings(x)

        pos = torch.arange(0, self.d_model, dtype=torch.float).view(1, -1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2, dtype=torch.float) * (-math.log(10000.0) / self.d_model))
        pos_enc = torch.zeros(self.d_model, self.vocab_size)
        pos_enc[:, 0::2] = torch.sin(pos * div_term)
        pos_enc[:, 1::2] = torch.cos(pos * div_term)

        return embed + pos_enc


# 3.1
class Encoder(nn.Module):
    def __init__(self, layer_no, d_model, vocab_size, h, d_k, d_v, d_ff):
        super(Encoder).__init__()
        self.layer_no = layer_no
        self.pos_enc = PositionalEncoding(d_model, vocab_size)
        self.multiheadattention = MultiHeadAttention(d_model, h, d_k, d_v)
        self.feedforward = FeedForward(d_model, d_ff)
        self.norm = nn.LayerNorm(d_model)

    @staticmethod
    def block(multihead, norm, feedforward, pos_enc, embeds):
        pos_enc_ = pos_enc(embeds)
        out = multihead(pos_enc_, pos_enc_, pos_enc_)
        out = norm(pos_enc_, out)
        feed_out = feedforward(out)
        out = norm(out, feed_out)

        return out

    def forward(self, embeds):
        output = embeds
        for i in range(self.layer_no):
            output = Encoder.block(self.multiheadattention, self.norm, self.feedforward, self.pos_enc, output)
        return output


# 3.1
class Decoder(nn.Module):
    def __init__(self, layer_no, d_model, vocab_size, h, d_k, d_v, d_ff):
        super(Decoder).__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.h = h
        self.d_k = d_k
        self.d_v = d_v
        self.d_ff = d_ff

        self.layer_no = layer_no
        self.norm = nn.LayerNorm(d_model)
        self.linear = nn.Linear(d_model, vocab_size)
        self.softmax = nn.Softmax()

    @staticmethod
    def block(multihead, norm, feedforward, pos_enc, encoder, embeds):
        pos_enc_ = pos_enc(embeds)
        out = multihead(pos_enc_, pos_enc_, pos_enc_)
        out = norm(pos_enc_, out)
        enc_out = encoder(embeds)
        multihead_out = multihead(enc_out, enc_out, out)
        out = norm(out, multihead_out)
        feed_out = feedforward(out)
        out = norm(out, feed_out)

        return out

    def forward(self, embeds):
        self.pos_enc = PositionalEncoding(self.d_model, self.vocab_size)

        self.multiheadattention = MultiHeadAttention(self.d_model, self.h, self.d_k, self.d_v)

        self.feedforward = FeedForward(self.d_model, self.d_ff)

        self.encoder = Encoder(self.layer_no, self.d_model, self.vocab_size, self.h, self.d_k, self.d_v, self.d_ff)

        output = embeds
        for i in range(self.layer_no):
            output = Decoder.block(self.multiheadattention,
                                   self.norm,
                                   self.feedforward,
                                   self.pos_enc,
                                   self.encoder,
                                   output)
        output = self.linear(output)
        output = self.softmax(output)
        return output


class Transformer(nn.Module):
    def __init__(self, layer_no, d_model, vocab_size, h, d_k, d_v, d_ff):
        super(Transformer).__init__()

        self.layer_no = layer_no
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.h = h
        self.d_k = d_k
        self.d_v = d_v
        self.d_ff = d_ff

    def forward(self, embeds):
        self.encoder = Encoder(self.layer_no, self.d_model, self.vocab_size, self.h, self.d_k, self.d_v, self.d_ff)
        self.decoder = Decoder(self.layer_no, self.d_model, self.vocab_size, self.h, self.d_k, self.d_v, self.d_ff)
        output = self.encoder(embeds)
        output = self.decoder(output)

        return output

