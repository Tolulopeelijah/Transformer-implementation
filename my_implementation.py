import torch
import torch.nn as nn
import math

class InputEmbeddings(nn.Module):
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.d_model = d_model
        self.voca_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x)#) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, seq_len, p): #p rep dropout
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(p)

        pos_enc = torch.zeros(seq_len, d_model)
        pos = torch.arange(0, seq_len, dtype = torch.float).unsqueeze(1)
        val = pos / (10000 ** (torch.arange(0, d_model, 2).float()/d_model * math.pi))
        pos_enc[:, 0::2] = torch.sin(pos * val)
        pos_enc[:, 1::2] = torch.cos(pos * val)

        pos_enc = pos_enc.unsqueeze(0) # shape: (1, seq_len, d_model)
        self.register_buffer('pos_enc', pos_enc)

    def forward(self, x):
        x += (self.pos_enc[:, :x.shape[1], :]).requires_grad(False)
        x = self.dropout(x)
        return x

class LayerNormalization(nn.Module):
    def __init__(self, eps = 10 ** -6):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        mean = x.mean(dim = -1, keepdim = True)
        std = x.std(dim = -1, keepdim = True)
        return ((x-mean) / (std + self.eps) * self.alpha) + self.bias

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, p):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p)

    def forward(self, x):
        x = self.linear1(x)
        x = nn.ReLU(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, h, p):
        super().__init__()
        self.d_model = d_model
        self.h = h
        assert d_model % h == 0, "d_model cannot be divided by h"

        self.d_k = d_model // h
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(p)

    @staticmethod
    def self_attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]

        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            attention_scores.masked_fill_(mask == 0, -1e9)
        attention_scores = attention_scores.softmax(dim = -1) # (batch, h, seq_len, seq_len)
        if dropout is not None:
            attention_scores = dropout(attention_scores)

        return (attention_scores @ value), attention_scores

    def forward(self, q, k, v, mask):
        query = self.w_q(q) #(batch, seq_len, d_model)
        key = self.w_k(k) #(batch, seq_len, d_model)
        value = self.w_v(v) #(batch, seq_len, d_model)

        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1,2)
        key  = query.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1,2)
        value = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1,2)

        x, self.attention_scores = MultiHeadAttention.self_attention(query, key, value, mask, self.dropout)

        # (batch, h, seq_len, d_k) to (batch, seq_len, h, d_k) to (batch, seq_len, d_model)
        x = x.transpose(1,2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

        # (batch, seq_len, d_model) to (batch, seq_len, d_model))
        return self.w_o(x)

class ResidualConnection(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.dropout = nn.Dropout(p)
        self.norm = LayerNormalization()

    def forward(self, x, sublayer):
        x = x + self.dropout(sublayer(self.norm(x)))
        return x

class EncoderBlock(nn.Module):
    def __init__(self, self_attention_block: MultiHeadAttention, feed_forward_block: FeedForward, p):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(p) for _ in range(2)])

    def forward(self, x, src_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x,x,x, src_mask))
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x

class Encoder(nn.Module):
    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class DecoderBlock(nn.Module):
    def __init__(self, self_attention_block: MultiHeadAttention, cross_attention_block: MultiHeadAttention, feed_forward_block: FeedForward, p):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.Module([ResidualConnection(p) for _ in range(3)])

    def forward(self, x, encoder_ouput, src_mask, tgt_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x,x,x, tgt_mask))
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x,encoder_ouput, encoder_ouput, src_mask))
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x

class Decoder(nn.Module):
    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)

class ProjectionLayer(nn.Module):
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        return torch.log_softmax(self.proj(x), dim = -1)

class Transformer(nn.Module):
    def __init(self, encoder: Encoder, decoder: Decoder, src_embed: InputEmbeddings, tgt_embed: InputEmbeddings, src_pos: PositionalEncoding, projection_layer: ProjectionLayer):
        super().__init__()