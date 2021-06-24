import torch
import torch.nn as nn
import torch.nn.functional as F

from parameters import *


class AttnDecoder(nn.Module):
    def __init__(self, embed_dim=d_k, dropout_p=0.1, num_heads=num_head):
        super(AttnDecoder, self).__init__()
        self.embed_dim = embed_dim
        self.dropout_p = dropout_p
        self.num_heads = num_heads

        self.multihead_attn = nn.MultiheadAttention(self.embed_dim, self.num_heads, self.dropout_p)
        self.ffn1 = nn.Conv1d(d_embedding, d_ff, 1)
        self.ffn2 = nn.Conv1d(d_ff, d_embedding, 1)

    def forward(self, query, key, value):
        output, attn_output_weights = self.multihead_attn(query, key, value)
        residual = output
        output = F.relu(self.ffn1(output), inplace=True)
        output = self.ffn2(output)

        output = output + residual
        return output
