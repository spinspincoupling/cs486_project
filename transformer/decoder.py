import torch
import torch.nn as nn
import torch.nn.functional as F

from parameters import *


class AttnDecoder(nn.Module):
    def __init__(self, embed_dim=d_embedding, dropout_p=0.1, num_heads=num_head):
        super(AttnDecoder, self).__init__()
        self.embed_dim = embed_dim
        self.dropout_p = dropout_p
        self.num_heads = num_heads
        # self.query = nn.Linear(d_embedding, embed_dim)
        # self.key = nn.Linear(d_embedding, embed_dim)
        # self.val = nn.Linear(d_embedding, embed_dim)

        self.multihead_attn = nn.MultiheadAttention(self.embed_dim, self.num_heads, self.dropout_p)
        self.ffn1 = nn.Conv1d(d_embedding, d_ff, 1)
        self.ffn2 = nn.Conv1d(d_ff, d_embedding, 1)

    def forward(self, input):
        # output, attn_output_weights = self.multihead_attn(query, key, value)
        # q = self.query(input)
        # k = self.key(input)
        # v = self.val(input)
        # print("q: " + str(q.size()))
        # print("k: " + str(k.size()))
        # print("v: " + str(v.size()))
        q = input
        k = input
        v = input
        output, attn_output_weights = self.multihead_attn(q, k, v)
        # [1, 103, 512] to [1, 512, 103] so it fits pytorch conv1d
        output = output.transpose(1, 2)
        residual = output
        output = F.relu(self.ffn1(output), inplace=True)
        output = self.ffn2(output)
        output = output + residual
        output = output.transpose(1, 2)
        return output
