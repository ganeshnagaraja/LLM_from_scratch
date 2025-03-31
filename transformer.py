import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


import sys
sys.path.append("/Users/ganeshnagaraja/Desktop/DeepLearning/LLM/LLM_from_scratch/transformers")
import MultiHead_attention_final as mhaf
import feed_forward as ff

# Configuration of the transformer block
cfg = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 768,
    "n_heads": 12,
    "drop_rate": 0.1,
    "n_layers": 12,
    "qkv_bias": False
}

# Dimensionlaity of the transformer block is preserved. 
class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = mhaf.MultiHeadAttention(d_in = cfg["emb_dim"],
                                           d_out = cfg["emb_dim"],
                                           context_length = cfg["context_length"],
                                           dropout = cfg["drop_rate"],
                                           num_heads = cfg["n_heads"],
                                           )
        self.ff = ff.FeedForward(cfg)
        self.norm1 = nn.LayerNorm(cfg["emb_dim"])
        self.norm2 = nn.LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        # shortcut connection for attention block
        shortcut = x
        x = self.norm1(x)
        x = self.att(x) # shape of x is preserved [batch, num_tokens, emb_dim]
        x = self.drop_shortcut(x)
        x += shortcut

        # shortcut connection for feed forward block
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x += shortcut


        return x
    
# testing the transformer block with a random tensor
# torch.manual_seed(143)
# x = torch.randn(2, 3, cfg["emb_dim"])
# print(x.shape)
# transformer_block = TransformerBlock(cfg)
# out = transformer_block(x)
# print(out.shape)