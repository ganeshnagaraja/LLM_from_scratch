import torch
import torch.nn as nn

class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            nn.GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"])
        )

    def forward(self, x):
        return self.layers(x)
    
# cfg = {
#     "vocab_size": 50257,
#     "context_length": 1024,
#     "emb_dim": 768,
#     "n_heads": 12,
#     "drop_rate": 0.1,
#     "n_layers": 12,
#     "qkv_bias": False
# }

# d_in = 768
# batch_example = torch.randn(2, 3, d_in)
# print(batch_example.shape)
# feed_forward = FeedForward(cfg)
# out = feed_forward(batch_example)
# print(out.shape)
