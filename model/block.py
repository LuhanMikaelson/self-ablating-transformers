import torch.nn as nn
from .attention import AttentionWithSelfAblation
from .mlp import MLPWithSelfAblation

class GPTNeoBlockWithSelfAblation(nn.Module):
    def __init__(self, config, layer_id):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.hidden_size, eps=1e-5)
        self.attn = AttentionWithSelfAblation(config, layer_id)
        self.ln_2 = nn.LayerNorm(config.hidden_size, eps=1e-5)
        self.mlp = MLPWithSelfAblation(config)

    def forward(self, x_ablated, x_clean, attention_ablations=None, neuron_ablations=None):
        x_ablated = x_ablated + self.attn(self.ln_1(x_ablated), self.ln_2(x_clean), attention_ablations)
        x_ablated = x_ablated + self.mlp(self.ln_2(x_ablated), neuron_ablations)
        return x_ablated
