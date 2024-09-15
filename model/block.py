import torch
import torch.nn as nn
from .attention import AttentionWithSelfAblation
from .mlp import MLPWithSelfAblation

def soft_top_k(x, k, temperature=1.0, eps=None):
    if eps is None:
        eps = x.new_tensor(1e-12) # Default epsilon value if not provided

    # Sort the input
    sorted_x, indices = torch.sort(x, descending=True)

    # Calculate the threshold as the midpoint between k-th and (k+1)-th largest values
    assert k < x.shape[-1]
    threshold = ((sorted_x[..., k-1] + sorted_x[..., k]) / 2).unsqueeze(-1)

    # Calculate temperature
    temperature = (sorted_x[..., k-1] - sorted_x[..., k]).unsqueeze(-1) * temperature
    assert torch.all(temperature >= 0)

    # Compute the difference from the threshold
    diff = (x - threshold) / (temperature + eps)

    # Apply sigmoid to get soft selection weights
    weights = torch.sigmoid(diff)

    # Normalize weights to sum to k
    weights = weights * (k / weights.sum(-1).unsqueeze(-1))

    return weights

class GPTNeoBlockWithSelfAblation(nn.Module):
    def __init__(self, config, layer_id):
        super().__init__()
        self.config = config
        self.ln_1 = nn.LayerNorm(config.hidden_size, eps=1e-5)
        self.attn = AttentionWithSelfAblation(config, layer_id)
        self.ln_2 = nn.LayerNorm(config.hidden_size, eps=1e-5)
        self.mlp = MLPWithSelfAblation(config)
        
        # Ablation heads
        self.attention_ablation_head = nn.Linear(config.hidden_size, config.hidden_size)
        self.neuron_ablation_head = nn.Linear(config.hidden_size, config.mlp_hidden_size)

    def forward(self, x_ablated, x_clean):
        
        # Generate ablation masks before passing through layers
        attn_ablation = self.attention_ablation_head(x_clean)
        attn_ablation = soft_top_k(attn_ablation, self.config.k_attention, self.config.temperature_attention, eps=self.config.top_k_epsilon)
        
        neuron_ablation = self.neuron_ablation_head(x_clean)
        neuron_ablation = soft_top_k(neuron_ablation, self.config.k_neurons, self.config.temperature_neurons, eps=self.config.top_k_epsilon)

        # Process x_clean
        attn_output_clean = self.attn(self.ln_1(x_clean), self.ln_1(x_clean))
        x_clean = x_clean + attn_output_clean
        x_clean = x_clean + self.mlp(self.ln_2(x_clean))

        # Process x_ablated with ablations
        attn_output_ablated = self.attn(self.ln_1(x_ablated), self.ln_1(x_clean), attn_ablation)
        x_ablated = x_ablated + attn_output_ablated
        x_ablated = x_ablated + self.mlp(self.ln_2(x_ablated), neuron_ablation)

        return x_ablated, x_clean