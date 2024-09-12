import torch
import torch.nn as nn
import math

class NewGELUActivation(nn.Module):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))

class MLPWithSelfAblation(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.hidden_size, config.mlp_hidden_size)
        self.c_proj = nn.Linear(config.mlp_hidden_size, config.hidden_size)
        self.act = NewGELUActivation()

    def forward(self, x, ablation_mask=None):
        activations = self.act(self.c_fc(x))

        if ablation_mask is not None:
            activations = activations * ablation_mask

        return self.c_proj(activations)
