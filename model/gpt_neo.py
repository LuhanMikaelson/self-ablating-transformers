import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .config import GPTNeoWithSelfAblationConfig
from .block import GPTNeoBlockWithSelfAblation

class GPTNeoWithSelfAblation(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.hidden_size),
            wpe = nn.Embedding(config.max_position_embeddings, config.hidden_size),
            h = nn.ModuleList([GPTNeoBlockWithSelfAblation(config, i) for i in range(config.num_layers)]),
            ln_f = nn.LayerNorm(config.hidden_size, eps=1e-5)
        ))
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.attention_ablations_head = nn.Linear(config.hidden_size, math.prod(self.get_attention_ablations_shape(1, 1)))
        self.neuron_ablations_head = nn.Linear(config.hidden_size, math.prod(self.get_neuron_ablations_shape(1, 1)))

        # tie weights
        self.transformer.wte.weight = self.lm_head.weight

    def get_attention_ablations_shape(self, batch_size, block_size):
        return torch.Size([batch_size, block_size, self.config.num_layers, self.config.hidden_size])

    def get_neuron_ablations_shape(self, batch_size, block_size):
        return torch.Size([batch_size, block_size, self.config.num_layers, self.config.mlp_hidden_size])

    def forward(self, input_ids, targets=None, attention_ablations=None, neuron_ablations=None):
        second_pass = attention_ablations is not None or neuron_ablations is not None

        device = input_ids.device
        b, t = input_ids.shape
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0)

        tok_emb = self.transformer.wte(input_ids)
        pos_emb = self.transformer.wpe(pos)

        x_clean = tok_emb + pos_emb
        if second_pass:
            x_ablated = x_clean

        for i, block in enumerate(self.transformer.h):
            x_clean = block(x_clean, x_clean)
            if second_pass:
                x_ablated = block(x_ablated, x_clean, attention_ablations[:,:,i,:], neuron_ablations[:,:,i,:])

        x = x_ablated if second_pass else x_clean

        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        L_base = None
        if targets is not None:
            logits_view = logits.view(-1, logits.size(-1))
            targets_view = targets.view(-1)
            L_base = F.cross_entropy(logits_view, targets_view)

        if second_pass:
            return {"logits": logits, "L_base": L_base}
        else:
            output_attention_ablations = self.attention_ablations_head(x)
            output_attention_ablations = torch.sigmoid(output_attention_ablations)
            output_attention_ablations = output_attention_ablations.reshape(self.get_attention_ablations_shape(b, t))
            output_neuron_ablations = self.neuron_ablations_head(x)
            output_neuron_ablations = torch.sigmoid(output_neuron_ablations)
            output_neuron_ablations = output_neuron_ablations.reshape(self.get_neuron_ablations_shape(b, t))
            second_pass_output = self.forward(input_ids, targets, output_attention_ablations, output_neuron_ablations)
            logits_ablated = second_pass_output["logits"]
            L_total = L_ablated = L_attention_density = L_neuron_density = None
            if targets is not None:
                L_ablated = second_pass_output["L_base"]
                L_attention_density = output_attention_ablations.mean()
                L_neuron_density = output_neuron_ablations.mean()
                L_total = sum([self.config.loss_coeff_base * L_base,
                               self.config.loss_coeff_ablated * L_ablated,
                               self.config.loss_coeff_attention_density * L_attention_density,
                               self.config.loss_coeff_neuron_density * L_neuron_density])
            return {"logits": logits,
                    "logits_ablated": logits_ablated,
                    "L_base": L_base,
                    "L_ablated": L_ablated,
                    "loss": L_total,
                    "attention_ablations": output_attention_ablations,
                    "neuron_ablations": output_neuron_ablations,
                    "attention_ablation_mask_density": L_attention_density,
                    "neuron_ablation_mask_density": L_neuron_density}

    def generate(self, input_ids, max_new_tokens, temperature=1.0, ablated=False):
        self.eval()
        device = next(self.parameters()).device

        x = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0) if isinstance(input_ids, list) else input_ids.to(device)

        for _ in range(max_new_tokens):
            x_crop = x[:, -self.config.max_position_embeddings:]

            ret = self(x_crop)
            logits = ret["logits_ablated"] if ablated else ret["logits"]

            logits = logits[:, -1, :] / temperature

            probs = F.softmax(logits, dim=-1)

            next_token = torch.multinomial(probs, num_samples=1)

            x = torch.cat((x, next_token), dim=1)

        return x[0].tolist()

    def generate_hard_cutoff_ablated(self, input_ids, max_new_tokens, temperature=1.0, attention_threshold=0.5, neuron_threshold=0.5):
        self.eval()
        device = next(self.parameters()).device

        x = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0) if isinstance(input_ids, list) else input_ids.to(device)

        att_densities = []
        neu_densities = []
        att_nonzeros = []
        neu_nonzeros = []

        for _ in range(max_new_tokens):
            x_crop = x[:, -self.config.max_position_embeddings:]

            ret = self(x_crop)
            attention_ablations, neuron_ablations = ret["attention_ablations"], ret["neuron_ablations"]
            attention_ablations_clipped = torch.where(attention_ablations > attention_threshold, attention_ablations, torch.zeros_like(attention_ablations))
            neuron_ablations_clipped = torch.where(neuron_ablations > neuron_threshold, neuron_ablations, torch.zeros_like(neuron_ablations))
            ret2 = self(x_crop, attention_ablations=attention_ablations_clipped, neuron_ablations=neuron_ablations_clipped)
            att_which_clipped = torch.where(attention_ablations > attention_threshold, torch.ones_like(attention_ablations), torch.zeros_like(attention_ablations))
            neu_which_clipped = torch.where(neuron_ablations > neuron_threshold, torch.ones_like(neuron_ablations), torch.zeros_like(neuron_ablations))
            att_densities.append(att_which_clipped[:,-1].mean().item())
            neu_densities.append(neu_which_clipped[:,-1].mean().item())
            att_nonzeros.append(att_which_clipped[:,-1].sum().item())
            neu_nonzeros.append(neu_which_clipped[:,-1].sum().item())
            logits = ret2["logits"]

            logits = logits[:, -1, :] / temperature

            probs = F.softmax(logits, dim=-1)

            next_token = torch.multinomial(probs, num_samples=1)

            x = torch.cat((x, next_token), dim=1)

        att_mean = torch.tensor(att_densities).mean()
        neu_mean = torch.tensor(neu_densities).mean()
        print(f"attention nonzero density {att_mean}, neuron nonzero density {neu_mean}")
        print(f"attention nonzero elements per token {att_nonzeros}, neuron nonzero elements per token {neu_nonzeros}")

        return x[0].tolist()