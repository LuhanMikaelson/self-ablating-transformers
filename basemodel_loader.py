# basemodel_loader.py
import torch
import os
import math
import torch.nn as nn
import yaml
from pathlib import Path
from model_config import ModelConfig

class BaseAttention(nn.Module):
    def __init__(self, config: ModelConfig, layer_id: int):
        super().__init__()
        self.is_local = (config.attention_layers[layer_id] == "local")
        self.num_heads = config.num_heads
        self.hidden_size = config.hidden_size
        self.head_dim = config.hidden_size // config.num_heads
        self.config = config

        self.attention = nn.ModuleDict({
            'k_proj': nn.Linear(config.hidden_size, config.hidden_size, bias=False),
            'v_proj': nn.Linear(config.hidden_size, config.hidden_size, bias=False),
            'q_proj': nn.Linear(config.hidden_size, config.hidden_size, bias=False),
            'out_proj': nn.Linear(config.hidden_size, config.hidden_size)
        })

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        q = self.attention.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.attention.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.attention.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-1, -2))

        if self.is_local:
            local_mask = torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device)
            local_mask = torch.triu(local_mask, diagonal=1) | torch.tril(local_mask, diagonal=-self.config.window_size)
            scores = scores.masked_fill(local_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        else:
            causal_mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device), diagonal=1)
            scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        attn = torch.nn.functional.softmax(scores, dim=-1)
        context = torch.matmul(attn, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        
        return self.attention.out_proj(context)

class BaseMLP(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.hidden_size, config.mlp_hidden_size)
        self.c_proj = nn.Linear(config.mlp_hidden_size, config.hidden_size)
        self.act = nn.GELU()

    def forward(self, x):
        hidden_states = self.c_fc(x)
        hidden_states = self.act(hidden_states)
        return self.c_proj(hidden_states)

class BaseBlock(nn.Module):
    def __init__(self, config: ModelConfig, layer_id: int):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.hidden_size, eps=1e-5)
        self.attn = BaseAttention(config, layer_id)
        self.ln_2 = nn.LayerNorm(config.hidden_size, eps=1e-5)
        self.mlp = BaseMLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class BaseGPTNeo(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        self.transformer = nn.ModuleDict({
            'wte': nn.Embedding(config.vocab_size, config.hidden_size),
            'wpe': nn.Embedding(config.max_position_embeddings, config.hidden_size),
            'drop': nn.Dropout(0.0),
            'h': nn.ModuleList([BaseBlock(config, i) for i in range(config.num_layers)]),
            'ln_f': nn.LayerNorm(config.hidden_size, eps=1e-5)
        })
        
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # Tie weights
        self.transformer.wte.weight = self.lm_head.weight
        
        # Initialize weights
        self.apply(self._init_weights)
        # Apply special scaled init to the residual projections
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.num_layers))

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(self, input_ids, targets=None):
        device = input_ids.device
        b, t = input_ids.shape
        
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0)
        tok_emb = self.transformer.wte(input_ids)
        pos_emb = self.transformer.wpe(pos)
        
        x = self.transformer.drop(tok_emb + pos_emb)
        
        for block in self.transformer.h:
            x = block(x)
            
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        
        loss = None
        if targets is not None:
            loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            
        return {
            "logits": logits,
            "loss": loss
        }

    def run_with_cache(self, input_ids, targets=None):
        """Forward pass with activation caching"""
        device = input_ids.device
        b, t = input_ids.shape
        
        cache = {}
        
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0)
        tok_emb = self.transformer.wte(input_ids)
        pos_emb = self.transformer.wpe(pos)
        
        x = self.transformer.drop(tok_emb + pos_emb)
        
        for i, block in enumerate(self.transformer.h):
            # Cache pre-attention layernorm
            ln1_out = block.ln_1(x)
            cache[f'transformer.h.{i}.ln_1'] = ln1_out
            
            # Process attention
            attn_out = block.attn(ln1_out)
            cache[f'transformer.h.{i}.attn'] = attn_out
            x = x + attn_out
            
            # Cache pre-MLP layernorm
            ln2_out = block.ln_2(x)
            cache[f'transformer.h.{i}.ln_2'] = ln2_out
            
            # Cache MLP intermediate activations
            mlp_fc = block.mlp.c_fc(ln2_out)
            mlp_mid = block.mlp.act(mlp_fc)
            cache[f'transformer.h.{i}.mlp'] = mlp_mid
            
            # Complete MLP
            mlp_out = block.mlp.c_proj(mlp_mid)
            x = x + mlp_out
        
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        
        loss = None
        if targets is not None:
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), 
                targets.view(-1)
            )
            
        return {
            "logits": logits,
            "loss": loss
        }, cache

    def __call__(self, *args, **kwargs):
        """Override call to maintain compatibility with both forward and run_with_cache"""
        if kwargs.get('return_cache', False):
            kwargs.pop('return_cache')
            return self.run_with_cache(*args, **kwargs)
        return self.forward(*args, **kwargs)

    def generate(self, input_ids, max_new_tokens, temperature=1.0):
        self.eval()
        for _ in range(max_new_tokens):
            # crop context if needed
            idx_cond = input_ids if input_ids.size(1) <= self.config.max_position_embeddings else input_ids[:, -self.config.max_position_embeddings:]
            # forward pass
            with torch.no_grad():
                logits = self(idx_cond)["logits"]
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # apply softmax to convert logits to (normalized) probabilities
            probs = torch.nn.functional.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            input_ids = torch.cat((input_ids, idx_next), dim=1)
        
        return input_ids[0].tolist()