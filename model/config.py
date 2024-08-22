import math

class GPTNeoConfig:
    def __init__(self, vocab_size=50257, hidden_size=64, mlp_hidden_size=None, num_layers=8,
                 num_heads=16, max_position_embeddings=2048, window_size=256, attention_layers=None,
                 loss_coeff_base=1.0, loss_coeff_ablated=0.1,
                 loss_coeff_attention_density=0.1, loss_coeff_neuron_density=0.1):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        if mlp_hidden_size == None:
            self.mlp_hidden_size = 4 * self.hidden_size
        else:
            self.mlp_hidden_size = mlp_hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.max_position_embeddings = max_position_embeddings
        self.window_size = window_size
        if attention_layers == None:
            self.attention_layers = ["global"] * num_layers
        else:
            self.attention_layers = attention_layers
        self.loss_coeff_base = loss_coeff_base
        self.loss_coeff_ablated = loss_coeff_ablated
        self.loss_coeff_attention_density = loss_coeff_attention_density
        self.loss_coeff_neuron_density = loss_coeff_neuron_density
