import math

class GPTNeoWithSelfAblationConfig:
    def __init__(
        self,
        vocab_size=50257,
        hidden_size=64,
        mlp_hidden_size=None,
        num_layers=8,
        num_heads=16,
        max_position_embeddings=2048,
        window_size=256,
        attention_layers=None,
        k_attention=32,
        k_neurons=32,
        temperature_attention=0.1,
        temperature_neurons=0.1,
        beta=0.9,
        reconstruction_coeff=0.1,
        top_k_epsilon=1e-12
    ):
        self.top_k_epsilon = top_k_epsilon
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.mlp_hidden_size = 4 * self.hidden_size if mlp_hidden_size is None else mlp_hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.max_position_embeddings = max_position_embeddings
        self.window_size = window_size
        self.attention_layers = ["global"] * num_layers if attention_layers is None else attention_layers
        
        # Ablation-specific parameters
        self.k_attention = k_attention
        self.k_neurons = k_neurons
        self.temperature_attention = temperature_attention
        self.temperature_neurons = temperature_neurons
        
        # Loss calculation parameters
        self.beta = beta
        self.reconstruction_coeff = reconstruction_coeff

    def __repr__(self):
        return f"GPTNeoWithSelfAblationConfig(vocab_size={self.vocab_size}, " \
               f"hidden_size={self.hidden_size}, mlp_hidden_size={self.mlp_hidden_size}, " \
               f"num_layers={self.num_layers}, num_heads={self.num_heads}, " \
               f"max_position_embeddings={self.max_position_embeddings}, " \
               f"window_size={self.window_size}, k_attention={self.k_attention}, " \
               f"k_neurons={self.k_neurons}, temperature_attention={self.temperature_attention}, " \
               f"temperature_neurons={self.temperature_neurons}, beta={self.beta}, " \
               f"reconstruction_coeff={self.reconstruction_coeff})"