import yaml
from pathlib import Path
from typing import Dict, Optional, Union
from dataclasses import dataclass

@dataclass
class ModelConfig:
    """Base configuration class for all models"""
    vocab_size: int = 50257
    hidden_size: int = 128
    mlp_hidden_size: int = 512
    num_layers: int = 8
    num_heads: int = 16
    max_position_embeddings: int = 256
    window_size: int = 256
    attention_layers: Optional[list] = None
    model_type: str = "base"  # "base" or "ablated"
    
    # Ablation-specific parameters
    k_attention: Optional[int] = None
    k_neurons: Optional[int] = None
    temperature_attention: Optional[float] = None
    temperature_neurons: Optional[float] = None
    loss_coeff_base: float = 1.0
    loss_coeff_ablated: float = 0.0
    reconstruction_coeff: float = 0.0
    top_k_epsilon: float = 1e-12
    has_layer_by_layer_ablation_mask: bool = False
    has_overall_ablation_mask: bool = False
    reconstruction_loss_type: Optional[str] = None

    def to_dict(self) -> Dict:
        """Convert config to dictionary"""
        return {k: v for k, v in vars(self).items() if v is not None}

    @classmethod
    def from_dict(cls, config_dict: Dict) -> 'ModelConfig':
        """Create config from dictionary"""
        return cls(**{
            k: v for k, v in config_dict.items()
            if k in cls.__dataclass_fields__
        })

def load_config(model_path: Union[str, Path], config_dir: Optional[str] = 'configs') -> ModelConfig:
    """
    Load model configuration from a YAML file based on model name.
    Falls back to default config if no matching config file found.
    
    Args:
        model_path: Path to model checkpoint
        config_dir: Directory containing config files
        
    Returns:
        ModelConfig instance
    """
    model_path = Path(model_path)
    config_dir = Path(config_dir)
    
    # Try to find matching config file
    model_name = model_path.stem  # Get filename without extension
    config_path = config_dir / f"{model_name}.yaml"
    
    # Load config file if it exists
    if config_path.exists():
        with open(config_path) as f:
            config_dict = yaml.safe_load(f)
        return ModelConfig.from_dict(config_dict)
    
    # Fall back to default base model config
    if "base_model" in model_name:
        return ModelConfig(model_type="base")
    else:
        raise ValueError(
            f"No config file found for {model_name} at {config_path}\n"
            f"Please create a config file or use base model."
        )

def save_config(config: ModelConfig, save_path: Union[str, Path]):
    """Save configuration to YAML file"""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(save_path, 'w') as f:
        yaml.dump(config.to_dict(), f, default_flow_style=False)
