import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from model.gpt_neo import GPTNeo
from model.config import GPTNeoConfig
from utils.data_preparation import prepare_data
import numpy as np

class BatchGenerator:
    def __init__(self, data_file, block_size, batch_size, device):
        self.data = np.memmap(data_file, dtype=np.uint16, mode='r')
        self.block_size = block_size
        self.batch_size = batch_size
        self.device = device

    def get_batch(self):
        ix = torch.randint(len(self.data) - self.block_size, (self.batch_size,))
        x = torch.stack([torch.from_numpy((self.data[i:i+self.block_size]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((self.data[i+1:i+1+self.block_size]).astype(np.int64)) for i in ix])
        return x.to(self.device), y.to(self.device)

class LossEstimator:
    def __init__(self, model, train_batch_gen, val_batch_gen, eval_iters):
        self.model = model
        self.train_batch_gen = train_batch_gen
        self.val_batch_gen = val_batch_gen
        self.eval_iters = eval_iters

    @torch.no_grad()
    def estimate_loss(self):
        out = {}
        self.model.eval()
        for split, batch_gen in [('train', self.train_batch_gen), ('val', self.val_batch_gen)]:
            losses = torch.zeros(self.eval_iters)
            L_bases = torch.zeros(self.eval_iters)
            L_ablateds = torch.zeros(self.eval_iters)
            attention_densities = torch.zeros(self.eval_iters)
            neuron_densities = torch.zeros(self.eval_iters)
            for k in range(self.eval_iters):
                X, Y = batch_gen.get_batch()
                with torch.no_grad():
                    ret = self.model(X, Y)
                    losses[k] = ret['loss'].item()
                    L_bases[k] = ret['L_base'].item()
                    L_ablateds[k] = ret['L_ablated'].item()
                    attention_densities[k] = ret['attention_ablation_mask_density'].item()
                    neuron_densities[k] = ret['neuron_ablation_mask_density'].item()
            out[split] = {
                'loss': losses.mean().item(),
                'L_base': L_bases.mean().item(),
                'L_ablated': L_ablateds.mean().item(),
                'attention_ablation_mask_density': attention_densities.mean().item(),
                'neuron_ablation_mask_density': neuron_densities.mean().item()
            }
        self.model.train()
        return out

def train_gptneo(model, config):
    train_batch_gen = BatchGenerator(config.train_file, config.block_size, config.batch_size, config.device)
    val_batch_gen = BatchGenerator(config.val_file, config.block_size, config.batch_size, config.device)
    loss_estimator = LossEstimator(model, train_batch_gen, val_batch_gen, config.eval_iters)
    model.to(config.device)
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.num_batches)
    best_val_loss = float('inf')
    
    for iteration in tqdm(range(config.num_batches)):
        model.train()
        # Get batch
        x, y = train_batch_gen.get_batch()
        # Forward pass
        loss = model(x, targets=y)["loss"]
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        if config.max_grad_norm:
            nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
        optimizer.step()
        scheduler.step()
        
        # Logging
        if (iteration + 1) % config.log_interval == 0:
            stats = loss_estimator.estimate_loss()
            print(f"Iteration {iteration}: train loss {stats['train']['loss']:.4f}, val loss {stats['val']['loss']:.4f}")
            print(f"train L_base {stats['train']['L_base']:.4f}, val L_base {stats['val']['L_base']:.4f}")
            print(f"train L_ablated {stats['train']['L_ablated']:.4f}, val L_ablated {stats['val']['L_ablated']:.4f}")
            print(f"train attention ablation mask density {stats['train']['attention_ablation_mask_density']:.4f}, val {stats['val']['attention_ablation_mask_density']:.4f}")
            print(f"train neuron ablation mask density {stats['train']['neuron_ablation_mask_density']:.4f}, val {stats['val']['neuron_ablation_mask_density']:.4f}")
            print(f"The current learning rate: {optimizer.param_groups[0]['lr']:.4f}")
            
            # Save best model
            if stats['val']['loss'] < best_val_loss:
                best_val_loss = stats['val']['loss']
                torch.save(model.state_dict(), config.save_path)
                print(f"New best model saved to {config.save_path}")
    
    print("Training completed!")

if __name__ == "__main__":
    # Set up configuration
    config = GPTNeoConfig(
        vocab_size=50257,  # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
        hidden_size=128,
        mlp_hidden_size=512,
        num_layers=8,
        num_heads=4,
        max_position_embeddings=2048,
        train_file='train.bin',
        val_file='val.bin',
        block_size=128,
        batch_size=64,
        learning_rate=3e-4,
        weight_decay=0.1,
        num_batches=100000,
        eval_iters=200,
        log_interval=100,
        max_grad_norm=1.0,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        save_path='best_model.pt'
    )

    # Initialize model
    model = GPTNeo(config)

    # Prepare data
    prepare_data(output_file=config.train_file)
    prepare_data(split='validation', output_file=config.val_file)

    # Train model
    train_gptneo(model, config)
