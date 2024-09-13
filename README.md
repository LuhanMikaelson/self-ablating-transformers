# self-ablating-transformers
A self-modeling transformer with an auxiliary output head that is an ablation mask for itself, used in a second forward pass.

See https://www.notion.so/apartresearch/Final-Self-Ablating-Transformer-4303c123a0ba4346bd7be95adecf6abf for detailed description (currently private to Apart lab fellows).

# Current status

The project's initial implementation used GPTNeo to match Ronen Eldan's pretrained models (e.g. https://huggingface.co/roneneldan/TinyStories-1M)

Activation function is replaced by NewGELUActivation

The initial model training stats matched other implementations 

Upon further inspection, evidence of data leakage was found. 

Currently refining the architecture design
