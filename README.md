# self-ablating-transformers
A self-modeling transformer with an auxiliary output head that is an ablation mask for itself, used in a second forward pass.

See https://www.notion.so/apartresearch/Final-Self-Ablating-Transformer-4303c123a0ba4346bd7be95adecf6abf for detailed description (currently private to my Apart fellows).

# Current status

I started out by just implementing GPTNeo since that is what Ronen Eldan's pretrained models (e.g. https://huggingface.co/roneneldan/TinyStories-1M) use. After fixing up a few discrepancies (e.g. use of NewGELUActivation rather than the other GELU, and particularly *not* scaling the attention scores by sqrt(num_heads)), I got the pretrained models to achieve identical loss and identical results in my implementation as in the reference implementation.

Then I added some training stuff, mostly cribbed from NanoGPT, and saw that I could train models that are kind of on a par with Ronen Eldan's in only an hour or two. In particular the memmap and memory pinning stuff speeds up the training by some huge factor so I'm glad I got that working.

Next the fun part... implementing the actual self-ablation stuff. As you can see, the main logic is just in GPTNeo.forward() and basically there's just two auxiliary output heads, which are sigmoid-ed and reshaped to become the ablation masks for the second pass.

The loss hyperparameters are there at the top in GPTNeoConfig... we definitely need to do searches over these to see what kinds of behaviors this exhibits.

So far everything seems like it's working (in that the training seems to be converging, and the ablated loss remains kind of near the base loss even when the mask densities are low (like a few percent). This is exciting but I'd be surprised if there weren't still serious bugs in my implementation.

Time for all the other fun stuff: hyperparameter searches, longer training runs, visualization, datasets to test specific capabilities...!
