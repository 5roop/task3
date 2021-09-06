# task3

# Optimization of hyperparameters

I needed to optimize the hyperparameters for specific model types. I started with model type `electra`, model name `classla/bcms-bertic`. Unlike the last time I stopped the optimization process after 100 runs.

It had been noticed that often an instance of the optimization process crashed, but the optimizer would just run on. I could not stop this pathology, but after reading through the traceback and searching online for a bit I was able to prevent the crashes with the following magic spell:

```python
import torch
torch.cuda.empty_cache()
```
This improved the graphics card memory allocation errors, but `wandb`, `runs`, `output` and `cache_dir` directories still needed to be purged manually to prevent the disk from filling up.

After succesful sweeps had ben run, the wandb website was inspected and the best performing hyperparameters were transcribed into the table below. The number of optimizer runs was capped at 50

Found optimal parameters for the models are as follows:

|Model name | Model type | epochs | batch size | learning rate|
|---|---|---|---|---|---|
|classla/bcms-bertic | electra | 12 | 74 | 1e-5 |
|EMBEDDIA/sloberta|camembert | 14| 21| 1e-5|
|roberta-base|roberta|4|69|3e-6||

It had been noted that instantiating `EMBEDDIA/sloberta` as `bert` or `roberta` instead of `camembert` renders the loading of the model impossible. `xlm-roberta` models also wouldn't load, the reason is as of yet unknown. `roberta-base` on the other hand works as it should.


