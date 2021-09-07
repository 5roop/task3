# task3

# Optimization of hyperparameters

I needed to optimize the hyperparameters for specific model types. I started with model type `electra`, model name `classla/bcms-bertic`. Unlike the last time I stopped the optimization process after 100 runs.

It had been noticed that often an instance of the optimization process crashed, but the optimizer would just run on. I could not stop this pathology, but after reading through the traceback and searching online for a bit I was able to prevent the crashes with the following magic spell:

```python
import torch
torch.cuda.empty_cache()
```
This improved the graphics card memory allocation errors, but `wandb`, `runs`, `output` and `cache_dir` directories still needed to be purged manually to prevent the disk from filling up.

After successful sweeps had been run, the wandb website was inspected and the best performing hyperparameters were transcribed into the table below. The number of optimizer runs was capped at 50

Found optimal parameters for the models are as follows:

|Model name | Model type | epochs | batch size | learning rate|
|---|---|---|---|---|---|
|classla/bcms-bertic | electra | 12 | 74 | 1e-5 |
|EMBEDDIA/sloberta|camembert | 14| 21| 1e-5|
|roberta-base|roberta|4|69|3e-6||

It had been noted that instantiating `EMBEDDIA/sloberta` as `bert` or `roberta` instead of `camembert` renders the loading of the model impossible. ~~`xlm-roberta` models also wouldn't load, the reason is as of yet unknown.~~ `roberta-base` on the other hand works as it should.

When I evaluated the models with the right metric for Task1, I noticed that the models that proved troublesome for wandb hyperparameters optimization worked as they should when loading them as model type `xlmroberta`.


# Finetuning

With merged data finetuning was performed 11 times. Special care was taken to assure that the best performing model instance (based on accuracy) was copied to a separate directory while the rest were being overwritten each time. The intermediate statistics were as following:

Model: xlm-roberta-base, xlmroberta, language='sl'
Accuracies: [0.660238751147842, 0.709366391184573, 0.6721763085399449, 0.6795224977043158, 0.6808999081726355, 0.6974288337924701, 0.689623507805326, 0.689623507805326, 0.657483930211203, 0.6960514233241506, 0.6887052341597796]
F1 scores: [0.6601559337129902, 0.7003895422308328, 0.6721738206544836, 0.6794249131212254, 0.6767424602390727, 0.6964607580962895, 0.688222518871888, 0.6885477968396336, 0.6572408613764471, 0.6921237931246541, 0.6873385012919897]
