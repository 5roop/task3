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

With merged data finetuning was performed 11 times. Special care was taken to assure that the best performing model instance (based on accuracy) was copied to a separate directory while the rest were being overwritten each time. The intermediate statistics are given below, grouped by the language of the data the model was being finetuned on.

## Slovenian

Model: xlm-roberta-base, xlmroberta, language='sl'
Accuracies: [0.682277318640955, 0.6882460973370065, 0.6983471074380165, 0.67722681359045, 0.6988062442607897, 0.6712580348943985, 0.6854912764003673]
F1 scores: [0.6717524932572289, 0.6862203507318707, 0.6965631457686351, 0.6769734278516519, 0.6937446894980026, 0.6708358662613981, 0.6828436062884093]


Model: EMBEDDIA/sloberta, camembert, language='sl'
Accuracies: [0.7704315886134068, 0.7777777777777778, 0.7736455463728191, 0.7764003673094583, 0.7800734618916437, 0.788337924701561, 0.7828282828282829]
F1 scores: [0.7688172956081655, 0.7759634520286623, 0.7711152460019659, 0.7747038368823205, 0.7772081128841954, 0.7857142399814923, 0.7810369032120938]


Model: EMBEDDIA/crosloengual-bert, bert, language='sl'
Accuracies: [0.7626262626262627, 0.77089072543618, 0.7626262626262627, 0.758494031221304, 0.7594123048668503, 0.7543617998163453, 0.7630853994490359]
F1 scores: [0.759113833199808, 0.7672336857525612, 0.7594826475863488, 0.7560500494020648, 0.7565539843532022, 0.7512137412224076, 0.7596067627228859]


### Statistical analysis

It would seem `EMBEDDIA/sloberta` bested the other three candidates in the tests, to confirm this, statistical tests were performed.

#### `Sloberta` vs `crosloengual-bert`:
| test | accuracy p-value | macro F1 p-value|
| --- | --- | --- |
|Wilcoxon|0.00781|0.00781|
|Mann Whithey|0.00163|0.00108|
|Student t-test |0.000101|3.95e-05|

#### `Sloberta` vs `xlm-roberta-base`:
| test | accuracy p-value | macro F1 p-value|
| --- | --- | --- |
|Wilcoxon|0.00781|0.00781|
|Mann Whithey|0.00108|0.00108|
|Student t-test |9.46e-11|6.94e-11|

## Croatian

Model: xlm-roberta-base, xlmroberta, language='hr'
Accuracies: [0.7084905660377359, 0.7160377358490566, 0.7174528301886792, 0.7084905660377359, 0.7415094339622641, 0.7018867924528301, 0.7283018867924528]
F1 scores: [0.6924610999064821, 0.7110693946004587, 0.7050340578489841, 0.6874543762971446, 0.7279074713025948, 0.6960410747456196, 0.714631848162606]



Model: classla/bcms-bertic, electra, language='hr'
Accuracies: [0.8325471698113207, 0.835377358490566, 0.8316037735849057, 0.8306603773584905, 0.8287735849056603, 0.8283018867924529, 0.8320754716981132]
F1 scores: [0.8223890350362764, 0.8258849830857479, 0.8221409218995981, 0.8217891286872718, 0.8194808914485652, 0.8191865456494926, 0.8224340315323582]


Model: EMBEDDIA/crosloengual-bert, bert, language='hr'
Accuracies: [0.8047169811320755, 0.8042452830188679, 0.8033018867924528, 0.8014150943396227, 0.8080188679245283, 0.8075471698113208, 0.8084905660377358]
F1 scores: [0.7952438186813187, 0.7953073309895347, 0.7943208602955085, 0.7919141395195504, 0.7985768907811182, 0.7977763251289849, 0.7990286728308582]

### Statistical analysis

#### `classla/bcms-bertic` vs `EMBEDDIA/crosloengual-bert`:
| test | accuracy p-value | macro F1 p-value|
| --- | --- | --- |
|Wilcoxon|0.00781|0.00781|
|Mann Whithey|0.00108|0.00108|
|Student t-test |2.43e-10 |1.27e-10|

#### `classla/bcms-bertic` vs `xlm-roberta-base` :
| test | accuracy p-value | macro F1 p-value|
| --- | --- | --- |
|Wilcoxon|0.00781|0.00781|
|Mann Whithey|0.00107|0.00108|
|Student t-test |4.83e-11 | 5.61e-11 |




# English
Model: xlm-roberta-base, xlmroberta, language='en'
Accuracies: [0.7510860121633363, 0.7558644656820156, 0.764118158123371, 0.761511728931364, 0.7528236316246742, 0.7602085143353605, 0.7580364900086881]
F1 scores: [0.725323441712606, 0.7375568511463544, 0.7483051180210519, 0.7440669841634362, 0.740430562037337, 0.7415839651189275, 0.7444385292878978]


Model: xlm-roberta-large, xlmroberta, language='en'
Accuracies: [0.787141615986099, 0.7936576889661164, 0.7875760208514335, 0.7953953084274544, 0.7945264986967854, 0.7827975673327541, 0.7919200695047784]
F1 scores: [0.7747465654233043, 0.7815003132271792, 0.7749624713549834, 0.78513672182704, 0.782420311908328, 0.7700501853982866, 0.7825030479728539]


Model: roberta-base, roberta, language='en'
Accuracies: [0.7927888792354474, 0.7962641181581234, 0.790616854908775, 0.7819287576020851, 0.7953953084274544, 0.792354474370113, 0.7910512597741095]
F1 scores: [0.779903005739679, 0.7842603092706253, 0.7780383684410571, 0.7678973127920354, 0.7823783904211664, 0.7787919676803026, 0.7783526833497377]


Model: distilbert-base-uncased-finetuned-sst-2-english, distilbert, language='en'
Accuracies: [0.7232841007819287, 0.7132927888792354, 0.7224152910512598, 0.7158992180712423, 0.7254561251086012, 0.7193744569939183, 0.7211120764552563]
F1 scores: [0.7044579789396375, 0.6905553514298262, 0.6993976623782681, 0.6936789340938113, 0.7055884733689493, 0.6957070128339444, 0.7009300631374454]



### Statistical analysis
#### `roberta-base` vs `xlm-roberta-base`:

| test | accuracy p-value | macro F1 p-value|
| --- | --- | --- |
|Wilcoxon|0.00781|0.00781|
|Mann Whithey|0.00108|0.00108|
|Student t-test | 1.35e-08 | 1.05e-07|

#### `roberta-base` vs `xlm-roberta-large`:
Notā: `roberta-base` has average accuracy 0.7915, while `xlm-roberta-large` has average accuracy of 0.7904. If macro F1 scores were to be compared, `roberta-base` actually has lower average than `xlm-roberta-large`: 0.77852 vs 0.77876 respectively. 
| test | accuracy p-value | macro F1 p-value|
| --- | --- | --- |
|Wilcoxon|0.188|0.406|
|Mann Whithey|0.375|0.649|
|Student t-test | 0.681| 0.934|

So can it be said that `xlm-roberta-large` is better than `roberta-base`? No, it can not, as the Wilcoxon p-value for this case reaches 0.656, Mann-Whithey p-value is 0.399, and of course the Student p-value stays the same.

#### `roberta-base` vs `distilbert-base-uncased-finetuned-sst-2-english`:

| test | accuracy p-value | macro F1 p-value|
| --- | --- | --- |
|Wilcoxon|0.00781|0.00781|
|Mann Whithey|0.00108|0.00108|
|Student t-test | 1.33e-12 	 | 3.03e-12|

I shall proceed with uploading the best performing models on ModelHub.