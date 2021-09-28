# task3

# Optimization of hyperparameters

I needed to optimize the hyperparameters for specific model types. I started with model type `electra`, model name `classla/bcms-bertic`. Unlike the last time I stopped the optimization process after 100 runs.

It had been noticed that often an instance of the optimization process crashed, but the optimizer would just run on. I could not stop this pathology, but after reading through the traceback and searching online for a bit I was able to prevent the crashes with the following magic spell:

```python
import torch
torch.cuda.empty_cache()
```
This improved the graphics card memory allocation errors, but `wandb`, `runs`, `output` and `cache_dir` directories still needed to be purged manually to prevent the disk from filling up.

After successful sweeps had been run, the `wandb` website was inspected and the best performing hyperparameters were transcribed into the table below. The number of optimizer runs was capped at 50.

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

| model | average accuracy | average macro F1|
|---|---|---|
|sloberta-frenk-hate|0.7785|0.7764|
|EMBEDDIA/crosloengual-bert |0.7616|0.7585|
|xlm-roberta-base |0.686|0.6827|

<!--- 
Model: xlm-roberta-base, xlmroberta, language='sl'
Accuracies: [0.682277318640955, 0.6882460973370065, 0.6983471074380165, 0.67722681359045, 0.6988062442607897, 0.6712580348943985, 0.6854912764003673]
F1 scores: [0.6717524932572289, 0.6862203507318707, 0.6965631457686351, 0.6769734278516519, 0.6937446894980026, 0.6708358662613981, 0.6828436062884093]


Model: EMBEDDIA/sloberta, camembert, language='sl'
Accuracies: [0.7704315886134068, 0.7777777777777778, 0.7736455463728191, 0.7764003673094583, 0.7800734618916437, 0.788337924701561, 0.7828282828282829]
F1 scores: [0.7688172956081655, 0.7759634520286623, 0.7711152460019659, 0.7747038368823205, 0.7772081128841954, 0.7857142399814923, 0.7810369032120938]


Model: EMBEDDIA/crosloengual-bert, bert, language='sl'
Accuracies: [0.7626262626262627, 0.77089072543618, 0.7626262626262627, 0.758494031221304, 0.7594123048668503, 0.7543617998163453, 0.7630853994490359]
F1 scores: [0.759113833199808, 0.7672336857525612, 0.7594826475863488, 0.7560500494020648, 0.7565539843532022, 0.7512137412224076, 0.7596067627228859]
--->

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
<!--- special completely ignored comment 
Model: xlm-roberta-base, xlmroberta, language='hr'
Accuracies: [0.7084905660377359, 0.7160377358490566, 0.7174528301886792, 0.7084905660377359, 0.7415094339622641, 0.7018867924528301, 0.7283018867924528]
F1 scores: [0.6924610999064821, 0.7110693946004587, 0.7050340578489841, 0.6874543762971446, 0.7279074713025948, 0.6960410747456196, 0.714631848162606]



Model: classla/bcms-bertic, electra, language='hr'
Accuracies: [0.8325471698113207, 0.835377358490566, 0.8316037735849057, 0.8306603773584905, 0.8287735849056603, 0.8283018867924529, 0.8320754716981132]
F1 scores: [0.8223890350362764, 0.8258849830857479, 0.8221409218995981, 0.8217891286872718, 0.8194808914485652, 0.8191865456494926, 0.8224340315323582]


Model: EMBEDDIA/crosloengual-bert, bert, language='hr'
Accuracies: [0.8047169811320755, 0.8042452830188679, 0.8033018867924528, 0.8014150943396227, 0.8080188679245283, 0.8075471698113208, 0.8084905660377358]
F1 scores: [0.7952438186813187, 0.7953073309895347, 0.7943208602955085, 0.7919141395195504, 0.7985768907811182, 0.7977763251289849, 0.7990286728308582]
--> 



| model | average accuracy | average macro F1|
|---|---|---|
|bcms-bertic-frenk-hate|0.8313|0.8219|
|EMBEDDIA/crosloengual-bert |0.8054|0.796|
|xlm-roberta-base |0.7175|0.7049|
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

| model | average accuracy | average macro F1|
|---|---|---|
|roberta-base-frenk-hate|0.7915|0.7785|
|xlm-roberta-large |0.7904|0.77876|
|xlm-roberta-base |0.7577|0.7402|

<!---
Model: xlm-roberta-base, xlmroberta, language='en'
Accuracies: [0.7510860121633363, 0.7558644656820156, 0.764118158123371, 0.761511728931364, 0.7528236316246742, 0.7602085143353605, 0.7580364900086881]
F1 scores: [0.725323441712606, 0.7375568511463544, 0.7483051180210519, 0.7440669841634362, 0.740430562037337, 0.7415839651189275, 0.7444385292878978]


Model: xlm-roberta-large, xlmroberta, language='en'
Accuracies: [0.787141615986099, 0.7936576889661164, 0.7875760208514335, 0.7953953084274544, 0.7945264986967854, 0.7827975673327541, 0.7919200695047784]
F1 scores: [0.7747465654233043, 0.7815003132271792, 0.7749624713549834, 0.78513672182704, 0.782420311908328, 0.7700501853982866, 0.7825030479728539]


Model: roberta-base, roberta, language='en'
Accuracies: [0.7927888792354474, 0.7962641181581234, 0.790616854908775, 0.7819287576020851, 0.7953953084274544, 0.792354474370113, 0.7910512597741095]
F1 scores: [0.779903005739679, 0.7842603092706253, 0.7780383684410571, 0.7678973127920354, 0.7823783904211664, 0.7787919676803026, 0.7783526833497377]

--->
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

# Afterword


I proceeded with uploading the best performing models on ModelHub. As it turned out, this can be done quite elegantly with drag-and-drop interface, but I plan to use `git` in the future as it is less fiddly and takes less time (you can only upload one file at a time with the drag and drop interface and large files block the pipeline.)

It was also determined that I need to specify a few things in the `config.json`. By adding the field `"id2labels": {"0": "Acceptable", "1": "Offensive"}` I was able to get the labels to display properly when run from the webpage API, but the behaviour when run locally remained the same and only output 0 or 1. So far we are pleased with the outcome.


It was also suggested to me that the default text can be set. I studied the rest of the published models and foun that some models prefix their model card with a field that looks like this:

```markdown
---
language: "sl"

tags:
- text-classification
- hate-speech

widget:
- text: "Silva, ti si grda in neprijazna."
---
```

With this addition the models featured the insert text on the webpage API for testing the model.

# Further research

I compared the following models:
* `xlm-roberta-large`
* `xlm-roberta-base`
* `roberta-base`
* `distilroberta-base`

I repeated the `wandb` optimization again, this time I was sure that the disk space will not be overwhelmed with the output models and that the model types are working, so the optimization was left running overnight and are therefore more reliable than in the first iteration of this task.

It was again necessary to repeat the optimization a few times because of unforeseen errors, and it had been noticed than some models run for a significantly longer time than the others (e.g. one evaluation run takes 20 minutes or more). Perhaps this could be improved by requiring a fixed value for batch size.

After numerous tries with careful monitoring the optimization was done. In preparation for future runs I encoded the optimal hyperparameters in a `dataclass` so that all the hyperparameters and the results are in one place, ready to be evaluated. Again an evaluation run of 7 runs was used and the metrics produced were recorded. They were analyzed in a more systematic and typo-resistant way to produce the following table:

|model name| model type| accuracy | macro f1 score|
| ---      | ---       | ---      | ---           |
|roberta-base| roberta| 0.803 +/- 0.00323| 0.791 +/- 0.00372|
|distilroberta-base| roberta| 0.798 +/- 0.00446| 0.786 +/- 0.00502|
|xlm-roberta-base| xlmroberta| 0.727 +/- 0.0756| 0.652 +/- 0.173|
|xlm-roberta-large| xlmroberta| 0.608 +/- 0.0| 0.378 +/- 5.55e-17|

Furthermore, the same statistical tests were done to compare models pair-wise, comparing the best performing one based on accuracy with all the rest:

#### `roberta-base` vs `distilroberta-base`:

| test | accuracy p-value | macro F1 p-value|
| --- | --- | --- |
|Wilcoxon|0.0156|0.0390625|
|Mann Whithney|0.0203|0.0367|
|Student t-test | 3.894e-02 | 7.056e-02|



#### `roberta-base` vs `xlm-roberta-base`:

| test | accuracy p-value | macro F1 p-value|
| --- | --- | --- |
|Wilcoxon|0.00781|0.0078125|
|Mann Whithney|0.00107|0.00107|
|Student t-test | 3.062e-02 | 7.135e-02|



#### `roberta-base` vs `xlm-roberta-large`:

| test | accuracy p-value | macro F1 p-value|
| --- | --- | --- |
|Wilcoxon|0.00781|0.0078125|
|Mann Whithney|0.000529|0.000529|
|Student t-test | 6.386e-21 | 4.070e-24|

## Concluding remarks

Again it was demonstrated that `roberta-base` outperforms other checkpoints. It is interesting to note, however, that `distilroberta-base` is not much worse off, while also reducing the model file size for a factor of 1/3.


`xlm-roberta-large` proved again difficult to handle to say the least. It needed long amounts of time when searching for optimal hyperparameters and when fine-tuning as well, in both cases constant supervision was necessary to make sure that the process is still running as it should.

A framework has been written that will probably be used in all future cases of such tasks, as of yet it is working, but not to its full potential. In the future most of what is now handled by helper functions could be included as dataclass methods which could make the workflow more elegant, albeit probably not faster.


## Addition of `fastext` comparison to model cards

Since the results, obtained in Task 1, were not directly comparable due to the fact that only `lgbt` subset had been used, the training and evaluation was performed anew. Hyperparameters were fiddled, but there was hardly any effect on the result. It was also discovered that the performance was repeatable, so repeated runs produced exactly the same statistics. This renders any attempts at comparing the distributions with statistical tools useless, but since the variances of the samples are small anyway, we can get away with only comparing the average metrics with the `fasttext` metrics.

### Language: sl
* Accuracy:  0.669
* F1 score: 0.659

### Language: hr
* Accuracy:  0.709
* F1 score: 0.691

### Language: en
* Accuracy:  0.712
* F1 score: 0.686


To optimize `fasttext` hyperparameters it seems we would need to provide a validation dataset, which we do not have. I used the test file in its stead, hoping that I'd be able to extract the hyperparameters afterwards. After reading through the console output, however, it seems that `fasttext` already does this for us; first the output took a long time to try and optimize parameters, and then it trains again with the best setup. After this pipeline runs for all the languages, the results look a tad better:


### Language: sl
* Accuracy:  0.717
* F1 score: 0.706

### Language: hr
* Accuracy:  0.775
* F1 score: 0.757

### Language: en
* Accuracy:  0.731
* F1 score: 0.715

The optimization is based on time, not number of runs. In my first run I let it run for 10 minutes. Not suprisingly if this number is increased, the metrics increase, but with diminishing returns. E.g., after doubling the optimization time, I only get about half a percentage point increase.

### Language: sl
* Accuracy:  0.721
* F1 score: 0.711

### Language: hr
* Accuracy:  0.781
* F1 score: 0.766


### Language: en
* Accuracy:  0.732
* F1 score: 0.712

~~I shall amend the model cards with these improved statistics.~~ 
The model cards have been amended with the latest data.

# Adendum 2021-09-27

In accordance with the workflow corrections, agreed upon in a skype meeting, the `Fasttext` results need a proper validation dataset. I prepared 3 validation files:

1. strategy: unshuffled train file was split into head (90%) and tail (10%), head was used for training, tail was used as validation
2. strategy: unshuffled train file was split into tail (90%) and head (10%), tail was used for training, head was used as validation
3. strategy: train file was shuffled, random 90% was used for training and the rest for testing

The code for the dev data preparation process is available in the codebase.

As before the optimization was being performed for 10 minutes. The first results are shown below.
|strategy|language|accuracy|macro F1 score |
|---|---|---|---|
|strategy=1|sl|0.705|0.695|
|strategy=1|hr|0.773|0.759|
|strategy=1|en|0.725|0.709|
|strategy=2|sl|0.706|0.699|
|strategy=2|hr|0.772|0.757|
|strategy=2|en|0.73|0.713|
|strategy=3|sl|0.566|0.385|
|strategy=3|hr|0.597|0.424|
|strategy=3|en|0.603|0.382|

As mentioned by Nikola in the skype meeting, it is beneficial if the train, dev and test data are contiguous and not randomly shuffled. The last strategy where the training and dev instances were shuffled proved to be the worst.

The analysis was repeated to assess the statistical fluctuations of the results. As can be seen below, there are differences, but not high enough that they would warrant further statistical analysis.

Second run:
|strategy|language|accuracy|macro F1 score |
|---|---|---|---|
|strategy=1|sl|0.709|0.701|
|strategy=1|hr|0.777|0.761|
|strategy=1|en|0.721|0.702|
|strategy=2|sl|0.713|0.706|
|strategy=2|hr|0.77|0.75|
|strategy=2|en|0.728|0.709|
|strategy=3|sl|0.566|0.385|
|strategy=3|hr|0.597|0.424|
|strategy=3|en|0.603|0.382|


Third run:
|strategy|language|accuracy|macro F1 score |
|---|---|---|---|
|strategy=1|sl|0.71|0.7|
|strategy=1|hr|0.761|0.744|
|strategy=1|en|0.723|0.703|
|strategy=2|sl|0.711|0.704|
|strategy=2|hr|0.771|0.754|
|strategy=2|en|0.722|0.703|
|strategy=3|sl|0.566|0.385|
|strategy=3|hr|0.597|0.424|
|strategy=3|en|0.603|0.382|