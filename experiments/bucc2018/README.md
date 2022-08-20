# Comparing Multilingual Sentence Embeddings

This experiments folder accompanies of our final DL4NLP paper.
We conducted a series of experiments including different models, metrics and datasets.

## Requirements and Installation
The requirements do not differ from the original repository and ca be installed the same way.

```shell
git clone https://github.com/potamides/unsupervised-metrics
pip install -e unsupervised-metrics[experiments]
```

If you want to use fast-align follow its install instruction and make sure that the fast_align and atools programs are on your PATH. This requirement is optional.


## Usage

To train the Scorer, just execute the `bucc2018.py` file:

```shell
python3 bucc2018.py
```


This will train a `sentence-transformers/distiluse-base-multilingual-cased-v2` transformer model based Contrastive 
Scorer from "de" (German) to "en" (English).

These (and more) settings can be adjusted in the file itself:

```python
scorer = train_contrastive("sentence-transformers/distiluse-base-multilingual-cased-v2", "de", "en", max_len=50,
                           train_batch_size=128, num_epochs=1, iterations=10)
```

## Acknowledgments

This experiment is mainly based on the original repository of the UScore paper:
* [Unsupervised Metrics](https://github.com/potamides/unsupervised-metrics.git)

Besides, several other projects build up the foundation of the library:
* [ACL20-Reference-Free-MT-Evaluation](https://github.com/AIPHES/ACL20-Reference-Free-MT-Evaluation)
* [Unsupervised-crosslingual-Compound-Method-For-MT](https://github.com/Rain9876/Unsupervised-crosslingual-Compound-Method-For-MT)
* [Seq2Seq examples](https://github.com/huggingface/transformers/tree/v4.5.1/examples/seq2seq) of [transformers](https://github.com/huggingface/transformers)
* [VecMap](https://github.com/artetxem/vecmap)
* [CRISS](https://github.com/pytorch/fairseq/tree/master/examples/criss)
