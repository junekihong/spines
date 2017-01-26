

# Tagging with constituent information

Currently we combine unary constituents with the POS tag of the word. And we train an LSTM tagger using Dynet.

#### Dependencies:

* Python 2.7
* Numpy
* [DyNet](http://dynet.readthedocs.io/en/latest/python.html)


#### Producing a data set:

```
python tags.py data/10sentences.sample
```

This will produce a file data/10sentences.sample.supertags

#### Training:

```
python dynet_bilstm_tagger.py --train data/10sentences.sample.supertags --dev data/10sentences.sample.supertags --model model
```