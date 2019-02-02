# REAT
PyTorch code for paper: On Attribution of Recurrent Neural Network Predictions via Additive Decomposition. It has been accepted in [WWW2019](https://www2019.thewebconf.org/).

We propose an attribution method, called REAT, to provide interpretations to RNN predictions in a faithful manner. REAT decomposes the final prediction of a RNN into additive contribution of each word
in the input text. This additive decomposition enables REAT to flexibly generate phrase-level attribution scores. In addition, REAT is generally applicable to various RNN architectures, including
GRU, LSTM and their bidirectional versions.

## Usage Instructions:
* Clone the code from Github:
```
git clone https://github.com/mndu/REAT.git
cd REAT
```

* Download the pretrained `word2vec` pre-trained using Google News corpus from [GoogleNews-vectors-negative300.bin.gz](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing). Unzip the .gz file:
```
gunzip GoogleNews-vectors-negative300.bin.gz
```

* Train RNN models, we consider the following three models, GRU, LSTM and Bidirectional GRU.  As the focus of this paper is to provide post-hoc attribution for the predictions of RNNs, rather than increase their predictive accuracy, thus we use [standard practices](https://github.com/clairett/pytorch-sentiment-classification) to train our models.
```
python train_batch.py --m gru
python train_batch.py --m lstm
python train_batch.py --m bigru
```

* Provide attribution for an input text for three kinds of RNN models, GRU, LSTM and Bidirectional GRU:
```
python reat_gru
python reat_lstm
python reat_bigru
```


## System requirement:
Python 2.7, torch 0.3, torchtext 0.2.3, nltk, and tqdm

## Reference:
```
@inproceedings{du2019www,
    author    = {Mengnan Du, Ninghao Liu, Fan Yang, Shuiwang Ji,  Xia Hu},
    title     = {On Attribution of Recurrent Neural Network Predictions via Additive Decomposition},
    booktitle = {The Web Conference (WWW)},
    year      = {2019}
}
```