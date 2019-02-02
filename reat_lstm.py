#from __future__ import unicode_literals  # to address string/unicode problem
import torch
import torch.nn as nn
from torchtext import data
import numpy as np
from torch.autograd import Variable
from nltk.tokenize import word_tokenize
import torch.nn.functional as F



def load_sst(text_field, label_field, batch_size):
    train, dev, test = data.TabularDataset.splits(path='./data/SST2/', train='train.tsv',
                                                  validation='dev.tsv', test='test.tsv', format='tsv',
                                                  fields=[('text', text_field), ('label', label_field)])
    text_field.build_vocab(train, dev, test)
    label_field.build_vocab(train, dev, test)
    train_iter, dev_iter, test_iter = data.BucketIterator.splits((train, dev, test),
                batch_sizes=(batch_size, len(dev), len(test)), sort_key=lambda x: len(x.text), repeat=False, device=-1) 
    return train_iter, dev_iter, test_iter

def sigmoid(x):
    return (1 / (1 + np.exp(-x)))


def attribution(model, text):
    # data for word-to-vec
    text_field = data.Field(lower=True)  # it is an object
    label_field = data.Field(sequential=False)   # it is also an object
    train_iter, dev_iter, test_iter = load_sst(text_field, label_field, 5)
    # word2vector
    word_to_idx = text_field.vocab.stoi  # example of word_to_idx: u'schools': 14512, len(word_to_idx) = 16190
    word_to_idx_dict = dict(word_to_idx)

    # processing text
    word_tokenize_list = word_tokenize(text)
    sentence = []
    for i in word_tokenize_list:
        sentence.append(word_to_idx_dict[i])
    sent = np.array(sentence)
    sent = Variable(torch.from_numpy(sent))
    sent = sent.cuda()
    length = len(word_tokenize_list)

    # model prediction
    best_model.batch_size = 1
    best_model.hidden = best_model.init_hidden()
    pred, hn, x = best_model(sent)
    hn = hn.cpu().data.numpy()
    x = x.cpu().data.numpy()
    pred = F.softmax(pred).cpu()
    pred_label = pred.data.max(1)[1].numpy()  
    if pred_label[0] == 0:
        print ("prediction category: positive sentiment with confidence of " + str(pred.data.numpy()[0, 0]))# 0 is positive, and 1 is negative
    else:
        print ("prediction category: negative sentiment with confidence of " + str(pred.data.numpy()[0, 1]))

    # attribution for the prediction
    weights = best_model.lstm.state_dict()
    W_ii, W_if, W_ig, W_io = np.split(weights['weight_ih_l0'], 4, 0)
    W_hi, W_hf, W_hg, W_ho = np.split(weights['weight_hh_l0'], 4, 0)
    b_i, b_f, b_g, b_o = np.split(weights['bias_ih_l0'].cpu().numpy() + weights['bias_hh_l0'].cpu().numpy(), 4)


    z_dict = []
    z_dict.append(np.ones(150))
    for i in range(length-1):
        i = i + 1
        z = sigmoid(np.matmul(W_if, x[i,0,:]) + np.matmul(W_hf, hn[i-1,0,:]) + b_f)
        z_dict.append(z)

    o_dict = []
    for i in range(length):
        if i == 0:
            o = sigmoid(np.matmul(W_io, x[i,0,:]) + b_o)
        else:
            o = sigmoid(np.matmul(W_io, x[i,0,:]) + np.matmul(W_ho, hn[i-1,0,:]) + b_o)
        o_dict.append(o)


    alpha_dict = []
    for i in range(len(z_dict)):
        if i == 0:
            alpha_dict.append(z_dict[0])
        else:
            alpha_dict.append(z_dict[i] * o_dict[i] / o_dict[i-1])

    weights_linear = best_model.hidden2label.state_dict()
    W = weights_linear['weight'].cpu().numpy()
    b= weights_linear['bias'].cpu().numpy()
    target_class = pred_label


    score_dict = []
    for i in range(len(alpha_dict)):
        if i == 0:
            updating = hn[0,0,:]
        else:
            updating = hn[i,0,:] - alpha_dict[i] * hn[i-1,0,:]
        forgetting = alpha_dict[0]
        for j in range(i+1, len(alpha_dict)):
            forgetting = forgetting*alpha_dict[j]
        score = np.matmul( W[target_class], updating * forgetting) 
        score_dict.append(score[0])
    return word_tokenize_list, score_dict


# testing text
text = "the fight scenes are fun but it grows tedious"
# text = "the story may be new, but it does not serve lots of laughs"
best_model = torch.load('models/lstm/best_model.pkl')
word_tokenize_list, score_dict = attribution(best_model, text) 


for i in range(len(score_dict)):
    print word_tokenize_list[i], score_dict[i]




