# -*- coding: utf-8 -*-
from __future__ import unicode_literals  # to address string/unicode problem
import torch
import torch.nn as nn
from torchtext import data
import numpy as np
from torch.autograd import Variable
from nltk.tokenize import word_tokenize
import torch.nn.functional as F
import pandas as pd
import spacy
import random
import csv
import argparse
import sys

reload(sys)
sys.setdefaultencoding('utf-8')

args = argparse.ArgumentParser()
args.add_argument('--data', dest='data', default='abstract', help='specify the dataset')
args = args.parse_args()

# torch.set_default_tensor_type('torch.DoubleTensor')
device = 'cpu'

spacy_en = spacy.load('en_core_web_sm')


# nlp = spacy.load('en')
def tokenizer(text):  # create a tokenizer function
    try:
        return [tok.text for tok in spacy_en.tokenizer(text)]
    except:
        return [tok.text for tok in spacy_en.tokenizer(unicode(text, "utf-8"))]
    # return [tok.text for tok in spacy_en.tokenizer(text)]


text_field = data.Field(tokenize=tokenizer, lower=True)
# text_field = data.Field(lower=True)
label_field = data.Field(sequential=False)


def load_sst(text_field, label_field, batch_size):
    train, dev, test = data.TabularDataset.splits(path='./data/SCORE/' + args.data, train='train.csv',
                                                  validation='dev.csv', test='test.csv', format='csv',
                                                  fields=[('text', text_field), ('label', label_field)])
    text_field.build_vocab(train, dev, test)
    label_field.build_vocab(train, dev, test)
    train_iter, dev_iter, test_iter = data.BucketIterator.splits((train, dev, test),
                                                                 batch_sizes=(batch_size, len(dev), len(test)),
                                                                 sort_key=lambda x: len(x.text), repeat=False,
                                                                 device=-1)
    return train_iter, dev_iter, test_iter


def sigmoid(x):
    return (1 / (1 + np.exp(-x)))


def attribution(model, text):
    # data for word-to-vec
    # text_field = data.Field(lower=True)  # it is an object
    # label_field = data.Field(sequential=False)  # it is also an object
    train_iter, dev_iter, test_iter = load_sst(text_field, label_field, 5)
    # word2vector
    word_to_idx = text_field.vocab.stoi  # example of word_to_idx: u'schools': 14512, len(word_to_idx) = 16190
    word_to_idx_dict = dict(word_to_idx)

    # processing text
    # word_tokenize_list = word_tokenize(text)
    word_tokenize_list = tokenizer(text)
    sentence = []
    for i in word_tokenize_list:
        try:
            sentence.append(word_to_idx_dict[i])
        except:
            print('not in word_to_idx_dict', i)
            sentence.append(random.choice(list(word_to_idx_dict.values())))
    sent = np.array(sentence)
    sent = Variable(torch.from_numpy(sent))
    sent = sent.to(device)
    length = len(word_tokenize_list)

    # model prediction
    best_model.batch_size = 1
    best_model.hidden = best_model.init_hidden()
    pred, hn, x = best_model(sent)
    hn = hn.cpu().data.numpy()
    x = x.cpu().data.numpy()
    pred = F.softmax(pred).cpu()
    pred_label = pred.data.max(1)[1].numpy()
    # if pred_label[0] == 0:
    #     print ("prediction category: positive sentiment with confidence of " + str(
    #         pred.data.numpy()[0, 0]))  # 0 is positive, and 1 is negative
    # else:
    #     print ("prediction category: negative sentiment with confidence of " + str(pred.data.numpy()[0, 1]))

    # attribution for the prediction
    weights = best_model.gru.state_dict()
    _, W_iz, _ = np.split(weights['weight_ih_l0'], 3, 0)
    _, W_hz, _ = np.split(weights['weight_hh_l0'], 3, 0)
    _, b_z, _ = np.split(weights['bias_ih_l0'].cpu().numpy() + weights['bias_hh_l0'].cpu().numpy(), 3)

    z_dict = []
    z_dict.append(np.ones(150))
    for i in range(length - 1):
        i = i + 1
        # print(type(b_z))
        z = sigmoid(
            np.matmul(W_iz, x[i, 0, :]) + np.matmul(W_hz, hn[i - 1, 0, :]) + torch.from_numpy(b_z))
        z_dict.append(z)
    alpha_dict = z_dict
    weights_linear = best_model.hidden2label.state_dict()
    W = weights_linear['weight'].cpu().numpy()
    b = weights_linear['bias'].cpu().numpy()

    target_class = pred_label
    score_dict = []
    # print('len(alpha_dict)', len(alpha_dict))
    # for i in range(len(alpha_dict)):
    #     if i == 0:
    #         updating = hn[0, 0, :]
    #     else:
    #         right_half = alpha_dict[i].cpu().numpy() * hn[i - 1, 0, :]
    #         updating = hn[i, 0, :] - alpha_dict[i].cpu().numpy() * hn[i - 1, 0, :]
    #     forgetting = alpha_dict[0]
    #     for j in range(i + 1, len(alpha_dict)):
    #         forgetting = forgetting * alpha_dict[j].type(torch.DoubleTensor)
    #         score = np.matmul(W[target_class], updating * forgetting)  # + b[target_class]
    #         try:
    #             score_0 = score.data.cpu().numpy()[0]
    #         except:
    #             print('i', i)
    #             score_0 = score[0]
    #             print(score_0)
    #     score_dict.append(score_0)
    # print('final i', i)
    # return word_tokenize_list, score_dict
    for i in range(len(alpha_dict)):
        if i == 0:
            updating = hn[0, 0, :]
        else:
            updating = hn[i, 0, :] - alpha_dict[i].cpu().numpy() * hn[i - 1, 0, :]
        forgetting = alpha_dict[0]
        for j in range(i + 1, len(alpha_dict)):
            forgetting = forgetting * alpha_dict[j].type(torch.DoubleTensor)
        score = np.matmul(W[target_class], updating * forgetting)  # + b[target_class]
        try:
            score_0 = score.data.cpu().numpy()[0]
        except:
            # print('i', i)
            score_0 = score[0]
            # print(score_0)
        score_dict.append(score_0)
    return word_tokenize_list, score_dict


# testing text
# text = "the fight scenes are fun but it grows tedious"
test_data = pd.read_csv('./data/SCORE/' + args.data + '/train.csv')
# print(test_data.head())
for i in range(0, 100):
    test_text = test_data.loc[i][0]
    # print(test_text)
    # punc = '''!()[]{};:'"\ <>/?@#$%^&*_~'''
    # for ele in test_text:
    #     if ele in punc:
    #         test_text = test_text.replace(ele, " ")
    ground_truth = test_data.loc[i][1]
    # print('ground_truth:', ground_truth)
    # text = "the story may be new, but it does not serve lots of laughs"
    best_model = torch.load('models/gru/' + args.data + '/best_model.pkl')
    word_tokenize_list, score_dict = attribution(best_model, test_text)
    # word_tokenize_list=[]
    word_tokenize_list = [str(word.encode('utf-8')) for word in word_tokenize_list]
    # except:print()
    # for i in range(len(score_dict)):
    #     print word_tokenize_list[i], score_dict[i]
    # print('len(word_tokenize_list)', len(word_tokenize_list))
    # print('len(score_dict)', len(score_dict))
    # print(word_tokenize_list)
    assert len(score_dict) == len(word_tokenize_list)
    print('==========len(score_dict)', len(score_dict))
    if i == 0:
        f = open(args.data + "_attri_data.csv", "w")
    else:
        f = open(args.data + "_attri_data.csv", "a")
    data_save = []
    for j in range(len(score_dict)):
        data_save.append(
            "{}*{}%{}".format(str(word_tokenize_list[j]).decode('utf-8'), str(score_dict[j]), ground_truth))
    data_save_join = ';'.join(data_save)
    print('++++++++sum(score_dict)', sum(score_dict))
    f.write("{}@".format(data_save_join))
    f.close()
# f.write(data_save)

# f.close()
