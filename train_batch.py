# -*- coding: utf-8 -*-
from __future__ import unicode_literals
import torch
import torch.nn as nn
from torch import optim
import time, random
import os
from tqdm import tqdm
from lstm import LSTMSentiment
from gru import GRUSentiment
# from bigru import BiGRUSentiment
from torchtext import data
import numpy as np
import argparse
from torch.autograd import Variable
import torch.nn.functional as F
import spacy

torch.set_num_threads(8)
torch.manual_seed(1)
random.seed(1)

spacy_en = spacy.load('en_core_web_sm')


# def tokenizer(text):  # create a tokenizer function
#     return [tok.text for tok in spacy_en.tokenizer(text)]
def tokenizer(text):  # create a tokenizer function
    try:
        return [tok.text for tok in spacy_en.tokenizer(text)]
    except:
        return [tok.text for tok in spacy_en.tokenizer(unicode(text, "utf-8"))]


text_field = data.Field(tokenize=tokenizer, lower=True)
# text_field = data.Field(lower=True)
label_field = data.Field(sequential=False)


def load_bin_vec(fname, vocab):
    """
    Loads 300x1 word vecs from Google (Mikolov) word2vec
    """
    word_vecs = {}
    with open(fname, "rb") as f:
        header = f.readline()
        vocab_size, layer1_size = map(int, header.split())
        binary_len = np.dtype('float32').itemsize * layer1_size
        for line in range(vocab_size):
            word = []
            while True:
                ch = f.read(1).decode('latin-1')
                if ch == ' ':
                    word = ''.join(word)
                    break
                if ch != '\n':
                    word.append(ch)
            if word in vocab:
                word_vecs[word] = np.fromstring(f.read(binary_len), dtype='float32')
            else:
                f.read(binary_len)
    return word_vecs


def get_accuracy(truth, pred):
    assert len(truth) == len(pred)
    right = 0
    for i in range(len(truth)):
        if truth[i] == pred[i]:
            right += 1.0
    return right / len(truth)


def train_epoch_progress(model, train_iter, loss_function, optimizer, text_field, label_field, epoch):
    model.train()
    avg_loss = 0.0
    truth_res = []
    pred_res = []
    count = 0
    for batch in tqdm(train_iter, desc='Train epoch ' + str(epoch + 1)):
        sent, label = batch.text, batch.label
        label.data.sub_(1)
        truth_res += list(label.data)
        model.batch_size = len(label.data)
        model.hidden = model.init_hidden()
        sent = sent.to(device)
        pred, _, _ = model(sent)
        pred = F.log_softmax(pred)
        pred = pred.cpu()
        pred_label = pred.data.max(1)[1].numpy()
        pred_res += [x for x in pred_label]
        model.zero_grad()
        loss = loss_function(pred, label)
        avg_loss += loss.data[0]
        count += 1
        loss.backward()
        optimizer.step()
    avg_loss /= len(train_iter)
    acc = get_accuracy(truth_res, pred_res)
    return avg_loss, acc


def evaluate(model, data, loss_function, name):
    model.eval()
    avg_loss = 0.0
    truth_res = []
    pred_res = []
    for batch in data:
        sent, label = batch.text, batch.label
        label.data.sub_(1)
        truth_res += list(label.data)
        model.batch_size = len(label.data)
        model.hidden = model.init_hidden()
        sent = sent.to(device)
        pred, _, _ = model(sent)
        pred = F.log_softmax(pred)
        pred = pred.cpu()
        pred_label = pred.data.max(1)[1].numpy()
        pred_res += [x for x in pred_label]
        loss = loss_function(pred, label)
        avg_loss += loss.data[0]
    avg_loss /= len(data)
    acc = get_accuracy(truth_res, pred_res)
    print(name + ': loss %.2f acc %.1f' % (avg_loss, acc * 100))
    return acc


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
    ## for GPU run
    #     train_iter, dev_iter, test_iter = data.BucketIterator.splits((train, dev, test),
    #                 batch_sizes=(batch_size, len(dev), len(test)), sort_key=lambda x: len(x.text), repeat=False, device=None)
    return train_iter, dev_iter, test_iter


args = argparse.ArgumentParser()
args.add_argument('--m', dest='model', default='gru', help='specify the mode to use (default: lstm)')
args.add_argument('--data', dest='data', default='abstract', help='specify the dataset')
args.add_argument('--epoch', dest='epoch', default=2, type=int, help='specify number of epoch')
args = args.parse_args()

# EPOCHS = 20
# EPOCHS = 2
USE_GPU = torch.cuda.is_available()
EMBEDDING_DIM = 300
HIDDEN_DIM = 150

if USE_GPU:
    device = 'cuda:' + args.cuda_id
else:
    device = 'cpu'

BATCH_SIZE = 5
timestamp = str(int(time.time()))
best_dev_acc = 0.0

# data is from torchtext.data, it is an object
train_iter, dev_iter, test_iter = load_sst(text_field, label_field, BATCH_SIZE)
print text_field, label_field

if args.model == 'lstm':
    print('model is lstm')
    model = LSTMSentiment(embedding_dim=EMBEDDING_DIM, hidden_dim=HIDDEN_DIM, vocab_size=len(text_field.vocab),
                          label_size=len(label_field.vocab) - 1, \
                          use_gpu=USE_GPU, batch_size=BATCH_SIZE)

if args.model == 'gru':
    print('model is gru')
    model = GRUSentiment(embedding_dim=EMBEDDING_DIM, hidden_dim=HIDDEN_DIM, vocab_size=len(text_field.vocab),
                         label_size=len(label_field.vocab) - 1, \
                         use_gpu=USE_GPU, batch_size=BATCH_SIZE)

# if args.model == 'bigru':
#     print('model is bigru')
#     model = BiGRUSentiment(embedding_dim=EMBEDDING_DIM, hidden_dim=HIDDEN_DIM, vocab_size=len(text_field.vocab), label_size=len(label_field.vocab)-1,\
#                           use_gpu=USE_GPU, batch_size=BATCH_SIZE)

if USE_GPU:
    print('use cuda')
    model = model.cuda()

print('Load word embeddings...')
# word2vector
word_to_idx = text_field.vocab.stoi
# example of word_to_idx: u'schools': 14512, u'mimicking': 12930
# len(word_to_idx) = 16190
pretrained_embeddings = np.random.uniform(-0.25, 0.25, (len(text_field.vocab), 300))
# pretrained_embeddings.shape = (16190, 300)

pretrained_embeddings[0] = 0  # set all pretrained_embeddings[0] to 0
word2vec = load_bin_vec('./data/GoogleNews-vectors-negative300.bin', word_to_idx)
for word, vector in word2vec.items():
    pretrained_embeddings[word_to_idx[word] - 1] = vector
# the embedding layer of this model is using the pretrained embeddings
model.embeddings.weight.data.copy_(torch.from_numpy(pretrained_embeddings))
# model.embeddings.weight.data = text_field.vocab.vectors
# model.embeddings.embed.weight.requires_grad = False


best_model = model
optimizer = optim.Adam(model.parameters(), lr=1e-3)
loss_function = nn.NLLLoss()

print('Training...')
if args.model == 'gru':
    out_dir = "models/gru" + "/" + args.data
elif args.model == 'lstm':
    out_dir = "models/lstm" + "/" + args.data
else:
    out_dir = "models/bigru" + "/" + args.data

# out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
print("Writing to {}\n".format(out_dir))
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
for epoch in range(args.epoch):
    avg_loss, acc = train_epoch_progress(model, train_iter, loss_function, optimizer, text_field, label_field, epoch)
    tqdm.write('Train: loss %.2f acc %.1f' % (avg_loss, acc * 100))
    dev_acc = evaluate(model, dev_iter, loss_function, 'Dev')
    if dev_acc > best_dev_acc:
        if best_dev_acc > 0:
            os.system('rm ' + out_dir + '/best_model' + '.pkl')
        best_dev_acc = dev_acc
        best_model = model
        torch.save(best_model, out_dir + '/best_model' + '.pkl')
        # evaluate on test with the best dev performance model
        test_acc = evaluate(best_model, test_iter, loss_function, 'Test')

best_model = torch.load(out_dir + '/best_model' + '.pkl')
test_acc = evaluate(best_model, test_iter, loss_function, 'Final Test')
