from metrics import confusion_dataframe, bleu
from visualizations import plot_confusion_dataframe, plot_history, COLORS
from drive import Drive
from models.seq2seq import Seq2seq
from preprocessing import unindex, read_langs, train_test_valid_split
from constants import *

import os
import time
import math
from collections import OrderedDict

import numpy as np
import pandas as pd

#from nltk.translate.bleu_score import corpus_bleu, sentence_bleu

import torch
import torch.nn as nn
from torch.optim import Adam


def time_str(diff):
    hours, rem = divmod(diff, 3600)
    minutes, seconds = divmod(rem, 60)
    return '{:0>2}:{:0>2}:{:05.2f}'.format(int(hours), int(minutes), seconds)


def batch_generator(batch_indices, batch_size):
    batches = math.ceil(len(batch_indices)/batch_size)
    for i in range(batches):
        batch_start = i*batch_size
        batch_end = (i+1)*batch_size
        if batch_end > len(batch_indices):
            yield batch_indices[batch_start:]
        else:
            yield batch_indices[batch_start:batch_end]


def evaluate(model, x, y):
    y_pred = [model(each_x.unsqueeze(0)) for each_x in x]
    
    y_true = [unindex(each, target_vocab) for each in y.numpy()]
    y_pred = [unindex(each, target_vocab) for each in y_pred]
    
    df = pd.DataFrame(OrderedDict([
        ('Source', [unindex(each, source_vocab) for each in x.numpy()]),
        ('True Name', y_true),
        ('Our Name', y_pred),
        ('BLEU', [bleu(y_true[i], y_pred[i]) for i in range(len(y_true))])
    ]))
    
    return df


def write_training_log(log_dict, bleu_history, loss_history, translations):
    log_template = ": {}\n".join(log_dict.keys()) + ": {}"
    log_string = log_template.format(*log_dict.values())

    drive.log(log_string, fname="train-log.txt")

    fig = plot_history(
        history = loss_history,
        color = COLORS['red'],
        title = 'Average Loss',
        ylabel = 'Loss')

    fname = '../img/loss.png'
    fig.savefig(fname)
    drive.upload_image(fname)

    fig = plot_history(
        history = bleu_history,
        color = COLORS['green'],
        title = 'Average BLEU',
        ylabel = 'BLEU')

    fname = '../img/bleu.png'
    fig.savefig(fname)
    drive.upload_image(fname)

    confusion_df = confusion_dataframe(
        translations['True Name'],
        translations['Our Name'],
        columns=['P', 'PP', 'TP', 'FP', 'FN'],
        orderby=['TP', 'PP'])

    fname = '../img/confusion.png'
    fig = plot_confusion_dataframe(confusion_df)
    fig.savefig(fname)
    drive.upload_image(fname)

    drive.log_dataframe(translations.sort_values('BLEU', ascending=False).head(20), 'translations.txt')
    drive.log_dataframe(confusion_df[confusion_df['PP'] > 0]['PP'], 'names.txt')


if __name__ == '__main__':
    total_time_start = time.time()

    drive = Drive('NameGen')

    # Loading data
    data_dir = '../data'
    methods = pd.read_csv(os.path.join(data_dir, 'methods_tokenized.csv'), delimiter='\t')

    drive.log('Data was loaded', 'log.txt')

    source_vocab, target_vocab, corpora = read_langs('source', 'name', methods)

    # Danger
    corpora = corpora[:300]

    (x_train, y_train), (x_valid, y_valid), (x_test, y_test) = train_test_valid_split(corpora, source_vocab, target_vocab)

    model = Seq2seq(source_vocab.n_words, target_vocab.n_words, 60)
    optimizer = Adam(model.parameters(), lr=0.001)

    # shuffle training indices
    assert x_train.shape[0] == y_train.shape[0]
    batch_indices = torch.randperm(x_train.shape[0])

    cross_entropy = nn.CrossEntropyLoss()
    nll = nn.NLLLoss()
    softmax = nn.Softmax(dim=1)

    loss_history = []
    bleu_history = []

    # training_dist = torch.zeros(target_vocab.n_words, MAX_SEQ_LENGTH)
    total_batches = int(len(batch_indices)/BATCH_SIZE)
    for epoch in range(100):
        epoch_start_time = time.time()
        
        total_loss = 0
        train_start_time = time.time()

        for batch in batch_generator(batch_indices, BATCH_SIZE):
            x_batch = x_train[batch, :] 
            # Here is the fix
            y_batch = y_train[batch, :-1] 
            y_true = y_train[batch, 1:]
            # (batch_size, vocab_size, seq_length)
            H = model.forward_train(x_batch, y_batch)
            loss = cross_entropy(H, y_true)
    #         H = softmax(H)
    #         likelihood = torch.gather(H, 1, y_true.unsqueeze(1))
    #         loss = torch.neg(torch.sum(torch.log(likelihood)))
            assert loss.item() > 0
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() #/(BATCH_SIZE*MAX_SEQ_LENGTH)

        train_time_elapsed = time.time() - train_start_time
        
        eval_start_time = time.time()
        translations = evaluate(model, x_valid, y_valid)
        eval_time_elapsed = time.time() - eval_start_time

        bleu_history.append(translations['BLEU'].mean())
        loss_history.append(total_loss / total_batches)

        epoch_time_elapsed = time.time() - epoch_start_time
        total_time_elapsed = time.time() - total_time_start

        log_dict = OrderedDict([
            ("Epoch", epoch + 1),
            ("Average loss", total_loss / total_batches),
            ("Average BLEU", bleu_history[-1]),
            ("Unique names", len(translations['Our Name'].unique())),
            ("Epoch time", time_str(epoch_time_elapsed)),
            ("Training time", time_str(train_time_elapsed)),
            ("Evaluation time", time_str(eval_time_elapsed)),
            ("Total time", time_str(total_time_elapsed))
        ])

        write_training_log(
            log_dict,
            bleu_history,
            loss_history,
            translations)