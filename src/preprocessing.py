from constants import *

import numpy as np

import torch


class Vocab:
    def __init__(self):
        self.index2word = {0: "SOS", 1: "EOS", 2: "UNK", 3: "PAD"}
        self.word2index = {v: k for k, v in self.index2word.items()}
        self.word2count = {}

        self.n_words = 4
      
    def index_words(self, tokenized):
        for word in tokenized:
            self.index_word(word)
        return tokenized

    def index_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1
    
    def unindex_words(self, indices):
        return ' '.join([self.index2word[i] for i in indices])


# def normalize_string(s):
#     s = s.lower().strip()
#     return s


def read_langs(source_lang, target_lang, corpora):
    source_vocab = Vocab()
    target_vocab = Vocab()
    
    source_corpora = []
    target_corpora = []
    for i, row in corpora.iterrows():
        source_sent = row[source_lang]
        target_sent = row[target_lang]
        
        source_tokenized = source_sent.split()
        target_tokenized = target_sent.split()
        if len(source_tokenized) > MAX_LENGTH or \
           len(target_tokenized) > MAX_LENGTH:
            continue
        
        source_corpora.append(source_tokenized)
        target_corpora.append(target_tokenized)
        
        target_vocab.index_words(target_tokenized)
        source_vocab.index_words(source_tokenized)
    return source_vocab, target_vocab, list(zip(source_corpora, target_corpora))


def __indexes_from_sentence(vocab, sentence):
    return [vocab.word2index[word] for word in sentence]


def __tensor_from_sentence(lang, sentence):
    indexes = __indexes_from_sentence(lang, sentence)
    indexes.append(EOS_token)
    indexes.insert(0, SOS_token)
    # we need to have all sequences the same length to process them in batches
    if len(indexes) < MAX_SEQ_LENGTH:
        indexes += [PAD_token]*(MAX_SEQ_LENGTH-len(indexes))
    tensor = torch.LongTensor(indexes)
    if USE_CUDA: var = tensor.cuda()
    return tensor


def __tensors_from_pair(source_sent, target_sent, source_vocab, target_vocab):
    source_tensor = __tensor_from_sentence(source_vocab, source_sent).unsqueeze(1)
    target_tensor = __tensor_from_sentence(target_vocab, target_sent).unsqueeze(1)
    
    return (source_tensor, target_tensor)


def __get_xy(corpora, source_vocab, target_vocab):
    tensors = []
    for source_sent, target_sent in corpora:
        tensors.append(__tensors_from_pair(source_sent, target_sent, source_vocab, target_vocab))

    x, y = zip(*tensors)
    x = torch.transpose(torch.cat(x, dim=-1), 1, 0)
    y = torch.transpose(torch.cat(y, dim=-1), 1, 0)
    
    return x, y


def __remove_sos_eos_pad(sentence):
    return [word_id for word_id in sentence if word_id > 3]


def unindex(sentence, vocab):
    sentence = __remove_sos_eos_pad(sentence)
    sentence = vocab.unindex_words(sentence)
    return sentence


def train_test_valid_split(corpora, source_vocab, target_vocab):
    test_size = int(TEST_SIZE * len(corpora))
    valid_size = int(VALID_SIZE * len(corpora))
    train_size = len(corpora) - (test_size + valid_size)

    indices = np.random.permutation(len(corpora))

    train_idx = indices[:train_size]
    valid_idx = indices[train_size:(train_size + valid_size)]
    test_idx = indices[-test_size:]

    # Sizes
    assert len(train_idx) == train_size
    assert len(valid_idx) == valid_size
    assert len(test_idx) == test_size

    # Coverage (union of train, validation, and test sets should cover all corpora)
    assert set.union(*map(set, [train_idx, valid_idx, test_idx])) == set(range(len(corpora)))

    corpora = np.array(corpora)
    train_corpora = corpora[train_idx]
    valid_corpora = corpora[valid_idx]
    test_corpora = corpora[test_idx]

    x_train, y_train = __get_xy(train_corpora, source_vocab, target_vocab)
    x_valid, y_valid = __get_xy(valid_corpora, source_vocab, target_vocab)
    x_test, y_test = __get_xy(test_corpora, source_vocab, target_vocab)

    return (x_train, y_train), (x_valid, y_valid), (x_test, y_test)