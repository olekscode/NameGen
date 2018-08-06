from constants import EOS_TOKEN, MAX_LENGTH, DEVICE

import numpy as np

import torch


class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence:
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


class TensorBuilder:
    def __init__(self, input_lang, output_lang):
        self.input_lang = input_lang
        self.output_lang = output_lang


    def indexesFromSentence(self, lang, sentence):
        return [lang.word2index[word] for word in sentence]


    def tensorFromSentence(self, lang, sentence):
        indexes = self.indexesFromSentence(lang, sentence)
        indexes.append(EOS_TOKEN)
        return torch.tensor(indexes, dtype=torch.long, device=DEVICE).view(-1, 1)


    def tensorsFromPair(self, pair):
        input_tensor = self.tensorFromSentence(self.input_lang, pair[0])
        target_tensor = self.tensorFromSentence(self.output_lang, pair[1])
        return (input_tensor, target_tensor)


def read_langs(lang1, lang2, corpora):
    input_lang = Lang(lang1)
    output_lang = Lang(lang2)
    
    input_sentences = []
    target_sentences = []

    for i, row in corpora.iterrows():
        input_sent = row[lang1]
        target_sent = row[lang2]
        
        input_tokenized = str(input_sent).split()
        target_tokenized = str(target_sent).split()

        if len(input_tokenized) > MAX_LENGTH and len(input_tokenized) > 0 and len(target_tokenized) > 0:
            continue
        
        input_sentences.append(input_tokenized)
        target_sentences.append(target_tokenized)
        
        output_lang.addSentence(target_tokenized)
        input_lang.addSentence(input_tokenized)

    return input_lang, output_lang, list(zip(input_sentences, target_sentences))


def train_validation_test_split(data, train_proportion, validation_proportion, test_proportion):
    if train_proportion + validation_proportion + test_proportion != 1.0:
        raise ValueError("Train, test, and validation proportions don't sum up to 1.")

    test_size = int(test_proportion * len(data))
    valid_size = int(validation_proportion * len(data))
    train_size = int(train_proportion * len(data))

    indices = np.random.permutation(len(data))

    train_idx = indices[:train_size]
    valid_idx = indices[train_size:(train_size + valid_size)]
    test_idx = indices[-test_size:]

    data = np.array(data)
    train_set = data[train_idx]
    valid_set = data[valid_idx]
    test_set = data[test_idx]

    return train_set, valid_set, test_set