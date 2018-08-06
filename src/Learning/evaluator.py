from metrics import confusion_dataframe, bleu, rouge, f1_score
from constants import SOS_TOKEN, EOS_TOKEN

from collections import OrderedDict

import pandas as pd


class Evaluator:
    def __init__(self, pairs, input_lang, output_lang):
        self.pairs = pairs
        self.input_lang = input_lang
        self.output_lang = output_lang


    def evaluate(self, model):
        names = self.__predict_names(model)
        # confusion_df = self.__build_confusion_dataframe(names)
        return names


    def __predict_names(self, model):
        x = []
        y_true = []
        y_pred = []

        for training_pair in self.pairs:
            input_tensor = training_pair[0]
            output_tensor = model(input_tensor)

            x.append(input_tensor)
            y_true.append(training_pair[1])
            y_pred.append(output_tensor)
        
        # Convert numbers to words and remove SOS and EOS tokens
        x = [[self.input_lang.index2word[index.item()]
                for index in sent
                if index.item() not in [SOS_TOKEN, EOS_TOKEN]
            ] for sent in x]

        y_true = [[self.output_lang.index2word[index.item()]
                for index in sent
                if index.item() not in [SOS_TOKEN, EOS_TOKEN]
            ] for sent in y_true]

        y_pred = [[self.output_lang.index2word[index]
                for index in sent
                if index not in [SOS_TOKEN, EOS_TOKEN]
            ] for sent in y_pred]
        
        names = pd.DataFrame(OrderedDict([
            ('Source', [' '.join(sent) for sent in x]),
            ('True Name', [' '.join(sent) for sent in y_true]),
            ('Our Name', [' '.join(sent) for sent in y_pred]),
            ('BLEU', [bleu(y_true[i], y_pred[i]) for i in range(len(y_true))]),
            ('ROUGE', [rouge(y_true[i], y_pred[i]) for i in range(len(y_true))]),
            ('F1', [f1_score(y_true[i], y_pred[i]) for i in range(len(y_true))])
        ]))
        
        return names

    
    def __build_confusion_dataframe(self, names):
        return confusion_dataframe(
            names['True Name'],
            names['Our Name'],
            columns=['P', 'PP', 'TP', 'FP', 'FN'],
            orderby=['TP', 'PP'])