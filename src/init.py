from models.seq2seq import Seq2Seq
from preprocessing import read_langs, train_validation_test_split, TensorBuilder
from evaluator import Evaluator
from logger import ConsoleLogger, DriveLogger

from constants import (
    WRITE_LOGS_TO_GOOGLE_DRIVE,
    TRAIN_PROP,
    VALID_PROP,
    TEST_PROP,
    HIDDEN_SIZE,
    LEARNING_RATE,
    TEACHER_FORCING_RATIO,
    DEVICE
)

import os
import time
import math
from collections import OrderedDict

import numpy as np
import pandas as pd


if __name__ == '__main__':

    logger = DriveLogger('NameGen') if WRITE_LOGS_TO_GOOGLE_DRIVE else ConsoleLogger()

    try:
        total_time_start = time.time()

        logger.log('Loading the data')
        data_dir = '../data'
        methods = pd.read_csv(os.path.join(data_dir, 'methods_tokenized.csv'), delimiter='\t')

        logger.log('Building input and output languages')
        input_lang, output_lang, pairs = read_langs('source', 'name', methods)

        logger.log('Splitting data into train, validation, and test sets')
        train_pairs, valid_pairs, test_pairs = train_validation_test_split(
            pairs, TRAIN_PROP, VALID_PROP, TEST_PROP)

        logger.log('Converting data entries to tensors')
        tensor_builder = TensorBuilder(input_lang, output_lang)
        train_pairs = [tensor_builder.tensorsFromPair(pair) for pair in train_pairs]
        valid_pairs = [tensor_builder.tensorsFromPair(pair) for pair in valid_pairs]
        test_pairs = [tensor_builder.tensorsFromPair(pair) for pair in test_pairs]

        logger.log('Building the model')
        model = Seq2Seq(
            input_size=input_lang.n_words,
            output_size=output_lang.n_words,
            hidden_size=HIDDEN_SIZE,
            learning_rate=LEARNING_RATE,
            teacher_forcing_ratio=TEACHER_FORCING_RATIO,
            device=DEVICE)

        logger.log(str(model))

        logger.log('Initializing evaluator')
        evaluator = Evaluator(valid_pairs, input_lang, output_lang)

        logger.log('Training the model')
        model.trainIters(train_pairs, 75000, logger, evaluator, log_every=250)

        logger.log('Saving the model')
        torch.save(model.state_dict(), '../data/trained_model.pt')

        logger.log('Done')

    except Exception as e:
        # Log the error message and raise it again so see more info
        logger.log("Error: " + str(e))
        raise e