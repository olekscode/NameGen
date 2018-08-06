import constants
from models.seq2seq import Seq2Seq
from preprocessing import read_langs, train_validation_test_split, TensorBuilder
from evaluator import Evaluator
from logger import DefaultLogger, DriveLogger

import os
import time
import math
from collections import OrderedDict

import numpy as np
import pandas as pd

import torch


if __name__ == '__main__':

    logger = DriveLogger('NameGen') if constants.WRITE_LOGS_TO_GOOGLE_DRIVE else DefaultLogger()

    try:
        total_time_start = time.time()

        logger.log('Loading the data')
        methods = pd.read_csv(os.path.join(constants.DATA_DIR, 'methods_tokenized.csv'), delimiter='\t')

        logger.log('Building input and output languages')
        input_lang, output_lang, pairs = read_langs('source', 'name', methods)

        logger.log('Number of unique input tokens: {}\n'
                   'Number of unique output tokens: {}'.format(
            input_lang.n_words,
            output_lang.n_words
        ))

        logger.log('Serializing input and output languages to pickles')

        logger.save_pickle(input_lang, os.path.join(constants.DATA_DIR, 'input_lang.pkl'))
        logger.save_pickle(output_lang, os.path.join(constants.DATA_DIR, 'output_lang.pkl'))

        logger.log('Splitting data into train, validation, and test sets')
        train_pairs, valid_pairs, test_pairs = train_validation_test_split(
            pairs, constants.TRAIN_PROP, constants.VALID_PROP, constants.TEST_PROP)

        logger.log('Train size: {}\n'
                   'Validation size: {}\n'
                   'Test size: {}'.format(
            len(train_pairs),
            len(valid_pairs),
            len(test_pairs)
        ))

        logger.log('Serializing train, validation, and test sets to pickles')

        logger.save_pickle(train_pairs, os.path.join(constants.DATA_DIR, 'train_pairs.pkl'))
        logger.save_pickle(valid_pairs, os.path.join(constants.DATA_DIR, 'valid_pairs.pkl'))
        logger.save_pickle(test_pairs, os.path.join(constants.DATA_DIR, 'test_pairs.pkl'))

        logger.log('Converting data entries to tensors')
        tensor_builder = TensorBuilder(input_lang, output_lang)
        train_pairs = [tensor_builder.tensorsFromPair(pair) for pair in train_pairs]
        valid_pairs = [tensor_builder.tensorsFromPair(pair) for pair in valid_pairs]
        test_pairs = [tensor_builder.tensorsFromPair(pair) for pair in test_pairs]

        logger.log('Building the model')
        model = Seq2Seq(
            input_size=input_lang.n_words,
            output_size=output_lang.n_words,
            hidden_size=constants.HIDDEN_SIZE,
            learning_rate=constants.LEARNING_RATE,
            teacher_forcing_ratio=constants.TEACHER_FORCING_RATIO,
            device=constants.DEVICE)

        logger.log(str(model))

        logger.log('Initializing evaluators')
        evaluator = Evaluator(valid_pairs, input_lang, output_lang)
        test_set_evaluator = Evaluator(test_pairs, input_lang, output_lang)

    except Exception as e:
        # Log the error message and raise it again so see more info
        logger.log("Error: " + str(e))
        raise e

    successful = False
    restarting_attempts = 10
    iters_completed = 0

    while not successful:
        try:
            logger.log('Training the model')
            model.trainIters(train_pairs, iters_completed, constants.NUM_ITER, logger, evaluator, constants.LOG_EVERY)

            logger.log('Saving the model')
            torch.save(model.state_dict(), os.path.join(constants.RESULTS_DIR, 'trained_model.pt'))

            successful = True

            logger.log('Evaluating on test set')
            names = test_set_evaluator.evaluate(model)
            logger.save_dataframe(names, os.path.join(constants.RESULTS_DIR, 'test_names.csv'))

            logger.log('Done')

        except Exception as e:
            restarted = False

            while not restarted and restarting_attempts > 0 or type(logger) == DriveLogger:
                # Log the error message and restart from the last successful state
                try:
                    print('Trying to restart', restarting_attempts)
                    if restarting_attempts == 0:
                        logger = DefaultLogger()
                        print("Switched to default logger")
                        restarting_attempts = 5

                    restarting_attempts -= 1

                    logger.log("Error during training: " + str(e))

                    try:
                        with open(os.path.join(constants.LOGS_DIR, 'iters_completed.txt'), 'r') as f:
                            iters_completed = int(f.read())
                    except Exception as e:
                        logger.log("Error: " + str(e))
                        logger.log("Can't read the number of completed iterations. Starting from 0")

                    logger.log("Restarting from iteration {}\n{} more iterations to go ({:.1f}%)".format(
                        iters_completed + 1,
                        constants.NUM_ITER - iters_completed,
                        (constants.NUM_ITER - iters_completed) / constants.NUM_ITER * 100))
            
                    try:
                        logger.log('Loading the last trained model')
                        model.load_state_dict(torch.load(os.path.join(constants.RESULTS_DIR, 'trained_model.pt')))
                    except Exception as e:
                        logger.log("Error: " + str(e))
                        logger.log("Can't load the last trained model. Starting from scratch")

                        model = Seq2Seq(
                            input_size=input_lang.n_words,
                            output_size=output_lang.n_words,
                            hidden_size=constants.HIDDEN_SIZE,
                            learning_rate=constants.LEARNING_RATE,
                            teacher_forcing_ratio=constants.TEACHER_FORCING_RATIO,
                            device=constants.DEVICE)

                    restarted = True

                except Exception as e:
                    logger = DefaultLogger()
                    print("Switched to default logger")


    