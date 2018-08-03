from constants import MAX_LENGTH, SOS_TOKEN, EOS_TOKEN
from models.encoder import EncoderRNN
from models.decoder import AttnDecoderRNN
from util import time_str

import time
import random
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.optim as optim


class Seq2Seq(nn.Module):
    def __init__(self, input_size, output_size, hidden_size,
                 learning_rate, teacher_forcing_ratio, device):
        super(Seq2Seq, self).__init__()

        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.device = device

        self.encoder = EncoderRNN(input_size, hidden_size)
        self.decoder = AttnDecoderRNN(hidden_size, output_size)

        self.encoder_optimizer = optim.SGD(self.encoder.parameters(), lr=learning_rate)
        self.decoder_optimizer = optim.SGD(self.decoder.parameters(), lr=learning_rate)

        self.criterion = nn.NLLLoss()


    def train(self, input_tensor, target_tensor, max_length=MAX_LENGTH):
        encoder_hidden = self.encoder.initHidden()

        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()

        input_length = input_tensor.size(0)
        target_length = target_tensor.size(0)

        encoder_outputs = torch.zeros(max_length + 1, self.encoder.hidden_size, device=self.device)

        loss = 0

        for ei in range(input_length):
            encoder_output, encoder_hidden = self.encoder(input_tensor[ei], encoder_hidden)
            encoder_outputs[ei] = encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_TOKEN]], device=self.device)
        decoder_hidden = encoder_hidden

        use_teacher_forcing = True if random.random() < self.teacher_forcing_ratio else False

        if use_teacher_forcing:
            # Teacher forcing: feed the target as the next input
            for di in range(target_length):
                decoder_output, decoder_hidden, decoder_attention = self.decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
                loss += self.criterion(decoder_output, target_tensor[di])
                decoder_input = target_tensor[di] # Teacher forcing
        else:
            # Without teacher forcing: use its own prediction as the next input
            for di in range(target_length):
                decoder_output, decoder_hidden, decoder_attention = self.decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
                topv, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze().detach() # detach from history as input

                loss += self.criterion(decoder_output, target_tensor[di])

                if decoder_input.item() == EOS_TOKEN:
                    break

        loss.backward()

        self.encoder_optimizer.step()
        self.decoder_optimizer.step()

        return loss.item() / target_length


    def trainIters(self, pairs, n_iters, logger, evaluator, log_every=100):
        start_total_time = time.time()
        start_epoch_time = time.time() # Reset every log_every
        start_train_time = time.time() # Reset every log_every

        total_loss = 0                 # Reset every log_every
        avg_loss_history = []
        avg_bleu_history = []
        avg_rouge_history = []


        for iter in range(1, n_iters + 1):
            training_pair = random.choice(pairs)
            input_tensor = training_pair[0]
            target_tensor = training_pair[1]

            loss = self.train(input_tensor, target_tensor)
            total_loss += loss

            if iter % log_every == 0:
                train_time_elapsed = time.time() - start_train_time

                start_eval_time = time.time()
                names, confusion_df = evaluator.evaluate(self)
                eval_time_elapsed = time.time() - start_eval_time

                avg_loss_history.append(total_loss / log_every)
                avg_bleu_history.append(names['BLEU'].mean())
                avg_rouge_history.append(names['ROUGE'].mean())

                epoch_time_elapsed = time.time() - start_epoch_time
                total_time_elapsed = time.time() - start_total_time

                log_dict = OrderedDict([
                    ("Iteration",  '{}/{} ({:.1f}%)'.format(iter, n_iters, iter / n_iters * 100)),
                    ("Average loss", avg_loss_history[-1]),
                    ("Average BLEU", avg_bleu_history[-1]),
                    ("Average ROUGE", avg_rouge_history[-1]),
                    ("Unique names", len(names['Our Name'].unique())),
                    ("Epoch time", time_str(epoch_time_elapsed)),
                    ("Training time", time_str(train_time_elapsed)),
                    ("Evaluation time", time_str(eval_time_elapsed)),
                    ("Total training time", time_str(total_time_elapsed))
                ])

                logger.write_training_log(
                    log_dict,
                    avg_bleu_history,
                    avg_rouge_history,
                    avg_loss_history,
                    names,
                    confusion_df)

                # Reseting counters
                total_loss = 0
                start_epoch_time = time.time()
                start_train_time = time.time()


    def forward(self, input_tensor, max_length=MAX_LENGTH):
        encoder_hidden = self.encoder.initHidden()

        input_length = input_tensor.size(0)

        encoder_outputs = torch.zeros(max_length + 1, self.encoder.hidden_size, device=self.device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = self.encoder(input_tensor[ei], encoder_hidden)
            encoder_outputs[ei] = encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_TOKEN]], device=self.device)
        decoder_hidden = encoder_hidden
        decoded_words = []

        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = self.decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_TOKEN:
                decoded_words.append(EOS_TOKEN)
                break
            else:
                decoded_words.append(topi.item())

            decoder_input = topi.squeeze().detach()

        return decoded_words