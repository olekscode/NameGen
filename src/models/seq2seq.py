from constants import *

import torch
import torch.nn as nn


class EncoderRNN(nn.Module):
    def __init__(self, vocab_size, hidden_size, n_layers=1):
        super(EncoderRNN, self).__init__()
        
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers=n_layers, batch_first=True, bidirectional=True)
        
    def forward(self, word_inputs, hidden): # word_inputs: (batch_size, seq_length), h: (h_or_c, layer_n_direction, batch, seq_length)
        # Note: we run this all at once (over the whole input sequence)
        seq_len = len(word_inputs)
        
        # embedded (batch_size, seq_length, hidden_size)
        embedded = self.embedding(word_inputs) 
        # output (batch_size, seq_length, hidden_size*directions)
        # hidden (h: (batch_size, num_layers*directions, hidden_size),
        #         c: (batch_size, num_layers*directions, hidden_size))
        output, hidden = self.lstm(embedded, hidden)
        return output, hidden

    def init_hidden(self, batches):
        hidden = torch.zeros(2, self.n_layers*2, batches, self.hidden_size)
        if USE_CUDA: hidden = hidden.cuda()
        return hidden


class DecoderRNN(nn.Module):
    def __init__(self, vocab_size, hidden_size, n_layers=1):
        super(DecoderRNN, self).__init__()
        
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers=n_layers, batch_first=True, bidirectional=False)
        
    def forward(self, word_inputs, hidden):
        # Note: we run this one by one
        # embedded (batch_size, 1, hidden_size)
        embedded = self.embedding(word_inputs).unsqueeze_(1)
        output, hidden = self.lstm(embedded, hidden)
        return output, hidden


class Seq2seq(nn.Module):
    def __init__(self, input_vocab_size, output_vocab_size, hidden_size):
        super(Seq2seq, self).__init__()
        
        self.n_layers = 1
        
        self.encoder = EncoderRNN(input_vocab_size, int(hidden_size/2), self.n_layers)
        self.decoder = DecoderRNN(output_vocab_size, hidden_size, self.n_layers)
        
        self.W = nn.Linear(hidden_size, output_vocab_size)
        self.softmax = nn.Softmax(dim=0)
        
        self.hidden_size = hidden_size
        
    def forward_encoder(self, x):
        batch_size = x.shape[0]
        init_hidden = self.encoder.init_hidden(batch_size)
        encoder_outputs, encoder_hidden = self.encoder(x, init_hidden)
        encoder_hidden_h, encoder_hidden_c = encoder_hidden
        
        decoder_hidden_h = encoder_hidden_h.reshape(self.n_layers, batch_size, self.hidden_size)
        decoder_hidden_c = encoder_hidden_c.reshape(self.n_layers, batch_size, self.hidden_size)
        return decoder_hidden_h, decoder_hidden_c
    
    def forward_train(self, x, y):
        decoder_hidden_h, decoder_hidden_c = self.forward_encoder(x)
        
        H = []
        for i in range(y.shape[1]):
            input = y[:, i]
            decoder_output, decoder_hidden = self.decoder(input, (decoder_hidden_h, decoder_hidden_c))
            decoder_hidden_h, decoder_hidden_c = decoder_hidden
            # h: (batch_size, vocab_size)
            h = self.W(decoder_output.squeeze(1))
            # h: (batch_size, vocab_size, 1)
            H.append(h.unsqueeze(2))
        
        # H: (batch_size, vocab_size, seq_len)
        return torch.cat(H, dim=2)
    
    def forward(self, x):
        decoder_hidden_h, decoder_hidden_c = self.forward_encoder(x)
        
        current_y = SOS_token
        result = [current_y]
        counter = 0
        while current_y != EOS_token and counter < 100:
            input = torch.tensor([current_y])
            decoder_output, decoder_hidden = self.decoder(input, (decoder_hidden_h, decoder_hidden_c))
            decoder_hidden_h, decoder_hidden_c = decoder_hidden
            # h: (vocab_size)
            h = self.W(decoder_output.squeeze(1)).squeeze(0)
            y = self.softmax(h)
            _, current_y = torch.max(y, dim=0)
            current_y = current_y.item()
            result.append(current_y)
            counter += 1
            
        return result