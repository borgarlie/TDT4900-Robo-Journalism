import time
import torch.nn as nn
import torch.nn.functional as F
import torch

from utils.logger import *


class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, max_length, n_layers=1, dropout_p=0.1):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size * 2  # * 2 because of bidirectional encoder
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size, self.n_layers)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs, batch_size=1):
        embedded = self.embedding(input).view(1, batch_size, self.hidden_size)
        embedded = self.dropout(embedded)

        cat = torch.cat((embedded[0], hidden[0]), 1)
        temp = self.attn(cat)
        attn_weights = F.softmax(temp, dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs.transpose(0, 1))

        output = torch.cat((embedded[0], attn_applied.squeeze(1)), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output, hidden = self.gru(output, hidden)
        output = F.log_softmax(self.out(output[0]), dim=1)

        return output, hidden, attn_weights


class PointerGeneratorDecoder(nn.Module):
    """
        The simplest pointer generator model, without coverage.

        max_length = max length of input to encoder (the actual input sequence)
    """
    def __init__(self, hidden_size, embedding_size, vocabulary_size, max_length, n_layers=1, dropout_p=0.1):
        super(PointerGeneratorDecoder, self).__init__()
        self.hidden_size = hidden_size * 2  # * 2 because of bidirectional encoder
        self.embedding_size = embedding_size
        self.vocabulary_size = vocabulary_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.vocabulary_size, self.embedding_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(input_size=self.embedding_size, hidden_size=self.hidden_size, num_layers=self.n_layers)

        # used to calculate W_s*s_t + b_attn
        self.decoder_state_linear = nn.Linear(self.hidden_size, self.hidden_size)
        # used to calculate w_h * e_t for all t
        self.w_h = nn.Parameter(torch.randn(self.hidden_size))
        self.attention_weight_v = nn.Parameter(torch.randn(self.hidden_size))

        self.out_hidden = nn.Linear(self.hidden_size * 2, self.embedding_size)
        self.out_vocabulary = nn.Linear(self.embedding_size, self.vocabulary_size)
        self.pointer_linear = nn.Linear(self.hidden_size * 2 + self.embedding_size, 1)

        self.full_vocab_padding = nn.ZeroPad2d((0, self.vocabulary_size + max_length, 0, 0))
        self.right_padding = nn.ZeroPad2d((0, max_length, 0, 0))

    def forward(self, input, hidden, encoder_outputs, full_input, batch_size=1, use_cuda=True):
        before1 = time.time()
        embedded_input = self.embedding(input).view(1, batch_size, self.embedding_size)
        embedded_input = self.dropout(embedded_input)
        timings[timings_var_decoder_test1] += time.time() - before1

        # calculate new decoder state
        before2 = time.time()
        decoder_output, decoder_hidden = self.gru(embedded_input, hidden)
        timings[timings_var_decoder_test2] += time.time() - before2

        # calculate attention weights
        before3 = time.time()
        decoder_state_linear = self.decoder_state_linear(decoder_hidden)
        attention_dist = F.tanh((self.w_h * encoder_outputs) + decoder_state_linear)
        attention_dist = (self.attention_weight_v * attention_dist).sum(2)
        attention_dist = F.softmax(attention_dist, dim=0).transpose(0, 1)
        timings[timings_var_decoder_test3] += time.time() - before3

        # calculate context vectors
        before4 = time.time()
        encoder_context = torch.bmm(attention_dist.unsqueeze(1), encoder_outputs.transpose(0, 1))
        combined_context = torch.cat((decoder_hidden.squeeze(0), encoder_context.squeeze(1)), 1)
        timings[timings_var_decoder_test4] += time.time() - before4

        before5 = time.time()
        p_vocab = F.softmax(self.out_vocabulary(self.out_hidden(combined_context)), dim=1)
        timings[timings_var_decoder_test5] += time.time() - before5

        before6 = time.time()
        pointer_combined = torch.cat((combined_context, embedded_input.squeeze(0)), 1)
        p_gen = F.sigmoid(self.pointer_linear(pointer_combined))
        timings[timings_var_decoder_test6] += time.time() - before6

        before7 = time.time()
        token_input_dist = self.full_vocab_padding(attention_dist)
        token_input_dist = token_input_dist.narrow(1, attention_dist.size()[1],
                                                   token_input_dist.size()[1] - attention_dist.size()[1])
        timings[timings_var_decoder_test7] += time.time() - before7

        # in place scatter add
        before8 = time.time()
        token_input_dist.scatter_add_(1, full_input, attention_dist)
        timings[timings_var_decoder_test8] += time.time() - before8

        before9 = time.time()
        p_final = self.right_padding((p_vocab * p_gen)) + (1 - p_gen) * token_input_dist
        timings[timings_var_decoder_test9] += time.time() - before9

        return p_final, decoder_hidden, attention_dist
