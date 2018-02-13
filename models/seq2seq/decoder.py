import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable


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

        # TODO: Should dim = 0 ?
        output = F.log_softmax(self.out(output[0]), dim=1)

        # print(output, flush=True)
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
        # TODO: Make sure we have fixed embedding things. Should also probably fix it for the other model (?)
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

    def forward(self, input, hidden, encoder_outputs, full_input, batch_size=1, use_cuda=True):
        embedded_input = self.embedding(input).view(1, batch_size, self.embedding_size)
        embedded_input = self.dropout(embedded_input)

        # TODO: Does view still work when we use embedded input here now? Or do we need to do embedded_input[0] ?
        # TODO: alternative: Do not use view ?

        # calculate new decoder state
        decoder_output, decoder_hidden = self.gru(embedded_input, hidden)

        # TODO: They are 100% alike the first round. What happens at round 2 ? Which one should we use?

        # calculate attention weights
        decoder_state_linear = self.decoder_state_linear(decoder_hidden)
        attention_dist = F.tanh((self.w_h * encoder_outputs) + decoder_state_linear)
        attention_dist = (self.attention_weight_v * attention_dist).sum(2)  # TODO: -1 ? 1? what dimension here?
        attention_dist = F.softmax(attention_dist, dim=1)  # TODO: dim = -1 ? what dim ?

        # calculate context vectors
        # TODO: Check the unsqueeze and squeeze here
        encoder_context = torch.bmm(attention_dist.transpose(0, 1).unsqueeze(1), encoder_outputs.transpose(0, 1))
        # sum over the length
        combined_context = torch.cat((decoder_hidden.squeeze(0), encoder_context.squeeze(1)), 1)

        p_vocab = F.softmax(self.out_vocabulary(self.out_hidden(combined_context)), dim=1)
        #
        # print("p_vocab", flush=True)
        # print(p_vocab, flush=True)

        pointer_combined = torch.cat((combined_context, embedded_input.squeeze(0)), 1)  # TODO: is [0] correct?

        p_gen = F.sigmoid(self.pointer_linear(pointer_combined))

        # create temporal variable to use for distributions
        token_input_dist = Variable(torch.zeros((batch_size, self.vocabulary_size + self.max_length)))
        padding_matrix = Variable(torch.zeros(batch_size, self.max_length))
        if use_cuda:
            token_input_dist = token_input_dist.cuda()
            padding_matrix = padding_matrix.cuda()

        # in place scatter add
        token_input_dist.scatter_add_(1, full_input.transpose(0, 1), attention_dist.transpose(0, 1))

        # print("token_input_dist", flush=True)
        # print(token_input_dist, flush=True)
        #
        # print("token[0]", flush=True)
        # print(token_input_dist.data[0][10], flush=True)
        #
        # print("full_input", flush=True)
        # print(full_input.transpose(0, 1), flush=True)
        # print("attention", flush=True)
        # print(attention_dist.transpose(0, 1), flush=True)

        p_final = torch.cat((p_vocab * p_gen, padding_matrix), 1) + (1 - p_gen) * token_input_dist

        # print("p_final", flush=True)
        # print(p_final, flush=True)
        # exit()

        return p_final, decoder_hidden, attention_dist
