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
        return output, hidden, attn_weights


class PointerGeneratorDecoder(nn.Module):
    def __init__(self, hidden_size, output_size, max_length, n_layers=1, dropout_p=0.1):
        super(PointerGeneratorDecoder, self).__init__()
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
        self.generator_layer = nn.Linear(self.hidden_size * 2, 1)

    def forward(self, input, hidden, encoder_outputs, full_input, batch_size=1, use_cuda=True):
        embedded = self.embedding(input).view(1, batch_size, self.hidden_size)
        embedded = self.dropout(embedded)

        cat = torch.cat((embedded[0], hidden[0]), 1)
        temp = self.attn(cat)
        attn_weights = F.softmax(temp, dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs.transpose(0, 1))

        combined_context = torch.cat((embedded[0], attn_applied.squeeze(1)), 1)
        combined_context = self.attn_combine(combined_context).unsqueeze(0)

        gru_output, gru_hidden = self.gru(combined_context, hidden)

        p_vocabulary = F.log_softmax(self.out(gru_output[0]), dim=1)

        # TODO: Fix all this shit. should it be hidden instead of gru_hidden? do hidden need a squeeze?
        generator_layer_input = torch.cat((combined_context, hidden), 1)
        p_generator = F.sigmoid(self.gen_layer(generator_layer_input))

        token_input_dist = Variable(torch.zeros((full_input.size()[0], self.output_size + 500)))
        padding_matrix_2 = Variable(torch.zeros(full_input.size()[0], 500))
        if use_cuda:
            token_input_dist = token_input_dist.cuda()
            padding_matrix_2 = padding_matrix_2.cuda()

        token_input_dist.scatter_add_(1, full_input, attn_weights)

        p_vocabulary_scaled = p_vocabulary * p_generator
        padded_vocabulary_matrix = torch.cat((p_vocabulary_scaled, padding_matrix_2), 1)

        p_final = padded_vocabulary_matrix + (1 - p_generator) * token_input_dist

        return p_final, hidden, attn_weights

