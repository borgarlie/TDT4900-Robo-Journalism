import torch.nn as nn
import torch.nn.functional as F
import torch


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
