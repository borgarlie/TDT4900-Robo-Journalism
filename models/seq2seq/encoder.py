import torch.nn as nn

######################################################################
# The Encoder
# -----------
#
# The encoder of a seq2seq network is a RNN that outputs some value for
# every word from the input sentence. For every input word the encoder
# outputs a vector and a hidden state, and uses the hidden state for the
# next input word.


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=1, batch_size=1):
        super(EncoderRNN, self).__init__()
        self.batch_size = batch_size
        self.n_layers = n_layers
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, bidirectional=True)

    def forward(self, input, input_lengths, hidden):
        embedded = self.embedding(input)
        outputs = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)  # packed
        outputs, hidden = self.gru(outputs, hidden)
        outputs, output_lengths = nn.utils.rnn.pad_packed_sequence(outputs)  # unpack (back to padded)
        return outputs, hidden
