import numpy as np

from utils.data_prep import *


def expand_and_prune_beams(vocabulary, beams, encoder_outputs, decoder, expansions=5, keep_beams=10):
    generated_beams = []
    for i in range(len(beams)):
        generated_beams += beams[i].expand(vocabulary, encoder_outputs, decoder, expansions)
    return prune_beams(generated_beams, keep_beams)


# Takes in a set of beams and returns the best scoring beams up until num_keep_beams
def prune_beams(beams, num_keep_beams):
    return sorted(beams, reverse=True)[:num_keep_beams]


class Beam:
    def __init__(self, decoded_word_sequence, decoded_outputs, attention_weights, scores, input_token, input_hidden,
                 full_input_variable, extended_vocab):
        self.decoded_word_sequence = decoded_word_sequence
        self.decoded_outputs = decoded_outputs
        self.scores = scores  # This is a list of log(output from softmax) for each word in the sequence
        self.input_token = input_token
        self.input_hidden = input_hidden
        self.attention_weights = attention_weights
        self.full_input_variable = full_input_variable
        self.extended_vocab = extended_vocab

    def get_avg_score(self):
        if len(self.scores) == 0:
            # TODO: Could this become a bug? Should it be -9999999 ?
            return 0.0
        return sum(self.scores) / len(self.scores)

    def __lt__(self, other):
        return self.get_avg_score().__lt__(other.get_avg_score())

    def generate_expanded_beams(self, vocabulary, topv, topi, decoder_hidden, decoder_attention, expansions=5):
        i = 0
        while i < expansions:
            next_word = topi[0][i]
            if has_trigram(self.decoded_outputs, next_word):
                expansions += 1
                i += 1
                continue
            if len(self.scores) < 4 and (next_word == EOS_token or next_word == PAD_token):
                expansions += 1
                i += 1
                continue
            decoded_outputs = list(self.decoded_outputs) + [next_word]
            next_unpacked_word = get_word_from_token(next_word, vocabulary, self.extended_vocab)
            decoded_words = list(self.decoded_word_sequence) + [next_unpacked_word]
            # Using log(score) to be able to sum instead of multiply,
            # so that we are able to take the average based on number of tokens in the sequence
            # also need to set a minimum value, since we can not take log of 0
            next_score = topv[0][i] if topv[0][i] > 1e-8 else 1e-8
            log_score = np.log(next_score)
            new_scores = list(self.scores) + [log_score]
            new_attention_weights = self.attention_weights.clone()
            new_attention_weights[len(self.decoded_word_sequence)] = decoder_attention.data
            i += 1
            yield Beam(decoded_words, decoded_outputs, new_attention_weights, new_scores, next_word, decoder_hidden,
                       self.full_input_variable, self.extended_vocab)

    # return list of expanded beams. return self if current beam is at end of sentence
    def expand(self, vocabulary, encoder_outputs, decoder, expansions=5):
        if self.input_token == EOS_token or self.input_token == PAD_token:
            return list([self])
        # expand beam
        if self.input_token >= vocabulary.n_words:
            self.input_token = UNK_token
        decoder_input = Variable(torch.LongTensor([self.input_token]))
        decoder_input = decoder_input.cuda() if use_cuda else decoder_input
        decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, self.input_hidden, encoder_outputs,
                                                                    self.full_input_variable, 1)
        # Increasing expansions with *= 10 to compensate for possible trigram overlaps
        topv, topi = decoder_output.data.topk(expansions * 10)
        return list(self.generate_expanded_beams(vocabulary, topv, topi, decoder_hidden, decoder_attention, expansions))

    def cut_attention_weights_at_sequence_length(self):
        self.attention_weights = self.attention_weights[:len(self.decoded_word_sequence)]

    def __repr__(self):
        return str(self.get_avg_score())

    def __str__(self):
        return self.__repr__()
