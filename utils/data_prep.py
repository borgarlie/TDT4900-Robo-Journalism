import re
import time
import torch
from torch.autograd import Variable
from utils.logger import *

#
# Functions used to prepare data
#

PAD_token = 0
SOS_token = 1
EOS_token = 2
UNK_token = 3

use_cuda = torch.cuda.is_available()


def pad_seq(seq, max_length):
    new_seq = seq.copy()
    new_seq += [PAD_token for i in range(max_length - len(seq))]
    return new_seq


def category_from_string(category_string):
    categories = []
    for cat in category_string:
        categories.append(int(cat))
    return categories


def indexes_from_sentence(vocabulary, sentence):
    return [vocabulary.word2index[word] for word in sentence.split(' ')] + [EOS_token]


def indexes_from_sentence_no_eos(vocabulary, sentence):
    return [vocabulary.word2index[word] for word in sentence.split(' ')]


def variable_from_sentence(vocabulary, sentence):
    indexes = indexes_from_sentence(vocabulary, sentence)
    indexes.append(EOS_token)
    result = Variable(torch.LongTensor(indexes).view(-1, 1))
    if use_cuda:
        return result.cuda()
    else:
        return result


def variables_from_pair(article, title, vocabulary):
    input_variable = variable_from_sentence(vocabulary, article)
    target_variable = variable_from_sentence(vocabulary, title)
    return input_variable, target_variable


# This can be optimized to instead search for ">>>" to just split on word position
def split_category_and_article(article):
    return article.split(">>>")


def concat_encoder_hidden_directions(h):
    """ do the following transformation:
        (#directions * #layers, #batch, hidden_size) -> (#layers, #batch, #directions * hidden_size)
        to compensate for two directions in a bidirectional encoder
    """
    return torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2)


# Used in the pointer-generator models to get the whole sentence when we have two district vocabularies
def get_sentence_from_tokens(tokens, vocabulary, extended_vocabulary):
    words = []
    for token in tokens:
        words.append(get_word_from_token(token, vocabulary, extended_vocabulary))
    return ' '.join(words)


def get_sentence_from_tokens_and_clean(tokens, vocabulary, extended_vocabulary):
    words = []
    for token in tokens:
        words.append(get_word_from_token(token, vocabulary, extended_vocabulary))
    sequence = ' '.join(words)
    sequence = clean_sequence(sequence)
    return sequence


def clean_sequence(line):
    line = re.sub(r'<EOS>', '', line)
    line = re.sub(r'<PAD>', '', line)
    line = line.strip()
    return line


# used in the pointer-generator models
def get_word_from_token(token, vocabulary, extended_vocabulary):
    if token >= vocabulary.n_words:
        next_unpacked_word = "<UNK>"
        for key, value in extended_vocabulary.items():
            if value == token:
                next_unpacked_word = key
                break
        return next_unpacked_word
    return vocabulary.index2word[token]


def get_sentence_from_tokens_unked(tokens, vocabulary):
    words = []
    for token in tokens:
        words.append(get_word_from_token_unked(token, vocabulary))
    return ' '.join(words)


def get_word_from_token_unked(token, vocabulary):
    if token >= vocabulary.n_words:
        return "<UNK>"
    return vocabulary.index2word[token]


def is_whole_batch_pad_or_eos(batched_input):
    before = time.time()
    is_only_pad_or_eos = True
    for token_index in range(0, len(batched_input)):
        # token_index = this batch element
        if batched_input[token_index][0] != PAD_token and batched_input[token_index][0] != EOS_token:
            is_only_pad_or_eos = False
            break
    timings[timings_var_check_eos_pad] += (time.time() - before)
    return is_only_pad_or_eos


# using indexes
def has_trigram(current_sequence, next_word):
    if len(current_sequence) < 3:
        return False
    word1 = current_sequence[-2]
    word2 = current_sequence[-1]
    word3 = next_word
    for i in range(2, len(current_sequence)):
        if current_sequence[i - 2] != word1:
            continue
        if current_sequence[i - 1] != word2:
            continue
        if current_sequence[i] != word3:
            continue
        # all equal = overlap
        return True
    return False
