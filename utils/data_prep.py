import torch
from torch.autograd import Variable

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
