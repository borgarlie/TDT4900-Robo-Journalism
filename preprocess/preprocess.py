from __future__ import unicode_literals, print_function, division
from io import open
from utils.data_prep import split_category_and_article


######################################################################
# We'll need a unique index per word to use as the inputs and targets of
# the networks later. To keep track of all this we will use a helper class
# called ``Lang`` which has word → index (``word2index``) and index → word
# (``index2word``) dictionaries, as well as a count of each word
# ``word2count`` to use to later replace rare words.

class Vocabulary:
    def __init__(self):
        self.word2index = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2}
        self.word2count = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2}
        self.index2word = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>"}
        self.n_words = 3  # Count PAD, SOS and EOS

    def add_sentence(self, sentence):
        for word in sentence.split(' '):
            self.add_word(word)

    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


def generate_vocabulary(relative_path, max_size=-1, with_categories=False):
    print("Reading lines...")
    article = open(relative_path + '.article.txt', encoding='utf-8').read().strip().split('\n')
    title = open(relative_path + '.title.txt', encoding='utf-8').read().strip().split('\n')
    print("Read %s articles" % len(article))
    print("Read %s title" % len(title))

    if max_size == -1:
        max_size = len(article)

    vocabulary = Vocabulary()

    longest_sentence = 0
    print("Counting words...")
    for i in range(0, max_size):
        if with_categories:
            _, art = split_category_and_article(article[i])
        else:
            art = article[i]
        vocabulary.add_sentence(art)
        sentence_length = len(art.strip().split(' '))
        if sentence_length > longest_sentence:
            longest_sentence = sentence_length
    for i in range(0, max_size):
        vocabulary.add_sentence(title[i])
        if len(title[i].split(' ')) > longest_sentence:
            longest_sentence = len(title[i].split(' '))

    print("longest sentence: ", longest_sentence)
    print("Counted words: %s" % vocabulary.n_words)
    return article[:max_size], title[:max_size], vocabulary


def generate_vocabulary_for_classifier(relative_path, relative_path_fake_data, max_size=-1, with_categories=False):
    print("Reading lines...")
    article = open(relative_path + '.article.txt', encoding='utf-8').read().strip().split('\n')
    title = open(relative_path + '.title.txt', encoding='utf-8').read().strip().split('\n')
    fake_titles = open(relative_path_fake_data + '.title.txt', encoding='utf-8').read().strip().split('\n')
    print("Read %s articles" % len(article))
    print("Read %s title" % len(title))
    print("Read %s fake_titles" % len(fake_titles))

    if max_size == -1:
        max_size = len(article)

    vocabulary = Vocabulary()

    longest_sentence = 0
    print("Counting words...")
    for i in range(0, max_size):
        if with_categories:
            _, art = split_category_and_article(article[i])
        else:
            art = article[i]
        vocabulary.add_sentence(art)
        sentence_length = len(art.strip().split(' '))
        if sentence_length > longest_sentence:
            longest_sentence = sentence_length
    for i in range(0, max_size):
        vocabulary.add_sentence(title[i])
        if len(title[i].split(' ')) > longest_sentence:
            longest_sentence = len(title[i].split(' '))
    for i in range(0, max_size):
        vocabulary.add_sentence(fake_titles[i])
        if len(fake_titles[i].split(' ')) > longest_sentence:
            longest_sentence = len(fake_titles[i].split(' '))

    print("longest sentence: ", longest_sentence)
    print("Counted words: %s" % vocabulary.n_words)
    return article[:max_size], title[:max_size], fake_titles[:max_size], vocabulary


class VocabularySizeItem:
    def __init__(self, key, value, num):
        self.key = key
        self.value = value
        self.num = num

    def __str__(self):
        return "%s %s" % (str(self.value), str(self.num))


def count_unk_items(vocab_items):
    vocab_items_single_char = []
    for item in vocab_items:
        if item.num == 1:
            vocab_items_single_char.append(item)

    print(vocab_items_single_char)

    unks = {}
    for item in vocab_items_single_char:
        unk = item.value[0]
        if unk not in unks:
            unks[unk] = []
        unks[unk].append(item)

    total_unks = 0
    for key, value in unks.items():
        print(key, len(value))
        total_unks += len(value)

    print("total = %d" % total_unks)


def get_list_to_unk(vocab_items, min_freq=3):
    vocab_items_single_char = {}
    for item in vocab_items:
        if item.num <= min_freq:
            vocab_items_single_char[item.value] = True
    return vocab_items_single_char


def replace_word_with_unk(word):
    return "<UNK" + word[0] + ">"


def save_articles_with_unk(articles, titles, relative_path, to_replace_vocabulary):
    articles_to_skip = []
    num_unks_10 = 0
    with open(relative_path + '.unk.article.txt', 'w') as f:
        for item in range(0, len(articles)):
            num_unk = 0
            words = articles[item].split(" ")
            unked_words = []
            for word in words:
                if word in to_replace_vocabulary:
                    unked_words.append(replace_word_with_unk(word))
                    num_unk += 1
                else:
                    unked_words.append(word)
            if num_unk >= 8:
                num_unks_10 += 1
                articles_to_skip.append(item)
            else:
                article = " ".join(unked_words)
                f.write(article)
                f.write("\n")
    print("Articles with 10 or more UNK: %d" % num_unks_10)
    with open(relative_path + '.unk.title.txt', 'w') as f:
        for item in range(0, len(titles)):
            if item in articles_to_skip:
                continue
            words = titles[item].split(" ")
            unked_words = []
            for word in words:
                if word in to_replace_vocabulary:
                    unked_words.append(replace_word_with_unk(word))
                else:
                    unked_words.append(word)
            title = " ".join(unked_words)
            f.write(title)
            f.write("\n")


def count_low_length(articles, titles):
    num_less_than_title = 0
    num_abit_more = 0
    num_too_short_article = 0
    num_too_short_title = 0
    for item in range(0, len(articles)):
        if len(articles[item].split(" ")) <= len(titles[item].split(" ")):
            num_less_than_title += 1
        elif len(articles[item].split(" ")) <= len(titles[item].split(" ")) + 10:
            num_abit_more += 1
        elif len(articles[item].split(" ")) < 25:
            num_too_short_article += 1
        elif len(titles[item].split(" ")) < 4:
            num_too_short_title += 1
    print("Articles less than 25 words: %d" % num_too_short_article)
    print("Titles less than 4 words: %d" % num_too_short_title)
    print("Articles with length==title length: %d" % num_less_than_title)
    print("Articles with length less than len(title) + 10: %d" % num_abit_more)


if __name__ == '__main__':

    relative_path_ntb = '../data/ntb_preprocessed/ntb_80_5cat'
    relative_path_ntb_unked = '../data/ntb_preprocessed/ntb_80_5cat.unk'

    with_categories = True
    article, title, vocabulary = generate_vocabulary(relative_path_ntb_unked, -1, with_categories)

    vocab_items = []
    for k, v in vocabulary.index2word.items():
        vocab_items.append(VocabularySizeItem(k, v, vocabulary.word2count[v]))

    minimum_frequency = 15
    unked_chars = get_list_to_unk(vocab_items, minimum_frequency)

    print("Unked chars: %d" % len(unked_chars))
    print("Remaining vocab: %d" % (len(vocab_items) - len(unked_chars)))

    # sorted_x = sorted(vocab_items, key=operator.attrgetter('num'), reverse=True)
    # for item in sorted_x:
    #     if item.num < minimum_frequency:
    #         print(item)

    count_low_length(article, title)

    # save_articles_with_unk(article, title, relative_path_ntb, unked_chars)
