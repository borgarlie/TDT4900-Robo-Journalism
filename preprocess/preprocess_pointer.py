import pickle

from utils.data_prep import split_category_and_article


class Vocabulary:
    def __init__(self):
        self.word2index = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
        self.word2count = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
        self.index2word = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.n_words = 4  # Count PAD, SOS, EOS and UNK

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

    def get_size(self):
        return self.n_words


def generate_vocabulary(articles, abstracts, max_size=-1, with_categories=False):
    if max_size == -1:
        max_size = len(articles)
    vocabulary = Vocabulary()
    for i in range(0, max_size):
        if with_categories:
            _, art = split_category_and_article(articles[i])
        else:
            art = articles[i]
        vocabulary.add_sentence(art)
    for i in range(0, max_size):
        vocabulary.add_sentence(abstracts[i])
    print("Done generating vocabulary")
    return vocabulary


def limit_vocabulary(vocabulary, limit):
    # limit = limit - 4
    vocab_words = [(w, vocabulary.word2count[w]) for w in vocabulary.word2count.keys()]
    vocab_words = sorted(vocab_words, key=lambda tup: tup[1], reverse=True)
    if len(vocab_words) > limit:
        vocab_words = vocab_words[:limit]
    new_vocab = Vocabulary()
    for word_tuple in vocab_words:
        new_vocab.word2index[word_tuple[0]] = new_vocab.n_words
        new_vocab.word2count[word_tuple[0]] = word_tuple[1]
        new_vocab.index2word[new_vocab.n_words] = word_tuple[0]
        new_vocab.n_words += 1
    return new_vocab


class SummaryPair:
    def __init__(self, article, abstract, vocabulary):
        self.article_tokens = []
        self.unked_article_tokens = []
        self.abstract_tokens = []
        self.unked_abstract_tokens = []
        self.unknown_tokens = dict()

        self.add_article_and_abstract(article, abstract, vocabulary)

    def add_article_and_abstract(self, article, abstract, vocabulary):
        self.article_tokens, self.unked_article_tokens = self.create_tokens(article, vocabulary)
        self.abstract_tokens, self.unked_abstract_tokens = self.create_tokens(abstract, vocabulary)

    def create_tokens(self, sentence, vocabulary):
        tokens = []
        unked_tokens = []
        for w in sentence.split(" "):
            if w in vocabulary.word2index:
                tokens.append(vocabulary.word2index[w])
                unked_tokens.append(vocabulary.word2index[w])
            else:
                if w not in self.unknown_tokens:
                    self.unknown_tokens[w] = len(vocabulary.index2word) + len(self.unknown_tokens)
                tokens.append(self.unknown_tokens[w])
                unked_tokens.append(3)  # <UNK>
        tokens.append(2)  # <EOS>
        unked_tokens.append(2)  # <EOS>
        return tokens, unked_tokens


class DataSet:
    def __init__(self, vocabulary, summary_pairs):
        self.vocabulary = vocabulary
        self.summary_pairs = summary_pairs


def create_summary_pairs(articles, abstracts, vocabulary, max_size=-1, with_categories=False):
    pairs = []
    if max_size == -1:
        max_size = len(articles)
    for i in range(0, max_size):
        if with_categories:
            _, art = split_category_and_article(articles[i])
        else:
            art = articles[i]
        pair = SummaryPair(art, abstracts[i], vocabulary)
        pairs.append(pair)
    return pairs


def save_dataset(dataset, path):
    total_path = path + ".pickle"  # serialized ?
    with open(total_path, 'wb') as handle:
        pickle.dump(dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("Done saving dataset")


def load_dataset(path):
    total_path = path + ".pickle"
    with open(total_path, 'rb') as f:
        dataset = pickle.load(f)
    print("Done loading dataset")
    return dataset.summary_pairs, dataset.vocabulary


def read_file(relative_path):
    print("Reading lines...")
    articles = open(relative_path + '.article.txt', encoding='utf-8').read().strip().split('\n')
    abstracts = open(relative_path + '.abstract.txt', encoding='utf-8').read().strip().split('\n')
    print("Read %s articles" % len(articles))
    print("Read %s abstracts" % len(abstracts))
    return articles, abstracts


if __name__ == '__main__':
    relative_path_cnn = '../data/ntb_preprocessed/ntb_80_5cat'
    articles, abstracts = read_file(relative_path_cnn)

    with_categories = True
    max_articles = -1
    limit = 30000

    vocabulary = generate_vocabulary(articles, abstracts, max_articles, with_categories)
    limited_vocabulary = limit_vocabulary(vocabulary, limit)

    summary_pairs = create_summary_pairs(articles, abstracts, limited_vocabulary, max_articles, with_categories)
    dataset = DataSet(limited_vocabulary, summary_pairs)

    # Test
    # vocab_words = [(w, limited_vocabulary.word2count[w]) for w in limited_vocabulary.word2count.keys()]
    # for tup in vocab_words:
    #     print("%s - %d" % (tup[0], tup[1]), flush=True)

    save_path_dataset = '../data/ntb_pickled/ntb_pointer_30k'
    save_dataset(dataset, save_path_dataset)

    # load_path = '../data/cnn_pickled/cnn_pointer_40k'
    # summary_pairs, vocabulary = load_dataset(load_path)
    # print(len(summary_pairs))
    # print(vocabulary.n_words)
