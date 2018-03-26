import sys

sys.path.append('../..')  # ugly dirtyfix for imports to work

from preprocess.preprocess_pointer import *
from utils.data_prep import *

stop_signs = ['.', '!', '?', '...']

if __name__ == '__main__':
    relative_path = "../../data/cnn_pickled/cnn_pointer_50k"
    summary_pairs, vocabulary = load_dataset(relative_path)
    print("Done loading dataset")
    summary_pairs = summary_pairs[-500:]
    stop_tokens = []
    for sign in stop_signs:
        token = vocabulary.word2index[sign]
        stop_tokens.append(token)

    references = []
    modelsummarys = []
    # Generate lead 3 and fetch reference summary
    for pair in summary_pairs:
        extended_vocab = pair.unknown_tokens
        # getting reference
        reference = []
        tokens_reference = pair.abstract_tokens[:-1]
        start = 0
        reference = []
        for i in range(1, len(tokens_reference)):
            if tokens_reference[i] in stop_tokens:
                next_sentence = get_sentence_from_tokens(tokens_reference[start:i + 1], vocabulary, extended_vocab)
                reference.append(next_sentence.strip())
                start = i + 1
        references.append(reference)
        # getting lead 3
        tokens_article = pair.article_tokens[:-1]
        i = 0
        num_stop_tokens = 0
        tokens_since_last = 0
        start = 0
        article = []
        while num_stop_tokens < 3:
            i += 1
            tokens_since_last += 1
            if tokens_article[i] in stop_tokens:
                if tokens_since_last > 7:
                    num_stop_tokens += 1
                    next_sentence = get_sentence_from_tokens(tokens_article[start:i+1], vocabulary, extended_vocab)
                    article.append(next_sentence.strip())
                start = i + 1
                tokens_since_last = 0
        modelsummarys.append(article)

    path_to_reference = "../for_rouge/lead3/reference/"
    path_to_modelsummary = "../for_rouge/lead3/modelsummary/"

    for i in range(0, len(references)):
        with open(path_to_reference + "%d_reference.txt" % i, 'w') as file:
            for j in range(0, len(references[i])):
                file.write(references[i][j])
                if j < len(references[i])-1:
                    file.write("\n")

    for i in range(0, len(modelsummarys)):
        with open(path_to_modelsummary + "%d_modelsummary.txt" % i, 'w') as file:
            for j in range(0, len(modelsummarys[i])):
                file.write(modelsummarys[i][j])
                if j < len(modelsummarys[i]) - 1:
                    file.write("\n")

    print("Done")
