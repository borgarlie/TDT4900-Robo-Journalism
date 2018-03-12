import nltk
from nltk import tokenize


def read_real_and_fake_data(path_real, path_fake):
    real_data = open(path_real, encoding='utf-8').read()
    fake_data = open(path_fake, encoding='utf-8').read()
    real_titles = tokenize_data_file(real_data)
    fake_titles = tokenize_data_file(fake_data)
    return real_titles, fake_titles


def tokenize_file(file):
    temp_list = []
    for line in file.split('\n'):
        temp_list.append(tokenize.sent_tokenize(line))
    return temp_list


def avg_bleu_score(titles, output):
    avg_bleu = 0
    num_examples = len(titles)
    cc = nltk.translate.bleu_score.SmoothingFunction()
    for i in range(num_examples):
        avg_bleu += nltk.translate.bleu_score.sentence_bleu(titles[i], output[i], smoothing_function=cc.method2)
    return avg_bleu/num_examples


def count_equal_titles(path_real, path_fake):
    real_data = open(path_real, encoding='utf-8').read()
    fake_data = open(path_fake, encoding='utf-8').read()
    real_titles = tokenize_file(real_data)
    fake_titles = tokenize_file(fake_data)
    equal_titles = 0
    for i in range(len(real_titles)):
        if real_titles[i] == fake_titles[i]:
            equal_titles += 1
    print("Equal titles: %d" % equal_titles)


def count_almost_equal_titles(path_real, path_fake):
    real_data = open(path_real, encoding='utf-8').read()
    fake_data = open(path_fake, encoding='utf-8').read()
    print("Done reading")
    real_titles = tokenize_data_file(real_data)
    fake_titles = tokenize_data_file(fake_data)
    print("Done tokenizing")
    equal_titles = 0
    for i in range(len(real_titles)):
        equal = True
        min_len = min(len(real_titles[i]), len(fake_titles[i]))
        for j in range(0, min_len):
            if real_titles[i][j] != fake_titles[i][j]:
                equal = False
                break
        if equal:
            equal_titles += 1
    print("Almost equal titles: %d" % equal_titles)


def tokenize_data_file(data):
    lines_and_words = []
    for line in data.split('\n'):
        words = []
        for word in line.split(' '):
            words.append(word)
        lines_and_words.append(words)
    return lines_and_words


if __name__ == '__main__':
    # nltk.download('punkt')

    path_real = "../../data/ntb_real_data/ntb_real_1.abstract.txt"
    path_fake = "../../data/ntb_fake_data/ntb_fake_1.abstract.txt"

    print("Started extracting titles...")
    references, samples = read_real_and_fake_data(path_real, path_fake)
    print("Done extracting titles...")
    print("starting to evaluate %d examples..." % len(references))
    print("Got a BLEU score equal: %.4f " % avg_bleu_score(references, samples))

    count_equal_titles(path_real, path_fake)
    count_almost_equal_titles(path_real, path_fake)
