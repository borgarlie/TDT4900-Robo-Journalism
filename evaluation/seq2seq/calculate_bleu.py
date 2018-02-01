import nltk
import re
from nltk import tokenize


def read_output_file(path):
    text = open(path, encoding='utf-8').read()
    titles = []
    output = []
    not_truth = False
    temp_title = None
    for line in text.split('\n'):
        if line.startswith('>'):
            temp_title = tokenize.sent_tokenize(line[1:])
            not_truth = True
        elif line.startswith('='):
            temp_title.append(line[1:])
        elif line.startswith('<') and not_truth:
            titles.append(temp_title)
            output.append(line[1:])
            not_truth = False

    output = clean_text(output)
    output = tokenize_list(output)

    tokenized_titles = []
    for title_list in titles:
        tokenized = tokenize_list(title_list)
        tokenized_titles.append(tokenized)

    return tokenized_titles, output


def read_real_and_fake_data(path_real, path_fake):
    real_data = open(path_real, encoding='utf-8').read()
    fake_data = open(path_fake, encoding='utf-8').read()

    # real_titles = tokenize_file(real_data)
    # fake_titles = tokenize_file(fake_data)
    # real_titles = tokenize_list(real_titles)
    # fake_titles = tokenize_list(fake_titles)

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

    # Without smoothingfunction we get around 0.3 and with it drops to 0.14. Not sure what is right though
    for i in range(num_examples):
        avg_bleu += nltk.translate.bleu_score.sentence_bleu(titles[i], output[i], smoothing_function=cc.method2)

    return avg_bleu/num_examples


def clean_text(input_txt):
    output_txt = []
    for line in input_txt:
        line = re.sub(r'\d+', ' ', line)
        line = re.sub(r'[-.]', ' ', line)
        line = re.sub(r'<EOS>', ' ', line)
        output_txt.append(line)
    return output_txt


def tokenize_list(input_list):
    output_list = []
    for line in input_list:
        output_list.append(nltk.wordpunct_tokenize(line[0]))  # should not be [0] on output data from model
    return output_list


def count_equal_titles(path_real, path_fake):
    real_data = open(path_real, encoding='utf-8').read()
    fake_data = open(path_fake, encoding='utf-8').read()
    real_titles = tokenize_file(real_data)
    fake_titles = tokenize_file(fake_data)

    equal_titles = 0

    for i in range(len(real_titles)):
        if real_titles[i] == fake_titles[i]:
            equal_titles += 1
    print(equal_titles)


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
    print(equal_titles)


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
    # path = 'experiments/ntb_paramsearch_1/output.txt'
    # print("Started extracting titles...")
    # titles, output = read_file(path)
    # print(titles[1])
    # print(output[1])
    # print(len(titles))
    # print(len(output))
    # print("Done extracting titles...")
    # print("starting to evaluate %d examples..." % len(titles))
    # print("Got a BLEU scorer equal: %.4f " % avg_bleu_score(titles, output))

    path_real = "../../data/ntb_preprocessed/ntb_80_5cat.unk.title.txt"
    path_fake = "../../data/ntb_fake_data/test_generated_fake_data.unk.title.txt"

    # print("Started extracting titles...")
    # references, samples = read_real_and_fake_data(path_real, path_fake)
    # print("Done extracting titles...")
    # print("starting to evaluate %d examples..." % len(references))
    # print("Got a BLEU scorer equal: %.4f " % avg_bleu_score(references, samples))

    # count_equal_titles(path_real, path_fake)
    count_almost_equal_titles(path_real, path_fake)
