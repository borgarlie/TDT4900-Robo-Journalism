import nltk
import re
from nltk import tokenize


def read_file(path):
    text = open(path, encoding='utf-8').read()
    text = clean_logger_output(text)  # when using logger
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
        title_list = clean_text(title_list)
        tokenized = tokenize_list(title_list)
        tokenized_titles.append(tokenized)
    return tokenized_titles, output


def avg_bleu_score(titles, output):
    smoothing = nltk.translate.bleu_score.SmoothingFunction()
    smoothing_functions = [smoothing.method0, smoothing.method1, smoothing.method2, smoothing.method3,
                           smoothing.method4]
    for i in range(0, len(smoothing_functions)):
        score = avg_bleu_score_smoothing_function(titles, output, smoothing_functions[i])
        print("Bleu score with smoothing %d: %.4f " % (i, score))


def avg_bleu_score_smoothing_function(titles, output, smoothing_function):
    avg_bleu = 0
    num_examples = len(titles)
    for i in range(num_examples):
        avg_bleu += nltk.translate.bleu_score.sentence_bleu(titles[i], output[i], smoothing_function=smoothing_function)
    return avg_bleu/num_examples


def clean_text(input_txt):
    output_txt = []
    for line in input_txt:
        line = re.sub(r'\d+', ' ', line)
        line = re.sub(r'[-.]', ' ', line)
        line = re.sub(r'<EOS>', ' ', line)
        line = line.strip()
        if len(line) > 0:
            output_txt.append(line)
    return output_txt


def clean_logger_output(text):
    output = ""
    for line in text.split('\n'):
        cleaned_line = re.sub('^(.*?)INFO - ', '', line)
        cleaned_line = re.sub('^(.*?)ERROR - ', '', cleaned_line)
        if len(cleaned_line) > 0:
            output += cleaned_line + '\n'
    return output


def tokenize_list(input_list):
    output_list = []
    for line in input_list:
        output_list.append(nltk.wordpunct_tokenize(line))
    return output_list


if __name__ == '__main__':
    # nltk.download('punkt')
    # path = 'experiments/ntb_paramsearch_1/output.txt'
    # path = '../output_for_eval/pointer_gen_ntb_baseline.txt'
    # path = '../output_for_eval/pointer_gen_ntb_baseline_2.txt'
    # path = '../output_for_eval/ntb_adagrad_test1.txt'
    # path = '../output_for_eval/cnn_pretrained_1.log'
    path = '../output_for_eval/ntb_beam_output_2.log'
    print("Started extracting titles...")
    titles, output = read_file(path)
    print(titles[1])
    print(output[1])
    print(len(titles))
    print(len(output))
    print("Done extracting titles...")
    print("starting to evaluate %d examples..." % len(titles))
    avg_bleu_score(titles, output)
