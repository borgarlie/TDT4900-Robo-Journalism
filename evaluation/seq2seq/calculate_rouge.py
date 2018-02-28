from rouge import Rouge
import re
import json


def read_file(path):
    text = open(path, encoding='utf-8').read()
    text = clean_logger_output(text)  # when using logger
    titles = []
    output = []
    not_truth = False
    for line in text.split('\n'):
        if line.startswith('>'):
            if not_truth:
                titles.pop()  # temp fix when we apparently do not get all the output...
            not_truth = True
        elif line.startswith('='):
            titles.append(line[1:])
        elif line.startswith('<') and not_truth:
            output.append(line[1:])
            not_truth = False
    output = clean_text(output)
    titles = clean_text(titles)
    return titles, output


def clean_text(input_txt):
    output_txt = []
    for line in input_txt:
        line = re.sub(r'\d+', ' ', line)
        line = re.sub(r'[-.]', ' ', line)
        line = re.sub(r'<EOS>', ' ', line)
        line = line.strip()
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


if __name__ == '__main__':
    # path = '../output_for_eval/pointer_gen_ntb_baseline_2.txt'
    path = '../output_for_eval/ntb_adagrad_test1.txt'
    print("Started extracting titles...")
    reference, hypothesis = read_file(path)
    print("Done extracting titles...")
    print("starting to evaluate %d examples..." % len(reference))
    rouge = Rouge()
    scores = rouge.get_scores(hypothesis, reference, avg=True)
    print(json.dumps(scores, indent=2))
