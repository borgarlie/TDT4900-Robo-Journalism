from rouge import Rouge
import re
import json


def read_file(path):
    text = open(path, encoding='utf-8').read()
    titles = []
    output = []
    not_truth = False
    for line in text.split('\n'):
        if line.startswith('>'):
            not_truth = True
        elif line.startswith('='):
            titles.append(line[1:])
        elif line.startswith('<') and not_truth:
            output.append(line[1:])
            not_truth = False
    output = clean_text(output)

    return titles, output


def clean_text(input_txt):
    output_txt = []
    for line in input_txt:
        line = re.sub(r'\d+', ' ', line)
        line = re.sub(r'[-.]', ' ', line)
        line = re.sub(r'<EOS>', ' ', line)
        output_txt.append(line)
    return output_txt


if __name__ == '__main__':
    path = '../../output_for_eval/baseline_adam_test.txt'
    print("Started extracting titles...")
    reference, hypothesis = read_file(path)
    print("Done extracting titles...")
    print("starting to evaluate %d examples..." % len(reference))
    rouge = Rouge()
    scores = rouge.get_scores(hypothesis, reference, avg=True)
    print(json.dumps(scores, indent=2))
