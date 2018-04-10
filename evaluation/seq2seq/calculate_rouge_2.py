import re
from sumeval.metrics.rouge import RougeCalculator


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

    output = clean_modelsummary(output)
    titles = clean_reference(titles)
    return titles, output


def clean_modelsummary(input_txt):
    output_txt = []
    for line in input_txt:
        line = line.split(" ")
        line = " ".join(line[2:])
        # line = line[16:]
        # line = re.sub(r'\d+', '', line)
        # line = line[3:]
        line = re.sub(r'<EOS>', '', line)
        line = re.sub(r'<PAD>', '', line)
        line = line.strip()
        output_txt.append(line)
    return output_txt


def clean_reference(input_txt):
    output_txt = []
    for line in input_txt:
        line = re.sub(r'<EOS>', '', line)
        line = re.sub(r'<PAD>', '', line)
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


def add_back_delimiter(sentence_list, delimiter):
    for i in range(0, len(sentence_list)-1):
        sentence_list[i] += " " + delimiter
    return sentence_list


def split_sentence(sentence):
    sentences = []
    temp = add_back_delimiter(sentence.split(" . "), ".")
    for k in temp:
        temp2 = add_back_delimiter(k.split(" ? "), "?")
        for j in temp2:
            temp3 = add_back_delimiter(j.split(" ! "), "!")
            for t in temp3:
                if t.startswith("<"):
                    t = t[1:]
                sentences.append(t)
    return "\n".join(sentences).strip()


if __name__ == '__main__':
    # path = '../output_for_eval/pointer_gen_ntb_baseline_2.txt'
    # path = '../output_for_eval/cnn_beam_output_GAN_4.log'
    path = '../output_for_eval/cnn_beam_output_rougetest_3.log'

    # path = '../output_for_eval/cnn_beam_output_1.log'
    print("Started extracting titles...")
    reference, hypothesis = read_file(path)
    print(len(hypothesis))

    # for i in range(0, len(reference)):
    #     reference[i] = split_sentence(reference[i])
    #
    # for i in range(0, len(hypothesis)):
    #     hypothesis[i] = split_sentence(hypothesis[i])

    print("Done extracting titles...")
    print("starting to evaluate %d examples..." % len(reference))

    rouge = RougeCalculator(stopwords=False, lang="en", stemming=False)

    rouge_1_points = 0.0
    for i in range(0, len(reference)):
        rouge_1 = rouge.rouge_n(
            summary=hypothesis[i],
            references=reference[i],
            n=1)
        rouge_1_points += rouge_1
    rouge_1_points = rouge_1_points / len(reference)

    print("ROUGE-1: {}".format(
        rouge_1_points
    ).replace(", ", "\n"))

    rouge_2_points = 0.0
    for i in range(0, len(reference)):
        rouge_2 = rouge.rouge_n(
            summary=hypothesis[i],
            references=reference[i],
            n=2)
        rouge_2_points += rouge_2
    rouge_2_points = rouge_2_points / len(reference)

    print("ROUGE-2: {}".format(
        rouge_2_points
    ).replace(", ", "\n"))

    rouge_l_points = 0.0
    for i in range(0, len(reference)):
        rouge_l = rouge.rouge_l(
            summary=hypothesis[i],
            references=reference[i])
        rouge_l_points += rouge_l
    rouge_l_points = rouge_l_points / len(reference)

    print("ROUGE-L: {}".format(
        rouge_l_points
    ).replace(", ", "\n"))
