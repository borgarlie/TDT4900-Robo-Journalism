import re


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
                sentences.append(t)
    return "\n".join(sentences).strip()


if __name__ == '__main__':

    path = '../output_for_eval/cnn_beam_output_2_13epoch_3_20_1000.log'
    # path = '../output_for_eval/cnn_pretrained_1.log'

    print("Started extracting titles...")
    reference, hypothesis = read_file(path)

    reference = reference[:1000]
    hypothesis = hypothesis[:1000]

    path_to_reference = "../for_rouge/pretrained1/reference_test/"
    path_to_modelsummary = "../for_rouge/pretrained1/cnn_pretrain_epoch13/"

    for i in range(0, len(reference)):
        reference[i] = split_sentence(reference[i])

    for i in range(0, len(hypothesis)):
        hypothesis[i] = split_sentence(hypothesis[i])

    for i in range(0, len(reference)):
        with open(path_to_reference + "%d_reference.txt" % i, 'w') as file:
            file.write(reference[i])

    for i in range(0, len(hypothesis)):
        with open(path_to_modelsummary + "%d_modelsummary.txt" % i, 'w') as file:
            file.write(hypothesis[i])

    print("Done")
