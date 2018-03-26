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
        line = line[16:]
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


if __name__ == '__main__':
    path = '../output_for_eval/cnn_beam_output_2_12epoch_5_30.log'
    print("Started extracting titles...")
    reference, hypothesis = read_file(path)

    reference = reference[:13000]
    hypothesis = hypothesis[:13000]

    path_to_reference = "../for_rouge/pretrained1/reference_test/"
    path_to_modelsummary = "../for_rouge/pretrained1/modelsummary_test/"

    for i in range(0, len(reference)):
        with open(path_to_reference + "%d_reference.txt" % i, 'w') as file:
            file.write(reference[i])

    for i in range(0, len(hypothesis)):
        with open(path_to_modelsummary + "%d_modelsummary.txt" % i, 'w') as file:
            file.write(hypothesis[i])

    print("Done")
