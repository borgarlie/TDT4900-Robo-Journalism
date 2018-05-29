import re


def read_file(path):
    text = open(path, encoding='utf-8').read()
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


if __name__ == '__main__':
    path = '../output_for_eval/rl_strat_test_data/beam_output_rl_strat_mle_epoch4_test_data.txt'

    print("Started extracting titles...")
    _, hypothesis = read_file(path)
    print(len(hypothesis))

    max_length = 120
    total_over_max = 0

    total_tokens = 0
    for hyp in hypothesis:
        tokens = len(hyp.split(" "))
        total_tokens += tokens
        if tokens >= max_length:
            total_over_max += 1
    average_tokens = total_tokens / len(hypothesis)

    print("Average tokens: ")
    print(average_tokens)

    print("Tokens over max:")
    print(total_over_max)
