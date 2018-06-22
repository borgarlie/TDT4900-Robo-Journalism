import re


def read_file(path, num_total):
    text = open(path, encoding='utf-8').read()
    if path.endswith(".log"):
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
    output = clean_modelsummary(output)[:num_total]
    titles = clean_reference(titles)[:num_total]
    for i in range(0, len(output)):
        output[i] = split_sentence(output[i])
        titles[i] = split_sentence(titles[i])
    return titles, output


def read_article(path, num_total):
    text = open(path, encoding='utf-8').read()
    if path.endswith(".log"):
        text = clean_logger_output(text)  # when using logger
    article = []
    for line in text.split('\n'):
        if line.startswith('>'):
            article.append(line[1:])
    article = clean_reference(article)[:num_total]
    for i in range(0, len(article)):
        article[i] = split_sentence(article[i])
    return article


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


def save_to_path(save_path, object_list, save_name):
    for i in range(0, len(object_list)):
        with open(save_path + "%d_" % i + save_name + ".txt", 'w', encoding='utf-8') as file:
            file.write(object_list[i])


def split_beamsearch_to_multiple_files(save_path, num_summaries):
    print("Started extracting titles...")

    # save paths
    path_to_article = save_path + '/article/'
    path_to_combined = save_path + '/combined_objective/'
    path_to_combined_mle = save_path + '/combined_objective_with_mle/'
    path_to_discriminator = save_path + '/discriminator_objective/'
    path_to_discriminator_mle = save_path + '/discriminator_objective_with_mle/'
    path_to_rouge_1 = save_path + '/rouge_1_objective/'
    path_to_rouge_1_mle = save_path + '/rouge_1_objective_with_mle/'
    path_to_rouge_2 = save_path + '/rouge_2_objective/'
    path_to_rouge_2_mle = save_path + '/rouge_2_objective_with_mle/'
    path_to_pretrained = save_path + '/pretrained/'
    path_to_ground_truth = save_path + '/ground_truth/'

    pretrained_path = '/baseline_pretrain/beam_output_baseline_epoch13.txt'
    rouge_1_path = '/eval_on_test_data/beam_output_rl_strat_rouge_1_argmax_negative_epoch5_test_data.txt'
    rouge_1_mle_path = '/rl_strat_test_data/beam_output_rl_strat_mle_epoch4_test_data.txt'
    rouge_2_path = '/eval_on_test_data/beam_output_rl_strat_rouge_2_argmax_negative_epoch10_test_data.txt'
    rouge_2_mle_path = '/eval_on_test_data/beam_output_rl_strat_rouge_2_mle_0.9984_argmax_negative_epoch3_test_data.txt'
    discriminator_path = '/4_16_4_seqGAN_test_data/beam_output_seqGAN_strat_4_16_4_epoch4_test_data.txt'
    discriminator_mle_path = '/4_16_4_seqGAN_test_data/beam_output_seqGAN_strat_4_16_4_mle_epoch5_test_data.txt'
    combined_path = '/combined_rouge_2_test_data/beam_output_rl_strat_combined_rouge_2_epoch7_test_data.txt'
    combined_mle_path = '/combined_rouge_2_test_data/beam_output_rl_strat_combined_rouge_2_mle_epoch7_test_data.txt'

    article = read_article('../output_for_eval' + pretrained_path, num_summaries)
    ground_truth, pretrained = read_file('../output_for_eval' + pretrained_path, num_summaries)

    _, rouge_1 = read_file('../output_for_eval' + rouge_1_path, num_summaries)
    _, rouge_1_mle = read_file('../output_for_eval' + rouge_1_mle_path, num_summaries)
    _, rouge_2 = read_file('../output_for_eval' + rouge_2_path, num_summaries)
    _, rouge_2_mle = read_file('../output_for_eval' + rouge_2_mle_path, num_summaries)
    _, discriminator = read_file('../output_for_eval' + discriminator_path, num_summaries)
    _, discriminator_mle = read_file('../output_for_eval' + discriminator_mle_path, num_summaries)
    _, combined = read_file('../output_for_eval' + combined_path, num_summaries)
    _, combined_mle = read_file('../output_for_eval' + combined_mle_path, num_summaries)

    print("Done reading files. starting saving.")

    save_to_path(path_to_article, article, 'article')
    save_to_path(path_to_ground_truth, ground_truth, 'ground_truth')
    save_to_path(path_to_pretrained, pretrained, 'pretrained')
    save_to_path(path_to_rouge_1, rouge_1, 'rouge_1')
    save_to_path(path_to_rouge_1_mle, rouge_1_mle, 'rouge_1_mle')
    save_to_path(path_to_rouge_2, rouge_2, 'rouge_2')
    save_to_path(path_to_rouge_2_mle, rouge_2_mle, 'rouge_2_mle')
    save_to_path(path_to_discriminator, discriminator, 'discriminator')
    save_to_path(path_to_discriminator_mle, discriminator_mle, 'discriminator_mle')
    save_to_path(path_to_combined, combined, 'combined')
    save_to_path(path_to_combined_mle, combined_mle, 'combined_mle')


if __name__ == '__main__':

    save_path = '../../results'
    num_summaries = 11000

    split_beamsearch_to_multiple_files(save_path, num_summaries)

    print("Done")
