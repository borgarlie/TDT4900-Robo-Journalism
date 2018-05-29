import re

import os


def read_file(path):
    text = open(path, encoding='utf-8').read()
    titles = []
    output = []
    articles = []
    not_truth = False
    for line in text.split('\n'):
        if line.startswith('>'):
            articles.append(line[1:])
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
    articles = clean_reference(articles)
    return articles, titles, output


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


def calculate_novel_percentage_n_grams(articles, hypotheses, n):
    all_total = 0
    all_novel = 0
    for i in range(0, len(articles)):
        total, novel = calculate_novel_n_grams(articles[i], hypotheses[i], n)
        all_total += total
        all_novel += novel
    if all_total == 0:  # Dirty fix
        return 0.0
    return all_novel / all_total


def calculate_novel_n_grams(article, hypothesis, n):
    article_n_grams = []
    article_split = article.split()
    for i in range(n, len(article_split)+1):
        n_gram = article_split[i-n:i]
        joined = ' '.join(n_gram)
        article_n_grams.append(joined)

    hypothesis_n_grams = []
    hypothesis_split = hypothesis.split()
    for i in range(n, len(hypothesis_split)+1):
        n_gram = hypothesis_split[i - n:i]
        joined = ' '.join(n_gram)
        hypothesis_n_grams.append(joined)

    total = len(hypothesis_n_grams)
    novel = 0
    for n_gram in hypothesis_n_grams:
        if n_gram not in article_n_grams:
            novel += 1

    return total, novel


def calculate_novel_percentage_sentences(articles, hypotheses):
    all_total = 0
    all_novel = 0
    for i in range(0, len(articles)):
        total, novel = calculate_novel_sentences(articles[i], hypotheses[i])
        all_total += total
        all_novel += novel
    if all_total == 0:  # Dirty fix
        return 0.0
    return all_novel / all_total


def calculate_novel_sentences(article, hypothesis):
    article_sentences = article.split("\n")
    hypothesis_sentences = hypothesis.split("\n")
    total = len(hypothesis_sentences)
    novel = 0
    for sentence in hypothesis_sentences:
        if sentence not in article_sentences:
            novel += 1
    return total, novel


def read_directory(directory):
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(".txt"):
            yield os.path.join(directory, file)


if __name__ == '__main__':
    path = '../output_for_eval/combined_objective/beam_output_combined_objective_0.75_epoch5.txt'
    test_data = True

    print("Started extracting titles...")
    article, reference, hypothesis = read_file(path)

    if test_data:
        article = article[:11000]
        reference = reference[:11000]
        hypothesis = hypothesis[:11000]
    else:
        article = article[:2000]
        reference = reference[:2000]
        hypothesis = hypothesis[:2000]

    assert len(article) == len(reference)
    assert len(reference) == len(hypothesis)

    print("Total = %d" % len(hypothesis))

    sentences_article = []
    sentences_reference = []
    sentences_hypothesis = []

    # article = []
    # hypothesis = []
    #
    # article_path = '../output_for_eval/sumGAN_output/article/'
    # article_files = list(read_directory(article_path))
    # article_files.sort()
    #
    # decoded_path = '../output_for_eval/sumGAN_output/decoded/'
    # decoded_files = list(read_directory(decoded_path))
    # decoded_files.sort()
    #
    # for tup in zip(article_files, decoded_files):
    #     art = open(tup[0], encoding='utf-8').read().strip()
    #     dec = open(tup[1], encoding='utf-8').read().strip()
    #     article.append(art)
    #     hypothesis.append(dec)
    # sentences_article = []
    # sentences_hypothesis = hypothesis

    for i in range(0, len(hypothesis)):
        sentences_article.append(split_sentence(article[i]))
        sentences_reference.append(split_sentence(reference[i]))
        sentences_hypothesis.append(split_sentence(hypothesis[i]))

    hypothesis = sentences_hypothesis
    reference = sentences_reference

    print("Done extracting titles...")
    print("starting to evaluate %d examples..." % len(hypothesis))

    n_gram_to_calculate = [1, 2, 3, 4]

    # Reference
    # for n in n_gram_to_calculate:
    #     percentage = calculate_novel_percentage_n_grams(article, reference, n)
    #     print("Percentage novelty for n = %d: %.4f" % (n, percentage))
    #
    # percentage = calculate_novel_percentage_sentences(sentences_article, sentences_reference)
    # print("Percentage novelty for sentences: %.4f" % percentage)

    # Hypothesis
    for n in n_gram_to_calculate:
        percentage = calculate_novel_percentage_n_grams(article, hypothesis, n)
        print("Percentage novelty for n = %d: %.4f" % (n, percentage))

    percentage = calculate_novel_percentage_sentences(sentences_article, sentences_hypothesis)
    print("Percentage novelty for sentences: %.4f" % percentage)
