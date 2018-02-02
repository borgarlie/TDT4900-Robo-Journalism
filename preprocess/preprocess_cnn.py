import os
import re

dm_single_close_quote = u'\u2019'  # unicode
dm_double_close_quote = u'\u201d'  # unicode
# acceptable ways to end a sentence
end_tokens = ['.', '!', '?', '...', "'", "`", '"', dm_single_close_quote, dm_double_close_quote, ")"]

max_article_tokens = 200
min_article_tokens = 40
max_abstract_tokens = 80
min_abstract_tokens = 10

cnn_directory = os.fsencode("../data/cnn_clean/cnn_stories_tokenized/")
dm_directory = os.fsencode("../data/cnn_clean/dm_stories_tokenized/")

relative_save_path = "../data/cnn_preprocessed/"
save_name = "cnn_preprocessed_200_80"


class Errors:
    too_short_articles = 0
    too_short_abstracts = 0


def read_directory(directory):
    idx = 0
    for file in os.listdir(directory):
        idx += 1
        filename = os.fsdecode(file)
        if filename.endswith(".story"):
            yield idx, os.path.join(directory, file)


def process_stories(directory):
    articles = []
    abstracts = []
    num_stories = len(os.listdir(directory))
    print("Processing directory: %s" % directory)
    for idx, filename in read_directory(directory):
        if idx % 1000 == 0:
            print("Processing story %i of %i; %.2f percent done" %
                  (idx, num_stories, float(idx) * 100.0 / float(num_stories)))
        lines = read_lines(filename)
        article, abstract = get_article_and_abstract(lines)
        if article is None or abstract is None:
            continue
        articles.append(article)
        abstracts.append(abstract)
    return articles, abstracts


def read_lines(filename):
    lines = []
    with open(filename, 'r') as file:
        for line in file:
            lines.append(line.strip())
    return lines


def get_article_and_abstract(lines):
    # Lowercase everything
    lines = [line.lower() for line in lines]
    # Put periods on the ends of lines that are missing them
    lines = [fix_missing_period(line) for line in lines]
    # Separate out article and abstract sentences
    article_lines = []
    highlights = []
    next_is_highlight = False
    for idx, line in enumerate(lines):
        if line == "":
            continue  # empty line
        elif line.startswith("@highlight"):
            next_is_highlight = True
        elif next_is_highlight:
            highlights.append(line)
        else:
            article_lines.append(line)
    # Make article into a single string
    article = ' '.join(article_lines)
    # Make abstract into a single string
    abstract = ' '.join(highlights)
    # Restrict minimum length of article
    if len(article.split(" ")) < min_article_tokens:
        Errors.too_short_articles += 1
        return None, None
    # Restrict minimum length of abstract
    if len(abstract.split(" ")) < min_abstract_tokens:
        Errors.too_short_abstracts += 1
        return None, None
    # Restrict length of article to max_article_tokens
    article = ' '.join(article.split(" ")[:max_article_tokens])
    # Restrict length of title to max_abstract_tokens
    abstract = ' '.join(abstract.split(" ")[:max_abstract_tokens])
    # Replace numbers with #
    article = re.sub('[0-9]', '#', article)
    abstract = re.sub('[0-9]', '#', abstract)
    return article, abstract


def fix_missing_period(line):
    if "@highlight" in line:
        return line
    elif line == "":
        return line
    elif line[-1] in end_tokens:
        return line
    return line + " ."


def save_articles(articles, abstracts, name, relative_path):
    with open(relative_path + name + '.article.txt', 'w') as f:
        for item in articles:
            f.write(item)
            f.write("\n")
    with open(relative_path + name + '.abstract.txt', 'w') as f:
        for item in abstracts:
            f.write(item)
            f.write("\n")


if __name__ == '__main__':
    cnn_articles, cnn_abstracts = process_stories(cnn_directory)
    dm_articles, dm_abstracts = process_stories(dm_directory)
    print("Too short articles: %d" % Errors.too_short_articles)
    print("Too short abstracts: %d" % Errors.too_short_abstracts)
    processed_articles = cnn_articles + dm_articles
    processed_abstracts = cnn_abstracts + dm_abstracts
    save_articles(processed_articles, processed_abstracts, save_name, relative_save_path)
    print("Number of saved articles: %d" % len(processed_articles))
    print("Number of saved abstracts: %d" % len(processed_abstracts))
    print("DONE")
    # cnn articles = 92579
    # daily mail articles = 219506
    # should be total = 312 085
