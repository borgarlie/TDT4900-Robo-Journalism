import json
import os


def read_directory(directory):
    idx = 0
    for file in os.listdir(directory):
        idx += 1
        filename = os.fsdecode(file)
        if filename.endswith(".txt"):
            yield idx, os.path.join(directory, file)


def count_endings(directory):
    endings = {}
    num = len(os.listdir(directory))
    print("Processing directory: %s" % directory)
    print("Length: %d" % num)
    for idx, filename in read_directory(directory):
        if idx % 1000 == 0:
            print("Processing %i of %i; %.2f percent done" %
                  (idx, num, float(idx) * 100.0 / float(num)))
        with open(filename, 'r') as file:
            for line in file:
                ending = line.strip()[-1]
                if ending in endings.keys():
                    endings[ending] += 1
                else:
                    endings[ending] = 1
    print(json.dumps(endings, indent=2))


if __name__ == '__main__':
    # directory = '../for_rouge/get-to-the-point/pointer-gen-cov/'
    directory = '../for_rouge/pretrained1/modelsummary_test'
    count_endings(directory)
