import os
import sys
import time

sys.path.append('../..')  # ugly dirtyfix for imports to work

from evaluation.seq2seq.calculate_rouge_3 import calculate_rouge_3
from evaluation.seq2seq.split_beamsearch_to_multiple_files import split_beamsearch_to_multiple_files


def read_directory(directory):
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(".txt"):
            yield os.path.join(directory, file)


if __name__ == '__main__':

    # Parameters
    test_data = False
    directory = "../output_for_eval/sr_0.10_0.20"

    # validation set
    path_to_reference = "../for_rouge/pretrained1/reference_new/"
    path_to_modelsummary = "../for_rouge/pretrained1/cnn_pretrain_new/"
    num_summaries = 2000

    # test set
    if test_data:
        path_to_reference = "../for_rouge/test_data/reference/"
        path_to_modelsummary = "../for_rouge/test_data/modelsummary/"
        num_summaries = 11000

    files = list(read_directory(directory))
    files.sort()

    print("Number of files to run: %d" % len(files), flush=True)

    for i in range(0, len(files)):
        f = files[i]
        print("######################################################", flush=True)
        print("######################################################", flush=True)
        print("Number %d of %d" % (i+1, len(files)))
        print("Evaluating file: %s" % f, flush=True)
        print("Splitting file", flush=True)
        split_beamsearch_to_multiple_files(f, path_to_reference, path_to_modelsummary, num_summaries)
        print("Sleeping for 3 seconds", flush=True)
        time.sleep(3)
        print("Calculating rouge", flush=True)
        output = calculate_rouge_3(path_to_reference, path_to_modelsummary)
        print(output)
        print("Done with file", flush=True)
        print("######################################################", flush=True)
        print("######################################################\n\n\n", flush=True)

    print("Done", flush=True)







