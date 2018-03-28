import sys

sys.path.append('../..')  # ugly dirtyfix for imports to work

from preprocess.preprocess_pointer import *
from utils.data_prep import *

if __name__ == '__main__':
    relative_path = "../../data/cnn_pickled/cnn_pointer_50k"
    # relative_path = "../../data/ntb_pickled/ntb_pointer_30k"
    summary_pairs, vocabulary = load_dataset(relative_path)
    print("Done loading dataset")
    real_data_save_file = "../../data/cnn_real_data/cnn_real_1.abstract.txt"
    # real_data_save_file = "../../data/ntb_real_data/ntb_real_1.abstract.txt"
    with open(real_data_save_file, 'w') as file:
        for pair in summary_pairs:
            if pair.unked_abstract_tokens[-1] == EOS_token:
                tokens = pair.unked_abstract_tokens[:-1]
            else:
                tokens = pair.unked_abstract_tokens
            sample = get_sentence_from_tokens_unked(tokens, vocabulary)
            file.write(sample)
            file.write("\n")
    print("Done")
