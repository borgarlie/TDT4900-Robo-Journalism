import sys
import os

sys.path.append('../..')  # ugly dirtyfix for imports to work

from models.seq2seq.decoder import *
from models.seq2seq.encoder import *
from preprocess.preprocess_pointer import *
from evaluation.seq2seq.evaluate import *


def load_state(filename):
    if os.path.isfile(filename):
        state = torch.load(filename)
        return state['model_state_encoder'], state['model_state_decoder']
    else:
        raise FileNotFoundError


if __name__ == '__main__':
    use_cuda = torch.cuda.is_available()

    filename = "beam_expansion2.log"
    init_logger(filename)

    if use_cuda:
        if len(sys.argv) < 2:
            log_message("Expected 1 argument: [0] = GPU (0 or 1)")
            exit()
        torch.cuda.set_device(int(sys.argv[1]))
        log_message("Using GPU: %s" % sys.argv[1])

    # relative_path = "../../data/cnn_pickled/cnn_pointer_50k"
    relative_path = "../../data/ntb_pickled/ntb_pointer_30k"
    hidden_size = 128
    embedding_size = 100
    n_layers = 1
    dropout_p = 0.0
    load_file = "../../models/pretrained_models/ntb/ntb_pretrain_2epochs.tar"

    summary_pairs, vocabulary = load_dataset(relative_path)

    encoder = EncoderRNN(vocabulary.n_words, embedding_size, hidden_size, n_layers=n_layers)

    max_article_length = max(len(pair.article_tokens) for pair in summary_pairs) + 1
    max_abstract_length = max(len(pair.abstract_tokens) for pair in summary_pairs) + 1

    decoder = PointerGeneratorDecoder(hidden_size, embedding_size, vocabulary.n_words, max_length=max_article_length,
                                      n_layers=n_layers, dropout_p=dropout_p)

    if use_cuda:
        encoder = encoder.cuda()
        decoder = decoder.cuda()

    try:
        model_state_encoder, model_state_decoder = load_state(load_file)
        encoder.load_state_dict(model_state_encoder)
        decoder.load_state_dict(model_state_decoder)
    except FileNotFoundError as e:
        log_error_message("No file found: exiting")
        exit()

    encoder.eval()
    decoder.eval()

    summary_pairs = summary_pairs[:5]

    config = {}
    config['evaluate'] = {}
    config['evaluate']['keep_beams'] = 20
    config['evaluate']['return_beams'] = 3

    for i in range(1, 5):
        log_message("Setting beam expansions to: %d" % i)
        config['evaluate']['expansions'] = i
        evaluate(config, summary_pairs, vocabulary, encoder, decoder, max_article_length)

    log_message("Done")
