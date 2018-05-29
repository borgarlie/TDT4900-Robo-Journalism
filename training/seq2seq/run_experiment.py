import json
import sys
import os
import torch

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from torch import optim
from tensorboardX import SummaryWriter

sys.path.append('../..')  # ugly dirtyfix for imports to work

from evaluation.seq2seq.evaluate import evaluate
from models.seq2seq.decoder import AttnDecoderRNN, PointerGeneratorDecoder
from models.seq2seq.encoder import EncoderRNN
from preprocess.preprocess_pointer import *
from training.seq2seq.train import train_iters
from utils.logger import *


def load_state(filename):
    if os.path.isfile(filename):
        state = torch.load(filename, map_location=lambda storage, loc: storage.cuda(0))
        return (state['epoch'], state['runtime'],
                state['model_state_encoder'], state['model_state_decoder'],
                state['optimizer_state_encoder'], state['optimizer_state_decoder'])
    else:
        raise FileNotFoundError


if __name__ == '__main__':

    experiment_path = sys.argv[1]
    config_file_path = experiment_path + "/config.json"
    with open(config_file_path) as config_file:
        config = json.load(config_file)

    config['experiment_path'] = experiment_path
    filename = config['log']['filename']
    init_logger(filename)

    use_cuda = torch.cuda.is_available()

    if use_cuda:
        if len(sys.argv) < 3:
            log_error_message("Expected 2 arguments: [0] = experiment path (e.g. test_experiment1), [1] = GPU (0 or 1)")
            exit()
        torch.cuda.set_device(int(sys.argv[2]))
        log_message("Using GPU: %s" % sys.argv[2])
    else:
        if len(sys.argv) < 2:
            log_error_message("Expected 1 argument: [0] = experiment path (e.g. test_experiment1)")
            exit()

    log_message(json.dumps(config, indent=2))

    writer = SummaryWriter(config['tensorboard']['log_path'])
    relative_path = config['train']['dataset']
    num_articles = config['train']['num_articles']
    num_evaluate = config['train']['num_evaluate']
    num_throw = config['train']['throw']

    batch_size = config['train']['batch_size']
    learning_rate = config['train']['learning_rate']

    embedding_size = config['model']['embedding_size']
    hidden_size = config['model']['hidden_size']
    n_layers = config['model']['n_layers']
    dropout_p = config['model']['dropout_p']

    load_model = config['train']['load']
    load_file = experiment_path + "/" + config['train']['load_file']

    summary_pairs, vocabulary = load_dataset(relative_path)

    if num_articles != -1:
        summary_pairs = summary_pairs[:num_articles]

    total_articles = len(summary_pairs) - num_throw
    train_articles_length = total_articles - num_evaluate

    # Append remainder to evaluate set so that the training set has exactly a multiple of batch size
    num_evaluate += train_articles_length % batch_size
    train_length = total_articles - num_evaluate
    test_length = num_evaluate
    log_message("Train length = %d" % train_length)
    log_message("Throw length = %d" % num_throw)
    log_message("Test length = %d" % test_length)

    train_articles = summary_pairs[0:train_length]
    log_message("Range train: %d - %d" % (0, train_length))

    train_length = train_length + num_throw  # compensate for thrown away articles
    test_articles = summary_pairs[train_length:train_length + test_length]

    log_message("Range test: %d - %d" % (train_length, train_length+test_length))

    encoder = EncoderRNN(vocabulary.n_words, embedding_size, hidden_size, n_layers=n_layers)

    max_article_length = max(len(pair.article_tokens) for pair in summary_pairs) + 1

    max_abstract_length = max(len(pair.abstract_tokens) for pair in summary_pairs) + 1

    decoder = PointerGeneratorDecoder(hidden_size, embedding_size, vocabulary.n_words, max_length=max_article_length,
                                      n_layers=n_layers, dropout_p=dropout_p)

    optimizer_state_encoder = None
    optimizer_state_decoder = None
    total_runtime = 0
    start_epoch = 1
    if load_model:
        try:
            (start_epoch, total_runtime, model_state_encoder, model_state_decoder,
             optimizer_state_encoder, optimizer_state_decoder) = load_state(load_file)
            encoder.load_state_dict(model_state_encoder)
            decoder.load_state_dict(model_state_decoder)
            log_message("Resuming training from epoch: %d" % start_epoch)
        except FileNotFoundError as e:
            log_error_message("No file found: exiting")
            exit()

    if use_cuda:
        encoder = encoder.cuda()
        decoder = decoder.cuda()

    encoder_optimizer = optim.Adagrad(encoder.parameters(), lr=learning_rate, weight_decay=1e-05)
    decoder_optimizer = optim.Adagrad(decoder.parameters(), lr=learning_rate, weight_decay=1e-05)

    if load_model:
        encoder_optimizer.load_state_dict(optimizer_state_encoder)
        decoder_optimizer.load_state_dict(optimizer_state_decoder)

    train_iters(config, train_articles, test_articles, vocabulary,
                encoder, decoder, max_article_length, max_abstract_length, encoder_optimizer, decoder_optimizer,
                writer, start_epoch=start_epoch, total_runtime=total_runtime)

    encoder.eval()
    decoder.eval()

    # evaluate(config, test_articles, vocabulary, encoder, decoder, max_length=max_article_length)

    log_message("Done")
