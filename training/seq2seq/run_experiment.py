import json
import sys
import os
import torch

from torch import optim
from tensorboardX import SummaryWriter

sys.path.append('../..')  # ugly dirtyfix for imports to work

from evaluation.seq2seq.evaluate import evaluate
from models.seq2seq.decoder import AttnDecoderRNN, PointerGeneratorDecoder
from models.seq2seq.encoder import EncoderRNN
from preprocess.preprocess_pointer import *
from training.seq2seq.train import train_iters


def load_state(filename):
    if os.path.isfile(filename):
        state = torch.load(filename)
        return (state['epoch'], state['runtime'],
                state['model_state_encoder'], state['model_state_decoder'],
                state['optimizer_state_encoder'], state['optimizer_state_decoder'])
    else:
        raise FileNotFoundError


if __name__ == '__main__':

    use_cuda = torch.cuda.is_available()

    if use_cuda:
        if len(sys.argv) < 3:
            print("Expected 2 arguments: [0] = experiment path (e.g. test_experiment1), [1] = GPU (0 or 1)", flush=True)
            exit()
        torch.cuda.set_device(int(sys.argv[2]))
        print("Using GPU: %s" % sys.argv[2], flush=True)
    else:
        if len(sys.argv) < 2:
            print("Expected 1 argument: [0] = experiment path (e.g. test_experiment1)", flush=True)
            exit()

    experiment_path = sys.argv[1]
    config_file_path = experiment_path + "/config.json"
    with open(config_file_path) as config_file:
        config = json.load(config_file)

    config['experiment_path'] = experiment_path
    print(json.dumps(config, indent=2), flush=True)

    writer = SummaryWriter(config['tensorboard']['log_path'])
    relative_path = config['train']['dataset']
    num_articles = config['train']['num_articles']
    num_evaluate = config['train']['num_evaluate']
    num_throw = config['train']['throw']
    with_categories = config['train']['with_categories']

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
    print("Train length = %d" % train_length, flush=True)
    print("Throw length = %d" % num_throw, flush=True)
    print("Test length = %d" % test_length, flush=True)

    train_articles = summary_pairs[0:train_length]
    print("Range train: %d - %d" % (0, train_length), flush=True)

    train_length = train_length + num_throw  # compensate for thrown away articles
    test_articles = summary_pairs[train_length:train_length + test_length]

    print("Range test: %d - %d" % (train_length, train_length+test_length), flush=True)

    encoder = EncoderRNN(vocabulary.n_words, embedding_size, hidden_size, n_layers=n_layers)

    max_article_length = max(len(pair.article_tokens) for pair in summary_pairs) + 1

    max_abstract_length = max(len(pair.abstract_tokens) for pair in summary_pairs) + 1

    decoder = PointerGeneratorDecoder(hidden_size, embedding_size, vocabulary.n_words, max_length=max_article_length,
                                      n_layers=n_layers, dropout_p=dropout_p)

    if use_cuda:
        encoder = encoder.cuda()
        decoder = decoder.cuda()

    encoder_optimizer = optim.Adagrad(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adagrad(decoder.parameters(), lr=learning_rate)

    total_runtime = 0
    start_epoch = 1
    if load_model:
        try:
            (start_epoch, total_runtime, model_state_encoder, model_state_decoder,
             optimizer_state_encoder, optimizer_state_decoder) = load_state(load_file)
            encoder.load_state_dict(model_state_encoder)
            decoder.load_state_dict(model_state_decoder)
            encoder_optimizer.load_state_dict(optimizer_state_encoder)
            decoder_optimizer.load_state_dict(optimizer_state_decoder)
            print("Resuming training from epoch: %d" % start_epoch, flush=True)
        except FileNotFoundError as e:
            print("No file found: exiting", flush=True)
            exit()

    # articles = summary. TODO: Fix naming conventions when stuff works
    train_iters(config, train_articles, test_articles, vocabulary,
                encoder, decoder, max_article_length, max_abstract_length, encoder_optimizer, decoder_optimizer,
                writer, start_epoch=start_epoch, total_runtime=total_runtime)

    encoder.eval()
    decoder.eval()

    evaluate(config, test_articles, vocabulary, encoder, decoder, max_length=max_article_length)

    print("Done", flush=True)
