import json
import sys
import os
import torch

from torch import optim
from tensorboardX import SummaryWriter

sys.path.append('../..')  # ugly dirtyfix for imports to work

from models.GAN.discriminator import Discriminator
from models.GAN.generator import Generator
from models.GAN.generator_beta import GeneratorBeta
from models.classifier.cnn_classifier import CNNDiscriminator
from models.seq2seq.decoder import PointerGeneratorDecoder
from models.seq2seq.encoder import EncoderRNN
from training.GAN.train import train_GAN
from evaluation.seq2seq.evaluate import evaluate
from preprocess.preprocess_pointer import *


def load_pretrained_generator(filename):
    if os.path.isfile(filename):
        state = torch.load(filename)
        return state['model_state_encoder'], state['model_state_decoder']
    else:
        raise FileNotFoundError


def load_pretrained_discriminator(filename):
    if os.path.isfile(filename):
        state = torch.load(filename)
        return state['model']
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

    # load shared parameters
    writer = SummaryWriter(config['tensorboard']['log_path'])
    relative_path = config['train']['dataset']
    num_articles = config['train']['num_articles']
    num_evaluate = config['train']['num_evaluate']
    num_throw = config['train']['throw']
    with_categories = config['train']['with_categories']
    batch_size = config['train']['batch_size']
    beta = config['train']['beta']
    num_monte_carlo_samples = config['train']['num_monte_carlo_samples']

    # load generator parameters
    generator_embedding_size = config['generator_model']['embedding_size']
    generator_hidden_size = config['generator_model']['hidden_size']
    generator_n_layers = config['generator_model']['n_layers']
    generator_dropout_p = config['generator_model']['dropout_p']
    generator_load_model = config['generator_model']['load']
    generator_load_file = experiment_path + "/" + config['generator_model']['load_file']

    # load discriminator parameters
    discriminator_hidden_size = config['discriminator_model']['hidden_size']
    discriminator_dropout_p = config['discriminator_model']['dropout_p']
    discriminator_num_kernels = config['discriminator_model']['num_kernels']
    discriminator_kernel_sizes = config['discriminator_model']['kernel_sizes']
    discriminator_load_model = config['discriminator_model']['load']
    discriminator_load_file = experiment_path + "/" + config['discriminator_model']['load_file']

    generator_learning_rate = config['train']['generator_learning_rate']
    discriminator_learning_rate = config['train']['discriminator_learning_rate']

    summary_pairs, vocabulary = load_dataset(relative_path)

    n_generator = config['train']['n_generator']

    if num_articles != -1:
        summary_pairs = summary_pairs[:num_articles]

    total_articles = len(summary_pairs) - num_throw
    train_articles_length = total_articles - num_evaluate

    # Append remainder to evaluate set so that the training set has exactly a multiple of batch size
    num_evaluate += train_articles_length % batch_size
    temp_train_length = total_articles - num_evaluate
    num_evaluate += int((temp_train_length / batch_size) % n_generator) * batch_size
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

    generator_encoder = EncoderRNN(vocabulary.n_words, generator_embedding_size, generator_hidden_size,
                                   n_layers=generator_n_layers)
    generator_beta_encoder = EncoderRNN(vocabulary.n_words, generator_embedding_size, generator_hidden_size,
                                        n_layers=generator_n_layers)

    max_article_length = max(len(pair.article_tokens) for pair in summary_pairs) + 1

    max_abstract_length = max(len(pair.abstract_tokens) for pair in summary_pairs) + 1

    generator_decoder = PointerGeneratorDecoder(generator_hidden_size, generator_embedding_size, vocabulary.n_words,
                                                max_length=max_article_length, n_layers=generator_n_layers,
                                                dropout_p=generator_dropout_p)
    generator_beta_decoder = PointerGeneratorDecoder(generator_hidden_size, generator_embedding_size, vocabulary.n_words
                                                     , max_length=max_article_length, n_layers=generator_n_layers,
                                                     dropout_p=generator_dropout_p)

    if generator_load_model:
        try:
            model_state_encoder, model_state_decoder = load_pretrained_generator(generator_load_file)
            generator_encoder.load_state_dict(model_state_encoder)
            generator_decoder.load_state_dict(model_state_decoder)
            print("Loaded pretrained generator", flush=True)
        except FileNotFoundError as e:
            print("No file found: exiting", flush=True)
            exit()

    discriminator_model = CNNDiscriminator(vocabulary.n_words, discriminator_hidden_size, discriminator_num_kernels,
                                           discriminator_kernel_sizes, discriminator_dropout_p)

    if discriminator_load_model:
        try:
            model_parameters = load_pretrained_discriminator(generator_load_file)
            discriminator_model.load_state_dict(model_parameters)
            print("Loaded pretrained discriminator", flush=True)
        except FileNotFoundError as e:
            print("No file found: exiting", flush=True)
            exit()

    if use_cuda:
        generator_encoder = generator_encoder.cuda()
        generator_decoder = generator_decoder.cuda()
        discriminator_model = discriminator_model.cuda()
        generator_beta_encoder = generator_beta_encoder.cuda()
        generator_beta_decoder = generator_beta_decoder.cuda()

    # generator_encoder_optimizer = optim.SGD(generator_encoder.parameters(), lr=generator_learning_rate)
    # generator_decoder_optimizer = optim.SGD(generator_decoder.parameters(), lr=generator_learning_rate)
    generator_encoder_optimizer = optim.Adam(generator_encoder.parameters(), lr=generator_learning_rate)
    generator_decoder_optimizer = optim.Adam(generator_decoder.parameters(), lr=generator_learning_rate)
    generator_mle_criterion = torch.nn.NLLLoss()
    policy_criterion = torch.nn.NLLLoss(reduce=False)

    # TODO: should this one be loaded?
    discriminator_optimizer = torch.optim.Adam(discriminator_model.parameters(), lr=discriminator_learning_rate)
    discriminator_criterion = torch.nn.BCEWithLogitsLoss()

    generator_beta = GeneratorBeta(vocabulary, generator_beta_encoder, generator_beta_decoder, batch_size, use_cuda)
    generator_beta.update_params(generator_encoder, generator_decoder)

    generator = Generator(vocabulary, generator_encoder, generator_decoder, generator_encoder_optimizer,
                          generator_decoder_optimizer, generator_mle_criterion, policy_criterion, batch_size, use_cuda,
                          beta, generator_beta, num_monte_carlo_samples)

    discriminator = Discriminator(discriminator_model, discriminator_optimizer, discriminator_criterion)

    # Train the generator and discriminator alternately in a standard GAN setup
    train_GAN(config, generator, discriminator, train_articles, test_articles, max_article_length, max_abstract_length,
              writer)

    # Evaluate the generator
    generator_encoder.eval()
    generator_decoder.eval()
    evaluate(config, test_articles, vocabulary, generator_encoder, generator_decoder, max_length=max_article_length)

    print("Done", flush=True)
