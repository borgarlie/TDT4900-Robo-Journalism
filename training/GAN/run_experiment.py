import json
import sys
import os

from torch import optim
from tensorboardX import SummaryWriter

sys.path.append('../..')  # ugly dirtyfix for imports to work

from models.GAN.discriminator import RougeDiscriminator
from models.GAN.generator_rl_strat import GeneratorRlStrat
from models.classifier.cnn_classifier import CNNDiscriminator
from models.seq2seq.decoder import PointerGeneratorDecoder
from models.seq2seq.encoder import EncoderRNN
from training.GAN.train import train_GAN
from evaluation.seq2seq.evaluate import *
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

    experiment_path = sys.argv[1]
    config_file_path = experiment_path + "/config.json"
    with open(config_file_path) as config_file:
        config = json.load(config_file)

    config['experiment_path'] = experiment_path
    log_file = config['log']['filename']
    init_logger(log_file)

    if use_cuda:
        if len(sys.argv) < 3:
            log_error_message("Expected 2 arguments: [0] = experiment path (e.g. test_experiment1), [1] = GPU (0 or 1)")
            exit()
        device_number = int(sys.argv[2])
        if device_number > -1:
            torch.cuda.set_device(device_number)
            log_message("Using GPU: %s" % sys.argv[2])
        else:
            log_message("Not setting specific GPU")
    else:
        if len(sys.argv) < 2:
            log_error_message("Expected 1 argument: [0] = experiment path (e.g. test_experiment1)")
            exit()

    log_message(json.dumps(config, indent=2))

    # load shared parameters
    writer = SummaryWriter(config['tensorboard']['log_path'])
    relative_path = config['train']['dataset']
    num_articles = config['train']['num_articles']
    num_evaluate = config['train']['num_evaluate']
    num_throw = config['train']['throw']
    batch_size = config['train']['batch_size']
    beta = config['train']['beta']
    num_monte_carlo_samples = config['train']['num_monte_carlo_samples']
    sample_rate = config['train']['sample_rate']
    allow_negative_reward = config['train']['allow_negative_reward']

    # load generator parameters
    generator_embedding_size = config['generator_model']['embedding_size']
    generator_hidden_size = config['generator_model']['hidden_size']
    generator_n_layers = config['generator_model']['n_layers']
    generator_dropout_p = config['generator_model']['dropout_p']
    generator_load_model = config['generator_model']['load']
    generator_load_file = config['generator_model']['load_file']

    # load discriminator parameters
    discriminator_hidden_size = config['discriminator_model']['hidden_size']
    discriminator_dropout_p = config['discriminator_model']['dropout_p']
    discriminator_num_kernels = config['discriminator_model']['num_kernels']
    discriminator_kernel_sizes = config['discriminator_model']['kernel_sizes']
    discriminator_load_model = config['discriminator_model']['load']
    discriminator_load_file = config['discriminator_model']['load_file']

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
    log_message("Train length = %d" % train_length)
    log_message("Throw length = %d" % num_throw)
    log_message("Test length = %d" % test_length)

    train_articles = summary_pairs[0:train_length]
    log_message("Range train: %d - %d" % (0, train_length))

    train_length = train_length + num_throw  # compensate for thrown away articles
    test_articles = summary_pairs[train_length:train_length + test_length]

    log_message("Range test: %d - %d" % (train_length, train_length+test_length))

    generator_encoder = EncoderRNN(vocabulary.n_words, generator_embedding_size, generator_hidden_size,
                                   n_layers=generator_n_layers)

    max_article_length = max(len(pair.article_tokens) for pair in summary_pairs) + 1

    max_abstract_length = max(len(pair.abstract_tokens) for pair in summary_pairs) + 1

    generator_decoder = PointerGeneratorDecoder(generator_hidden_size, generator_embedding_size, vocabulary.n_words,
                                                max_length=max_article_length, n_layers=generator_n_layers,
                                                dropout_p=generator_dropout_p)

    if generator_load_model:
        try:
            model_state_encoder, model_state_decoder = load_pretrained_generator(generator_load_file)
            generator_encoder.load_state_dict(model_state_encoder)
            generator_decoder.load_state_dict(model_state_decoder)
            log_message("Loaded pretrained generator")
        except FileNotFoundError as e:
            log_error_message("No generator model file found: exiting")
            exit()

    discriminator_model = CNNDiscriminator(vocabulary.n_words, discriminator_hidden_size, discriminator_num_kernels,
                                           discriminator_kernel_sizes, discriminator_dropout_p)

    if discriminator_load_model:
        try:
            model_parameters = load_pretrained_discriminator(discriminator_load_file)
            discriminator_model.load_state_dict(model_parameters)
            log_message("Loaded pretrained discriminator")
        except FileNotFoundError as e:
            log_message("No discriminator model file found: exiting")
            exit()

    if use_cuda:
        generator_encoder = generator_encoder.cuda()
        generator_decoder = generator_decoder.cuda()
        discriminator_model = discriminator_model.cuda()

    # generator_encoder_optimizer = optim.SGD(generator_encoder.parameters(), lr=generator_learning_rate)
    # generator_decoder_optimizer = optim.SGD(generator_decoder.parameters(), lr=generator_learning_rate)
    generator_encoder_optimizer = optim.Adagrad(generator_encoder.parameters(), lr=generator_learning_rate,
                                                weight_decay=1e-05)
    generator_decoder_optimizer = optim.Adagrad(generator_decoder.parameters(), lr=generator_learning_rate,
                                                weight_decay=1e-05)
    generator_mle_criterion = torch.nn.NLLLoss()

    # TODO: should this one be loaded?
    # discriminator_optimizer = torch.optim.Adam(discriminator_model.parameters(), lr=discriminator_learning_rate,
    #                                            weight_decay=1e-05)
    # discriminator_criterion = torch.nn.BCEWithLogitsLoss()

    generator = GeneratorRlStrat(vocabulary, generator_encoder, generator_decoder, generator_encoder_optimizer,
                                 generator_decoder_optimizer, generator_mle_criterion, batch_size,
                                 use_cuda, beta, num_monte_carlo_samples, sample_rate, allow_negative_reward)

    # discriminator = Discriminator(discriminator_model, discriminator_optimizer, discriminator_criterion)

    # TEST
    discriminator = RougeDiscriminator(vocabulary)

    # Train the generator and discriminator alternately in a standard GAN setup
    train_GAN(config, generator, discriminator, train_articles, test_articles, max_article_length, max_abstract_length,
              writer)

    # Evaluate the generator
    generator_encoder.eval()
    generator_decoder.eval()

    # evaluate(config, test_articles, vocabulary, generator_encoder, generator_decoder, max_length=max_article_length)

    log_message("Done")
