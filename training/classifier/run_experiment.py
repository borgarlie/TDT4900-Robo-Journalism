import json
import random
import sys
import os
import torch
from tensorboardX import SummaryWriter

sys.path.append('../..')  # ugly dirtyfix for imports to work

from models.classifier.cnn_classifier import CNNDiscriminator
from training.classifier.train import train_iters
from preprocess.preprocess_pointer import *


def save_state(state, filename):
    torch.save(state, filename)


def load_classifier(filename):
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

    writer = SummaryWriter(config['tensorboard']['log_path'])

    relative_path = config['train']['dataset']
    relative_path_fake_data = config['train']['fake_dataset']
    num_articles = config['train']['num_articles']
    num_evaluate = config['train']['num_evaluate']

    batch_size = config['train']['batch_size']
    learning_rate = config['train']['learning_rate']

    hidden_size = config['model']['hidden_size']
    dropout_p = config['model']['dropout_p']
    num_kernels = config['model']['num_kernels']
    kernel_sizes = config['model']['kernel_sizes']

    print("Using cuda: " + str(use_cuda), flush=True)

    vocabulary_path = config['train']['vocabulary_path']
    _, vocabulary = load_dataset(vocabulary_path)

    titles = open(relative_path + '.abstract.txt', encoding='utf-8').read().strip().split('\n')
    fake_titles = open(relative_path_fake_data + '.abstract.txt', encoding='utf-8').read().strip().split('\n')

    if num_articles != -1:
        titles = titles[:num_articles]
        fake_titles = fake_titles[:num_articles]

    all_titles = titles + fake_titles
    ground_truth = [1 for i in titles] + [0 for i in fake_titles]

    # shuffle titles and ground truth equally
    c = list(zip(all_titles, ground_truth))
    random.shuffle(c)
    all_titles_shuffled, ground_truth_shuffled = zip(*c)

    train_length = len(all_titles_shuffled) - num_evaluate
    # Append remainder to evaluate set so that the training set has exactly a multiple of batch size
    num_evaluate += train_length % batch_size
    train_length = len(all_titles_shuffled) - num_evaluate

    print("Train length: ", train_length, flush=True)
    print("Num eval: ", num_evaluate, flush=True)
    print("Range train: %d - %d" % (0, train_length), flush=True)
    print("Range test: %d - %d" % (train_length, train_length + num_evaluate), flush=True)

    ground_truth_train = ground_truth_shuffled[0:train_length]
    train_titles = all_titles_shuffled[0:train_length]

    ground_truth_eval = ground_truth_shuffled[train_length:train_length+num_evaluate]
    train_eval = all_titles_shuffled[train_length:train_length+num_evaluate]

    fake_count = 0
    for gt in ground_truth_eval:
        if gt == 0:
            fake_count += 1
    print("fake_count_eval: %d / %d" % (fake_count, len(ground_truth_eval)), flush=True)

    model = CNNDiscriminator(vocabulary.n_words, hidden_size, num_kernels, kernel_sizes, dropout_p)
    if use_cuda:
        model = model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    train_iters(config, ground_truth_train, train_titles, vocabulary, model, optimizer, ground_truth_eval, train_eval,
                writer)

    writer.close()

    print("Saving model", flush=True)

    save_state({
        'model': model.state_dict()
    }, "model" + config['save']['save_file'])

    print("Model saved", flush=True)
    print("Done", flush=True)
