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
        device_number = int(sys.argv[2])
        if device_number > -1:
            torch.cuda.set_device(device_number)
            print("Using GPU: %s" % sys.argv[2], flush=True)
        else:
            print("Not setting specific GPU", flush=True)
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

    relative_path = config['train']['real_data_file']
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

    real_data = open(relative_path + '.abstract.txt', encoding='utf-8').read().strip().split('\n')

    # TODO: Remove 620 , make it train_length instead ?
    max_train_examples = 620
    real_data = real_data[:max_train_examples]

    # Add EOS to real dataset (can probably do this in the files later instead
    # TODO: Fix in file
    for i in range(0, len(real_data)):
        real_data[i] += " <EOS>"

    train_length = len(real_data) - num_evaluate
    # Append remainder to evaluate set so that the training set has exactly a multiple of batch size
    num_evaluate += train_length % batch_size
    train_length = len(real_data) - num_evaluate

    real_training_data = real_data[:train_length]
    real_validation_data = real_data[train_length:]

    # TODO: Load validation data from validation folder
    fake_validation_data = []

    all_validation_data = real_validation_data + fake_validation_data

    print("Train length: ", train_length, flush=True)
    print("Num eval: ", num_evaluate, flush=True)
    print("Range train: %d - %d" % (0, train_length), flush=True)
    print("Range test: %d - %d" % (train_length, train_length + num_evaluate), flush=True)

    ground_truth = [[1, 0] for i in real_training_data] + [[0, 1] for i in real_training_data]
    # len(real_training_data) should be equal to len(fake_training_data) - ensured by 'max_train_examples'

    ground_truth_eval = [[1, 0] for i in real_validation_data] + [[0, 1] for i in fake_validation_data]

    fake_count = 0
    for gt in ground_truth_eval:
        if gt[1] == 1:
            fake_count += 1
    print("fake_count_eval: %d / %d" % (fake_count, len(ground_truth_eval)), flush=True)

    model = CNNDiscriminator(vocabulary.n_words, hidden_size, num_kernels, kernel_sizes, dropout_p)
    if use_cuda:
        model = model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-05)
    train_iters(config, vocabulary, model, optimizer, writer, real_training_data, ground_truth,
                all_validation_data, ground_truth_eval, max_train_examples)

    writer.close()

    print("Done", flush=True)
