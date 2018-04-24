import sys
import os

sys.path.append('../..')  # ugly dirtyfix for imports to work

from models.classifier.cnn_classifier import CNNDiscriminator
from preprocess.preprocess_pointer import *
from utils.batching import chunks
from utils.data_prep import *


def load_model(filename):
    if os.path.isfile(filename):
        state = torch.load(filename)
        return state['model']
    else:
        raise FileNotFoundError


def load_pretrained_classifier(vocabulary, discriminator_load_file):
    discriminator_hidden_size = 128
    discriminator_dropout_p = 0.5
    discriminator_num_kernels = 100
    discriminator_kernel_sizes = [3, 4, 5]
    discriminator_model = CNNDiscriminator(vocabulary.n_words, discriminator_hidden_size, discriminator_num_kernels,
                                           discriminator_kernel_sizes, discriminator_dropout_p)
    try:
        model_parameters = load_model(discriminator_load_file)
        discriminator_model.load_state_dict(model_parameters)
        print("Loaded pretrained discriminator", flush=True)
    except FileNotFoundError as e:
        print("No discriminator model file found: exiting", flush=True)
        exit()
    return discriminator_model


def prepare_batch(vocabulary, abstracts):
    sequences = []
    batches = len(abstracts)
    for i in range(batches):
        sequence = indexes_from_sentence(vocabulary, abstracts[i])
        sequences.append(sequence)

    seq_lengths = [len(s) for s in sequences]
    seq_padded = [pad_seq(s, max(seq_lengths)) for s in sequences]

    sequences = Variable(torch.cuda.LongTensor(seq_padded))
    return sequences


def evaluate_all_data(sequences, batch_size, model, vocabulary):
    batches = list(chunks(sequences, batch_size))
    total_score = 0.0
    for i in range(0, len(batches)):
        if i % 10 == 0:
            print("Processing batch %d of %d" % (i, len(batches)), flush=True)
        prepared_batch = prepare_batch(vocabulary, batches[i])
        scores = evaluate_batch(prepared_batch, model)
        total_score += scores.mean().data[0]
    mean_score = total_score / len(batches)
    print("Scores %.10f" % mean_score, flush=True)


def evaluate_batch(sequences, model):
    scores = model(sequences)
    scores = scores.squeeze()
    return torch.nn.functional.sigmoid(scores)


if __name__ == '__main__':
    cuda_device = 1
    torch.cuda.set_device(cuda_device)

    max_abstracts = 1000
    batch_size = 32
    vocabulary_path = '../../data/cnn_pickled/cnn_pointer_50k'

    # dataset_path = '../../data/cnn_fake_data/cnn_13epoch'
    # dataset_path = '../../data/cnn_real_data/cnn_real_1'
    dataset_path = '../../data/cnn_fake_data/cnn_13epoch_sampled'

    print("Loading datasets", flush=True)
    _, vocabulary = load_dataset(vocabulary_path)

    # discriminator_load_file = '../../models/pretrained_models/classifier/cnn/cnn_classifier_13epoch.tar'
    discriminator_load_file = '../../models/pretrained_models/classifier/cnn/cnn_classifier_13epoch_sampled.tar'

    model = load_pretrained_classifier(vocabulary, discriminator_load_file)
    model.eval()
    model.cuda()

    fake_abstracts = open(dataset_path + '.abstract.txt', encoding='utf-8').read().strip().split('\n')
    fake_abstracts = fake_abstracts[0:max_abstracts]

    print("Evaluating data", flush=True)
    evaluate_all_data(fake_abstracts, batch_size, model, vocabulary)
    print("Done", flush=True)
