import random

from torch import nn

from evaluation.classifier.evaluate import evaluate
from utils.batching import *
from utils.data_prep import *
from utils.time_utils import *


# Train one batch
# Example:
# scores = [[0.2], [123.2], [-500]]
# ground_truth = [[0], [1], [0]], 0 = false data, 1 = real data (original)
def train(ground_truth, sequences, model, optimizer, criterion):
    optimizer.zero_grad()
    scores = model(sequences)
    loss = criterion(scores, ground_truth)
    loss.backward()
    optimizer.step()
    return loss.data[0]


# ground_truth = [1, 1, 1, 1, ..., 0, 0, 0, 0, ...]
def train_iters(ground_truth, titles, vocabulary, model, optimizer, ground_truth_eval, eval_titles, writer, batch_size,
                n_epochs):

    start = time.time()
    print_loss_total = 0  # Reset every print_every
    print_every = 500

    criterion = nn.BCEWithLogitsLoss()

    num_batches = int(len(titles) / batch_size)
    n_iters = num_batches * n_epochs

    lowest_loss = 999
    total_runtime = 0.0

    # Evaluate once before starting to train
    model.eval()
    evaluate(ground_truth_eval, eval_titles, vocabulary, model, writer, 1, 0)
    model.train()

    print("Starting training", flush=True)
    for epoch in range(1, n_epochs + 1):
        print("Starting epoch: %d" % epoch, flush=True)
        batch_loss_avg = 0

        # shuffle ground_truth and titles equally
        c = list(zip(ground_truth, titles))
        random.shuffle(c)
        ground_truth_shuffled, titles_shuffled = zip(*c)

        # split into batches
        ground_truth_batches = list(chunks(ground_truth_shuffled, batch_size))
        title_batches = list(chunks(titles_shuffled, batch_size))

        for batch in range(num_batches):
            ground_truth, sequences = batch_sequences(vocabulary, title_batches[batch], ground_truth_batches[batch])
            loss = train(ground_truth, sequences, model, optimizer, criterion)

            print_loss_total += loss
            batch_loss_avg += loss
            # calculate number of batches processed
            itr = (epoch-1) * num_batches + batch + 1

            if itr % print_every == 0:
                print_loss_avg = print_loss_total / print_every
                print_loss_total = 0
                progress, total_runtime = time_since(start, itr / n_iters, total_runtime)
                start = time.time()
                print('%s (%d %d%%) %.4f' % (progress, itr, itr / n_iters * 100, print_loss_avg), flush=True)
                if print_loss_avg < lowest_loss:
                    lowest_loss = print_loss_avg
                    print(" ^ Lowest loss so far", flush=True)

        batch_loss_avg /= num_batches

        # evaluate epoch on test set
        model.eval()
        evaluate(ground_truth_eval, eval_titles, vocabulary, model, writer, batch_loss_avg, epoch)
        model.train()

    print("Done with training")


def batch_sequences(vocabulary, titles, ground_truth):
    sequences = []
    batch_size = len(titles)
    for i in range(batch_size):
        sequence = indexes_from_sentence(vocabulary, titles[i])
        sequences.append(sequence)

    seq_lengths = [len(s) for s in sequences]
    seq_padded = [pad_seq(s, max(seq_lengths)) for s in sequences]

    sequences = Variable(torch.LongTensor(seq_padded))
    ground_truth_batched = Variable(torch.FloatTensor([ground_truth]))

    if use_cuda:
        sequences = sequences.cuda()
        ground_truth_batched = ground_truth_batched.cuda()

    return ground_truth_batched, sequences
