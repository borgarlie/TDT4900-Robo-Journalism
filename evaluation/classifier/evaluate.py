import numpy as np
from sklearn.metrics import accuracy_score
from torch import nn

from utils.data_prep import *


def eval_single_article(ground_truth, sequence, model, criterion):
    scores = model(sequence)
    loss = criterion(scores, ground_truth)
    return loss.data[0], scores.data.cpu().numpy()[0]


def evaluate(ground_truth, titles, vocabulary, model, writer, train_loss, epoch):
    print("Evaluating", flush=True)
    criterion = nn.BCEWithLogitsLoss()
    total_loss = 0
    scores = []
    for i in range(len(titles)):
        sequence = indexes_from_sentence(vocabulary, titles[i])
        sequence = Variable(torch.LongTensor([sequence]))
        current_ground_truth = Variable(torch.FloatTensor([ground_truth[i]]))
        if use_cuda:
            sequence = sequence.cuda()
            current_ground_truth = current_ground_truth.cuda()
        loss, score = eval_single_article(current_ground_truth, sequence, model, criterion)
        scores.append(score)
        total_loss += loss
    avg_loss = total_loss / len(titles)
    print("Avg evaluation loss: %0.4f" % avg_loss, flush=True)
    # Making a loss dictionary for tensorboard
    loss_dict = {'train_loss': train_loss, 'eval_loss': avg_loss}
    writer.add_scalars('Loss', loss_dict, epoch)

    print("Calculating accuracy on 0.00 confidence", flush=True)
    np_gold_truth = np.array(ground_truth)
    np_predicted = get_predictions(scores, 0.00)
    calculate_accuracy(np_gold_truth, np_predicted)


def get_predictions(model_scores, min_score):
    predicted = []
    for example in range(0, len(model_scores)):
        example_predictions = []
        for score in range(0, len(model_scores[example])):
            if model_scores[example][score] > min_score:
                example_predictions.append(1)
            else:
                example_predictions.append(0)
        predicted.append(example_predictions)
    return np.array(predicted)


def calculate_accuracy(gold_truth, predictions):
    # calculate total accuracy
    accuracy = accuracy_score(gold_truth, predictions)
    print("Total Accuracy: %0.4f" % accuracy, flush=True)
