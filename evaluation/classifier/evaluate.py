from torch import nn

from utils.data_prep import *


def eval_single_article(ground_truth, sequence, model, criterion):
    scores = model(sequence)
    loss = criterion(scores, ground_truth)
    return loss.data[0]


def evaluate(ground_truth, titles, vocabulary, model, writer, train_loss, epoch):
    print("Evaluating", flush=True)
    criterion = nn.BCEWithLogitsLoss()
    total_loss = 0
    for i in range(len(titles)):
        sequence = indexes_from_sentence(vocabulary, titles[i])
        sequence = Variable(torch.LongTensor([sequence]))
        current_ground_truth = Variable(torch.FloatTensor([[ground_truth[i]]]))
        if use_cuda:
            sequence = sequence.cuda()
            current_ground_truth = current_ground_truth.cuda()
        loss = eval_single_article(current_ground_truth, sequence, model, criterion)
        total_loss += loss
    avg_loss = total_loss / len(titles)
    print("Avg evaluation loss: %0.4f" % avg_loss, flush=True)
    # Making a loss dictionary for tensorboard
    loss_dict = {'train_loss': train_loss, 'eval_loss': avg_loss}
    writer.add_scalars('Loss', loss_dict, epoch)
