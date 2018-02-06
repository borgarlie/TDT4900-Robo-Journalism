from torch import nn


class Discriminator:

    def __init__(self, model, optimizer, criterion):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion

    def train(self, ground_truth, sequences):
        self.optimizer.zero_grad()
        scores = self.model(sequences)
        loss = self.criterion(scores, ground_truth)
        loss.backward()
        self.optimizer.step()
        return loss.data[0]

    def evaluate(self, sequences):
        self.model.eval()
        scores = self.model(sequences)
        self.model.train()
        scores = scores.squeeze()
        return nn.functional.sigmoid(scores)
