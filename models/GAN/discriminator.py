import time
import torch
from torch import nn
from sumeval.metrics.rouge import RougeCalculator
from torch.autograd import Variable

from utils.logger import *
from utils.data_prep import get_sentence_from_tokens_and_clean_unked


class GANDiscriminator:
    def __init__(self, vocabulary, model, optimizer, criterion):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.vocabulary = vocabulary

    def train(self, ground_truth, sequences):
        self.optimizer.zero_grad()
        scores = self.model(sequences)
        loss = self.criterion(scores, ground_truth)
        loss.backward()
        self.optimizer.step()
        return loss.data[0]

    # reference_batch and extended_vocabs are not used here, but included to make
    # the eval function general across discriminators
    def evaluate(self, sequences, reference_batch, extended_vocabs):

        # Insert UNK instead of using out of vocabulary words
        # copy_params_time = time.time()
        # for batch_index in range(0, len(sequences)):
        #     for token_index in range(0, len(sequences[batch_index])):
        #         if sequences[batch_index][token_index].data[0] >= self.vocabulary.n_words:
        #             sequences[batch_index][token_index].data[0] = UNK_token
        # timings[timings_var_copy_params] += time.time() - copy_params_time

        self.model.eval()
        scores = self.model(sequences)
        self.model.train()
        scores = Variable(scores.data.narrow(1, 0, 1)).squeeze()
        sigm = nn.functional.sigmoid(scores).data
        gan_reward = Variable(torch.cuda.FloatTensor(sigm))
        return gan_reward, gan_reward, gan_reward


class JointRougeAndGANDiscriminator:
    def __init__(self, vocabulary, model, optimizer, criterion, phi):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.vocabulary = vocabulary
        self.rouge = RougeCalculator(stopwords=False, lang="en", stemming=False)
        self.phi = phi

    def train(self, ground_truth, sequences):
        self.optimizer.zero_grad()
        scores = self.model(sequences)
        loss = self.criterion(scores, ground_truth)
        loss.backward()
        self.optimizer.step()
        return loss.data[0]

    # TODO: Probably don't need extended_vocabs anymore?
    # reference_batch and extended_vocabs are not used here, but included to make
    # the eval function general across discriminators
    def evaluate(self, sequences, reference_batch, extended_vocabs):
        gan_rewards = self.evaluate_gan(sequences)
        rouge_reward = self.evaluate_rouge(sequences, reference_batch, extended_vocabs)
        joint_reward = self.phi * rouge_reward + (1 - self.phi) * gan_rewards
        return joint_reward, gan_rewards, rouge_reward

    def evaluate_gan(self, sequences):
        self.model.eval()
        scores = self.model(sequences)
        self.model.train()
        scores = Variable(scores.data.narrow(1, 0, 1)).squeeze()
        sigm = nn.functional.sigmoid(scores).data
        return Variable(torch.cuda.FloatTensor(sigm))

    def evaluate_rouge(self, generated_batch, reference_batch, extended_vocabs):
        before_transfer = time.time()
        generated_batch = generated_batch.cpu()
        timings[timings_var_discriminator_transfer] += time.time() - before_transfer
        generated_batch, reference_batch = self.convert_batch(generated_batch, reference_batch)
        rouge_l_points = []
        for i in range(0, len(generated_batch)):
            rouge_l = self.rouge.rouge_2(
                summary=generated_batch[i],
                references=reference_batch[i])
            rouge_l_points.append(rouge_l)
        return Variable(torch.cuda.FloatTensor(rouge_l_points))

    def convert_batch(self, generated_batch, reference_batch):
        converted_generated_batch = []
        converted_reference_batch = []
        for i in range(0, len(generated_batch)):
            full_generated_sequence_unpacked, full_reference_sequence_unpacked \
                = self.convert_single_sequence(generated_batch.data[i], reference_batch[i])
            converted_generated_batch.append(full_generated_sequence_unpacked)
            converted_reference_batch.append(full_reference_sequence_unpacked)
        return converted_generated_batch, converted_reference_batch

    def convert_single_sequence(self, generated_sequence, reference_sequence):
        full_generated_sequence_unpacked = get_sentence_from_tokens_and_clean_unked(generated_sequence,
                                                                                    self.vocabulary)
        full_reference_sequence_unpacked = get_sentence_from_tokens_and_clean_unked(reference_sequence,
                                                                                    self.vocabulary)
        return full_generated_sequence_unpacked, full_reference_sequence_unpacked


class RougeDiscriminator:
    def __init__(self, vocabulary):
        self.rouge = RougeCalculator(stopwords=False, lang="en", stemming=False)
        self.vocabulary = vocabulary

    def train(self, ground_truth, sequences):
        pass

    def convert_batch(self, generated_batch, reference_batch):
        converted_generated_batch = []
        converted_reference_batch = []
        for i in range(0, len(generated_batch)):
            full_generated_sequence_unpacked, full_reference_sequence_unpacked \
                = self.convert_single_sequence(generated_batch.data[i], reference_batch[i])
            converted_generated_batch.append(full_generated_sequence_unpacked)
            converted_reference_batch.append(full_reference_sequence_unpacked)
        return converted_generated_batch, converted_reference_batch

    def convert_single_sequence(self, generated_sequence, reference_sequence):
        full_generated_sequence_unpacked = get_sentence_from_tokens_and_clean_unked(generated_sequence,
                                                                                    self.vocabulary)
        full_reference_sequence_unpacked = get_sentence_from_tokens_and_clean_unked(reference_sequence,
                                                                                    self.vocabulary)
        return full_generated_sequence_unpacked, full_reference_sequence_unpacked

    def evaluate(self, generated_batch, reference_batch, extended_vocabs):
        before_transfer = time.time()
        generated_batch = generated_batch.cpu()
        timings[timings_var_discriminator_transfer] += time.time() - before_transfer
        generated_batch, reference_batch = self.convert_batch(generated_batch, reference_batch)
        rouge_l_points = []
        for i in range(0, len(generated_batch)):
            rouge_l = self.rouge.rouge_1(
                summary=generated_batch[i],
                references=reference_batch[i])
            rouge_l_points.append(rouge_l)
        rouge_reward = Variable(torch.cuda.FloatTensor(rouge_l_points))
        return rouge_reward, rouge_reward, rouge_reward
