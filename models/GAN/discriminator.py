import torch
from torch import nn
from sumeval.metrics.rouge import RougeCalculator
from torch.autograd import Variable

from utils.data_prep import get_sentence_from_tokens_and_clean
from utils.logger import log_message


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


class RepetitionAvoidanceDiscriminator:
    def __init__(self):
        pass

    def train(self, ground_truth, sequences):
        pass

    def evaluate(self, sequence_batch):
        pass


class RougeDiscriminator:
    def __init__(self, vocabulary):
        self.rouge = RougeCalculator(stopwords=False, lang="en", stemming=False)
        self.vocabulary = vocabulary

    def train(self, ground_truth, sequences):
        pass

    def convert_batch(self, generated_batch, reference_batch, extended_vocabs):
        converted_generated_batch = []
        converted_reference_batch = []
        for i in range(0, len(generated_batch)):
            full_generated_sequence_unpacked, full_reference_sequence_unpacked \
                = self.convert_single_sequence(generated_batch.data[i], reference_batch[i], extended_vocabs[i])
            converted_generated_batch.append(full_generated_sequence_unpacked)
            converted_reference_batch.append(full_reference_sequence_unpacked)
        return converted_generated_batch, converted_reference_batch

    def convert_single_sequence(self, generated_sequence, reference_sequence, extended_vocab):
        full_generated_sequence_unpacked = get_sentence_from_tokens_and_clean(generated_sequence, self.vocabulary, extended_vocab)
        full_reference_sequence_unpacked = get_sentence_from_tokens_and_clean(reference_sequence, self.vocabulary, extended_vocab)
        return full_generated_sequence_unpacked, full_reference_sequence_unpacked

    def evaluate(self, generated_batch, reference_batch, extended_vocabs):
        generated_batch, reference_batch = self.convert_batch(generated_batch, reference_batch, extended_vocabs)
        # log_message(generated_batch[0])
        # log_message(reference_batch[0])

        rouge_l_points = []
        for i in range(0, len(generated_batch)):
            rouge_l = self.rouge.rouge_l(
                summary=generated_batch[i],
                references=reference_batch[i])
            rouge_l_points.append(rouge_l)

        # log_message(rouge_l_points)
        # exit()
        return Variable(torch.cuda.FloatTensor(rouge_l_points))


if __name__ == '__main__':
    pass
    # discriminator = RepetitionAvoidanceDiscriminator()

    # TARGET SENTENCE >>> british model yasmin le bon has posed on location in sri lanka . this is her third season as the face of monsoon . the mother-of-three has opened up on ageing in the fashion industry . clique magazine issue 3 is out now . <EOS>
    # GENERATED SENTENCE >>> yasmin le bon is the face of monsoon for a third season . she is the face of monsoon for a third season . she is yet to succumb to the surgeon 's knife . <EOS>
    #  SENTENCE >>> lesley emerson , 59 , was buried with her phone after dying in july 2011 . her family occasionally sent her texts as a way of dealing with their grief . but mrs emerson 's grandmother , sheri , received a haunting reply . mystery user sent a text to ms emerson , 22 , saying : ` i 'm watching over you ' the family say o2 had promised to put the phone number out of service . but instead the number was recycled , and passed on to a new customer . <EOS>
    # GENERATED SENTENCE >>> lesley emerson , 22 , died from bowel cancer in july 2011 . she was buried with her mobile phone when she died in 2011 . her family decided to bury her with her mobile phone . she was buried with her mobile phone and her family could text her . <EOS>
    # TARGET SENTENCE >>> catherine goins , 37 , is facing murder charges in the shooting death of natalia roberts , 30 , a mother of two from georgia . sheriff 's officials say goins lured roberts to an ex-boyfriend 's home under the pretense of wanting to give her some baby clothes . she then shot the mother in the back of the head and fled the scene with her three-week-old and three-year-old kids , police said . goins initially told police she heard a noise and shot an intruder in a hallway . <EOS>
    # GENERATED SENTENCE >>> catherine goins , 37 , is accused of shooting dead natalia roberts , 30 , last week . she was charged with murder for allegedly luring a young mother from georgia to her death . goins was a mother of two , last friday morning and lured her to the la fayette home . goins was a mother of two , last friday morning and lured her to the la fayette home . <EOS>
    #

    # reference_batch = []
    # reference_batch.append("british model yasmin le bon has posed on location in sri lanka . this is her third season as the face of monsoon . the mother-of-three has opened up on ageing in the fashion industry . clique magazine issue 3 is out now .")
    # reference_batch.append("lesley emerson , 59 , was buried with her phone after dying in july 2011 . her family occasionally sent her texts as a way of dealing with their grief . but mrs emerson 's grandmother , sheri , received a haunting reply . mystery user sent a text to ms emerson , 22 , saying : ` i 'm watching over you ' the family say o2 had promised to put the phone number out of service . but instead the number was recycled , and passed on to a new customer .")
    # reference_batch.append("catherine goins , 37 , is facing murder charges in the shooting death of natalia roberts , 30 , a mother of two from georgia . sheriff 's officials say goins lured roberts to an ex-boyfriend 's home under the pretense of wanting to give her some baby clothes . she then shot the mother in the back of the head and fled the scene with her three-week-old and three-year-old kids , police said . goins initially told police she heard a noise and shot an intruder in a hallway .")
    #
    # generated_batch = []
    # generated_batch.append("yasmin le bon is the face of monsoon for a third season . she is the face of monsoon for a third season . she is yet to succumb to the surgeon 's knife .")
    # generated_batch.append("lesley emerson , 22 , died from bowel cancer in july 2011 . she was buried with her mobile phone when she died in 2011 . her family decided to bury her with her mobile phone . she was buried with her mobile phone and her family could text her .")
    # generated_batch.append("catherine goins , 37 , is accused of shooting dead natalia roberts , 30 , last week . she was charged with murder for allegedly luring a young mother from georgia to her death . goins was a mother of two , last friday morning and lured her to the la fayette home . goins was a mother of two , last friday morning and lured her to the la fayette home .")
    #
    # discriminator = RougeDiscriminator()
    #
    # rouge_scores = discriminator.evaluate(generated_batch, reference_batch)
    # print(rouge_scores)
