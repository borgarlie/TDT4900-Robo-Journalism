import numpy as np

from utils.data_prep import *


class Generator:
    def __init__(self, encoder, decoder, encoder_optimizer, decoder_optimizer, mle_criterion,
                 batch_size, use_cuda, beta, generator_beta):
        self.encoder = encoder
        self.decoder = decoder
        self.encoder_optimizer = encoder_optimizer
        self.decoder_optimizer = decoder_optimizer
        self.mle_criterion = mle_criterion
        self.batch_size = batch_size
        self.use_cuda = use_cuda
        self.beta = beta
        self.generator_beta = generator_beta

    # discriminator is used to calculate reward
    # summary batch is used for MLE
    def train_on_batch(self, input_variable_batch, input_lengths, target_variable_batch, target_lengths, discriminator):

        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()

        max_target_length = max(target_lengths)
        encoder_outputs, encoder_hidden = self.encoder(input_variable_batch, input_lengths, None)

        encoder_hidden = concat_encoder_hidden_directions(encoder_hidden)
        # Multiple layers are currently removed for simplicity

        decoder_input = Variable(torch.LongTensor([SOS_token] * self.batch_size))
        decoder_input = decoder_input.cuda() if self.use_cuda else decoder_input
        decoder_hidden = encoder_hidden

        mle_loss = 0
        policy_loss = 0
        accumulated_sequence = None

        # Without teacher forcing: use its own predictions as the next input
        for di in range(max_target_length):
            decoder_output, decoder_hidden, decoder_attention = self.decoder(decoder_input, decoder_hidden,
                                                                             encoder_outputs, self.batch_size)
            topv, topi = decoder_output.data.topk(1)
            ni = topi  # next input, batch of top softmax
            decoder_input = Variable(ni)
            mle_loss += self.mle_criterion(decoder_output, target_variable_batch[di])

            ni_transposed = ni.transpose(0, 1)
            decoder_input_batch = Variable(ni_transposed)
            if di == 0:
                accumulated_sequence = decoder_input_batch
            else:
                accumulated_sequence = torch.cat((accumulated_sequence, decoder_input_batch), 0)

            # print(accumulated_sequence, flush=True)
            # calculate policy value
            policy_target = decoder_input.squeeze(1)
            current_policy_value = self.mle_criterion(decoder_output, policy_target)
            # TODO: Is it possible to use a max() function or something and keep the gradient here? might be cheaper

            # calculate policy loss using monte carlo search
            accumulated_reward = 0
            num_samples = 2
            for _ in range(num_samples):
                sample = self.generator_beta.generate_sequence(input_variable_batch, input_lengths, max_target_length,
                                                               accumulated_sequence)
                sample = sample.transpose(1, 0)
                accumulated_reward += discriminator.evaluate(sample)

                print(sample, flush=True)

            reward = accumulated_reward / num_samples
            policy_loss += reward * current_policy_value

            print(reward, flush=True)
            print(policy_loss, flush=True)

        exit()

        total_loss = self.beta * policy_loss + (1 - self.beta) * mle_loss
        total_loss.backward()

        self.encoder_optimizer.step()
        self.decoder_optimizer.step()

        return total_loss.data[0]

    # Used to create fake data samples to train the discriminator
    # Returned values as batched sentences as variables
    def create_samples(self, input_variable_batch, input_lengths, max_sample_length):

        self.encoder.eval()
        self.decoder.eval()

        encoder_outputs, encoder_hidden = self.encoder(input_variable_batch, input_lengths, None)
        encoder_hidden = concat_encoder_hidden_directions(encoder_hidden)
        # Multiple layers are currently removed for simplicity
        decoder_input = Variable(torch.LongTensor([SOS_token] * self.batch_size))
        decoder_input = decoder_input.cuda() if self.use_cuda else decoder_input
        decoder_hidden = encoder_hidden

        decoder_outputs = [[] for _ in range(0, self.batch_size)]

        # Without teacher forcing: use its own predictions as the next input
        for di in range(max_sample_length):
            decoder_output, decoder_hidden, decoder_attention = self.decoder(decoder_input, decoder_hidden,
                                                                             encoder_outputs, self.batch_size)
            topv, topi = decoder_output.data.topk(1)
            ni = topi  # next input, batch of top softmax scores
            decoder_input = Variable(torch.cuda.LongTensor(ni)) if self.use_cuda else Variable(torch.LongTensor(ni))

            decoder_output_data = ni.cpu().numpy()
            for batch_index in range(0, len(decoder_output_data)):
                decoder_outputs[batch_index].append(decoder_output_data[batch_index].item())

        decoder_output_variables = Variable(torch.LongTensor(decoder_outputs))
        if self.use_cuda:
            decoder_output_variables = decoder_output_variables.cuda()

        self.encoder.train()
        self.decoder.train()

        return decoder_output_variables

    def update_generator_beta_params(self):
        self.generator_beta.update_params(self.encoder, self.decoder)
