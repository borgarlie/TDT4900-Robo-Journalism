from torch.distributions import Categorical

from utils.data_prep import *
from utils.logger import *
import time


class GeneratorBase:
    def __init__(self, vocabulary, encoder, decoder, encoder_optimizer, decoder_optimizer, mle_criterion,
                 batch_size, use_cuda, beta, num_monte_carlo_samples, sample_rate, negative_reward, use_trigram_check,
                 use_running_avg_baseline):
        self.vocabulary = vocabulary
        self.encoder = encoder
        self.decoder = decoder
        self.encoder_optimizer = encoder_optimizer
        self.decoder_optimizer = decoder_optimizer
        self.mle_criterion = mle_criterion
        self.batch_size = batch_size
        self.use_cuda = use_cuda
        self.beta = beta
        self.num_monte_carlo_samples = num_monte_carlo_samples
        self.updates = 0
        self.cumulative_reward = 0.0
        self.sample_rate = sample_rate
        self.allow_negative_rewards = negative_reward
        self.use_trigram_check = use_trigram_check
        self.use_running_avg_baseline = use_running_avg_baseline

    def get_teacher_forcing_mle(self, encoder_hidden, encoder_outputs, max_target_length, full_input_variable_batch,
                                full_target_variable_batch, target_variable):
        mle_loss = 0
        decoder_input = Variable(torch.LongTensor([SOS_token] * self.batch_size))
        decoder_input = decoder_input.cuda() if self.use_cuda else decoder_input
        decoder_hidden = encoder_hidden
        for di in range(max_target_length):
            decoder_output, decoder_hidden, decoder_attention \
                = self.decoder(decoder_input, decoder_hidden, encoder_outputs, full_input_variable_batch,
                               self.batch_size)
            log_output = torch.log(decoder_output.clamp(min=1e-8))
            mle_loss += self.mle_criterion(log_output, full_target_variable_batch[di])
            decoder_input = target_variable[di]
        return mle_loss

    def get_argmax_baseline(self, encoder_hidden, encoder_outputs, max_target_length, full_input_variable_batch,
                            discriminator, full_target_variable_batch_2, extended_vocabs):
        decoder_input = Variable(torch.LongTensor([SOS_token] * self.batch_size))
        decoder_input = decoder_input.cuda() if self.use_cuda else decoder_input
        decoder_hidden = encoder_hidden
        accumulated_sequence = None
        for di in range(max_target_length):
            decoder_output, decoder_hidden, decoder_attention \
                = self.decoder(decoder_input, decoder_hidden, encoder_outputs, full_input_variable_batch,
                               self.batch_size)

            if self.use_trigram_check:
                # Using topk > 1 to compensate for possible trigram overlaps
                topv, topi = decoder_output.data.topk(100)
                next_words = []
                for batch in range(0, self.batch_size):
                    topk = 0
                    while 1:
                        next_word = topi[batch][topk]
                        if di > 2 and has_trigram(accumulated_sequence[batch].data, next_word):
                            topk += 1
                            continue
                        break
                    next_words.append(next_word)
                ni = torch.cuda.LongTensor(next_words).unsqueeze(1)
            else:
                topv, topi = decoder_output.data.topk(1)
                ni = topi  # next input, batch of top softmax

            if accumulated_sequence is None:
                accumulated_sequence = Variable(ni).clone()
            else:
                accumulated_sequence = torch.cat((accumulated_sequence, Variable(ni)), 1)

            for token_index in range(0, len(ni)):
                if ni[token_index][0] >= self.vocabulary.n_words:
                    ni[token_index][0] = UNK_token
            decoder_input = Variable(ni)

        baseline = discriminator.evaluate(accumulated_sequence, full_target_variable_batch_2, extended_vocabs)
        return baseline

    # Used to create fake data samples to train the discriminator
    # Returned values as batched sentences as variables
    def create_samples(self, input_variable_batch, full_input_variable_batch, input_lengths, max_sample_length,
                       pad_length, sample=False):

        self.encoder.eval()
        self.decoder.eval()

        encoder_outputs, encoder_hidden = self.encoder(input_variable_batch, input_lengths, None)
        encoder_hidden = concat_encoder_hidden_directions(encoder_hidden)
        # Multiple layers are currently removed for simplicity
        decoder_input = Variable(torch.LongTensor([SOS_token] * self.batch_size))
        decoder_input = decoder_input.cuda() if self.use_cuda else decoder_input
        decoder_hidden = encoder_hidden

        decoder_outputs = [[] for _ in range(0, self.batch_size)]
        create_fake_sample_break_early = False

        # Without teacher forcing: use its own predictions as the next input
        for di in range(max_sample_length):

            create_fake_inner_time_start = time.time()
            decoder_output, decoder_hidden, decoder_attention \
                = self.decoder(decoder_input, decoder_hidden, encoder_outputs, full_input_variable_batch,
                               self.batch_size)
            timings[timings_var_create_fake_inner] += (time.time() - create_fake_inner_time_start)

            if sample:
                m = Categorical(decoder_output)
                ni = m.sample()
                for token_index in range(0, len(ni)):
                    if ni[token_index].data[0] >= self.vocabulary.n_words:
                        ni[token_index].data[0] = UNK_token
                decoder_input = ni.unsqueeze(1)
                ni = ni.unsqueeze(1).data
            else:
                topv, topi = decoder_output.data.topk(1)
                ni = topi  # next input, batch of top softmax scores
                for token_index in range(0, len(ni)):
                    if ni[token_index][0] >= self.vocabulary.n_words:
                        ni[token_index][0] = UNK_token
                decoder_input = Variable(ni)

            decoder_output_data = ni.cpu().numpy()
            for batch_index in range(0, len(decoder_output_data)):
                decoder_outputs[batch_index].append(decoder_output_data[batch_index].item())

            if is_whole_batch_pad_or_eos(ni):
                decode_breakings[decode_breaking_fake_sampling] += di
                create_fake_sample_break_early = True
                break

        if not create_fake_sample_break_early:
            decode_breakings[decode_breaking_fake_sampling] += max_sample_length - 1

        decoder_outputs_padded = [pad_seq(s, pad_length) for s in decoder_outputs]
        decoder_output_variables = Variable(torch.LongTensor(decoder_outputs_padded))
        if self.use_cuda:
            decoder_output_variables = decoder_output_variables.cuda()

        self.encoder.train()
        self.decoder.train()

        return decoder_output_variables
